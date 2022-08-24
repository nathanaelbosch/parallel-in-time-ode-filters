from collections import defaultdict
import itertools
from pathlib import Path
import diffrax

# import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import trange

from pof.diffrax import solve_diffrax

import pof.initialization as init
from pof.ivp import *
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.transitions import *
from pof.utils import MVNSqrt, tria
import pof.iterators

fs = linear_filtsmooth
if jax.lib.xla_bridge.get_backend().platform == "gpu":
    fs = jax.jit(linear_filtsmooth)

lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))
lom_reg = jax.jit(
    jax.vmap(linearize_regularized, in_axes=[None, 0, None]), static_argnums=(0,)
)

MAXITER = 500


def set_up(ivp, dt, order):
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

    iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))

    x0 = init.taylor_mode_init(ivp.f, ivp.y0, order)
    # x0 = uncertain_init(ivp.f, ivp.y0, order, var=1e0)
    x0 = jax.tree_map(lambda x: PI @ x, x0)

    return {
        "ts": ts,
        "dtm": dtm,
        "om": om,
        "x0": x0,
        "E0": E0,
        "ivp": ivp,
        "P": P,
        "PI": PI,
        "order": order,
        "iwp": iwp,
    }


def calibrate(out, dtm, ssq):
    sigma = jnp.sqrt(ssq)
    out = MVNSqrt(out.mean, sigma * out.chol)
    dtm = TransitionModel(dtm.F, sigma * dtm.QL)
    return out, dtm


def ieks_iterator(dtm, om, x0, init_traj, maxiter=MAXITER):
    bar = trange(maxiter)
    iterator = pof.iterators.ieks_iterator(dtm, om, x0, init_traj)
    for _, (out, nll, obj) in zip(bar, iterator):
        yield out, nll, obj
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e}]")


def lm_ieks_iterator(dtm, om, x0, init_traj, maxiter=MAXITER):
    bar = trange(maxiter)
    iterator = pof.iterators.lm_ieks_iterator(dtm, om, x0, init_traj)
    for _, (out, nll, obj, reg) in zip(bar, iterator):
        yield out, nll, obj
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} reg={reg:.4e}]")


def qpm_ieks_iterator(dtm, om, x0, init_traj, maxiter=MAXITER):
    bar = trange(maxiter)
    iterator = pof.iterators.qpm_ieks_iterator(
        dtm,
        om,
        x0,
        init_traj,
        #
        # WIP for VdP:
        # reg_start=1e20,
        # reg_final=1e-20,
        # steps=100,
        # tau_start=1e5,
        # tau_final=1e-2,
        #
        # Best so far:
        reg_start=1e10,
        reg_final=1e-10,
        steps=20,
        tau_start=1e5,
        tau_final=1e-5,
    )
    for _, (out, nll, obj, reg) in zip(bar, iterator):
        yield out, nll, obj
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} reg={reg:.4e}]")


def evaluate(init_traj, setup, ys_true, opt="ieks"):
    results = defaultdict(lambda: [])

    mse = get_mse(init_traj.mean, setup["E0"], ys_true[1:])
    results["nll"].append(np.inf)
    results["obj"].append(np.inf)
    results["mse"].append(mse.item())

    _iterator = {
        "ieks": ieks_iterator,
        "lm": lm_ieks_iterator,
        "qpm": qpm_ieks_iterator,
    }[opt]

    ieks_iter = _iterator(setup["dtm"], setup["om"], setup["x0"], init_traj)
    for (k, (out, nll, obj)) in enumerate(ieks_iter):
        mse = get_mse(out.mean, setup["E0"], ys_true)
        results["nll"].append(nll.item())
        results["obj"].append(obj.item())
        results["mse"].append(mse.item())
    return results, k


@jax.jit
def get_mse(out, E0, ys_true):
    ys = jnp.dot(E0, out.T).T
    es = ys - ys_true
    return jax.vmap(lambda e: jnp.sqrt(jnp.mean(e ** 2)), in_axes=0)(es).mean()


########################################################################################
# Init options
########################################################################################
def _precondition(setup, raw_states):
    precondition = lambda v: setup["PI"] @ v
    preconditioned_states = jax.tree_map(jax.vmap(precondition), raw_states)
    return preconditioned_states


def constant_init(setup):
    ivp = setup["ivp"]
    raw_traj = init.constant_init(
        y0=ivp.y0, f=ivp.f, order=setup["order"], ts=setup["ts"][1:]
    )
    return _precondition(setup, raw_traj)


def prior_init(setup):
    return init.prior_init(x0=setup["x0"], dtm=setup["dtm"])


def updated_prior_init(setup):
    return init.updated_prior_init(x0=setup["x0"], dtm=setup["dtm"], om=setup["om"])


def coarse_solver_init(setup):
    ivp, order, ts = setup["ivp"], setup["order"], setup["ts"]
    raw_traj = init.coarse_rk_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts[1:], fact=100)
    return _precondition(setup, raw_traj)


def coarse_ekf_init(setup):
    ivp, order, ts = setup["ivp"], setup["order"], setup["ts"]
    raw_traj = init.coarse_ekf_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts[1:])
    return _precondition(setup, raw_traj)


def merge_dataframes(dfs, names):
    n = names[0]
    df = dfs[n]
    add_suffix = lambda df, n: df.rename(columns=lambda c: f"{c}_{n}")
    df = add_suffix(df, n)
    for i in range(1, len(names)):
        n = names[i]
        df = pd.merge(
            df,
            add_suffix(dfs[n], n),
            "outer",
            left_index=True,
            right_index=True,
        )
    return df


def run_exp(*, ivp, dts, order, probname, inits, opt="ieks", sol_true):
    for dt, dt_str in dts:
        setup = set_up(ivp, dt, order=order)
        ys_true = jax.vmap(sol_true.evaluate)(setup["ts"])

        dfs = {}
        for n, initf in inits:
            print(f"initialization: {n}; optimizer: {opt}")
            traj = initf(setup)
            res, k = evaluate(traj, setup, ys_true, opt=opt)
            df = pd.DataFrame(res)
            dfs[n] = df

        # Merge dataframes
        df = merge_dataframes(dfs, [i[0] for i in inits])

        # Save
        path = Path("experiments/2_init_comparison/data")
        filename = f"prob={probname}_dt={dt_str}_order={order}_opt={opt}.csv"
        df.to_csv(path / filename)
        print(f"Saved to {path / filename}")


if __name__ == "__main__":
    INITS = (
        # ("constant", constant_init),
        ("prior", prior_init),
        # ("updated_prior", updated_prior_init),
        # ("coarse_dopri5", coarse_solver_init),
        ("coarse_ekf", coarse_ekf_init),
    )
    ORDERS = (
        # 1,
        2,
        3,
        5,
    )
    DTS = (
        # (1e-0, "1e-0"),
        # (1e-1, "1e-1"),
        (1e-2, "1e-2"),
        (1e-3, "1e-3"),
        # (1e-4, "1e-4"),
    )
    OPTS = ("ieks", "qpm")
    for IVP, PROBNAME in (
        (lotkavolterra(), "lotkavolterra"),
        (vanderpol(), "vanderpol"),
    ):
        sol_true = solve_diffrax(IVP.f, IVP.y0, IVP.t_span, atol=1e-20, rtol=1e-20)

        for (order, opt) in itertools.product(ORDERS, OPTS):
            run_exp(
                ivp=IVP,
                dts=DTS,
                order=order,
                probname=PROBNAME,
                opt=opt,
                inits=INITS,
                sol_true=sol_true,
            )
