from collections import defaultdict
from pathlib import Path

# import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from tqdm import trange

from pof.diffrax import solve_diffrax
from pof.initialization import get_initial_trajectory, taylor_mode_init, uncertain_init
from pof.ivp import logistic, lotkavolterra
from pof.observations import NonlinearModel, linearize, uncertain_linearize
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth.filter import _sqrt_update, _sqrt_predict
from pof.transitions import (
    IWP,
    TransitionModel,
    get_transition_model,
    nordsieck_preconditioner,
    preconditioned_discretize,
    projection_matrix,
)
from pof.utils import MVNSqrt

fs = linear_filtsmooth
if jax.lib.xla_bridge.get_backend().platform == "gpu":
    fs = jax.jit(linear_filtsmooth)

lom_uncertain = jax.jit(
    jax.vmap(uncertain_linearize, in_axes=[None, 0]), static_argnums=(0,)
)
lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))


def set_up(ivp, dt, order):
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

    iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))

    x0 = taylor_mode_init(ivp.f, ivp.y0, order)
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
    }


def eks_step(dtm, om, x0, traj, uncertain):
    if uncertain:
        dom = lom_uncertain(om, traj)
    else:
        dom = lom(om, traj)
    return fs(x0, dtm, dom)


def calibrate(out, dtm, ssq):
    sigma = jnp.sqrt(ssq)
    out = MVNSqrt(out.mean, sigma * out.chol)
    dtm = TransitionModel(dtm.F, sigma * dtm.QL)
    return out, dtm


def ieks_iterator(dtm, om, x0, init_traj, maxiter=250):
    uncertain = True
    bar = trange(maxiter)

    out, nll, obj, ssq = eks_step(dtm, om, x0, init_traj, uncertain=uncertain)
    out, dtm = calibrate(out, dtm, ssq)
    bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} ssq={ssq:.4e}]")
    bar.update(1)
    yield out, nll, obj
    for _ in bar:
        nll_old, obj_old, mean_old = nll, obj, out.mean
        out, nll, obj, ssq = eks_step(
            dtm, om, x0, jax.tree_map(lambda l: l[1:], out), uncertain=uncertain
        )
        out, dtm = calibrate(out, dtm, ssq)

        rel_obj_change = (obj - obj_old) / obj_old * 100
        rel_nll_change = (nll - nll_old) / nll_old * 100
        threshold = 10
        if (
            uncertain
            and jnp.abs(rel_obj_change) < threshold
            and jnp.abs(rel_nll_change) < threshold
        ):
            uncertain = False

        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} ssq={ssq:.4e}]")
        bar.write(f"OBJ: {rel_obj_change:.1}%; NLL: {rel_nll_change:.1}%")

        yield out, nll, obj
        if (
            (
                jnp.isclose(nll_old, nll)
                and jnp.isclose(obj_old, obj)
                # and jnp.isclose(mean_old, out.mean).all()
            )
            or jnp.isnan(nll)
            or jnp.isnan(obj)
        ):
            bar.close()
            break


def evaluate(init_traj, setup, ys_true):
    results = defaultdict(lambda: [])

    mse = get_mse(init_traj.mean, setup["E0"], ys_true[1:])
    results["nll"].append(np.inf)
    results["obj"].append(np.inf)
    results["mse"].append(mse.item())

    ieks_iter = ieks_iterator(setup["dtm"], setup["om"], setup["x0"], init_traj)
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
    return jax.vmap(lambda e: jnp.sqrt(jnp.mean(e**2)), in_axes=0)(es).mean()


########################################################################################
# Init options
########################################################################################
def precondition(setup, raw_states):
    precondition = lambda v: setup["PI"] @ v
    preconditioned_states = jax.tree_map(jax.vmap(precondition), raw_states)
    return preconditioned_states


def constant_init(setup):
    ivp = setup["ivp"]
    raw_traj = get_initial_trajectory(
        ivp.y0, ivp.f, order=setup["order"], N=len(setup["ts"]) - 1
    )
    return precondition(setup, raw_traj)


def _prior_extrapolation(x0, iwp, ts):
    def predict(t):
        tm = get_transition_model(iwp, t)
        return _sqrt_predict(tm.F, tm.QL, x0)

    return jax.vmap(predict)(ts)


def prior_init(setup):
    ivp = setup["ivp"]
    assert ivp.t0 == 0
    iwp = IWP(num_derivatives=setup["order"], wiener_process_dimension=ivp.y0.shape[0])
    x0_raw = jax.tree_map(lambda l: setup["P"] @ l, setup["x0"])
    states_raw = _prior_extrapolation(x0_raw, iwp, setup["ts"][1:])
    return precondition(setup, states_raw)


def updated_prior_init(setup):
    ivp = setup["ivp"]
    assert ivp.t0 == 0
    iwp = IWP(num_derivatives=setup["order"], wiener_process_dimension=ivp.y0.shape[0])
    x0_raw = jax.tree_map(lambda l: setup["P"] @ l, setup["x0"])
    states_raw = _prior_extrapolation(x0_raw, iwp, setup["ts"][1:])
    states = precondition(setup, states_raw)

    def update(x, om):
        H, b, cholR = linearize(om, x)
        return _sqrt_update(H, cholR, b, x)[0]

    states = jax.vmap(update, in_axes=[0, None])(states, setup["om"])
    return states


def coarse_solver_init(setup, dt):
    sol_init = solve_diffrax(ivp, dt=dt)
    ys = jax.vmap(sol_init.evaluate)(setup["ts"][1:])
    raw_traj = get_initial_trajectory(ys, ivp.f, order=setup["order"])
    return precondition(setup, raw_traj)


INITS = (
    ("constant", constant_init),
    ("prior", prior_init),
    ("updated_prior", updated_prior_init),
    # ("coarse_solver_2p-0", lambda s: coarse_solver_init(s, 2.0**-0)),
    ("coarse_solver_2p-1", lambda s: coarse_solver_init(s, 2.0**-1)),
    ("coarse_solver_2p-2", lambda s: coarse_solver_init(s, 2.0**-2)),
    ("coarse_solver_2p-3", lambda s: coarse_solver_init(s, 2.0**-3)),
    ("coarse_solver_2p-4", lambda s: coarse_solver_init(s, 2.0**-4)),
)
INIT_NAMES = [i[0] for i in INITS]


########################################################################################
# Evaluation
########################################################################################
# Setup
ORDER = 4

PROBS = {
    # "logistic": (
    #     logistic(),
    #     (
    #         (1e-0, "1e-0"),
    #         (1e-1, "1e-1"),
    #         (1e-2, "1e-2"),
    #         (1e-3, "1e-3"),
    #     ),
    # ),
    "lotkavolterra": (
        lotkavolterra(),
        (
            (1e-0, "1e-0"),
            (1e-1, "1e-1"),
            (1e-2, "1e-2"),
            (1e-3, "1e-3"),
            # (1e-4, "1e-4"),
        ),
    ),
}

for (probname, (ivp, dts)) in PROBS.items():
    print(
        f"""
        ###############################################
        # Prob: {probname}
        ###############################################
        """
    )

    sol_true = solve_diffrax(ivp, atol=1e-20, rtol=1e-20)
    for dt, dt_str in dts:
        setup = set_up(ivp, dt, order=ORDER)
        ys_true = jax.vmap(sol_true.evaluate)(setup["ts"])

        # Eval each INIT
        dfs = {}
        for n, init in INITS:
            print(f"initialization: {n}")
            traj = init(setup)
            res, k = evaluate(traj, setup, ys_true)
            df = pd.DataFrame(res)
            dfs[n] = df

        # Merge dataframes
        n = INIT_NAMES[0]
        df = dfs[n]
        add_suffix = lambda df, n: df.rename(columns=lambda c: f"{c}_{n}")
        df = add_suffix(df, n)
        for i in range(1, len(INITS)):
            n = INIT_NAMES[i]
            df = pd.merge(
                df, add_suffix(dfs[n], n), "outer", left_index=True, right_index=True
            )
            # Save
        path = Path("experiments/2_init_comparison/data")
        filename = f"prob={probname}_dt={dt_str}_order={ORDER}_dev.csv"
        df.to_csv(path / filename)
        print(f"Saved to {path / filename}")
