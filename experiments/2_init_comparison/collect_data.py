import time
from collections import defaultdict
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import diffrax
import jax
import jax.numpy as jnp
from scipy.integrate import solve_ivp
import probnum

import pof
from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import get_initial_trajectory, get_x0
from pof.ivp import logistic, lotkavolterra
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth
from pof.solver import make_continuous_models
from pof.diffrax import solve_diffrax
from pof.transitions import projection_matrix, IWP, get_transition_model
from pof.utils import MVNSqrt
from pof.observations import linearize
from pof.sequential_filtsmooth.filter import _sqrt_update

ORDER = 2

fs = linear_filtsmooth
if jax.lib.xla_bridge.get_backend().platform == "gpu":
    fs = jax.jit(linear_filtsmooth)
lom = jax.jit(linearize_observation_model, static_argnums=(0,))


def set_up(ivp, dt, order):
    iwp, om = make_continuous_models(ivp.f, ivp.y0, order)
    E0 = projection_matrix(iwp, 0)
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    dtm = discretize_transitions(iwp, ts)
    x0 = get_x0(ivp.f, ivp.y0, order)
    return {"ts": ts, "dtm": dtm, "om": om, "x0": x0, "E0": E0, "ivp": ivp}


def eks_step(dtm, om, x0, traj):
    dom = lom(om, traj)
    return fs(x0, dtm, dom)


def ieks_iterator(dtm, om, x0, init_traj):
    nll_old, obj_old = 0, 0
    out, nll, obj = eks_step(dtm, om, x0, init_traj)
    yield out, nll, obj
    while not (
        (jnp.isclose(nll_old, nll) and jnp.isclose(obj_old, obj))
        or jnp.isnan(nll)
        or jnp.isnan(obj)
    ):
        nll_old, obj_old = nll, obj
        out, nll, obj = eks_step(dtm, om, x0, out.mean[1:])
        yield out, nll, obj


def evaluate(init_traj, setup, sol_true, maxiter=100):
    results = defaultdict(lambda: [])

    mse = get_mse(init_traj, setup["E0"], setup["ts"][1:], sol_true)
    results["nll"].append(np.inf)
    results["obj"].append(np.inf)
    results["mse"].append(mse.item())

    ieks_iter = ieks_iterator(setup["dtm"], setup["om"], setup["x0"], init_traj)
    for (k, (out, nll, obj)) in enumerate(ieks_iter):
        print(k)
        mse = get_mse(out.mean, setup["E0"], setup["ts"], sol_true)
        results["nll"].append(nll.item())
        results["obj"].append(obj.item())
        results["mse"].append(mse.item())
        if k > maxiter:
            break
    return results, k


def get_mse(out, E0, ts, sol_true):
    ys = jnp.dot(E0, out.T).T
    es = ys - jax.vmap(sol_true.evaluate)(ts)
    return jax.vmap(lambda e: jnp.sqrt(jnp.mean(e**2)), in_axes=0)(es).mean()


########################################################################################
# Init options
########################################################################################
def constant_init(setup):
    ivp = setup["ivp"]
    return get_initial_trajectory(ivp.y0, ivp.f, order=ORDER, N=len(setup["ts"]) - 1)


def _prior_extrapolation(x0, iwp, ts):
    def predict(t):
        tm = get_transition_model(iwp, t)
        return MVNSqrt(tm.F @ x0.mean, tm.F @ x0.chol + tm.QL)

    return jax.vmap(predict)(ts)


def prior_init(setup):
    ivp = setup["ivp"]
    assert ivp.t0 == 0
    iwp = IWP(num_derivatives=ORDER, wiener_process_dimension=ivp.y0.shape[0])
    return _prior_extrapolation(setup["x0"], iwp, setup["ts"][1:]).mean


def updated_prior_init(setup):
    ivp = setup["ivp"]
    assert ivp.t0 == 0
    iwp = IWP(num_derivatives=ORDER, wiener_process_dimension=ivp.y0.shape[0])
    states = _prior_extrapolation(setup["x0"], iwp, setup["ts"][1:])

    om = setup["om"]

    def update(x, om):
        H, b, cholR = linearize(om, x.mean)
        return _sqrt_update(H, cholR, b, x)[0]

    states = jax.vmap(update, in_axes=[0, None])(states, om)
    return states.mean


def coarse_solver_init(setup, dt):
    sol_init = solve_diffrax(ivp, dt=dt)
    ys = jax.vmap(sol_init.evaluate)(setup["ts"][1:])
    return get_initial_trajectory(ys, ivp.f, order=ORDER)


INITS = (
    ("constant", constant_init),
    ("prior", prior_init),
    ("updated_prior", updated_prior_init),
    ("coarse_solver_2p-0", lambda s: coarse_solver_init(s, 2.0**-0)),
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
ivp, probname = lotkavolterra(), "lotkavolterra"
dts = ((1e-1, "1e-1"), (1e-2, "1e-2"), (1e-3, "1e-3"), (1e-4, "1e-4"))
# ivp, probname = logistic(), "logistic"
# dts = ((1e-0, "1e-0"), (1e-1, "1e-1"), (1e-2, "1e-2"), (1e-3, "1e-3"))

sol_true = solve_diffrax(ivp, atol=1e-20, rtol=1e-20)
for dt, dt_str in dts:
    setup = set_up(ivp, dt, order=ORDER)

    # Eval each INIT
    dfs = {}
    for n, init in INITS:
        print(f"initialization: {n}")
        traj = init(setup)
        res, k = evaluate(traj, setup, sol_true)
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
    df.to_csv(f"experiments/2_init_comparison/{probname}_{dt_str}_dev.csv")
