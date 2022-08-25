import time
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", message=r"jax.tree_leaves", category=FutureWarning)

import jax

jax.config.update("jax_enable_x64", True)

import tqdm
import diffrax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import probnum
from scipy.integrate import solve_ivp

from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import constant_init, taylor_mode_init, coarse_ekf_init
from pof.ivp import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth
from pof.solver import make_continuous_models
from pof.transitions import *
import pof.iterators
import pof.diffrax


ORDER = 4
SETUPS = (
    # {
    #     "ivp": logistic(),
    #     "ivp_name": "logistic",
    #     "dts": 2.0 ** -np.arange(0, 5),
    #     # "dts": 2.0 ** -np.arange(0, 10),
    #     "coarse_N": 10,
    # },
    # {
    #     "ivp": lotkavolterra(),
    #     "ivp_name": "lotkavolterra",
    #     "dts": 2.0 ** -np.arange(3, 7),
    #     # "dts": 2.0 ** -np.arange(3, 13),
    #     "coarse_N": 50,
    # },
    {
        "ivp": vanderpol,
        "ivp_name": "vanderpol",
        # "dts": 2.0 ** -np.arange(3, 7),
        "dt": 0.05,
        "tmax_range": 2.0 ** jnp.arange(0, 8),
        # "coarse_N": 50,
    },
    {
        "ivp": fitzhughnagumo,
        "ivp_name": "fitzhughnagumo",
        # "dts": 2.0 ** -np.arange(3, 7),
        "dt": 0.05,
        "tmax_range": 2.0 ** jnp.arange(0, 8),
        # "coarse_N": 50,
    },
)


def solve_parallel_eks(f, y0, ts, order, coarse_N=10):
    out, info = pof.solver.solve(
        f=f,
        y0=y0,
        ts=ts,
        order=order,
        init="constant",
        # coarse_N=coarse_N,
    )
    return out.mean, info, 0


def solve_sequential_eks(f, y0, ts, order):
    out, ts, info = pof.solver.sequential_eks_solve(
        f, y0, ts, order, return_full_states=False
    )
    return out.mean, info, 0


def solve_diffrax(f, y0, ts, solver):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: f(t, y)),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=diffrax.SaveAt(steps=True, t0=True, dense=True),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=2 * ts.shape[0],
        throw=False,
    )
    return sol.ys, sol.result


def remove_infs(ys):
    idxs = jnp.isfinite(ys).all(axis=1)
    return ys[idxs]


def solve_scipy(f, y0, ts, solver):
    dt = ts[1] - ts[0]
    tspan = (ts[0], ts[-1])
    sol = solve_ivp(f, tspan, y0, method=solver, max_step=dt)
    return sol.y.T, sol.status


def solve_probnum(f, y0, ts):
    dt = ts[1] - ts[0]
    sol = probnum.diffeq.probsolve_ivp(
        f,
        ts[0],
        ts[-1],
        y0,
        adaptive=False,
        step=dt,
        diffusion_model="constant",
        algo_order=ORDER,
    )
    return sol.states.mean, 0


def timeit(f, N):
    times = []
    for _ in range(N):
        start = time.time()
        f()
        t = time.time() - start
        times.append(t)
    return min(times)


def benchmark(methods, get_ivp, tmax_range, dt, N=3):
    results = defaultdict(lambda: [])

    bar = tqdm.tqdm(tmax_range, desc="tmax range")
    for tmax in bar:
        ivp = get_ivp(tmax=tmax)
        truesol = get_truesol(ivp)
        bar.write(f"tmax={tmax}, dt={dt}")
        results["tmax"].append(tmax)
        ts = jnp.arange(ivp.t0, tmax + dt, dt)
        bar2 = tqdm.tqdm(methods.items(), desc="Methods")
        for k, v in bar2:
            bar2.write(f"{k} start")
            f = lambda: v(ts)
            out, *_, status = f()
            if status != 0:
                t = np.nan
            else:
                t = timeit(f, N)
            results[f"{k}_time"].append(t)

            if jnp.isinf(out[-1]).all():
                out = remove_infs(out)

            rmse = jnp.sqrt(jnp.mean((out[-1] - truesol.ys[-1]) ** 2))
            results[f"{k}_err"].append(rmse)

            bar2.write(f"{k} results: time={t}; rmse={rmse}\n")
    return results


def block_and_return_state(f):
    def f2(*args, **kwargs):
        a, *b = f(*args, **kwargs)
        a.block_until_ready()
        return a, *b

    return f2


def get_truesol(ivp):
    truesol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: ivp.f(t, y)),
        diffrax.Dopri5(),
        t0=ivp.t0,
        t1=ivp.tmax,
        y0=ivp.y0,
        dt0=1e-3,
        stepsize_controller=diffrax.PIDController(rtol=1e-7, atol=1e-9),
        max_steps=100_000,
    )
    return truesol


for setup in tqdm.tqdm(SETUPS, desc="Setup"):
    ivp, name = setup["ivp"](), setup["ivp_name"]
    print(f"ivp={name}")
    f, y0 = ivp.f, ivp.y0

    peks = jax.jit(lambda ts: solve_parallel_eks(f, y0, ts, order=ORDER))
    # peks = jax.jit(lambda ts: pof.solver.solve(f=f, y0=y0, ts=ts, order=ORDER))
    seks = jax.jit(lambda ts: solve_sequential_eks(f, y0, ts, order=ORDER))
    dp5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=diffrax.Dopri5()))
    Kv5 = diffrax.Kvaerno5(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    kv5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=Kv5))
    rk45 = lambda ts: solve_scipy(f, y0, ts, "RK45")
    lsoda = lambda ts: solve_scipy(f, y0, ts, "LSODA")
    methods = {
        "pEKS": block_and_return_state(peks),
        "sEKS": block_and_return_state(seks),
        "dp5": block_and_return_state(dp5),
        "kv5": block_and_return_state(kv5),
        # "rk45": rk45,
        "lsoda": lsoda,
        # "probnumek0": lambda ts: solve_probnum(f, y0, ts)[1],
    }

    res = benchmark(methods, setup["ivp"], setup["tmax_range"], dt=setup["dt"], N=5)

    df = pd.DataFrame(res)
    df["N"] = (setup["tmax_range"] - ivp.t0) / setup["dt"]

    filename = f"experiments/2_solver_scaling/{name}.csv"
    df.to_csv(filename)
    print(f"Saved file to {filename}")
