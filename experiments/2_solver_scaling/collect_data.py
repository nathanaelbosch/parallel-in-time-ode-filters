import time
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", message=r"jax.tree_leaves", category=FutureWarning)


import tqdm
import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import probnum
from scipy.integrate import solve_ivp

from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import constant_init, taylor_mode_init, coarse_ekf_init
from pof.ivp import logistic, lotkavolterra
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth
from pof.solver import make_continuous_models
import pof.iterators


DTS = 2.0 ** -np.arange(0, 13)
# DTS = 2.0 ** -np.arange(0, 4)
ORDER = 4


def solve_parallel_eks(f, y0, ts, order, coarse_N=10):
    out, info = pof.solver.solve(f=f, y0=y0, ts=ts, order=order, coarse_N=coarse_N)
    return out.mean, 0


def solve_sequential_eks(f, y0, ts, order):
    iwp, om = make_continuous_models(f, y0, order)
    dtm = discretize_transitions(iwp, ts)
    x0 = taylor_mode_init(f, y0, order)
    traj = constant_init(y0=y0, f=f, order=order, ts=ts)
    out, _ = filtsmooth(x0, dtm, om)
    return out.mean, 0


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


def solve_scipy(f, y0, ts, solver):
    dt = ts[1] - ts[0]
    tspan = (ts[0], ts[-1])
    sol = solve_ivp(f, tspan, y0, method=solver, max_step=dt)
    return sol.y, sol.status


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


def benchmark(methods, N=3):
    results = defaultdict(lambda: [])

    bar = tqdm.tqdm(DTS, desc="DTs")
    for dt in bar:
        bar.write(f"dt={dt}")
        results["dt"].append(dt)
        ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
        bar2 = tqdm.tqdm(methods.items(), desc="Methods")
        for k, v in bar2:
            bar2.write(f"method={k}\n")
            f = lambda: v(ts)
            status = f()
            if status != 0:
                t = np.nan
            else:
                t = timeit(f, N)
            bar2.write(f"t={t}\n")
            results[k].append(t)
    return results


def block_and_return_state(f):
    def f2(*args, **kwargs):
        a, b = f(*args, **kwargs)
        a.block_until_ready()
        return b

    return f2


for ivp, name in tqdm.tqdm(
    (
        (logistic(), "logistic"),
        (lotkavolterra(), "lotkavolterra"),
    ),
    desc="IVP",
):
    print(f"ivp={name}")
    f, y0 = ivp.f, ivp.y0

    peks = jax.jit(lambda ts: solve_parallel_eks(f, y0, ts, order=ORDER))
    # peks = jax.jit(lambda ts: pof.solver.solve(f=f, y0=y0, ts=ts, order=ORDER))
    seks = jax.jit(lambda ts: solve_sequential_eks(f, y0, ts, order=ORDER))
    dp5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=diffrax.Dopri5()))
    Kv5 = diffrax.Kvaerno5(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    kv5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=Kv5))
    rk45 = lambda ts: solve_scipy(f, y0, ts, "RK45")[1]
    lsoda = lambda ts: solve_scipy(f, y0, ts, "LSODA")[1]
    methods = {
        "pEKS": block_and_return_state(peks),
        "sEKS": block_and_return_state(seks),
        "dp5": block_and_return_state(dp5),
        "kv5": block_and_return_state(kv5),
        "rk45": rk45,
        "lsoda": lsoda,
        # "probnumek0": lambda ts: solve_probnum(f, y0, ts)[1],
    }

    res = benchmark(methods, N=5)

    df = pd.DataFrame(res)
    df["N"] = (ivp.tmax - ivp.t0) / df.dt

    filename = f"experiments/2_solver_scaling/{name}.csv"
    df.to_csv(filename)
    print(f"Saved file to {filename}")
