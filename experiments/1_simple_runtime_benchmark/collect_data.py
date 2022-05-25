import time
from collections import defaultdict

import diffrax
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import probnum
from scipy.integrate import solve_ivp

from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import get_initial_trajectory, taylor_mode_init
from pof.ivp import logistic, lotkavolterra
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth
from pof.solver import make_continuous_models


def parallel_eks(f, y0, ts, order):
    iwp, om = make_continuous_models(f, y0, order)
    dtm = discretize_transitions(iwp, ts)
    x0 = taylor_mode_init(f, y0, order)
    traj = get_initial_trajectory(y0, f, order, N=ts.shape[0])
    dom = linearize_observation_model(om, traj[1:])
    out, _, _ = linear_filtsmooth(x0, dtm, dom)
    return out.mean, 0


def sequential_eks(f, y0, ts, order):
    iwp, om = make_continuous_models(f, y0, order)
    dtm = discretize_transitions(iwp, ts)
    x0 = taylor_mode_init(f, y0, order)
    traj = get_initial_trajectory(y0, f, order, N=ts.shape[0])
    dom = linearize_observation_model(om, traj[1:])
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
        algo_order=4,
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


def benchmark(methods, N=10):
    results = defaultdict(lambda: [])

    dts = 2.0 ** -np.arange(0, 12)
    for dt in dts:
        print(f"dt={dt}")
        results["dt"].append(dt)
        ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
        for k, v in methods.items():
            print(k)
            f = lambda: v(ts)
            status = f()
            if status != 0:
                t = np.nan
            else:
                t = timeit(f, N)
            print(t)
            results[k].append(t)
    return results


def block_and_return_state(f):
    def f2(*args, **kwargs):
        a, b = f(*args, **kwargs)
        a.block_until_ready()
        return b

    return f2


for ivp, name in ((logistic(), "logistic"), (lotkavolterra(), "lotkavolterra")):
    print(f"ivp={name}")
    f, y0 = ivp.f, ivp.y0
    if name == "logistic":
        continue

    peks = jax.jit(lambda ts: parallel_eks(f, y0, ts, order=4))
    seks = jax.jit(lambda ts: sequential_eks(f, y0, ts, order=4))
    dp5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=diffrax.Dopri5()))
    kv5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=diffrax.Kvaerno5()))
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

    df.to_csv(f"experiments/1_simple_runtime_benchmark/{name}_dev.csv")
