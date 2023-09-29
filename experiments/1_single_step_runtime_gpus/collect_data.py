import time
from collections import defaultdict
import os

import plac
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import diffrax
from scipy.integrate import solve_ivp

from pof.convenience import linearize_observation_model
from pof.ivp import lotkavolterra, logistic
from pof.parallel_filtsmooth import linear_filtsmooth

from pof.convenience import set_up_solver, get_initial_trajectory
from pof.sequential_filtsmooth import filtsmooth as seq_fs
from pof.utils import _gmul


def parallel_eks(f, y0, ts, order):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]
    states = get_initial_trajectory(setup, method="constant")

    dom = linearize_observation_model(om, jax.tree_map(lambda l: l[1:], states))
    out, _, _, _ = linear_filtsmooth(x0, dtm, dom)
    return out.mean, 0


def sequential_eks(f, y0, ts, order):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]

    states, *_ = seq_fs(x0, dtm, om)

    states = jax.vmap(_gmul, in_axes=[None, 0])(setup["E0"], states)

    return states.mean, 0


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


def timeit(f, N):
    times = []
    for _ in range(N):
        start = time.time()
        f()
        t = time.time() - start
        times.append(t)
    return min(times)


def benchmark(methods, dts, ivp, N=10):
    results = defaultdict(lambda: [])
    for dt in dts:
        print(f"dt = 2^{int(jnp.round(jnp.log2(dt)))}")
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


@plac.pos("gpu_name", "GPU Name")
def main(gpu_name):
    dts = 2.0 ** -np.arange(0, 19)
    # 1060 can only handle up to 14; 15 if we try manually afterwards; 16 does not work
    # dts = 2.0 ** -np.arange(0, 14)
    # dts = 2.0 ** -np.arange(17, 19)
    # dts = 2.0 ** -np.arange(13, 14)

    ivp = logistic()
    f, y0 = ivp.f, ivp.y0

    peks = jax.jit(lambda ts: parallel_eks(f, y0, ts, order=4))
    seks = jax.jit(lambda ts: sequential_eks(f, y0, ts, order=4))
    dp5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=diffrax.Dopri5()))
    Kv5 = diffrax.Kvaerno5(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    kv5 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=Kv5))
    Kv3 = diffrax.Kvaerno5(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    kv3 = jax.jit(lambda ts: solve_diffrax(f, y0, ts, solver=Kv3))
    rk45 = lambda ts: solve_scipy(f, y0, ts, "RK45")[1]
    lsoda = lambda ts: solve_scipy(f, y0, ts, "LSODA")[1]
    methods = {
        "pEKS": block_and_return_state(peks),
        "sEKS": block_and_return_state(seks),
        "dp5": block_and_return_state(dp5),
        "kv5": block_and_return_state(kv5),
        # "kv3": block_and_return_state(kv3),
    }

    res = benchmark(methods, dts, ivp, N=3)

    df = pd.DataFrame(res)
    df["N"] = (ivp.tmax - ivp.t0) / df.dt

    current_dir = os.path.dirname(os.path.realpath(__file__))
    filepath = os.path.join(current_dir, "results", f"{gpu_name}.csv")
    df.to_csv(filepath)


if __name__ == "__main__":
    plac.call(main)
