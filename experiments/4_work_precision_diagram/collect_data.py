import time
import os
from collections import defaultdict

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax
import pandas as pd
import numpy as np
from tqdm import tqdm

from pof.ivp import lotkavolterra, fitzhughnagumo
from pof.diffrax import solve_diffrax, get_ts_ys
from pof.solver import solve, sequential_eks_solve


def timeit(f, N):
    times = []
    for _ in range(N):
        start = time.time()
        f()
        t = time.time() - start
        times.append(t)
    return min(times)


def solve_diffrax_(f, y0, ts, solver):
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
    return sol.ys, {}, sol.result


def _diffrax(solver):
    def f(ivp):
        @jax.jit
        def inner(ts):
            return solve_diffrax_(ivp.f, ivp.y0, ts, solver)

        return inner

    return f


def _ieks(order, maxiters):
    def f(ivp):
        @jax.jit
        def inner(ts):
            ys, info_dict = solve(
                f=ivp.f,
                y0=ivp.y0,
                ts=ts,
                order=order,
                init="constant",
                maxiters=maxiters,
            )
            return ys.mean, info_dict, 0

        return inner

    return f


def _eks(order):
    def f(ivp):
        @jax.jit
        def inner(ts):
            ys, info_dict = sequential_eks_solve(
                f=ivp.f,
                y0=ivp.y0,
                ts=ts,
                order=order,
            )
            return ys.mean, info_dict, 0

        return inner

    return f


IVP = ivp = fitzhughnagumo()
Ns = 2 ** jnp.arange(8, 18)
METHODS = {
    "DP5": _diffrax(diffrax.Dopri5()),
    "Heun": _diffrax(diffrax.Heun()),
    "ImplicitEuler": _diffrax(
        diffrax.ImplicitEuler(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    ),
    "KV3": _diffrax(
        diffrax.Kvaerno3(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    ),
    "KV5": _diffrax(
        diffrax.Kvaerno5(diffrax.NewtonNonlinearSolver(rtol=1e-6, atol=1e-9))
    ),
    "EKS(1)": _eks(1),
    "EKS(2)": _eks(2),
    "EKS(3)": _eks(3),
    "EKS(5)": _eks(5),
    "IEKS(1)": _ieks(1, 1000),
    "IEKS(2)": _ieks(2, 1000),
    "IEKS(3)": _ieks(3, 1000),
    "IEKS(5)": _ieks(5, 1000),
    # "IEKS(1)_10": _ieks(1, 10),
    # "IEKS(1)_20": _ieks(1, 20),
    # "IEKS(1)_50": _ieks(1, 50),
    # "IEKS(1)_100": _ieks(1, 100),
    # "IEKS(1)_200": _ieks(1, 200),
    # "IEKS(1)_500": _ieks(1, 500),
    # "IEKS(1)_1000": _ieks(1, 1000),
    # "IEKS(2)_10": _ieks(2, 10),
    # "IEKS(2)_20": _ieks(2, 20),
    # "IEKS(2)_50": _ieks(2, 50),
    # "IEKS(2)_100": _ieks(2, 100),
    # "IEKS(2)_200": _ieks(2, 200),
    # "IEKS(2)_500": _ieks(2, 500),
    # "IEKS(2)_1000": _ieks(2, 1000),
    # "IEKS(3)_10": _ieks(3, 10),
    # "IEKS(3)_20": _ieks(3, 20),
    # "IEKS(3)_50": _ieks(3, 50),
    # "IEKS(3)_100": _ieks(3, 100),
    # "IEKS(3)_200": _ieks(3, 200),
    # "IEKS(3)_500": _ieks(3, 500),
    # "IEKS(3)_1000": _ieks(3, 1000),
    # "IEKS(5)_10": _ieks(5, 10),
    # "IEKS(5)_20": _ieks(5, 20),
    # "IEKS(5)_50": _ieks(5, 50),
    # "IEKS(5)_100": _ieks(5, 100),
    # "IEKS(5)_200": _ieks(5, 200),
    # "IEKS(5)_500": _ieks(5, 500),
    # "IEKS(5)_1000": _ieks(5, 1000),
}


def main():

    yref_final = get_ts_ys(
        solve_diffrax(ivp.f, ivp.y0, ivp.t_span, atol=1e-20, rtol=1e-20)
    )[1][-1]

    def evaluate_method(method, Ns):
        results = defaultdict(list)
        for N in tqdm(Ns):
            ts = jnp.linspace(0, ivp.tmax, N)
            f = lambda: method(ts)
            ys, info, status = f()
            if status != 0:
                t = np.nan
                rmse = np.nan
            else:
                t = timeit(f, 3)
                idxs = jnp.isfinite(ys[:, 0])
                y = ys[idxs][-1]
                rmse = jnp.linalg.norm(y - yref_final)
            print(t)
            print(rmse)
            results["runtimes"].append(t)
            results["rmses"].append(rmse)
            if "iterations" in info.keys():
                results["iterations"].append(info["iterations"])

        return results

    # initialize dataframe that will be filled column by column
    df = pd.DataFrame({"Ns": Ns})

    for k, v in METHODS.items():
        print(f"\nEvaluating {k}\n")
        results = evaluate_method(v(IVP), Ns)
        df[f"{k}_time"] = results["runtimes"]
        df[f"{k}_rmse"] = results["rmses"]
        if "iterations" in results.keys():
            df[f"{k}_iterations"] = results["iterations"]

    return df


def save_df(df):
    # save dataframe to csv file in the same directory as this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(current_dir, "data.csv")
    df.to_csv(filename, index=False)
    print(f"Saved data to {filename}")


def plot(df):
    fig, ax = plt.subplots(1, 3)

    # 1: error vs runtime
    for key in METHODS.keys():
        label = key
        ax[0].plot(df[f"{key}_time"], df[f"{key}_rmse"], label=label)
    ax[0].set_xlabel("Runtime (s)")
    ax[0].set_ylabel("RMSE")
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].legend()

    # 2: error vs N
    for key in METHODS.keys():
        label = key
        ax[1].plot(df["Ns"], df[f"{key}_rmse"], label=label)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_xlabel("N")
    ax[1].set_ylabel("RMSE")
    ax[1].legend()

    # 3: runtime vs N
    for key in METHODS.keys():
        label = key
        ax[2].plot(df["Ns"], df[f"{key}_time"], label=label)
    ax[2].set_xscale("log")
    ax[2].set_yscale("log")
    ax[2].set_xlabel("N")
    ax[2].set_ylabel("Runtime (s)")
    ax[2].legend()

    return ax


if __name__ == "__main__":
    df = main()
    save_df(df)
    # plot(df)
