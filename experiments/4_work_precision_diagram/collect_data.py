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

from pof.ivp import lotkavolterra, fitzhughnagumo, logistic, vanderpol
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
        saveat=diffrax.SaveAt(
            # ts=ts,
            steps=True,
            t0=True,
            t1=True,
            dense=True,
        ),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=ts.shape[0] - 1,
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
}


def main():

    ref = solve_diffrax(IVP.f, IVP.y0, IVP.t_span, atol=1e-20, rtol=1e-20)
    yref_final = get_ts_ys(ref)[1][-1]

    def evaluate_method(method, methodname, Ns):
        results = defaultdict(list)
        for N in tqdm(Ns):
            ts = jnp.linspace(IVP.t0, IVP.tmax, N)
            f = lambda: method(ts)
            ys, info, status = f()
            if status != 0:
                t = np.nan
                rmse_final = np.nan
                rmse_traj = np.nan
            else:
                t = timeit(f, 3)
                idxs = jnp.isfinite(ys[:, 0])
                ys = ys[idxs]
                rmse_final = jnp.linalg.norm(ys[-1] - yref_final)
                yref = jax.vmap(ref.evaluate)(ts)
                rmse_traj = jax.vmap(jnp.linalg.norm)(ys - yref).mean()

            print(f"[{methodname}] runtime: {t}")
            print(f"[{methodname}] rmse_final: {rmse_final}")
            print(f"[{methodname}] rmse_traj: {rmse_traj}")
            results["runtime"].append(t)
            results["rmse_final"].append(rmse_final)
            results["rmse_traj"].append(rmse_traj)
            if "iterations" in info.keys():
                results["iterations"].append(info["iterations"])
                print(f"[{methodname}] iterations: {info['iterations']}")

        return results

    # initialize dataframe that will be filled column by column
    df = pd.DataFrame({"Ns": Ns})

    for k, v in METHODS.items():
        print(f"\nEvaluating {k}\n")
        results = evaluate_method(v(IVP), k, Ns)
        for k2, v2 in results.items():
            df[f"{k}_{k2}"] = v2

    return df


def save_df(df):
    # save dataframe to csv file in the same directory as this script
    current_dir = os.path.dirname(os.path.realpath(__file__))
    filename = os.path.join(current_dir, f"data_{IVP_NAME}.csv")
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
