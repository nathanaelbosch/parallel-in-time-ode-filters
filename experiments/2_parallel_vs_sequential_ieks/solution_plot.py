from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import diffrax

from tueplots import axes, bundles, figsizes
from tueplots.constants import markers

from pof.ivp import *
from pof.diffrax import solve_diffrax
from pof.solver import solve, sequential_eks_solve

plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(axes.lines())
plt.rcParams.update(
    {
        # "lines.linewidth": 1.5,
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.2,
        # "lines.markersize": 4,
        "axes.grid": False,
        "axes.grid.which": "both",
        # "axes.grid.which": "major",
    }
)


def plot_result(ts, ys, ax):
    means, chol_covs = ys
    covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

    for i in range(means.shape[1]):
        ax.plot(ts_dense, ys_true[:, i], "--k", linewidth=0.5)
        ax.plot(
            ts,
            means[:, i],
            # marker="o",
            color=f"C{i}",
        )
        ax.fill_between(
            ts,
            means[:, i] - 1.96 * jnp.sqrt(covs[:, i, i]),
            means[:, i] + 1.96 * jnp.sqrt(covs[:, i, i]),
            alpha=0.2,
            color=f"C{i}",
        )
    return ax


def plot_truesol(ts, ys, ax):
    for i in range(ys.shape[1]):
        ax.plot(ts, ys[:, i], "--k", linewidth=0.5)
    return ax


def plot_errors(ts, ys, ax):
    means, chol_covs = ys
    covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

    errs = means - jax.vmap(sol_true.evaluate)(ts)

    for i in range(means.shape[1]):
        ax.plot(
            ts,
            errs[:, i],
            # marker="o",
            color=f"C{i}",
        )
        ax.fill_between(
            ts,
            # errs[:, i] - 1.96 * jnp.sqrt(covs[:, i, i]),
            # errs[:, i] + 1.96 * jnp.sqrt(covs[:, i, i]),
            -1.96 * jnp.sqrt(covs[:, i, i]),
            +1.96 * jnp.sqrt(covs[:, i, i]),
            alpha=0.2,
            color=f"C{i}",
        )
    ax.hlines(
        [0],
        xmin=ivp.t0,
        xmax=ivp.tmax,
        linestyles="dashed",
        colors="black",
        linewidth=0.5,
    )
    return ax


Ns = [30, 200, 100]
ORDER = 2
IVPS = (logistic(), rigid_body(), vanderpol(stiffness_constant=1))
IVPNAMES = ("Logistic", "Rigid Body", "Van der Pol")

plt.rcParams.update(figsizes.jmlr2001(nrows=2, ncols=3))
fig, axes = plt.subplots(2, len(IVPS), sharex="col")

for i, (ivp, N) in enumerate(zip(IVPS, Ns)):

    sol_true = solve_diffrax(
        ivp.f,
        ivp.y0,
        ivp.t_span,
        solver=diffrax.Kvaerno5,
        atol=1e-12,
        rtol=1e-12,
    )
    ts_dense = jnp.linspace(ivp.t0, ivp.tmax, 100)
    ys_true = jax.vmap(sol_true.evaluate)(ts_dense)

    ts = jnp.linspace(ivp.t0, ivp.tmax, N)
    ys_par, info_par = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant")
    ys_seq, info_seq = solve(
        f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant", sequential=True
    )
    # ys_eks, info_eks = sequential_eks_solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER)

    plot_truesol(ts_dense, ys_true, ax=axes[0, i])
    plot_result(ts, ys_par, ax=axes[0, i])
    plot_errors(ts, ys_par, ax=axes[1, i])
    axes[0, i].set_xlim(ivp.t0, ivp.tmax)
    axes[1, i].set_xlim(ivp.t0, ivp.tmax)
    axes[1, i].set_xlabel("$t$")
    axes[0, i].set_xticks(ivp.t_span)
    axes[1, i].set_xticks(ivp.t_span)
    if i == 0:
        axes[0, i].set_ylabel("$y(t)$")
        axes[1, i].set_ylabel("$y(t) - y^*(t)$")

    axes[0, i].set_title(rf"$\bf {chr(ord('a') + i)}.$ {IVPNAMES[i]}", loc="left")

DIR = Path("experiments/2_parallel_vs_sequential_ieks")
filepath = DIR / "solutions.pdf"
fig.savefig(filepath, bbox_inches="tight")
print(f"Saved plot to {filepath}")
