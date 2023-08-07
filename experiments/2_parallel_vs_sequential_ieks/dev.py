import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pof.ivp import *
from pof.diffrax import solve_diffrax
from pof.solver import solve, sequential_eks_solve

ORDER = 2

ivp = rigid_body()


sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, atol=1e-6, rtol=1e-3)
ts_dense = jnp.linspace(ivp.t0, ivp.tmax, 1000)
ys_true = jax.vmap(sol_true.evaluate)(ts_dense)


def plot_result(ts, ys, ax=None):
    means, chol_covs = ys
    covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    for i in range(means.shape[1]):
        ax.plot(ts_dense, ys_true[:, i], "--k", linewidth=0.5)
        ax.plot(ts, means[:, i], marker="o", color=f"C{i}")
        ax.fill_between(
            ts,
            means[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
            means[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
            alpha=0.2,
            color=f"C{i}",
        )
    return ax


Ns = [10, 25, 50, 100]
fig, axes = plt.subplots(2, len(Ns))
for i, N in enumerate(Ns):
    ts = jnp.linspace(ivp.t0, ivp.tmax, N)
    ys_par, info_par = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant")
    ys_seq, info_seq = solve(
        f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant", sequential=True
    )
    ys_eks, info_eks = sequential_eks_solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER)

    plot_result(ts, ys_seq, ax=axes[0, i])
    plot_result(ts, ys_seq, ax=axes[0, i])
    plot_result(ts, ys_eks, ax=axes[1, i])
    axes[0, i].set_ylim(-2.2, 2.2)
    axes[1, i].set_ylim(-2.2, 2.2)
plt.show()
