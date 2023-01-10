import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pof.ivp import lotkavolterra
from pof.solver import solve, sequential_eks_solve

ivp = lotkavolterra()

N = 50
ts = jnp.linspace(0, 10, N)

ys_par, info_par = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=3, init="constant")

ys_seq, info_seq = sequential_eks_solve(f=ivp.f, y0=ivp.y0, ts=ts, order=3)


def plot_result(ys, ax=None):
    means, chol_covs = ys
    covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(ts, means, marker="o")
    for i in range(means.shape[1]):
        ax.fill_between(
            ts,
            means[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
            means[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
            alpha=0.2,
            color=f"C{i}",
        )
    return ax


fig, axes = plt.subplots(2, 1)
plot_result(ys_par, ax=axes[0])
plot_result(ys_seq, ax=axes[1])
axes[0].set_ylim(0, 6)
axes[1].set_ylim(0, 6)
plt.show()
