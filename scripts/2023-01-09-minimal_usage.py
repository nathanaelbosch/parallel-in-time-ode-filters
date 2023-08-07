import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pof.ivp import *
from pof.diffrax import solve_diffrax
from pof.solver import solve, sequential_eks_solve

ORDER = 2

# ivp = lotkavolterra()
# ivp = fitzhughnagumo()
# ivp = threebody()
ivp = henonheiles(tmax=10.0)

# N = 100
N = 2**5
ts = jnp.linspace(ivp.t0, ivp.tmax, N)

# sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, atol=1e-9, rtol=1e-9)
sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, max_steps=int(1e5))
ys_true = jax.vmap(sol_true.evaluate)(ts)

ys_par, info_par = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant")
ys_seq, info_seq = solve(
    f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER, init="constant", sequential=True
)

ys_eks, info_eks = sequential_eks_solve(f=ivp.f, y0=ivp.y0, ts=ts, order=ORDER)


def plot_result(ys, ax=None):
    means, chol_covs = ys
    covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    ax.plot(ts, means, marker="o")
    for i in range(means.shape[1]):
        ax.plot(ts, ys_true[:, i], "--k", linewidth=0.5)
        ax.fill_between(
            ts,
            means[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
            means[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
            alpha=0.2,
            color=f"C{i}",
        )
    return ax


fig, axes = plt.subplots(3, 1)
plot_result(ys_par, ax=axes[0])
plot_result(ys_seq, ax=axes[1])
plot_result(ys_eks, ax=axes[2])
axes[0].set_ylim(-0.6, 0.6)
axes[1].set_ylim(-0.6, 0.6)
axes[2].set_ylim(-0.6, 0.6)
plt.show()
