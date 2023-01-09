import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pof.ivp import lotkavolterra
from pof.solver import solve

ivp = lotkavolterra()

ts = jnp.linspace(0, 10, 100)
ys, info = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=3, init="constant")

# the rest is just plotting:
means, chol_covs = ys
covs = jax.vmap(lambda c: c @ c.T, in_axes=0)(chol_covs)

plt.plot(ts, means, marker="o")
for i in range(means.shape[1]):
    plt.fill_between(
        ts,
        means[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
        means[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
        alpha=0.2,
        color=f"C{i}",
    )
plt.show()
