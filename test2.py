import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from tqdm import trange

import pof
import pof.ieks
import pof.transitions
from pof.ieks import filtsmooth
from pof.main import (  # get_sigmasq_estimate,
    discretize_transitions,
    get_constant_initial_trajectory,
    get_x0,
    linearize_observation_model,
    make_continuous_models,
)
from pof.utils import tria

ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e0, tmax=8)
ts_true, ys_true, sol_true = solve_diffrax(ivp, rtol=1e-10, atol=1e-10)

order = 3
dt = 1e-1
sigma = 1

ts = jnp.arange(0, ivp.tmax + dt, dt)
traj = get_constant_initial_trajectory(ivp.y0, order, len(ts))

x0 = get_x0(ivp, order)
iwp, om = make_continuous_models(ivp, order)
E0 = pof.transitions.projection_matrix(iwp, 0)
dtm = discretize_transitions(iwp, ts)
_linearize = jax.jit(linearize_observation_model, static_argnums=0)
dom = _linearize(om, traj[1:])

out = pof.ieks.linear_noiseless_filtering(x0, dtm, dom)
for i in trange(10):
    dom = _linearize(om, out.mean[1:])
    out = filtsmooth(x0, dtm, dom)

ys = jnp.dot(E0, out.mean.T).T

plt.plot(ts, ys, marker=".")


@jax.jit
def get_std(cholP, E0):
    cholP0 = E0 @ cholP
    P0 = cholP0 @ cholP0.T
    return jnp.sqrt(jnp.diagonal(P0))


stds = jax.vmap(get_std, in_axes=[0, None])(out.chol, E0)
for i in range(ys.shape[1]):
    plt.fill_between(
        ts, ys[:, i] - 3 * stds[:, i], ys[:, i] + 3 * stds[:, i], alpha=0.2
    )

# Likelihoods?


F = dtm.F[1]
cholQ = dtm.QL[1]
H = dom.H[1]
c = dom.b[1]
m, cholP = out.mean[1], out.chol[1]
get_whitened_residual(F, cholQ, H, c, m, cholP)


###################
# go through it step by step for a moment
dom = _linearize(om, traj[1:])
out = pof.ieks.linear_noiseless_filtering(x0, dtm, dom)

m0, cholP0 = out.mean[0], out.chol[0]
mp = dtm.F[0] @ m0
cholPp = tria(jnp.concatenate([dtm.F[0] @ cholP, dtm.QL[0]], axis=1))
z = dom.H[0] @ mp + dom.b[0]
cholS = tria(dom.H[0] @ cholPp)
S = cholS @ cholS.T


# how calibrated is the filtering state actually?
diff = sol_true.evaluate(ts[1]) - E0 @ out.mean[1]
C = E0 @ out.chol[1] @ out.chol[1].T @ E0.T
std = jnp.sqrt(jnp.diagonal(C))
