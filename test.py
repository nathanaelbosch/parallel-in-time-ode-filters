import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from tqdm import trange

import pof
from pof.ieks import filtsmooth
from pof.main import (
    discretize_transitions,
    get_constant_initial_trajectory,
    get_x0,
    linearize_observation_model,
    make_continuous_models,
)
from pof.solve import solve_diffrax

ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e0, tmax=8)

ts_true, ys_true, sol_true = solve_diffrax(ivp, rtol=1e-10, atol=1e-10)
plt.plot(ts_true, ys_true, "--", color="black")


order = 3
# dt = 5e-1
# times = jnp.arange(0, ivp.tmax + dt, dt)
# ts, ys = solve_diffrax(ivp, ts=times)
_ts, _ys, _sol = solve_diffrax(ivp, rtol=1e-3, atol=1e-6)
# it's DP5, so it takes 6 evaluations per step!
ts = jnp.concatenate(
    [
        *jax.vmap(jnp.linspace, in_axes=(0, 0, None, None))(
            _ts[:-1], _ts[1:], 6, False
        ),
        _ts[-1:],
    ]
)
ys = jax.vmap(sol.evaluate)(ts)
dys = jax.vmap(lambda y: ivp.f(None, y))(ys)
# dys = jnp.zeros_like(ys)
N, d = ys.shape
traj = jnp.concatenate(
    [ys[:, :, None], dys[:, :, None], jnp.zeros((N, d, order - 1))], axis=2
).reshape(N, -1)

ts = jnp.arange(0, ivp.tmax + dt, dt)
traj = get_constant_initial_trajectory(ivp.y0, order, len(ts))

x0 = get_x0(ivp, order)
iwp, om = make_continuous_models(ivp, order)
dtm = discretize_transitions(iwp, ts)
dom = linearize_observation_model(om, traj[1:])
# out = filtsmooth(x0, dtm, dom)
out, etas = pof.ieks.linear_noiseless_filtering(x0, dtm, dom)
E0 = pof.transitions.projection_matrix(iwp, 0)

# iterate
for i in trange(100):
    dom = linearize_observation_model(om, out.mean[1:])
    out = filtsmooth(x0, dtm, dom)

plt.plot(_ts, _ys, marker="o", color="blue")
plt.ylim(-3, 3)
plt.plot(ts, jnp.dot(E0, out.mean.T).T, marker=".", color="orange")


# Likelihoods?
def _sqrt_loglikelihood(F, cholQ, H, c, m_t_1, cholP_t_1):
    predicted_mean = F @ m_t_1
    predicted_chol = tria(jnp.concatenate([F @ cholP_t_1, cholQ], axis=1))
    obs_mean = H @ predicted_mean + c
    obs_chol = tria(H @ predicted_chol)
    return mvn_loglikelihood(obs_mean, obs_chol)
