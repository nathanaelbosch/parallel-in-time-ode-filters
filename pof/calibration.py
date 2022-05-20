import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from pof.utils import tria


@jax.jit
def whiten(m, cholP):
    return solve_triangular(cholP.T, m)


def get_whitened_residual(F, cholQ, H, c, m, cholP):
    predicted_mean = F @ m
    predicted_chol = tria(jnp.concatenate([F @ cholP, cholQ], axis=1))
    obs_mean = H @ predicted_mean + c
    obs_chol = tria(H @ predicted_chol)
    return whiten(obs_mean, obs_chol)


def get_sigma_estimate(filtered_states, discrete_transitions, discrete_observations):
    N, d = discrete_observations.b.shape
    F, cholQ = discrete_transitions
    H, b = discrete_observations
    m, cholP = filtered_states
    whitened_residuals = jax.vmap(get_whitened_residual)(
        F, cholQ, H, b, m[:-1], cholP[:-1]
    )
    chisquares = jax.vmap(lambda r: jnp.dot(r, r))(whitened_residuals)
    return jnp.sqrt(jnp.sum(chisquares) / d / (N + 2))
