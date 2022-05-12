import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

import tornadox

from pof.utils import MVNSqrt, tria
from pof.transitions import IWP, get_transition_model, projection_matrix
from pof.observations import NonlinearModel, linearize


def make_continuous_models(ivp, order):
    d = ivp.y0.shape[0]
    iwp = IWP(num_derivatives=order, wiener_process_dimension=d)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    observation_model = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))
    return iwp, observation_model


def discretize_transitions(iwp, times):
    steps = times[1:] - times[:-1]
    get_transitions = jax.vmap(get_transition_model, in_axes=[None, 0])
    return get_transitions(iwp, steps)


def get_constant_initial_trajectory(y0, dy0, order, N, with_dy=True):
    d = y0.shape[0]
    if with_dy:
        _x0 = jnp.concatenate(
            [y0[:, None], dy0[:, None], jnp.zeros((d, (order - 1)))], axis=1
        )
    else:
        _x0 = jnp.concatenate([y0[:, None], jnp.zeros((d, (order)))], axis=1)
    return jnp.repeat(_x0.reshape(1, -1), N, axis=0)


def linearize_observation_model(observation_model, trajectory):
    return jax.vmap(linearize, in_axes=[None, 0])(observation_model, trajectory)


def get_x0(ivp, num_derivatives):
    d = ivp.y0.shape[0]
    m0, P0 = tornadox.init.TaylorMode()(
        f=ivp.f, df=None, y0=ivp.y0, t0=0, num_derivatives=num_derivatives
    )
    m0, P0 = jnp.concatenate(m0.T), jnp.kron(jnp.eye(d), P0)
    x0 = MVNSqrt(m0, P0)
    return x0


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
