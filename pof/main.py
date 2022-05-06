import jax
import jax.numpy as jnp

import tornadox

from pof.utils import MVNSqrt
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


def get_constant_initial_trajectory(y0, order, N):
    d = y0.shape[0]
    x0 = jnp.concatenate([y0, jnp.zeros(d * order)])
    return jnp.repeat(x0.reshape(1, -1), N, axis=0)


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
