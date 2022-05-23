from typing import Optional, Callable, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlag

from pof.utils import MVNSqrt, tria


def smoothing(discrete_transition_models, filter_trajectory):
    last_state = jax.tree_map(lambda z: z[-1], filter_trajectory)
    previous_states = jax.tree_map(lambda z: z[:-1], filter_trajectory)

    def body(smoothed, inputs):
        transition_model, filtered = inputs
        F, QL = transition_model
        smoothed_state = _sqrt_smooth(F, QL, filtered, smoothed)
        return smoothed_state, smoothed_state

    _, smoothed_states = jax.lax.scan(
        body, last_state, (discrete_transition_models, previous_states), reverse=True
    )
    smoothed_states = jax.tree_map(
        lambda a, b: jnp.concatenate([a, b[None, ...]]), smoothed_states, last_state
    )
    return smoothed_states


@jax.jit
def _sqrt_smooth(F, cholQ, xf, xs):
    mf, cholPf = xf
    ms, cholPs = xs

    nx = F.shape[0]
    Phi = jnp.block([[F @ cholPf, cholQ], [cholPf, jnp.zeros_like(F)]])
    tria_Phi = tria(Phi)
    Phi11 = tria_Phi[:nx, :nx]
    Phi21 = tria_Phi[nx:, :nx]
    Phi22 = tria_Phi[nx:, nx:]
    gain = jlag.solve_triangular(Phi11, Phi21.T, trans=True, lower=True).T

    mean_diff = ms - F @ mf
    mean = mf + gain @ mean_diff
    chol = tria(jnp.concatenate([Phi22, gain @ cholPs], axis=1))

    return MVNSqrt(mean, chol)
