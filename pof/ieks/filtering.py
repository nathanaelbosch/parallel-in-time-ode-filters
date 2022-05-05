from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._utils import none_or_concat

from pof.utils import (
    tria,
    linearize,
    AffineModel,
    MVNSqrt,
    append_zeros_along_new_axis,
)
from pof.ieks.operators import sqrt_filtering_operator


def filtering(
    x0: MVNSqrt,
    transition_model: AffineModel,
    observation_model: Callable,
    cholQ: jnp.ndarray,
    cholR: jnp.ndarray,
    nominal_trajectory: jnp.ndarray,
):
    lin_obs_mod = jax.vmap(linearize, in_axes=[None, 0])(
        observation_model, nominal_trajectory
    )
    return linear_filtering(x0, transition_model, lin_obs_mod, cholQ, cholR)


def linear_filtering(
    x0: MVNSqrt,
    transition_model: AffineModel,
    observation_model: AffineModel,
    cholQ: jnp.ndarray,
    cholR: jnp.ndarray,
):
    """Filter a linear state-space model.

    The `transition_model` is assumed to be constant over time;
    the `observation_model` is assumed to change each step.
    Filtering is done in parallel via `jax.lax.associative_scan`.
    """
    elems = get_elements(transition_model, observation_model, cholQ, cholR, x0)

    _, means, cholcovs, _, _ = jax.lax.associative_scan(
        jax.vmap(sqrt_filtering_operator), elems
    )

    means = jnp.concatenate([x0.mean[None, ...], means])
    cholcovs = jnp.concatenate([x0.chol[None, ...], cholcovs])
    return MVNSqrt(means, cholcovs)


@jax.jit
def get_elements(transition_model, observation_model, cholQ, cholR, x0):
    N = observation_model.A.shape[0]
    xs = jax.tree_map(lambda z: append_zeros_along_new_axis(z, N - 1), x0)
    fn = jax.vmap(_get_elem, in_axes=[None, 0, None, None, 0])
    return fn(transition_model, observation_model, cholQ, cholR, xs)


@jax.jit
def _get_elem(transition_model, observation_model, cholQ, cholR, xs):
    F, b = transition_model
    H, c = observation_model

    ms, Ls = xs

    nx = cholQ.shape[0]
    ny = cholR.shape[0]

    m1 = F @ ms
    N1_ = tria(jnp.concatenate((F @ Ls, cholQ), axis=1))
    Psi_ = jnp.block([[H @ N1_, cholR], [N1_, jnp.zeros((nx, ny))]])
    Tria_Psi_ = tria(Psi_)
    Psi11 = Tria_Psi_[:ny, :ny]
    Psi21 = Tria_Psi_[ny : ny + nx, :ny]
    U = Tria_Psi_[ny : ny + nx, ny:]

    K = jlinalg.solve_triangular(Psi11, Psi21.T, trans=True, lower=True).T

    A = F - K @ H @ F
    b_sqr = m1 + K @ (-H @ m1 - c)

    Z = jlinalg.solve_triangular(Psi11, H @ F, lower=True).T
    eta = jlinalg.solve_triangular(Psi11, Z.T, trans=True, lower=True).T @ (-c)

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    return A, b_sqr, U, eta, Z
