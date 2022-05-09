from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._utils import none_or_concat

from pof.utils import tria, MVNSqrt, append_zeros_along_new_axis
from pof.ieks.operators import sqrt_filtering_operator


def linear_noiseless_filtering(
    x0: MVNSqrt,
    transition_models,
    observation_models,
):
    """Filter a linear state-space model with exact observations.

    The `transition_model` is assumed to be constant over time;
    the `observation_model` is assumed to change each step.
    Filtering is done in parallel via `jax.lax.associative_scan`.
    """
    N = observation_models.H.shape[0]
    xs = jax.tree_map(lambda z: append_zeros_along_new_axis(z, N - 1), x0)

    elems = jax.vmap(_get_element)(transition_models, observation_models, xs)

    _, means, cholcovs, _, _ = jax.lax.associative_scan(sqrt_filtering_operator, elems)

    means = jnp.concatenate([x0.mean[None, ...], means])
    cholcovs = jnp.concatenate([x0.chol[None, ...], cholcovs])
    return MVNSqrt(means, cholcovs)


@jax.jit
def _get_element(transition_model, observation_model, xs):
    F, cholQ = transition_model
    H, c = observation_model

    ms, Ls = xs  # for any but the first, we have (0, 0)

    nx = F.shape[0]
    ny = c.shape[0]

    m1 = F @ ms  # = 0
    N1_ = tria(jnp.concatenate((F @ Ls, cholQ), axis=1))  # = cholQ
    Psi_ = jnp.block([[H @ N1_, jnp.zeros((ny, ny))], [N1_, jnp.zeros((nx, ny))]])
    Tria_Psi_ = tria(Psi_)
    Psi11 = Tria_Psi_[:ny, :ny]
    Psi21 = Tria_Psi_[ny:, :ny]
    U = Tria_Psi_[ny:, ny:]

    K = jlinalg.solve_triangular(Psi11, Psi21.T, trans=True, lower=True).T

    A = F - K @ H @ F
    b_sqr = m1 + K @ (-H @ m1 - c)  # -K @ c

    Z = jlinalg.solve_triangular(Psi11, H @ F, lower=True).T  # sqrt(J)
    eta = jlinalg.solve_triangular(Psi11, Z.T, trans=True, lower=True).T @ (-c)

    if nx > ny:
        Z = jnp.concatenate([Z, jnp.zeros((nx, nx - ny))], axis=1)
    else:
        Z = tria(Z)

    return A, b_sqr, U, eta, Z
