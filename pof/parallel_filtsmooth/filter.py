from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from pof.calibration import whiten
from pof.observations import AffineModel
from pof.utils import (
    MVNSqrt,
    append_zeros_along_new_axis,
    mvn_loglikelihood,
    objective_function_value,
    tria,
)


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
    d = observation_models.b.shape[1]
    xs = jax.tree_map(lambda z: append_zeros_along_new_axis(z, N - 1), x0)

    elems = jax.vmap(_get_element)(transition_models, observation_models, xs)

    _, means, cholcovs, _, _ = jax.lax.associative_scan(sqrt_filtering_operator, elems)

    means = jnp.concatenate([x0.mean[None, ...], means])
    cholcovs = jnp.concatenate([x0.chol[None, ...], cholcovs])

    ssq = _get_sigma_squared(transition_models, observation_models, means, cholcovs)
    nll = _get_nll(transition_models, observation_models, means, cholcovs)

    obj = jax.vmap(objective_function_value)(
        means[:-1], means[1:], transition_models
    ).sum()

    return MVNSqrt(means, cholcovs), nll, obj, ssq


@jax.jit
def _get_element(transition_model, observation_model, xs):
    F, cholQ = transition_model
    H, c, cholR = observation_model

    ms, Ls = xs  # for any but the first, we have (0, 0)

    nx = F.shape[0]
    ny = c.shape[0]

    m1 = F @ ms  # = 0
    N1_ = tria(jnp.concatenate((F @ Ls, cholQ), axis=1))  # = cholQ
    Psi_ = jnp.block([[H @ N1_, cholR], [N1_, jnp.zeros((nx, ny))]])
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


@jax.jit
def _get_obs(transition_model, observation_model, m, cholP):
    F, cholQ = transition_model
    H, c, cholR = observation_model
    ny = c.shape[0]
    predicted_mean = F @ m
    predicted_chol = tria(jnp.concatenate([F @ cholP, cholQ], axis=1))
    obs_mean = H @ predicted_mean + c
    obs_chol = tria(jnp.concatenate([H @ predicted_chol, cholR], axis=1))
    return obs_mean, obs_chol


@jax.jit
def _get_nll(transition_models, observation_models, means, cholcovs):
    obs_mean, obs_chol = jax.vmap(_get_obs)(
        transition_models, observation_models, means[:-1], cholcovs[:-1]
    )
    nll = -jax.vmap(mvn_loglikelihood)(obs_mean, obs_chol).sum()
    return nll


@jax.jit
def _get_sigma_squared(transition_models, observation_models, means, cholcovs):
    om = observation_models
    N, d = om.b.shape
    # noiseless_om = AffineModel(om.H, om.b, jnp.zeros_like(om.cholR))
    obs_mean, obs_chol = jax.vmap(_get_obs)(
        transition_models, om, means[:-1], cholcovs[:-1]
    )
    ress = jax.vmap(whiten)(obs_mean, obs_chol)
    sigma_squared = jax.vmap(jnp.dot)(ress, ress).sum() / N / d
    return sigma_squared


@jax.jit
@jax.vmap
def sqrt_filtering_operator(elem1, elem2):
    A1, b1, U1, eta1, Z1 = elem1
    A2, b2, U2, eta2, Z2 = elem2

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)], [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx : nx + nx, :nx]
    Xi22 = tria_xi[nx : nx + nx, nx:]

    M = jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True)
    A = A2 @ A1 - M.T @ Xi21.T @ A1
    m = jlinalg.solve_triangular(Xi11, U1.T, lower=True)
    b = A2 @ (jnp.eye(nx) - m.T @ Xi21.T) @ (b1 + U1 @ U1.T @ eta2) + b2

    _U = jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True).T
    U = tria(jnp.concatenate([_U, U2], axis=1))
    _e = jlinalg.solve_triangular(Xi11, Xi21.T, lower=True, trans=True)
    eta = A1.T @ (jnp.eye(nx) - _e.T @ U1.T) @ (eta2 - Z2 @ Z2.T @ b1) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    return A, b, U, eta, Z
