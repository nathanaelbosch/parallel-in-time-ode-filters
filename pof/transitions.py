"""
All things transitions.

This file contains everything that tornadox.iwp does, but in a more functional manner.
"""
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy


class TransitionModel(NamedTuple):
    """Linear transition model in square-root form: x' | x ~ N(Fx, QL*QLt)"""

    F: jnp.ndarray
    QL: jnp.ndarray


class IWP(NamedTuple):
    wiener_process_dimension: int
    num_derivatives: int


@partial(jax.jit, static_argnames="n")
def hilbert(n):
    return jnp.array(scipy.linalg.hilbert(n))


@partial(jax.jit, static_argnames="n")
def pascal(n):
    return jnp.array(scipy.linalg.pascal(n, kind="lower", exact=False))


@partial(jax.jit, static_argnames="iwp")
def preconditioned_discretize_1d(iwp: IWP):
    A_1d = jnp.flip(pascal(iwp.num_derivatives + 1))
    Q_1d = jnp.flip(hilbert(iwp.num_derivatives + 1))
    return A_1d, jnp.linalg.cholesky(Q_1d)


@partial(jax.jit, static_argnames="iwp")
def preconditioned_discretize(iwp: IWP):
    A_1d, L_Q1d = preconditioned_discretize_1d(iwp)
    id_factor = jnp.eye(iwp.wiener_process_dimension)
    A = jnp.kron(id_factor, A_1d)
    L_Q = jnp.kron(id_factor, L_Q1d)
    return A, L_Q


@partial(jax.jit, static_argnames="iwp")
def nordsieck_preconditioner_1d(iwp: IWP, dt: float):
    powers = np.arange(iwp.num_derivatives, -1, -1)
    scales = jnp.array(scipy.special.factorial(powers))
    powers = powers + 0.5

    scaling_vector = (jnp.abs(dt) ** powers) / scales
    scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales
    return jnp.diag(scaling_vector), jnp.diag(scaling_vector_inv)


@partial(jax.jit, static_argnames="iwp")
def nordsieck_preconditioner(iwp: IWP, dt: float):
    P, PI = nordsieck_preconditioner_1d(iwp, dt)
    Id = jnp.eye(iwp.wiener_process_dimension)
    return jnp.kron(Id, P), jnp.kron(Id, PI)


@partial(jax.jit, static_argnames="iwp")
def non_preconditioned_discretize(iwp: IWP, dt: float):
    P, PI = nordsieck_preconditioner(iwp, dt)
    F, QL = preconditioned_discretize(iwp)
    state_trans_mat = P @ F @ PI
    proc_noise_cov_cholesky = P @ QL
    return (state_trans_mat, proc_noise_cov_cholesky)


@partial(jax.jit, static_argnames=("iwp", "derivative_to_project_onto"))
def projection_matrix_1d(iwp: IWP, derivative_to_project_onto):
    return jnp.eye(1, iwp.num_derivatives + 1, derivative_to_project_onto)


@partial(jax.jit, static_argnames=("iwp", "derivative_to_project_onto"))
def projection_matrix(iwp: IWP, derivative_to_project_onto):
    I_d = jnp.eye(iwp.wiener_process_dimension)
    return jnp.kron(I_d, projection_matrix_1d(iwp, derivative_to_project_onto))


@partial(jax.jit, static_argnames=("iwp",))
def get_transition_model(iwp, dt):
    return TransitionModel(*non_preconditioned_discretize(iwp, dt))


def discretize_transitions(iwp, times=None, steps=None):
    if steps is None:
        steps = times[1:] - times[:-1]
    get_transitions = jax.vmap(get_transition_model, in_axes=[None, 0])
    return get_transitions(iwp, steps)
