from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from pof.utils import MVNSqrt, objective_function_value, tria


def smoothing(
    transition_models,
    filter_trajectory: MVNSqrt,
):
    assert isinstance(filter_trajectory, MVNSqrt)

    elems = get_elements(transition_models, filter_trajectory)
    means, _, chols = jax.lax.associative_scan(
        sqrt_smoothing_operator, elems, reverse=True
    )
    res = jax.vmap(MVNSqrt)(means, chols)

    obj = jax.vmap(objective_function_value)(means[:-1], means[1:], transition_models)

    return res, jnp.sum(obj)


@jax.jit
def get_elements(transition_models, filtering_trajectory):
    ms, Ps = filtering_trajectory
    vmapped_fn = jax.vmap(_sqrt_associative_params)
    gs, Es, Ls = vmapped_fn(transition_models, ms[:-1], Ps[:-1])
    g_T, E_T, L_T = ms[-1], jnp.zeros_like(Ps[-1]), Ps[-1]
    gs = jnp.concatenate([gs, g_T[None, ...]])
    Es = jnp.concatenate([Es, E_T[None, ...]])
    Ls = jnp.concatenate([Ls, L_T[None, ...]])
    return (gs, Es, Ls)


@jax.jit
def _sqrt_associative_params(transition_model, m, chol_P):
    F, cholQ = transition_model
    nx = cholQ.shape[0]

    Phi = jnp.block([[F @ chol_P, cholQ], [chol_P, jnp.zeros_like(cholQ)]])
    Tria_Phi = tria(Phi)
    Phi11 = Tria_Phi[:nx, :nx]
    Phi21 = Tria_Phi[nx:, :nx]
    D = Tria_Phi[nx:, nx:]

    E = jlinalg.solve(Phi11.T, Phi21.T).T
    g = m - E @ (F @ m)
    return g, E, D


@jax.jit
@jax.vmap
def sqrt_smoothing_operator(elem1, elem2):
    g1, E1, D1 = elem1
    g2, E2, D2 = elem2

    g = E2 @ g1 + g2
    E = E2 @ E1
    D = tria(jnp.concatenate([E2 @ D1, D2], axis=1))

    return g, E, D
