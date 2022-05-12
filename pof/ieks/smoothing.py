from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._utils import none_or_concat

from pof.utils import tria, MVNSqrt, objective_function_value
from pof.ieks.operators import sqrt_smoothing_operator


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
    return none_or_concat((gs, Es, Ls), (g_T, E_T, L_T), -1)


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
