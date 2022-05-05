from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from parsmooth._utils import none_or_concat

from pof.utils import tria, linearize, AffineModel, MVNSqrt
from pof.ieks.operators import sqrt_smoothing_operator


def smoothing(
    transition_model: AffineModel,
    cholQ: jnp.ndarray,
    filter_trajectory: MVNSqrt,
):
    assert isinstance(filter_trajectory, MVNSqrt)

    associative_params = _associative_params(
        transition_model,
        cholQ,
        filter_trajectory,
    )
    smoothed_means, _, smoothed_chols = jax.lax.associative_scan(
        jax.vmap(sqrt_smoothing_operator), associative_params, reverse=True
    )
    res = jax.vmap(MVNSqrt)(smoothed_means, smoothed_chols)

    return res


def _associative_params(
    transition_model,
    cholQ,
    filtering_trajectory,
):
    ms, Ps = filtering_trajectory
    vmapped_fn = jax.vmap(_sqrt_associative_params, in_axes=[None, None, 0, 0])
    gs, Es, Ls = vmapped_fn(transition_model, cholQ, ms[:-1], Ps[:-1])
    g_T, E_T, L_T = ms[-1], jnp.zeros_like(Ps[-1]), Ps[-1]
    return none_or_concat((gs, Es, Ls), (g_T, E_T, L_T), -1)


def _sqrt_associative_params(transition_model, cholQ, m, chol_P):
    F, b = transition_model
    nx = cholQ.shape[0]

    Phi = jnp.block([[F @ chol_P, cholQ], [chol_P, jnp.zeros_like(cholQ)]])
    Tria_Phi = tria(Phi)
    Phi11 = Tria_Phi[:nx, :nx]
    Phi21 = Tria_Phi[nx:, :nx]
    D = Tria_Phi[nx:, nx:]

    E = jlinalg.solve(Phi11.T, Phi21.T).T
    g = m - E @ (F @ m + b)
    return g, E, D
