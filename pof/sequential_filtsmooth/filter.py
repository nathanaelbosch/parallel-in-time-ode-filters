from typing import Callable, Optional, Union

import jax
import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular

from pof.observations import linearize
from pof.utils import MVNSqrt, mvn_loglikelihood, tria


def extended_kalman_filter(
    x0, discrete_transition_models, continuous_observation_model
):
    predict = _sqrt_predict
    update = _sqrt_update

    def body(carry, inp):
        x, ell = carry
        F, QL = inp

        x = predict(F, QL, x)
        H, b, R = linearize(continuous_observation_model, x)
        x, ell_inc = update(H, R, b, x)

        return (x, ell + ell_inc), x

    (_, ell), xs = jax.lax.scan(body, (x0, 0.0), discrete_transition_models)
    means = jnp.concatenate([x0.mean[None, ...], xs.mean])
    cholcovs = jnp.concatenate([x0.chol[None, ...], xs.chol])
    return MVNSqrt(means, cholcovs), ell


@jax.jit
def _sqrt_predict(F, cholQ, x):
    m, cholP = x

    m = F @ m
    cholP = tria(jnp.concatenate([F @ cholP, cholQ], axis=1))

    return MVNSqrt(m, cholP)


@jax.jit
def _sqrt_update(H, cholR, c, x):
    m, cholP = x
    nx = m.shape[0]
    ny = c.shape[0]

    y_hat = H @ m + c
    y_diff = 0 - y_hat

    M = jnp.block([[H @ cholP, cholR], [cholP, jnp.zeros_like(cholP, shape=(nx, ny))]])
    chol_S = tria(M)

    cholP = chol_S[ny:, ny:]

    G = chol_S[ny:, :ny]
    I = chol_S[:ny, :ny]

    m = m + G @ solve_triangular(I, y_diff, lower=True)
    ell = mvn_loglikelihood(y_diff, I)
    return MVNSqrt(m, cholP), ell
