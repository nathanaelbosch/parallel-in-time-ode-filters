"""
Models in this section are all assumed to be noiseless!
"""
from typing import NamedTuple, Callable
from functools import partial

import jax
import jax.numpy as jnp

from pof.utils import MVNSqrt, tria


class NonlinearModel(NamedTuple):
    f: Callable

    def __call__(self, x):
        return self.f(x)


class AffineModel(NamedTuple):
    H: jnp.ndarray
    b: jnp.ndarray
    cholR: jnp.ndarray


@partial(jax.jit, static_argnums=(0,))
def linearize(f: NonlinearModel, x: jnp.ndarray):
    res, F_x = f(x), jax.jacfwd(f, 0)(x)
    cholR = jnp.zeros((res.shape[0], res.shape[0]))
    return AffineModel(F_x, res - F_x @ x, cholR)


@partial(jax.jit, static_argnums=(0,))
def linearize_uncertain(f: NonlinearModel, x: MVNSqrt):
    m, C = x
    res, F_m = f(m), jax.jacfwd(f, 0)(m)
    cholR = tria(F_m @ C)
    return AffineModel(F_m, res - F_m @ m, cholR)
