"""
Models in this section are all assumed to be noiseless!
"""
from typing import NamedTuple, Callable
from functools import partial

import jax
import jax.numpy as jnp


class NonlinearModel(NamedTuple):
    f: Callable

    def __call__(self, x):
        return self.f(x)


class AffineModel(NamedTuple):
    H: jnp.ndarray
    b: jnp.ndarray


@partial(jax.jit, static_argnums=(0,))
def linearize(f: NonlinearModel, x: jnp.ndarray):
    res, F_x = f(x), jax.jacfwd(f, 0)(x)
    return AffineModel(F_x, res - F_x @ x)
