"""
Models in this section are all assumed to be noiseless!
"""
from functools import partial
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from pof.transitions import IWP, projection_matrix
from pof.utils import MVNSqrt, tria


class NonlinearModel(NamedTuple):
    """Nonlinear noiseless observation model: y = f(x)"""

    f: Callable

    def __call__(self, x):
        return self.f(x)


class AffineModel(NamedTuple):
    """Affine approximation of a nonlinear model

    The original model (y = f(x)) is approximated by an affine model
    y = H x + b, where H and b are computed from f.
    """

    H: jnp.ndarray
    b: jnp.ndarray
    cholR: jnp.ndarray


@partial(jax.jit, static_argnums=(0,))
def linearize(f: NonlinearModel, x: MVNSqrt):
    m = x.mean
    res, F_x = f(m), jax.jacfwd(f, 0)(m)
    cholR = jnp.zeros((res.shape[0], res.shape[0]))
    return AffineModel(F_x, res - F_x @ m, cholR)


@partial(jax.jit, static_argnums=(0,))
def uncertain_linearize(f: NonlinearModel, x: MVNSqrt):
    """WIP of an uncertainty-aware linearization"""
    m, CL = x
    res, F_m = f(m), jax.jacfwd(f, 0)(m)
    cholR = tria(F_m @ CL)
    return AffineModel(F_m, res - F_m @ m, cholR)


@partial(jax.jit, static_argnums=(0,))
def linearize_ek0(f: NonlinearModel, x: MVNSqrt):
    m = x.mean
    res = f(m)
    d = res.shape[0]
    q = (m.shape[0] // d) - 1
    _iwp = IWP(wiener_process_dimension=d, num_derivatives=q)
    F_x = E1 = projection_matrix(_iwp, 1)
    cholR = jnp.zeros((res.shape[0], res.shape[0]))
    return AffineModel(F_x, res - F_x @ m, cholR)
