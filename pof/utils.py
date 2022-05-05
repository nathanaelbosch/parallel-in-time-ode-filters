from typing import NamedTuple, Callable, Any, Union, Tuple
import itertools
from functools import partial

import jax
import jax.numpy as jnp


class MVNSqrt(NamedTuple):
    mean: Any
    chol: Any


class AffineModel(NamedTuple):
    A: jnp.ndarray
    b: MVNSqrt

    def __call__(self, x):
        return self.A @ x + self.b


# @jax.jit
def tria(A):
    return qr(A.T).T


def qr(A: jnp.ndarray):
    return _qr(A)


def _qr(A: jnp.ndarray, return_q=False):
    m, n = A.shape
    min_ = min(m, n)
    if return_q:
        Q = jnp.eye(m)

    for j in range(min_):
        # Apply Householder transformation.
        v, tau = _householder(A[j:, j])

        H = jnp.eye(m)
        H = H.at[j:, j:].add(-tau * (v[:, None] @ v[None, :]))

        A = H @ A
        if return_q:
            Q = H @ Q  # noqa

    R = jnp.triu(A[:min_, :min_])
    if return_q:
        return Q[:n].T, R  # noqa
    else:
        return R


def _householder(a):

    alpha = a[0]
    s = jnp.sum(a[1:] ** 2)

    v = a
    t = (alpha**2 + s) ** 0.5
    v0 = jax.lax.cond(alpha <= 0, lambda _: alpha - t, lambda _: -s / (alpha + t), None)
    tau = 2 * v0**2 / (s + v0**2)
    v = v / v0
    v = v.at[0].set(1.0)

    v = jnp.nan_to_num(v)
    tau = jnp.nan_to_num(tau)

    return v, tau


@partial(jax.jit, static_argnums=(0,))
def linearize(f: Callable, x: jnp.ndarray):
    assert isinstance(f, Callable)
    res, F_x = f(x), jax.jacfwd(f, 0)(x)
    return AffineModel(F_x, res - F_x @ x)


def append_zeros_along_new_axis(z, N):
    return jnp.concatenate([z[None, ...], jnp.zeros_like(z, shape=(N,) + z.shape)])
