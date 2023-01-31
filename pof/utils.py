from functools import partial
from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg


class MVNSqrt(NamedTuple):
    mean: Any
    chol: Any


class AffineModel(NamedTuple):
    A: jnp.ndarray
    b: MVNSqrt

    def __call__(self, x):
        return self.A @ x + self.b


@jax.jit
def mvn_loglikelihood(x, chol_cov):
    dim = chol_cov.shape[0]
    y = jax.scipy.linalg.solve_triangular(chol_cov, x, lower=True)
    normalizing_constant = (
        jnp.sum(jnp.log(jnp.abs(jnp.diag(chol_cov)))) + dim * jnp.log(2 * jnp.pi) / 2.0
    )
    norm_y = jnp.sum(y * y, -1)
    return -0.5 * norm_y - normalizing_constant


@jax.jit
def tria(A):
    return qr(A.T).T


def qr(A: jnp.ndarray):
    # return _qr(A)
    # return jlinalg.qr(A, mode="economic")[1]
    return jnp.linalg.qr(A, mode="r")


@partial(jax.jit, static_argnames="return_q")
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


@jax.jit
def objective_function_value(mnext, m, transition_model):
    F, QL = transition_model
    r = jlinalg.solve_triangular(QL, mnext - F @ m, lower=True)
    return jnp.dot(r, r)


@jax.jit
def _gmul(A: jnp.ndarray, x: MVNSqrt):
    """Multiply a Gaussian with a matrix: A * x"""
    return jax.tree_map(lambda l: A @ l, x)


@jax.jit
def whiten(m, cholP):
    return jlinalg.solve_triangular(cholP.T, m)


def cholesky_update_many(chol_init, update_vectors, multiplier):
    def body(chol, update_vector):
        res = _cholesky_update(chol, update_vector, multiplier=multiplier)
        return res, None

    final_chol, _ = jax.lax.scan(body, chol_init, update_vectors)
    return final_chol


def _cholesky_update(chol, update_vector, multiplier=1.0):
    chol_diag = jnp.diag(chol)

    # The algorithm in [1] is implemented as a double for loop. We can treat
    # the inner loop in Algorithm 3.1 as a vector operation, and thus the
    # whole algorithm as a single for loop, and hence can use a `tf.scan`
    # on it.

    # We use for accumulation omega and b as defined in Algorithm 3.1, since
    # these are updated per iteration.

    def scan_body(carry, inp):
        _, _, omega, b = carry
        index, diagonal_member, col = inp
        omega_at_index = omega[..., index]

        # Line 4
        new_diagonal_member = jnp.sqrt(
            jnp.square(diagonal_member) + multiplier / b * jnp.square(omega_at_index)
        )
        # `scaling_factor` is the same as `gamma` on Line 5.
        scaling_factor = jnp.square(diagonal_member) * b + multiplier * jnp.square(
            omega_at_index
        )

        # The following updates are the same as the for loop in lines 6-8.
        omega = omega - (omega_at_index / diagonal_member)[..., None] * col
        new_col = new_diagonal_member[..., None] * (
            col / diagonal_member[..., None]
            + (multiplier * omega_at_index / scaling_factor)[..., None] * omega
        )
        b = b + multiplier * jnp.square(omega_at_index / diagonal_member)
        return (new_diagonal_member, new_col, omega, b), (
            new_diagonal_member,
            new_col,
            omega,
            b,
        )

    # We will scan over the columns.
    chol = chol.T

    _, (new_diag, new_chol, _, _) = jax.lax.scan(
        scan_body,
        (0.0, jnp.zeros_like(chol[0]), update_vector, 1.0),
        (jnp.arange(0, chol.shape[0]), chol_diag, chol),
    )

    new_chol = new_chol.T
    new_chol = _set_diagonal(new_chol, new_diag)
    new_chol = _set_triu(new_chol, 0.0)
    new_chol = jnp.where(jnp.isfinite(new_chol), new_chol, 0.0)
    return new_chol


def _set_diagonal(x, y):
    N, _ = x.shape
    i, j = jnp.diag_indices(N)
    return x.at[i, j].set(y)


def _set_triu(x, val):
    N, _ = x.shape
    i = jnp.triu_indices(N, 1)
    return x.at[i].set(val)
