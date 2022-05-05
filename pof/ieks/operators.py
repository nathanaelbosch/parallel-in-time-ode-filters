import jax
import jax.numpy as jnp
import jax.scipy.linalg as jlinalg

from pof.utils import tria


@jax.jit
@jax.vmap
def sqrt_filtering_operator(elem1, elem2):
    A1, b1, U1, eta1, Z1 = elem1
    A2, b2, U2, eta2, Z2 = elem2

    nx = Z2.shape[0]

    Xi = jnp.block([[U1.T @ Z2, jnp.eye(nx)], [Z2, jnp.zeros_like(A1)]])
    tria_xi = tria(Xi)
    Xi11 = tria_xi[:nx, :nx]
    Xi21 = tria_xi[nx : nx + nx, :nx]
    Xi22 = tria_xi[nx : nx + nx, nx:]

    M = jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True)
    A = A2 @ A1 - M.T @ Xi21.T @ A1
    m = jlinalg.solve_triangular(Xi11, U1.T, lower=True)
    b = A2 @ (jnp.eye(nx) - m.T @ Xi21.T) @ (b1 + U1 @ U1.T @ eta2) + b2

    _U = jlinalg.solve_triangular(Xi11, U1.T @ A2.T, lower=True).T
    U = tria(jnp.concatenate([_U, U2], axis=1))
    _e = jlinalg.solve_triangular(Xi11, Xi21.T, lower=True, trans=True)
    eta = A1.T @ (jnp.eye(nx) - _e.T @ U1.T) @ (eta2 - Z2 @ Z2.T @ b1) + eta1
    Z = tria(jnp.concatenate([A1.T @ Xi22, Z1], axis=1))

    return A, b, U, eta, Z


@jax.jit
@jax.vmap
def sqrt_smoothing_operator(elem1, elem2):
    g1, E1, D1 = elem1
    g2, E2, D2 = elem2

    g = E2 @ g1 + g2
    E = E2 @ E1
    D = tria(jnp.concatenate([E2 @ D1, D2], axis=1))

    return g, E, D
