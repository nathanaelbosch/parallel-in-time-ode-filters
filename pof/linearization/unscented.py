from functools import partial
from typing import Tuple, Union, Optional, Callable, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pof.observations import NonlinearModel, AffineModel
from pof.utils import MVNSqrt, tria, cholesky_update_many
import jax.scipy.linalg as jlinalg


class SigmaPoints(NamedTuple):
    points: np.ndarray
    wm: np.ndarray
    wc: np.ndarray


def linearize_unscented(
    f: NonlinearModel,
    x: MVNSqrt,
    alpha: float = 1.0,
    beta: float = 0.0,
    kappa: float = None,
):
    # from: https://github.com/EEA-sensors/sqrt-parallel-smoothers/blob/main/parsmooth/linearization/_unscented.py

    m_x, chol_x = x
    x_pts = _get_sigma_points(x, alpha, beta, kappa)

    f_pts = jax.vmap(f)(x_pts.points)
    m_f = jnp.dot(x_pts.wm, f_pts)

    Psi_x = _cov(x_pts.wc, x_pts.points, m_x, f_pts, m_f)
    F_x = jlinalg.cho_solve((chol_x, True), Psi_x).T

    sqrt_Phi = jnp.sqrt(x_pts.wc[:, None]) * (f_pts - m_f[None, :])
    sqrt_Phi = tria(sqrt_Phi.T)

    chol_L = sqrt_Phi
    # print(chol_L)
    chol_L = cholesky_update_many(chol_L, (F_x @ chol_x).T, -1.0)

    return AffineModel(F_x, m_f - F_x @ m_x, chol_L)


def _get_sigma_points(mvn: MVNSqrt, alpha: float, beta: float, kappa: Optional[float]):
    mean, chol = mvn
    n_dim = mean.shape[0]
    if kappa is None:
        kappa = 3.0 + n_dim
    wm, wc, lamda = _unscented_weights(n_dim, alpha, beta, kappa)
    scaled_chol = jnp.sqrt(n_dim + lamda) * chol

    zeros = jnp.zeros((1, n_dim))
    sigma_points = mean[None, :] + jnp.concatenate(
        [zeros, scaled_chol.T, -scaled_chol.T], axis=0
    )
    return SigmaPoints(sigma_points, wm, wc)


def _unscented_weights(n_dim: int, alpha: float, beta: float, kappa: float):
    lamda = alpha**2 * (n_dim + kappa) - n_dim
    wm = jnp.full(2 * n_dim + 1, 1 / (2 * (n_dim + lamda)))

    wm = wm.at[0].set(lamda / (n_dim + lamda))
    wc = wm.at[0].set(lamda / (n_dim + lamda) + (1 - alpha**2 + beta))
    return wm, wc, lamda


def _cov(wc, x_pts, x_mean, y_points, y_mean):
    one = (x_pts - x_mean[None, :]).T * wc[None, :]
    two = y_points - y_mean[None, :]
    return jnp.dot(one, two)
