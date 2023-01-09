import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import taylor_mode_init
from pof.ivp import logistic
from pof.parallel_filtsmooth import linear_noiseless_filtering as pfilt
from pof.parallel_filtsmooth import smoothing as psmooth
from pof.sequential_filtsmooth import extended_kalman_filter as sfilt
from pof.sequential_filtsmooth import smoothing as ssmooth
from pof.solver import sequential_eks_solve
from tests.simple_linear_model import get_model, linearize_model


@pytest.fixture
def ivp():
    return logistic()


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("dt", [0.5])
def test_sequential_eks(ivp, order, dt):
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    out, _, info = sequential_eks_solve(
        ivp.f, ivp.y0, time_grid, order, return_full_states=False
    )

    assert out.mean.shape[0] == len(time_grid)


def test_equality():
    x0, disc_transmod, obsmod = get_model()

    out_ekf, _ = sfilt(x0, disc_transmod, obsmod)
    out_eks = ssmooth(disc_transmod, out_ekf)
    assert out_ekf.mean.shape == out_eks.mean.shape
    assert out_ekf.chol.shape == out_eks.chol.shape

    lin_obsmod = linearize_model(obsmod, x0, disc_transmod.F.shape[0])
    out_pkf, _, _, _ = pfilt(x0, disc_transmod, lin_obsmod)
    out_pks, _ = psmooth(disc_transmod, out_pkf)
    assert out_pkf.mean.shape == out_pks.mean.shape
    assert out_pkf.chol.shape == out_pks.chol.shape

    assert out_ekf.mean.shape == out_pkf.mean.shape
    assert out_ekf.chol.shape == out_pkf.chol.shape

    assert all(out_ekf.mean == out_pkf.mean)
    assert all(out_ekf.chol == out_pkf.chol)

    assert all(out_eks.mean == out_pks.mean)
    assert all(out_eks.chol == out_pks.chol)
