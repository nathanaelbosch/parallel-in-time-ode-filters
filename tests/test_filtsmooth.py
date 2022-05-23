import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import get_initial_trajectory, get_x0
from pof.ivp import logistic
from pof.sequential_filtsmooth import (
    extended_kalman_filter as sfilt,
    smoothing as ssmooth,
)
from pof.parallel_filtsmooth import (
    linear_noiseless_filtering as pfilt,
    smoothing as psmooth,
)
from pof.solver import make_continuous_models

from tests.simple_linear_model import get_model, linearize_model


@pytest.fixture
def ivp():
    return logistic()


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("dt", [0.5])
def test_sequential_eks(ivp, order, dt):
    transition_model, observation_model = make_continuous_models(ivp.f, ivp.y0, order)
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    discrete_transition_models = pof.discretize_transitions(transition_model, time_grid)

    x0 = get_x0(ivp.f, ivp.y0, order)

    out, nll = sfilt(x0, discrete_transition_models, observation_model)

    assert out.mean.shape[0] == len(time_grid)

    ssmooth(discrete_transition_models, out)


def test_equality():
    x0, disc_transmod, obsmod = get_model()

    out_ekf, _ = sfilt(x0, disc_transmod, obsmod)
    out_eks = ssmooth(disc_transmod, out_ekf)
    assert out_ekf.mean.shape == out_eks.mean.shape
    assert out_ekf.chol.shape == out_eks.chol.shape

    lin_obsmod = linearize_model(obsmod, x0, disc_transmod.F.shape[0])
    out_pkf, _, _ = pfilt(x0, disc_transmod, lin_obsmod)
    out_pks, _ = psmooth(disc_transmod, out_pkf)
    assert out_pkf.mean.shape == out_pks.mean.shape
    assert out_pkf.chol.shape == out_pks.chol.shape

    assert out_ekf.mean.shape == out_pkf.mean.shape
    assert out_ekf.chol.shape == out_pkf.chol.shape

    assert all(out_ekf.mean == out_pkf.mean)
    assert all(out_ekf.chol == out_pkf.chol)

    assert all(out_eks.mean == out_pks.mean)
    assert all(out_eks.chol == out_pks.chol)
