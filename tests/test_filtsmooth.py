import jax
import jax.numpy as jnp
import pytest

from pof.convenience import discretize_transitions, linearize_observation_model
from pof.ivp import logistic
from pof.observations import NonlinearModel
from pof.parallel_filtsmooth import linear_noiseless_filtering as pfilt
from pof.parallel_filtsmooth import smoothing as psmooth
from pof.sequential_filtsmooth import extended_kalman_filter as sfilt
from pof.sequential_filtsmooth import smoothing as ssmooth
from pof.transitions import IWP, projection_matrix
from pof.utils import MVNSqrt


def get_model():
    iwp = IWP(num_derivatives=0, wiener_process_dimension=1)

    E0 = projection_matrix(iwp, 0)
    obsmod = NonlinearModel(lambda x: E0 @ x)

    x0 = MVNSqrt(jnp.zeros(1), jnp.zeros((1, 1)))

    time_grid = jnp.arange(10)
    disc_transmod = discretize_transitions(iwp, time_grid)

    return x0, disc_transmod, obsmod


def linearize_model(obsmod, x0, N):
    traj = jax.tree_map(lambda l: jnp.repeat(l[None, ...], N, axis=0), x0)
    lin_obsmod = linearize_observation_model(obsmod, traj)
    return lin_obsmod


@pytest.fixture
def ivp():
    return logistic()


def test_equality():
    x0, disc_transmod, obsmod = get_model()

    out_ekf, _, _ = sfilt(x0, disc_transmod, obsmod)
    out_eks, _ = ssmooth(disc_transmod, out_ekf)
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
