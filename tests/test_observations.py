import jax.numpy as jnp
import pytest

from pof.observations import AffineModel, NonlinearModel, linearize
from pof.utils import MVNSqrt


@pytest.fixture
def f():
    return lambda x: jnp.sin(x)


@pytest.fixture
def nlm(f):
    return NonlinearModel(f)


def test_nonlinearmodel(f, nlm):
    x = jnp.array([1.0, 2.0, 3.0])
    assert all(f(x) == nlm(x))


def test_linearize(nlm):
    x = MVNSqrt(jnp.array([1.0]), jnp.zeros((1, 1)))
    linearized_model = linearize(nlm, x)
    H, b, cholR = linearized_model
    assert isinstance(linearized_model, AffineModel)
    assert H.shape == (1, 1)
    assert b.shape == (1,)
    assert cholR.shape == (1, 1)
    assert all(cholR == 0)

    assert H @ x.mean + b == nlm(x.mean)
