import jax.numpy as jnp
import pytest

from pof.ivp import logistic
from pof.solver import sequential_eks_solve, solve


@pytest.fixture
def ivp():
    return logistic()


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("init", ["constant", "prior"])
@pytest.mark.parametrize("dt", [0.5])
def test_full_solve(ivp, order, init, dt):
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    out, info = solve(f=ivp.f, y0=ivp.y0, ts=time_grid, order=order, init=init)
    assert out.mean.shape[0] == len(time_grid)


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("dt", [0.5])
def test_sequential_solve(ivp, order, dt):
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    out, info = sequential_eks_solve(f=ivp.f, y0=ivp.y0, ts=time_grid, order=order)
    assert out.mean.shape[0] == len(time_grid)
