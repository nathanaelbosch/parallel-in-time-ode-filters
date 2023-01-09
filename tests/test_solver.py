import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import constant_init, taylor_mode_init
from pof.ivp import logistic
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.solver import solve


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
