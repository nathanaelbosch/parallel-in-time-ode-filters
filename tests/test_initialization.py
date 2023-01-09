import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import *
from pof.ivp import logistic
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.solver import make_continuous_models
from pof.transitions import *


@pytest.fixture
def ivp():
    return logistic()


orders = [1, 3, 5]
dts = [1e-1, 1e-3]


@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("dt", dts)
def test_constant_init(ivp, order, dt):
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    initial_trajectory = constant_init(y0=ivp.y0, f=ivp.f, order=order, ts=time_grid)


@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("dt", dts)
def test_prior_init(ivp, order, dt):
    x0 = taylor_mode_init(ivp.f, ivp.y0, order)
    iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])

    initial_trajectory = prior_init(x0=x0, dtm=dtm)


@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("dt", dts)
def test_updated_prior_init(ivp, order, dt):
    x0 = taylor_mode_init(ivp.f, ivp.y0, order)
    iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])

    P, PI = nordsieck_preconditioner(iwp, dt)
    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))

    initial_trajectory = updated_prior_init(x0=x0, dtm=dtm, om=om)


@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("dt", dts)
def test_coarse_ekf_init(ivp, order, dt):
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    initial_trajectory = coarse_ekf_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts)


@pytest.mark.parametrize("order", orders)
@pytest.mark.parametrize("dt", dts)
def test_coarse_rk_init(ivp, order, dt):
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    initial_trajectory = coarse_rk_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts)
