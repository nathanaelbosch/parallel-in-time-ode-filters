import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import get_initial_trajectory, taylor_mode_init
from pof.ivp import logistic
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.solver import make_continuous_models


@pytest.fixture
def ivp():
    return logistic()


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("dt", [0.5])
def test_full_solve(ivp, order, dt):
    transition_model, observation_model = make_continuous_models(ivp.f, ivp.y0, order)
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    discrete_transition_models = pof.discretize_transitions(transition_model, time_grid)
    initial_trajectory = get_initial_trajectory(ivp.y0, ivp.f, order, N=len(time_grid))
    linearized_observation_models = pof.linearize_observation_model(
        observation_model, jax.tree_map(lambda l: l[1:], initial_trajectory)
    )

    x0 = taylor_mode_init(ivp.f, ivp.y0, order)

    out, nll, obj = linear_filtsmooth(
        x0, discrete_transition_models, linearized_observation_models
    )
