import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pof
from pof.initialization import get_initial_trajectory, get_x0
from pof.ivp import logistic
from pof.sequential_filtsmooth import extended_kalman_filter, smoothing
from pof.solver import make_continuous_models


@pytest.fixture
def ivp():
    return logistic()


@pytest.mark.parametrize("order", [1, 3])
@pytest.mark.parametrize("dt", [0.5])
def test_sequential_eks(ivp, order, dt):
    transition_model, observation_model = make_continuous_models(ivp, order)
    time_grid = jnp.arange(0, ivp.tmax + dt, dt)
    discrete_transition_models = pof.discretize_transitions(transition_model, time_grid)

    x0 = get_x0(ivp, order)

    out, nll = extended_kalman_filter(x0, discrete_transition_models, observation_model)

    assert out.mean.shape[0] == len(time_grid)

    smoothing(discrete_transition_models, out)
