import jax.numpy as jnp
import numpy as np
import pytest

from pof.transitions import (
    IWP,
    TransitionModel,
    get_transition_model,
    projection_matrix,
)


@pytest.mark.parametrize("dim", [1, 3, 5])
@pytest.mark.parametrize("order", [1, 3, 5])
def test_iwp(dim, order):
    iwp = IWP(num_derivatives=order, wiener_process_dimension=dim)


@pytest.mark.parametrize("dim", [1, 3, 5])
@pytest.mark.parametrize("order", [1, 3, 5])
def test_projection_matrix(dim, order):
    iwp = IWP(num_derivatives=order, wiener_process_dimension=dim)
    D = dim * (order + 1)

    E0 = projection_matrix(iwp, 0)
    E1 = projection_matrix(iwp, 1)
    assert E0.shape == (dim, D)
    assert E1.shape == (dim, D)

    x = np.random.uniform(size=D)
    assert all(E0 @ x == x[0 :: order + 1])
    assert all(E1 @ x == x[1 :: order + 1])


def test_discretize():
    d, q = 2, 3
    iwp = IWP(num_derivatives=q, wiener_process_dimension=d)
    D = d * (q + 1)
    dt = 0.1
    tm = get_transition_model(iwp, dt)
    assert isinstance(tm, TransitionModel)
    F, QL = tm
    assert F.shape == (D, D)
    assert QL.shape == (D, D)
