import pytest

import tornadox

from pof.diffrax import solve_diffrax, get_ts_ys


@pytest.fixture
def ivp():
    return tornadox.ivp.vanderpol_julia(stiffness_constant=1e1, tmax=8)


def test_diffrax(ivp):
    sol = solve_diffrax(ivp, rtol=1e-10, atol=1e-10)
    ts, ys = get_ts_ys(sol)
    assert ts.shape[0] == ys.shape[0]
