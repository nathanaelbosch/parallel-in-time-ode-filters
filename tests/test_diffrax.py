import pytest
import tornadox

from pof.diffrax import get_ts_ys, solve_diffrax


@pytest.fixture
def ivp():
    return tornadox.ivp.vanderpol_julia(stiffness_constant=1e1, tmax=8)


def test_diffrax(ivp):
    sol = solve_diffrax(ivp, rtol=1e-10, atol=1e-10)
    ts, ys = get_ts_ys(sol)
    assert ts.shape[0] == ys.shape[0]
