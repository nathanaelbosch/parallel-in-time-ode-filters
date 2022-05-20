import pytest
import tornadox

from pof.ivp import logistic, lotkavolterra


@pytest.mark.parametrize("ivp", [logistic, lotkavolterra])
def test_ivp(ivp):
    ivp = ivp()
    assert isinstance(ivp, tornadox.ivp.InitialValueProblem)
