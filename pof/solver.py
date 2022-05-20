import jax
import jax.numpy as jnp
import tornadox
from jax.scipy.linalg import solve_triangular

from pof.observations import NonlinearModel
from pof.transitions import IWP, projection_matrix


def make_continuous_models(ivp, order):
    d = ivp.y0.shape[0]
    iwp = IWP(num_derivatives=order, wiener_process_dimension=d)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    observation_model = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))
    return iwp, observation_model
