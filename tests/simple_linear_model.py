import jax
import jax.numpy as jnp
import tornadox
from jax.scipy.linalg import solve_triangular

from pof.observations import NonlinearModel
from pof.transitions import IWP, projection_matrix
from pof.utils import MVNSqrt
from pof.convenience import linearize_observation_model, discretize_transitions


def get_model():
    iwp = IWP(num_derivatives=0, wiener_process_dimension=1)

    E0 = projection_matrix(iwp, 0)
    obsmod = NonlinearModel(lambda x: E0 @ x)

    x0 = MVNSqrt(jnp.zeros(1), jnp.zeros((1, 1)))

    time_grid = jnp.arange(10)
    disc_transmod = discretize_transitions(iwp, time_grid)

    return x0, disc_transmod, obsmod


def linearize_model(obsmod, x0, N):
    traj = jnp.repeat(x0.mean.reshape(1, -1), N, axis=0)
    lin_obsmod = linearize_observation_model(obsmod, traj)
    return lin_obsmod
