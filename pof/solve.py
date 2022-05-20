import jax.numpy as jnp

# import parsmooth
import tornadox

from .utils import AffineModel, MVNSqrt, linearize

# from parsmooth.linearization import cubature, extended



def make_filter_args(f, y0, T, order, dt):
    d, q = y0.shape[0], order
    D = d * (q + 1)

    iwp = tornadox.iwp.IntegratedWienerTransition(
        num_derivatives=q, wiener_process_dimension=d
    )
    P, PI = iwp.nordsieck_preconditioner(dt)
    E0 = iwp.projection_matrix(0) @ P
    E1 = iwp.projection_matrix(1) @ P
    A, QL = iwp.preconditioned_discretize

    transition_model = AffineModel(A, jnp.zeros(D))

    c = jnp.zeros(d)
    RS = 0 * jnp.eye(d)
    observation_model = lambda x: E1 @ x - f(None, E0 @ x)

    times = jnp.arange(0, T + dt, dt)
    data = jnp.zeros((len(times) - 1, d))

    m0, P0 = tornadox.init.TaylorMode()(f=f, df=None, y0=y0, t0=0, num_derivatives=q)
    m0, P0 = jnp.concatenate(m0.T), jnp.kron(jnp.eye(d), P0)
    m0, P0 = PI @ m0, PI @ P0 @ PI
    x0 = MVNSqrt(m0, P0)
    return transition_model, observation_model, QL, RS, data, x0, times, E0


def solve_ek(
    f,
    y0,
    T,
    order=3,
    dt=1e-2,
    diffusion=0.1,
    smooth=True,
    return_full_states=False,
):
    tm, om, data, x0, ts, E0 = make_filter_args(f, y0, T, order, dt)

    kwargs = {}
    # method = parsmooth.filter_smoother if smooth else parsmooth.filter
    states = method(
        observations=data,
        x0=x0,
        transition_model=tm,
        observation_model=om,
        linearization_method=extended,
        nominal_trajectory=None,
        parallel=False,
    )

    if return_full_states:
        return ts, states
    else:
        return ts, jnp.dot(E0, states.mean.T).T


def mse_criterion(i, prev_traj, curr_traj, tol=1e-6):
    return jnp.mean((pref_traj.mean - curr_traj.mean) ** 2) > 1e-6


def get_solver_iterator(
    ivp,
    order=3,
    dt=1e-2,
    diffusion=0.1,
    parallel=True,
):

    d, q = ivp.y0.shape[0], order
    D = d * (q + 1)

    iwp = tornadox.iwp.IntegratedWienerTransition(
        num_derivatives=q, wiener_process_dimension=d
    )
    P, PI = iwp.nordsieck_preconditioner(dt)
    E0 = iwp.projection_matrix(0) @ P
    E1 = iwp.projection_matrix(1) @ P
    A, QL = iwp.preconditioned_discretize

    b = jnp.zeros(D)
    transition_model = FunctionalModel(lambda x: A @ x, MVNSqrt(b, QL))

    c = jnp.zeros(d)
    RS = 0 * jnp.eye(d)
    observation_model = FunctionalModel(
        lambda x: E1 @ x - ivp.f(None, E0 @ x), MVNSqrt(c, RS)
    )

    times = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    data = jnp.zeros((len(times) - 1, d))
    N = len(times)

    m0, P0 = tornadox.init.TaylorMode()(
        f=ivp.f, df=None, y0=ivp.y0, t0=0, num_derivatives=q
    )
    m0, P0 = jnp.concatenate(m0.T), jnp.kron(jnp.eye(d), P0)
    m0, P0 = PI @ m0, PI @ P0 @ PI
    x0 = MVNSqrt(m0, P0)

    kwargs = {
        "observations": data,
        "x0": x0,
        "transition_model": transition_model,
        "observation_model": observation_model,
        "linearization_method": extended,
    }

    init_lin = MVNSqrt(
        jnp.repeat(x0.mean.reshape(1, -1), data.shape[0] + 1, axis=0),
        jnp.zeros((data.shape[0] + 1, d * (order + 1), d * (order + 1))),
    )

    # @jax.jit
    def refine(trajectory):
        # out = parsmooth.filter_smoother(
        #     **kwargs,
        #     nominal_trajectory=trajectory,
        #     parallel=parallel,
        # )
        return

    def project(trajectory):
        return jnp.dot(E0, trajectory.mean.T).T

    return init_lin, refine, project
