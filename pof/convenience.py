import jax

import pof.initialization as init
from pof.observations import *
from pof.transitions import *
from pof.utils import _gmul


def linearize_observation_model(observation_model, trajectory):
    return jax.vmap(linearize, in_axes=[None, 0])(observation_model, trajectory)


def set_up_solver(*, f, y0, ts, order):
    dts = ts[1:] - ts[:-1]
    # assert jnp.all(jnp.isclose(dts, dts[0]))
    dt = dts[0]

    iwp = IWP(num_derivatives=order, wiener_process_dimension=y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    # N = dtm.QL.shape[0]
    # dtm = TransitionModel(
    #     dtm.F, (1.0 ** jnp.linspace(ts[0], ts[-1], N)[:, None, None]) * dtm.QL
    # )
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))

    x0 = init.taylor_mode_init(f, y0, order)
    x0 = jax.tree_map(lambda x: PI @ x, x0)

    return {
        "f": f,
        "y0": y0,
        "ts": ts,
        "dtm": dtm,
        "om": om,
        "x0": x0,
        "E0": E0,
        "P": P,
        "PI": PI,
        "order": order,
        "iwp": iwp,
    }


def set_up_solver_no_precond(*, f, y0, ts, order):
    dts = ts[1:] - ts[:-1]

    iwp = IWP(num_derivatives=order, wiener_process_dimension=y0.shape[0])
    get_transition_models = jax.vmap(get_transition_model, in_axes=[None, 0])
    dtm = discrete_transition_models = get_transition_models(iwp, dts)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)

    om = observation_model = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))

    x0 = init.taylor_mode_init(f, y0, order)

    P = PI = jnp.eye(x0.mean.shape[0])

    return {
        "ts": ts,
        "dtm": dtm,
        "om": om,
        "x0": x0,
        "E0": E0,
        "P": P,
        "PI": PI,
        "order": order,
        "iwp": iwp,
    }


def get_initial_trajectory(setup, method="prior"):
    f, y0, order = setup["f"], setup["y0"], setup["order"]
    ts = setup["ts"]
    PI = setup["PI"]
    if method == "coarse":
        states = init.coarse_ekf_init(y0=y0, order=order, ts=ts, f=f, N=100)
        states = jax.vmap(_gmul, in_axes=[None, 0])(PI, states)
        return states
    elif method == "constant":
        states = init.constant_init(y0=y0, order=order, ts=ts, f=f)
        states = jax.vmap(_gmul, in_axes=[None, 0])(PI, states)
        return states
    elif method == "prior":
        states = init.prior_init(f=f, y0=y0, order=order, ts=ts)
        return states
    else:
        raise Exception(f"method={method} not found")
