from functools import partial

import jax
import jax.numpy as jnp
from pof.convenience import linearize_observation_model

from pof.diffrax import solve_diffrax
from pof.initialization import classic_to_init, taylor_mode_init
from pof.observations import NonlinearModel, linearize
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth as seq_fs
from pof.transitions import (
    IWP,
    TransitionModel,
    nordsieck_preconditioner,
    preconditioned_discretize,
    projection_matrix,
)
from pof.utils import MVNSqrt
from pof.initialization import *


def make_continuous_models(f, y0, order):
    d = y0.shape[0]
    iwp = IWP(num_derivatives=order, wiener_process_dimension=d)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    observation_model = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))
    return iwp, observation_model


@jax.jit
def _gmul(A: jnp.ndarray, x: MVNSqrt):
    return jax.tree_map(lambda l: A @ l, x)


@partial(jax.jit, static_argnames=("f",))
def get_coarse_sol(f, y0, tspan, dt_fine):
    t0, tmax = tspan
    dt_coarse = (tmax - t0) / jnp.ceil(jnp.log2((tmax - t0) / dt_fine))
    sol = solve_diffrax(f, y0, tspan, dt=dt_coarse)
    return sol


def solve(*, f, y0, ts, order, coarse_N=10):
    dt = ts[1] - ts[0]

    iwp = IWP(num_derivatives=order, wiener_process_dimension=y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))

    x0 = taylor_mode_init(f, y0, order)
    x0 = _gmul(PI, x0)

    # # sol_init = get_coarse_sol(f, y0, tspan, dt)
    # # ys = jax.vmap(sol_init.evaluate)(ts)
    # # states = classic_to_init(ys=ys, f=f, order=order)
    # states = constant_init(y0=y0, order=order, ts=ts, f=f)
    states = coarse_ekf_init(y0=y0, order=order, ts=ts, f=f, N=coarse_N)
    states = jax.vmap(_gmul, in_axes=[None, 0])(PI, states)

    j0 = jnp.zeros(())
    states, nll, obj, nll_old, obj_old, k = val = (states, j0, j0, j0 + 1, j0 + 1, j0)

    @jax.jit
    def cond(val):
        states, nll, obj, nll_old, obj_old, k = val
        # converged = jnp.logical_and(
        #     jnp.isclose(nll_old, nll), jnp.isclose(obj_old, obj)
        # )
        # isnan = jnp.logical_or(jnp.isnan(nll), jnp.isnan(obj))
        # return ~jnp.logical_or(jnp.logical_or(converged, isnan), k > 10)
        return ~jnp.isclose(obj_old, obj)
        # return k < 20

    @jax.jit
    def body(val):
        states, nll_old, obj_old, _, _, k = val

        dom = linearize_at_previous_states(om, states)
        states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)

        return states, nll, obj, nll_old, obj_old, k + 1

    states, nll, obj, _, _, k = val = jax.lax.while_loop(cond, body, val)

    info_dict = {"iterations": k, "nll": nll, "obj": obj}
    states = jax.vmap(_gmul, in_axes=[None, 0])(E0, states)

    return states, info_dict


@partial(jax.jit, static_argnames="om")
def linearize_at_previous_states(om, prev_states):
    lin_traj = jax.tree_map(lambda l: l[1:], prev_states)
    vlinearize = jax.vmap(linearize, in_axes=[None, 0])
    dom = vlinearize(om, lin_traj)
    return dom


def sequential_eks_solve(f, y0, ts, order, return_full_states=False):
    # t0, tmax = tspan
    # ts = jnp.arange(t0, tmax + dt, dt)
    dt = ts[1] - ts[0]

    iwp = IWP(num_derivatives=order, wiener_process_dimension=y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))

    x0 = taylor_mode_init(f, y0, order)
    x0 = _gmul(PI, x0)

    states, nll = seq_fs(x0, dtm, om)

    info_dict = {"nll": nll}
    if not return_full_states:
        states = jax.vmap(_gmul, in_axes=[None, 0])(E0, states)
    else:
        states = jax.vmap(_gmul, in_axes=[None, 0])(P, states)

    return states, ts, info_dict
