import jax
import jax.numpy as jnp

import pof.convergence_criteria
from pof.convenience import get_initial_trajectory, set_up_solver
from pof.sequential_filtsmooth import filtsmooth as seq_fs
from pof.step import ieks_step
from pof.utils import MVNSqrt, _gmul


def solve(
    *, f, y0, ts, order, init="prior", calibrate=True, maxiters=10_000, sequential=False
):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]

    PI = setup["PI"]

    states = get_initial_trajectory(setup, method=init)

    j0 = jnp.zeros(())
    states, nll, obj, ssq, nll_old, obj_old, states_old, k = val = (
        states,
        j0,
        j0,
        j0,
        states,
        j0,
        j0,
        j0,
    )

    @jax.jit
    def cond(val):
        # continue while loop while this is true
        states, nll, obj, ssq, states_old, nll_old, obj_old, k = val
        first_iteration = k < 1
        not_maxiter = k <= maxiters
        converged = pof.convergence_criteria.crit(
            obj, obj_old, nll, nll_old, states, states_old
        )
        return jnp.logical_or(first_iteration, jnp.logical_and(~converged, not_maxiter))

    @jax.jit
    def body(val):
        states_old, nll_old, obj_old, _, _, _, _, k = val

        states, nll, obj, ssq = ieks_step(
            om=om, dtm=dtm, x0=x0, states=states_old, sequential=sequential
        )

        return states, nll, obj, ssq, states_old, nll_old, obj_old, k + 1

    states, nll, obj, ssq, _, _, _, k = val = jax.lax.while_loop(cond, body, val)
    info_dict = {
        "iterations": k,
        "nll": nll,
        "obj": obj,
        "sigma_squared": ssq,
        "calibrated": False,
    }

    if calibrate:
        chols = ssq**0.5 * states.chol
        states = MVNSqrt(states.mean, chols)
        info_dict["calibrated"] = True

    ys = jax.vmap(_gmul, in_axes=[None, 0])(setup["E0"], states)

    return ys, info_dict


def sequential_eks_solve(*, f, y0, ts, order, return_full_states=False, calibrate=True):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]

    states, nll, obj, ssq = seq_fs(x0, dtm, om)
    info_dict = {"nll": nll, "obj": obj, "sigma_squared": ssq, "calibrated": False}

    if calibrate:
        chols = ssq**0.5 * states.chol
        states = MVNSqrt(states.mean, chols)
        info_dict["calibrated"] = True

    if not return_full_states:
        states = jax.vmap(_gmul, in_axes=[None, 0])(setup["E0"], states)
    else:
        states = jax.vmap(_gmul, in_axes=[None, 0])(setup["P"], states)

    return states, info_dict
