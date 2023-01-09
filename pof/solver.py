import jax
import jax.numpy as jnp

import pof.convergence_criteria
from pof.convenience import get_initial_trajectory, set_up_solver
from pof.sequential_filtsmooth import filtsmooth as seq_fs
from pof.step import ieks_step
from pof.utils import _gmul


def solve(*, f, y0, ts, order, init="prior"):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]

    PI = setup["PI"]

    states = get_initial_trajectory(setup, method=init)

    j0 = jnp.zeros(())
    states, nll, obj, nll_old, obj_old, k = val = (states, j0, j0, j0 + 1, j0 + 1, j0)

    @jax.jit
    def cond(val):
        states, nll, obj, nll_old, obj_old, k = val
        return ~pof.convergence_criteria.crit(obj, obj_old, nll, nll_old)

    @jax.jit
    def body(val):
        states, nll_old, obj_old, _, _, k = val

        states, nll, obj = ieks_step(om=om, dtm=dtm, x0=x0, states=states)

        return states, nll, obj, nll_old, obj_old, k + 1

    states, nll, obj, _, _, k = val = jax.lax.while_loop(cond, body, val)

    info_dict = {"iterations": k, "nll": nll, "obj": obj}
    ys = jax.vmap(_gmul, in_axes=[None, 0])(setup["E0"], states)

    return ys, info_dict


def sequential_eks_solve(f, y0, ts, order, return_full_states=False):

    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)

    dtm = setup["dtm"]
    om = setup["om"]
    x0 = setup["x0"]

    states, nll = seq_fs(x0, dtm, om)

    info_dict = {"nll": nll}
    if not return_full_states:
        states = jax.vmap(_gmul, in_axes=[None, 0])(setup["E0"], states)
    else:
        states = jax.vmap(_gmul, in_axes=[None, 0])(setup["P"], states)

    return states, ts, info_dict
