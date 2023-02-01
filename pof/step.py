from functools import partial

import jax

from pof.observations import linearize, linearize_regularized
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.linearization.unscented import linearize_unscented
from pof.utils import MVNSqrt


@partial(jax.jit, static_argnames="om")
def linearize_at_previous_states(om, prev_states):
    lin_traj = jax.tree_map(lambda l: l[1:], prev_states)
    vlinearize = jax.vmap(
        linearize,
        # linearize_unscented,
        # lambda f, x: linearize_regularized(f, x, 1e3),
        in_axes=[None, 0],
    )
    dom = vlinearize(om, lin_traj)
    return dom


@partial(jax.jit, static_argnames=["om", "calibrate"])
def ieks_step(*, om, dtm, x0, states, calibrate=True):
    dom = linearize_at_previous_states(om, inflate(states))
    states, nll, obj, ssq = linear_filtsmooth(x0, dtm, dom)
    if calibrate:
        chols = ssq**0.5 * states.chol
        states = MVNSqrt(states.mean, chols)
    return states, nll, obj, ssq
