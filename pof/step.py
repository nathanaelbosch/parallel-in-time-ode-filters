from functools import partial

import jax

from pof.observations import linearize
from pof.parallel_filtsmooth import linear_filtsmooth


@partial(jax.jit, static_argnames="om")
def linearize_at_previous_states(om, prev_states):
    lin_traj = jax.tree_map(lambda l: l[1:], prev_states)
    vlinearize = jax.vmap(linearize, in_axes=[None, 0])
    dom = vlinearize(om, lin_traj)
    return dom


@partial(jax.jit, static_argnames="om")
def ieks_step(*, om, dtm, x0, states):
    dom = linearize_at_previous_states(om, states)
    states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)
    return states, nll, obj
