import jax

from pof.observations import linearize
from pof.transitions import get_transition_model


def discretize_transitions(iwp, times=None, steps=None):
    if steps is None:
        steps = times[1:] - times[:-1]
    get_transitions = jax.vmap(get_transition_model, in_axes=[None, 0])
    return get_transitions(iwp, steps)


def linearize_observation_model(observation_model, trajectory):
    return jax.vmap(linearize, in_axes=[None, 0])(observation_model, trajectory)
