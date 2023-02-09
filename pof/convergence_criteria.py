import jax.numpy as jnp


def crit(obj, obj_old, nll, nll_old, states, states_old, *, rtol=1e-6, atol=1e-9):
    # rtol=1e-5 and atol=1e-8 are the numpy defaults
    isnan = jnp.logical_or(jnp.isnan(obj), jnp.isnan(nll))
    obj_converged = jnp.isclose(obj_old, obj, rtol=rtol, atol=atol)

    means, means_old = states.mean, states_old.mean
    means_converged = jnp.isclose(means_old, means, rtol=1e-13).all()

    converged = jnp.logical_or(obj_converged, means_converged)
    return jnp.logical_or(isnan, converged)
