import jax.numpy as jnp


def crit(obj, obj_old, nll, nll_old, *, rtol=1e-5, atol=1e-8):
    # rtol=1e-5 and atol=1e-8 are the numpy defaults
    isnan = jnp.logical_or(jnp.isnan(obj), jnp.isnan(nll))
    converged = jnp.isclose(obj_old, obj, rtol=rtol, atol=atol)
    # converged = jnp.logical_and(
    #     jnp.isclose(obj_old, obj, rtol=rtol, atol=atol),
    #     jnp.isclose(nll_old, nll, rtol=rtol, atol=atol),
    # )
    return jnp.logical_or(converged, isnan)
