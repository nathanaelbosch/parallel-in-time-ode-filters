import jax
import jax.numpy as jnp
import tornadox


def logistic(t0=0, tmax=10.0, y0=None):
    y0 = y0 or jnp.array([0.01])

    @jax.jit
    def f(t, y):
        return 1.0 * y * (1 - y)

    return tornadox.ivp.InitialValueProblem(
        f=f,
        t0=t0,
        tmax=tmax,
        y0=y0,
    )
