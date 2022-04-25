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


def lotkavolterra(t0=0.0, tmax=7, y0=None, p=None):

    y0 = y0 or jnp.array([1.0, 1.0])
    p = p or jnp.array([1.5, 1.0, 3.0, 1.0])

    @jax.jit
    def f(_, Y, p=p):
        a, b, c, d = p
        return jnp.array([a * Y[0] - b * Y[0] * Y[1], -c * Y[1] + d * Y[0] * Y[1]])

    return tornadox.ivp.InitialValueProblem(
        f=f,
        t0=t0,
        tmax=tmax,
        y0=y0,
    )
