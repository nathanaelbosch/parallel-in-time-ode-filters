""" Additional initial value problems of the same form as in tornadox """
import jax
import jax.numpy as jnp
import tornadox


def logistic(t0=0, tmax=10.0, y0=None):
    y0 = y0 or jnp.array([0.01])

    @jax.jit
    def f(t, y):
        return 1.0 * y * (1 - y)

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def lotkavolterra(t0=0.0, tmax=7, y0=None, p=None):
    y0 = y0 or jnp.array([1.0, 1.0])
    p = p or jnp.array([1.5, 1.0, 3.0, 1.0])

    @jax.jit
    def f(_, Y, p=p):
        a, b, c, d = p
        return jnp.array(
            [
                a * Y[0] - b * Y[0] * Y[1],
                -c * Y[1] + d * Y[0] * Y[1],
            ]
        )

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def vanderpol(t0=0.0, tmax=6.3, y0=None, stiffness_constant=1e1):
    y0 = y0 or jnp.array([2.0, 0.0])

    @jax.jit
    def f_vanderpol(_, Y, mu=stiffness_constant):
        return jnp.array([Y[1], mu * ((1.0 - Y[0] ** 2) * Y[1] - Y[0])])

    return tornadox.ivp.InitialValueProblem(f=f_vanderpol, t0=t0, tmax=tmax, y0=y0)


def fitzhughnagumo(t0=0.0, tmax=100.0, y0=None, p=None):
    y0 = y0 or jnp.array([1.0, 1.0])
    p = p or jnp.array([0.7, 0.8, 1 / 12.5, 0.5])

    @jax.jit
    def f(_, Y, p=p):
        a, b, tinv, l = p
        v = Y[0]
        w = Y[1]
        return jnp.array(
            [
                v - (v**3) / 3 - w + l,
                tinv * (v + a - b * w),
            ]
        )

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def rober(t0=0.0, tmax=1e11, y0=None, p=None):
    y0 = y0 or jnp.array([1.0, 0.0, 0.0])
    p = p or jnp.array([0.04, 3e7, 1e4])

    @jax.jit
    def f(_, Y, p=p):
        k1, k2, k3 = p
        y1, y2, y3 = Y
        return jnp.array(
            [
                -k1 * y1 + k3 * y2 * y3,
                k1 * y1 - k2 * y2**2 - k3 * y2 * y3,
                k2 * y2**2,
            ]
        )

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def rigid_body(t0=0.0, tmax=20.0, y0=None, p=None):
    y0 = y0 or jnp.array([1.0, 0.0, 0.9])
    p = p or jnp.array([-2.0, 1.25, -0.5])

    @jax.jit
    def f(_, y, p=p):
        return jnp.array([p[0] * y[1] * y[2], p[1] * y[0] * y[2], p[2] * y[0] * y[1]])

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def seir(t0=0.0, tmax=200.0, y0=None, p=None):
    y0 = y0 or jnp.array([998.0, 1.0, 1.0, 1.0])
    p = p or jnp.array([0.3, 0.3, 0.1, y0.sum()])

    @jax.jit
    def f(_, y, p=p):
        return jnp.array(
            [
                -p[1] * y[0] * y[2] / p[3],
                p[1] * y[0] * y[2] / p[3] - p[0] * y[1],
                p[0] * y[1] - p[2] * y[2],
                p[2] * y[2],
            ]
        )

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


def threebody(t0=0.0, tmax=17.0652165601579625588917206249, y0=None, p=None):
    Y0 = (0.994, 0)
    DY0 = (0, -2.00158510637908252240537862224)
    y0 = y0 or jnp.array([*Y0, *DY0])
    p = p or jnp.array([0.012277471])

    @jax.jit
    def f(_, y, p=p):
        mu, mp = p[0], 1.0 - p[0]
        D1 = jnp.linalg.norm(jnp.asarray([y[0] + mu, y[1]])) ** 3.0
        D2 = jnp.linalg.norm(jnp.asarray([y[0] - mp, y[1]])) ** 3.0
        du0p = y[0] + 2 * y[3] - mp * (y[0] + mu) / D1 - mu * (y[0] - mp) / D2
        du1p = y[1] - 2 * y[2] - mp * y[1] / D1 - mu * y[1] / D2
        return jnp.asarray([y[2], y[3], du0p, du1p])

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)


# def henon_heiles(u, /, p):
#     """Henon-Heiles dynamics as a second-order differential equation."""
#     x, y = u[0], u[1]
#     ddx = -x - 2 * p * x * y
#     ddy = -y - p * (x**2.0 - y**2.0)
#     return backend.numpy.asarray([ddx, ddy])


def henonheiles(t0=0.0, tmax=100.0, y0=None, p=None):
    y0 = y0 or jnp.array([0.5, 0.0, 0.0, 0.1])
    p = p or 1.0

    @jax.jit
    def f(_, y, p=p):
        return jnp.array(
            [
                y[2],
                y[3],
                -y[0] - 2 * p * y[0] * y[1],
                -y[1] - p * (y[0] ** 2 - y[1] ** 2),
            ]
        )

    return tornadox.ivp.InitialValueProblem(f=f, t0=t0, tmax=tmax, y0=y0)
