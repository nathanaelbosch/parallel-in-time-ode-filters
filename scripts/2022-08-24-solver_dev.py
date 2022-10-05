import diffrax
import jax
import jax.numpy as jnp

from pof.convenience import *
from pof.initialization import *
from pof.ivp import *
from pof.parallel_filtsmooth import *
from pof.sequential_filtsmooth import *
from pof.solver import *
from pof.solver import _gmul
from pof.iterators import *


ivp, name = (logistic(), "logistic")
f, y0, t0, tmax = ivp.f, ivp.y0, ivp.t0, ivp.tmax
dt = 1e-2
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
order = 3


def solve(*, f, ts, order):
    dt = ts[1] - ts[0]

    iwp = IWP(num_derivatives=order, wiener_process_dimension=y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - f(None, E0 @ x))

    x0 = taylor_mode_init(f, y0, order)
    x0 = _gmul(PI, x0)

    # # sol_init = get_coarse_sol(f, y0, tspan, dt)
    # # ys = jax.vmap(sol_init.evaluate)(ts)
    # # states = classic_to_init(ys=ys, f=f, order=order)

    # states = coarse_ekf_init(y0=y0, order=order, ts=ts, f=f)
    states = coarse_ekf_init(y0=y0, order=order, ts=ts, f=f)
    # states = constant_init(y0=y0, order=order, ts=ts, f=f)
    states = jax.vmap(_gmul, in_axes=[None, 0])(PI, states)

    j0 = jnp.zeros(())
    states, nll, obj, nll_old, obj_old, k = val = (states, j0, j0, j0 + 1, j0 + 1, j0)

    # @jax.jit
    def cond(val):
        states, nll, obj, nll_old, obj_old, k = val
        # converged = jnp.logical_and(
        #     jnp.isclose(nll_old, nll), jnp.isclose(obj_old, obj)
        # )
        # isnan = jnp.logical_or(jnp.isnan(nll), jnp.isnan(obj))
        # return ~jnp.logical_or(jnp.logical_or(converged, isnan), k > 10)
        return ~jnp.isclose(obj_old, obj)
        # return k < 20

    @jax.jit
    def body(val):
        states, nll_old, obj_old, _, _, k = val

        dom = linearize_at_previous_states(om, states)
        states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)

        return states, nll, obj, nll_old, obj_old, k + 1

    # while cond(val):
    # states, nll, obj, _, _, k = val = body(val)
    states, nll, obj, _, _, k = val = jax.lax.while_loop(cond, body, val)

    info_dict = {"iterations": k, "nll": nll, "obj": obj}
    states = jax.vmap(_gmul, in_axes=[None, 0])(E0, states)

    return states, info_dict


# solve(f=f, ts=ts, order=order)
states, info = jax.jit(solve, static_argnames=["f", "order"])(f=f, ts=ts, order=order)
