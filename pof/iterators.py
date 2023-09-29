import jax
import jax.numpy as jnp

import pof.convergence_criteria
import pof.initialization as init
from pof.convenience import get_initial_trajectory, set_up_solver
from pof.observations import AffineModel, linearize, linearize_regularized
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.step import ieks_step

fs = linear_filtsmooth
if jax.lib.xla_bridge.get_backend().platform == "gpu":
    fs = jax.jit(linear_filtsmooth)
lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))
lom_reg = jax.jit(
    jax.vmap(linearize_regularized, in_axes=[None, 0, None]), static_argnums=(0,)
)


def ieks_iterator(*, f, y0, ts, order, init="prior"):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)
    states = get_initial_trajectory(setup, method=init)
    iterator = _ieks_iterator(setup["dtm"], setup["om"], setup["x0"], states)
    return iterator, setup


def _ieks_iterator(dtm, om, x0, init_traj):
    states, nll, obj, ssq = ieks_step(om=om, dtm=dtm, x0=x0, states=init_traj)
    yield states, nll, obj, ssq

    while True:
        nll_old, obj_old, states_old = nll, obj, states

        states, nll, obj, ssq = ieks_step(om=om, dtm=dtm, x0=x0, states=states_old)

        yield states, nll, obj, ssq

        if pof.convergence_criteria.crit(
            obj, obj_old, nll, nll_old, states, states_old
        ):
            break


def qpm_ieks_iterator(*, f, y0, ts, order, init="prior", **kwargs):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)
    states = get_initial_trajectory(setup, method=init)
    iterator = _qpm_ieks_iterator(
        setup["dtm"], setup["om"], setup["x0"], states, **kwargs
    )
    return iterator, setup


def _qpm_ieks_iterator(
    dtm,
    om,
    x0,
    init_traj,
    reg_start=1e20,
    reg_final=1e-20,
    steps=40,
    tau_start=None,
    tau_final=None,
):
    N = dtm.F.shape[0]

    dom = lom(om, jax.tree_map(lambda l: l[1:], init_traj))

    reg_fact = (reg_final / reg_start) ** (1 / steps)
    if tau_start is None:
        # tau_start, tau_final = jnp.sqrt(reg_start), jnp.sqrt(reg_final)
        tau_start, tau_final = 1e5, 1e-5
    tau_fact = (tau_final / tau_start) ** (1 / steps)
    reg, tau = reg_start, tau_start

    init_covs = init_traj.chol
    if jnp.all(init_covs == 0):
        init_covs = init._prior_init(x0=x0, dtm=dtm).chol
    # cholR = jax.vmap(lambda H, cP: reg * tria(H @ cP))(dom.H, init_covs)
    cholR = jax.vmap(lambda cR: jnp.eye(*cR.shape))(dom.cholR)

    dom = jax.vmap(AffineModel)(dom.H, dom.b, reg * cholR / N)
    states, nll, obj, ssq = fs(x0, dtm, dom)
    yield states, nll, obj, reg

    while True:
        nll_old, obj_old, states_old = nll, obj, states

        dom = lom(om, jax.tree_map(lambda l: l[1:], states))
        dom = jax.vmap(AffineModel)(dom.H, dom.b, reg * cholR / N)
        states, nll, obj, ssq = fs(x0, dtm, dom)

        yield states, nll, obj, reg

        if pof.convergence_criteria.crit(
            obj, obj_old, nll, nll_old, states, states_old, rtol=tau, atol=tau
        ):
            reg *= reg_fact
            tau *= tau_fact
            if reg == 0:
                break
            elif reg < reg_final:
                reg = 0.0
                tau = min(tau_final, 1e-5)

        if jnp.isnan(nll) or jnp.isnan(obj):
            break


def lm_ieks_iterator(dtm, om, x0, init_traj, reg=1e0, nu=10.0):
    dom = lom_reg(om, init_traj, reg)
    out, nll, obj, ssq = fs(x0, dtm, dom)
    yield out, nll, obj, reg

    while True:
        nll_old, obj_old, out_old = nll, obj, out

        dom = lom_reg(om, jax.tree_map(lambda l: l[1:], out), reg)
        out, nll, obj, ssq = fs(x0, dtm, dom)

        # if obj < obj_old:
        #     reg /= nu
        # else:
        #     reg *= nu
        #     out = out_old

        yield out, nll, obj, reg

        if (
            (jnp.isclose(nll_old, nll) and jnp.isclose(obj_old, obj))
            or jnp.isnan(nll)
            or jnp.isnan(obj)
        ):
            break


def admm_ieks_iterator(*, f, y0, ts, order, init="prior"):
    setup = set_up_solver(f=f, y0=y0, ts=ts, order=order)
    states = get_initial_trajectory(setup, method=init)
    iterator = _admm_ieks_iterator(setup["dtm"], setup["om"], setup["x0"], states)
    return iterator, setup


def _admm_ieks_iterator(dtm, om, x0, init_traj, rho=1):
    dom = linearize_at_previous_states(om, init_traj)
    states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)
    yield states, nll, obj

    while True:
        nll_old, obj_old, mean_old = nll, obj, states.mean

        dom = linearize_at_previous_states(om, states)
        states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)

        yield states, nll, obj

        if pof.convergence_criteria.crit(obj, obj_old, nll, nll_old):
            break
