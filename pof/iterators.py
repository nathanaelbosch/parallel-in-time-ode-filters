import jax
import jax.numpy as jnp

import pof.initialization as init
from pof.observations import AffineModel
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.observations import linearize, linearize_regularized

fs = linear_filtsmooth
lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))
lom_reg = jax.jit(
    jax.vmap(linearize_regularized, in_axes=[None, 0, None]), static_argnums=(0,)
)


def ieks_iterator(dtm, om, x0, init_traj):

    dom = lom(om, init_traj)
    out, nll, obj, ssq = fs(x0, dtm, dom)
    yield out, nll, obj

    while True:

        nll_old, obj_old, mean_old = nll, obj, out.mean

        dom = lom(om, jax.tree_map(lambda l: l[1:], out))
        out, nll, obj, ssq = fs(x0, dtm, dom)

        yield out, nll, obj

        if (
            (jnp.isclose(nll_old, nll) and jnp.isclose(obj_old, obj))
            or jnp.isnan(nll)
            or jnp.isnan(obj)
        ):
            break


def qpm_ieks_iterator(dtm, om, x0, init_traj):

    dom = lom(om, init_traj)

    reg_start, reg_final, steps = 1e0, 1e-20, 20
    reg_fact = (reg_final / reg_start) ** (1 / steps)
    tau_start, tau_final = 1e10, 1e-10
    tau_fact = (tau_final / tau_start) ** (1 / steps)
    reg, tau = reg_start, tau_start

    init_covs = init_traj.chol
    if jnp.all(init_covs == 0):
        init_covs = init.prior_init(x0=x0, dtm=dtm).chol
    # cholR = jax.vmap(lambda H, cP: reg * tria(H @ cP))(dom.H, init_covs)
    cholR = jax.vmap(lambda cR: jnp.eye(*cR.shape))(dom.cholR)
    cholR = cholR / cholR.shape[0]
    dom = jax.vmap(AffineModel)(dom.H, dom.b, reg * cholR)

    out, nll, obj, ssq = fs(x0, dtm, dom)
    yield out, nll, obj, reg

    while True:

        nll_old, obj_old, out_old = nll, obj, out

        dom = lom(om, jax.tree_map(lambda l: l[1:], out))
        dom = jax.vmap(AffineModel)(
            dom.H,
            dom.b,
            # jax.vmap(lambda H, cP: reg * tria(H @ cP))(dom.H, init_covs),
            # jax.vmap(lambda H, cP: reg * tria(H @ cP))(dom.H, out_old.chol[1:]),
            reg * cholR,
        )
        out, nll, obj, ssq = fs(x0, dtm, dom)

        yield out, nll, obj, reg

        if jnp.isclose(obj_old - nll_old, obj - nll, rtol=tau):
            if reg < 1e-20:
                break
            else:
                reg *= reg_fact
                tau *= tau_fact


def lm_ieks_iterator(dtm, om, x0, init_traj):
    reg, nu = 1e0, 10.0

    dom = lom_reg(om, init_traj, reg)
    out, nll, obj, ssq = fs(x0, dtm, dom)
    yield out, nll, obj

    while True:

        nll_old, obj_old, out_old = nll, obj, out

        dom = lom_reg(om, jax.tree_map(lambda l: l[1:], out), reg)
        out, nll, obj, ssq = fs(x0, dtm, dom)

        if obj < obj_old:
            reg /= nu
        else:
            reg *= nu
            out = out_old

        yield out, nll, obj

        if (
            (jnp.isclose(nll_old, nll) and jnp.isclose(obj_old, obj))
            or jnp.isnan(nll)
            or jnp.isnan(obj)
        ):
            break
