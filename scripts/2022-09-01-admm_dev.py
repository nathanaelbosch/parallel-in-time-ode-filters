from functools import partial

# import diffrax
import jax
import jax.numpy as jnp
from tqdm import trange
import matplotlib.pyplot as plt

import pof
from pof.diffrax import solve_diffrax
import pof.initialization as init
from pof.ivp import *
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.transitions import *
from pof.iterators import *
from pof.step import *
import pof.convergence_criteria

MAXITER = 1000


def wrap_iterator(iterator, maxiter=MAXITER):
    """Wrap the iterator to add a TQDM bar"""
    bar = trange(maxiter)
    for _, (out, nll, obj, *_) in zip(bar, iterator):
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} other={_}]")
        yield out, nll, obj


@partial(jax.vmap, in_axes=[0, None])
def project(states, setup):
    mean, cholcov = states
    pmean = jnp.dot(setup["E0"], mean.T).T
    pchol = setup["E0"] @ cholcov
    pcov = pchol @ pchol.T
    return pmean, pcov


ivp, ylims = vanderpol(tmax=100, stiffness_constant=1e0), (-2.5, 2.5)
order = 3
dt = 1e-1
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, max_steps=int(1e6))
ys_true = jax.vmap(sol_true.evaluate)(ts)


def _admm_adjust_dom(dom, Y, rho_sqrt):
    d = Y.shape[-1]
    _cholR = jnp.eye(d) / rho_sqrt
    cholR = jax.vmap(lambda _: _cholR)(dom.cholR)
    bnew = dom.b + Y / (rho_sqrt ** 2)
    dom = jax.vmap(AffineModel)(dom.H, bnew, cholR)
    return dom


def _inner_admm_ieks_iterator(dtm, om, x0, states, Y, rho_sqrt=1e2):

    dom = linearize_at_previous_states(om, states)
    dom = _admm_adjust_dom(dom, Y, rho_sqrt)
    states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)

    yield states, nll, obj

    while True:

        nll_old, obj_old, mean_old = nll, obj, states.mean

        dom = linearize_at_previous_states(om, states)
        dom = _admm_adjust_dom(dom, Y, rho_sqrt)
        states, nll, obj, _ = linear_filtsmooth(x0, dtm, dom)

        yield states, nll, obj, Y.mean(axis=0)

        if pof.convergence_criteria.crit(obj, obj_old, nll, nll_old, rtol=1e-3):
            break


def _admm_ieks_iterator(dtm, om, x0, states, Y0, rho_sqrt=1e-1):
    Y = Y0
    for _ in range(50):
        __iter = _inner_admm_ieks_iterator(dtm, om, x0, states, Y, rho_sqrt)

        for states, *stuff in __iter:
            yield states, *stuff

        e = jax.vmap(om)(states.mean[1:])
        Y = Y + rho_sqrt ** 2 * e


setup = set_up_solver(f=ivp.f, y0=ivp.y0, ts=ts, order=order)
states = get_initial_trajectory(setup, method="prior")
Y = jnp.zeros((states.mean.shape[0] - 1, ivp.y0.shape[0]))
_iter = _admm_ieks_iterator(setup["dtm"], setup["om"], setup["x0"], states, Y)
_iter = wrap_iterator(_iter)


from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)

for (k, (states, nll, obj)) in enumerate(_iter):
    ys, covs = project(states, setup)
    d = ys.shape[1]
    plt.plot(setup["ts"], ys_true[:, 0], "--k", linewidth=0.5)
    for i in range(1):
        plt.plot(setup["ts"], ys[:, i], color=f"C{i}")
        plt.fill_between(
            setup["ts"],
            ys[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
            ys[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
            color=f"C{i}",
        )
    plt.text(0, ylims[1] - 0.5, f"Iteration {k}")
    plt.ylim(*ylims)
    camera.snap()

animation = camera.animate(interval=50)
