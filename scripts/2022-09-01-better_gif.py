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
from pof.solver import solve


def wrap_iterator(iterator, maxiter=250):
    """Wrap the iterator to add a TQDM bar"""
    bar = trange(maxiter)
    for _, (out, nll, obj, *_) in zip(bar, iterator):
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} other={_}]")
        yield out, nll, obj, *_


@partial(jax.vmap, in_axes=[0, None])
def project(states, setup):
    mean, cholcov = states
    pmean = jnp.dot(setup["E0"], mean.T).T
    pchol = setup["E0"] @ cholcov
    pcov = pchol @ pchol.T
    return pmean, pcov


def _ieks(ivp, ts, order):
    ys, info_dict = solve(
        f=ivp.f,
        y0=ivp.y0,
        ts=ts,
        order=order,
        init="constant",
        maxiters=1000,
    )
    return ys.mean, info_dict


ivp, ylims, Ns = (
    vanderpol(tmax=20, stiffness_constant=1e2),
    (-2.5, 2.5),
    2 ** jnp.arange(8, 18),
)
# ivp, ylims, Ns = fitzhughnagumo(tmax=500), (-2, 2), 2 ** jnp.arange(8, 18)
# ivp, ylims, Ns = logistic(), (-0.2, 1.2), 2 ** jnp.arange(3, 15)
# Ns = 2 ** jnp.arange(8, 18)
order = 2
N = Ns[5]
ts = jnp.linspace(ivp.t0, ivp.tmax, N)

sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, max_steps=int(1e6))
ys_true = jax.vmap(sol_true.evaluate)(ts)

_ys, info = _ieks(ivp, ts, order)
info

_iter, setup = qpm_ieks_iterator(
    f=ivp.f,
    y0=ivp.y0,
    ts=ts,
    order=order,
    init="constant",
    reg_start=1e20,
    reg_final=1e-20,
    steps=40,
    tau_start=1e5,
    tau_final=1e-5,
)
_iter, setup = ieks_iterator(f=ivp.f, y0=ivp.y0, ts=ts, order=order, init="constant")
# _iter, setup = ieks_iterator(f=ivp.f, y0=ivp.y0, ts=ts, order=order, init="prior")
_iter = wrap_iterator(_iter)


from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)

for (k, (states, nll, obj, *_)) in enumerate(_iter):
    ys, covs = project(states, setup)
    d = ys.shape[1]
    for i in range(d):
        plt.plot(setup["ts"], ys_true[:, i], "--k", linewidth=0.5)
        plt.plot(setup["ts"], ys[:, i], color=f"C{i}", marker="o", markersize=2)
        plt.fill_between(
            setup["ts"],
            ys[:, i] - 2 * jnp.sqrt(covs[:, i, i]),
            ys[:, i] + 2 * jnp.sqrt(covs[:, i, i]),
            color=f"C{i}",
            alpha=0.2,
        )
    plt.text(0, ylims[1] - 0.5, f"Iteration {k}")
    plt.ylim(*ylims)
    camera.snap()

animation = camera.animate(interval=50)
plt.show()
# animation.save("animation.gif")
