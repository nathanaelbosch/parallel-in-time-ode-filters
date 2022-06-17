# import diffrax
import jax
import jax.numpy as jnp
from tqdm import trange
import matplotlib.pyplot as plt

import pof
import pof.initialization as init
from pof.ivp import *
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.transitions import *
from pof.iterators import *

fs = linear_filtsmooth
lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))


def set_up(ivp, dt, order):
    ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

    iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
    dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
    P, PI = nordsieck_preconditioner(iwp, dt)

    E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
    E0, E1 = E0 @ P, E1 @ P
    om = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))

    x0 = init.taylor_mode_init(ivp.f, ivp.y0, order)
    # x0 = uncertain_init(ivp.f, ivp.y0, order, var=1e0)
    x0 = jax.tree_map(lambda x: PI @ x, x0)

    return {
        "ts": ts,
        "dtm": dtm,
        "om": om,
        "x0": x0,
        "E0": E0,
        "ivp": ivp,
        "P": P,
        "PI": PI,
        "order": order,
        "iwp": iwp,
    }


def ieks_iterator(dtm, om, x0, init_traj, maxiter=250):
    bar = trange(maxiter)
    iterator = pof.iterators.ieks_iterator(dtm, om, x0, init_traj)
    for _, (out, nll, obj) in zip(bar, iterator):
        yield out, nll, obj
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e}]")


def qpm_ieks_iterator(dtm, om, x0, init_traj, maxiter=500):
    bar = trange(maxiter)
    iterator = pof.iterators.qpm_ieks_iterator(dtm, om, x0, init_traj)
    for _, (out, nll, obj, reg) in zip(bar, iterator):
        yield out, nll, obj
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} reg={reg:.4}]")


def _precondition(setup, raw_states):
    precondition = lambda v: setup["PI"] @ v
    preconditioned_states = jax.tree_map(jax.vmap(precondition), raw_states)
    return preconditioned_states


def prior_init(setup):
    return init.prior_init(x0=setup["x0"], dtm=setup["dtm"])


def coarse_solver_init(setup):
    ivp, order, ts = setup["ivp"], setup["order"], setup["ts"]
    raw_traj = init.coarse_rk_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts[1:])
    return _precondition(setup, raw_traj)


ivp = lotkavolterra()
dt = 1e-2
order = 2
setup = set_up(ivp, dt, order=order)
init_traj = prior_init(setup)
ieks_iter = qpm_ieks_iterator(setup["dtm"], setup["om"], setup["x0"], init_traj)

project = lambda states: jnp.dot(setup["E0"], states.mean.T).T

from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)

for (k, (states, nll, obj)) in enumerate(ieks_iter):
    ys = project(states)
    d = ys.shape[1]
    for i in range(d):
        plt.plot(setup["ts"], ys[:, i], color=f"C{i}")
    plt.ylim(-2, 10)
    camera.snap()

animation = camera.animate()
# animation.save("ieks_lv_solve.mp4")
