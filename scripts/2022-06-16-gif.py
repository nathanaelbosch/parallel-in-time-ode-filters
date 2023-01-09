# import diffrax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from pof.ivp import *
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.transitions import *
from pof.iterators import *

fs = linear_filtsmooth
lom = jax.jit(jax.vmap(linearize, in_axes=[None, 0]), static_argnums=(0,))

MAXITER = 1000


# def set_up(ivp, dt, order):
#     ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

#     iwp = IWP(num_derivatives=order, wiener_process_dimension=ivp.y0.shape[0])
#     dtm = jax.vmap(lambda _: TransitionModel(*preconditioned_discretize(iwp)))(ts[1:])
#     P, PI = nordsieck_preconditioner(iwp, dt)

#     E0, E1 = projection_matrix(iwp, 0), projection_matrix(iwp, 1)
#     E0, E1 = E0 @ P, E1 @ P
#     om = NonlinearModel(lambda x: E1 @ x - ivp.f(None, E0 @ x))

#     x0 = init.taylor_mode_init(ivp.f, ivp.y0, order)
#     # x0 = uncertain_init(ivp.f, ivp.y0, order, var=1e0)
#     x0 = jax.tree_map(lambda x: PI @ x, x0)

#     return {
#         "ts": ts,
#         "dtm": dtm,
#         "om": om,
#         "x0": x0,
#         "E0": E0,
#         "ivp": ivp,
#         "P": P,
#         "PI": PI,
#         "order": order,
#         "iwp": iwp,
#     }


# def ieks_iterator(dtm, om, x0, init_traj, maxiter=MAXITER):
#     bar = trange(maxiter)
#     iterator = pof.iterators.ieks_iterator(dtm, om, x0, init_traj)
#     for _, (out, nll, obj) in zip(bar, iterator):
#         yield out, nll, obj
#         bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e}]")


# def qpm_ieks_iterator(dtm, om, x0, init_traj, maxiter=MAXITER):
#     bar = trange(maxiter)
#     iterator = pof.iterators.qpm_ieks_iterator(
#         dtm,
#         om,
#         x0,
#         init_traj,
#         reg_start=1e20,
#         reg_final=1e-20,
#         steps=100,
#         # steps=40,
#         # tau_start=1e0,
#         # tau_final=1e-10,
#     )
#     for _, (out, nll, obj, reg) in zip(bar, iterator):
#         yield out, nll, obj
#         bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} reg={reg:.4}]")


# def lm_ieks_iterator(dtm, om, x0, init_traj, maxiter=250):
#     bar = trange(maxiter)
#     iterator = pof.iterators.lm_ieks_iterator(
#         dtm,
#         om,
#         x0,
#         init_traj,
#         reg=1e-100,
#         nu=10,
#     )
#     for _, (out, nll, obj, reg) in zip(bar, iterator):
#         yield out, nll, obj
#         bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} reg={reg:.4}]")


# def _precondition(setup, raw_states):
#     precondition = lambda v: setup["PI"] @ v
#     preconditioned_states = jax.tree_map(jax.vmap(precondition), raw_states)
#     return preconditioned_states


# def prior_init(setup):
#     states = init.prior_init(
#         f=setup["ivp"].f, y0=setup["ivp"].y0, order=setup["order"], ts=setup["ts"]
#     )
#     return _precondition(setup, states)


# def coarse_solver_init(setup):
#     ivp, order, ts = setup["ivp"], setup["order"], setup["ts"]
#     raw_traj = init.coarse_rk_init(f=ivp.f, y0=ivp.y0, order=order, ts=ts[1:])
#     return _precondition(setup, raw_traj)


ivp, ylims = vanderpol(tmax=10, stiffness_constant=1e0), (-2.5, 2.5)
dt = 1e-1
print(f"Number of steps: {ivp.tmax // dt}")
order = 3
# setup = set_up(ivp, dt, order=order)
# sol_true = solve_diffrax(ivp.f, ivp.y0, ivp.t_span, max_steps=int(1e6))
# ys_true = jax.vmap(sol_true.evaluate)(setup["ts"])
# init_traj = prior_init(setup)
# init_traj = jax.tree_map(lambda l: l[:-1], init_traj)
# init_traj = MVNSqrt(
#     jnp.repeat(_states.mean[:-1, None, ...], 10, axis=1).reshape(-1, 8),
#     jnp.repeat(_states.chol[:-1, None, ...], 10, axis=1).reshape(-1, 8, 8),
# )
# init_traj
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
ieks_iter, setup = ieks_iterator(f=ivp.f, y0=ivp.y0, ts=ts, order=order)

# setup["dtm"], setup["om"], setup["x0"], init_traj)
# ieks_iter = qpm_ieks_iterator(setup["dtm"], setup["om"], setup["x0"], init_traj)

project = lambda states: jnp.dot(setup["E0"], states.mean.T).T


@jax.vmap
def project_cov(cholcov):
    out_chol = setup["E0"] @ cholcov
    return out_chol @ out_chol.T


from celluloid import Camera

fig = plt.figure()
camera = Camera(fig)

for (k, (states, nll, obj)) in enumerate(ieks_iter):
    ys = project(states)
    covs = project_cov(states.chol)
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
filename = f"vdp_{ivp.tmax}_{dt}.gif"
# animation.save(filename)
# print(f"Saved to {filename}")
