import jax
import jax.numpy as jnp

import tornadox
import matplotlib.pyplot as plt
from tqdm import trange

import pof
from pof.main import (
    make_continuous_models,
    discretize_transitions,
    get_constant_initial_trajectory,
    linearize_observation_model,
    get_x0,
    # get_sigmasq_estimate,
)
import pof.ieks
import pof.transitions
from pof.ieks import filtsmooth
from pof.ivp import lotkavolterra, logistic


# ivp = lotkavolterra()
# ivp = logistic()
ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e1, tmax=8)

order = 3
dt = 1e-2

x0 = get_x0(ivp, order)

ts = jnp.arange(0, ivp.tmax + dt, dt)
traj = get_constant_initial_trajectory(ivp.y0, ivp.f(None, ivp.y0), order, len(ts))
# traj = jnp.repeat(x0.mean.reshape(1, -1), traj.shape[0], axis=0)

iwp, om = make_continuous_models(ivp, order)
E0 = pof.transitions.projection_matrix(iwp, 0)
dtm = discretize_transitions(iwp, ts)
lom = jax.jit(linearize_observation_model, static_argnums=0)
dom = lom(om, traj[1:])
# fs = jax.jit(filtsmooth)
out, nll, obj = filtsmooth(x0, dtm, dom)

nlls = [nll]
objs = [obj]
print(f"{nll}")
pbar = trange(2000)
nll_old, obj_old = 0, 0
for i in pbar:
    dom = lom(om, out.mean[1:])
    out, nll, obj = filtsmooth(x0, dtm, dom)
    pbar.write(f"nll={nll}, obj={obj}")
    nlls.append(nll)
    objs.append(obj)
    if jnp.isclose(nll_old, nll) and jnp.isclose(obj_old, obj):
        break
    nll_old, obj_old = nll, obj


ys = jnp.dot(E0, out.mean.T).T
fig, ax = plt.subplots(1, 1)
ax.plot(ts, ys, marker=".")

fig, axes = plt.subplots(2, 1)
axes[0].plot(nlls, marker="o", label="observation likelihood", color="C0")
axes[0].set_yscale("log")
axes[0].legend()
axes[1].plot(objs, marker="o", label="objective value", color="C1")
axes[1].set_yscale("log")
axes[1].legend()
# plt.yscale("log")
