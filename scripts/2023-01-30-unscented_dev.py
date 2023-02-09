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
from pof.linearization.unscented import linearize_unscented
from pof.solver import solve


ivp = lotkavolterra()
order = 2
dt = 1e-1
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

_iter, setup = ieks_iterator(f=ivp.f, y0=ivp.y0, ts=ts, order=order)

om = setup["om"]
traj = get_initial_trajectory(setup, method="constant")
traj = MVNSqrt(traj.mean[:2], traj.chol[:2])
# traj = get_initial_trajectory(setup, method="prior")

lin_traj = jax.tree_map(lambda l: l[1:], traj)
vlinearize = jax.vmap(
    # linearize,
    linearize_unscented,
    in_axes=[None, 0],
)
dom = vlinearize(om, lin_traj)
# dom.cholR

# N = 200
# ts = jnp.linspace(0, 10, N)
# ys_par, info_par = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=3, init="prior")


# from pof.linearization.unscented import *
# from pof.linearization.unscented import _get_sigma_points, _cov

# x = MVNSqrt(traj.mean[1], traj.chol[1])
# f = om
# alpha, beta, kappa = 1.0, 0.0, None
# linearize_unscented(f, x)
