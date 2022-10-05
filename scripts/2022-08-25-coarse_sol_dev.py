import time
from collections import defaultdict

import warnings

warnings.filterwarnings("ignore", message=r"jax.tree_leaves", category=FutureWarning)

import jax

jax.config.update("jax_enable_x64", True)

import tqdm
import diffrax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import probnum
from scipy.integrate import solve_ivp

from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import constant_init, taylor_mode_init, coarse_ekf_init
from pof.ivp import logistic, lotkavolterra
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.sequential_filtsmooth import filtsmooth
from pof.solver import make_continuous_models
from pof.transitions import *
import pof.iterators
import pof.diffrax


# DTS = 2.0 ** -np.arange(0, 12)
DTS = 2.0 ** -np.arange(1, 5)
ORDER = 4


def solve_parallel_eks(f, y0, ts, order, coarse_N=10):
    out, info = pof.solver.solve(f=f, y0=y0, ts=ts, order=order, coarse_N=coarse_N)
    return out.mean, info, 0


def solve_sequential_eks(f, y0, ts, order):
    iwp, om = make_continuous_models(f, y0, order)
    dtm = discretize_transitions(iwp, ts)
    x0 = taylor_mode_init(f, y0, order)
    # traj = constant_init(y0=y0, f=f, order=order, ts=ts)
    out, _ = filtsmooth(x0, dtm, om)

    E0 = projection_matrix(iwp, 0)

    return jax.vmap(lambda x: E0 @ x)(out.mean), 0


def solve_diffrax(f, y0, ts, solver):
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(lambda t, y, args: f(t, y)),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=ts[1] - ts[0],
        y0=y0,
        saveat=diffrax.SaveAt(steps=True, t0=True, dense=True),
        stepsize_controller=diffrax.ConstantStepSize(),
        max_steps=2 * ts.shape[0],
        throw=False,
    )
    return sol.ys, sol.result


ivp = lotkavolterra()

dt = DTS[0] * 5
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
ys_diffrax, *_ = solve_diffrax(ivp.f, ivp.y0, ts, diffrax.Dopri5())
# ys_eks, *_ = solve_sequential_eks(ivp.f, ivp.y0, ts, order=4)
