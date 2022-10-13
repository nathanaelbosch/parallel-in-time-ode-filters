import jax.numpy as jnp

import pof.solver
from pof.ivp import *


ivp, ylims = vanderpol(tmax=100, stiffness_constant=1e0), (-2.5, 2.5)
order = 3
dt = 1e-1
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

pof.solver.solve(f=ivp.f, y0=ivp.y0, ts=ts, order=3)

"""
Solved: set XLA_PYTHON_CLIENT_PREALLOCATE=false
i.e.
`XLA_PYTHON_CLIENT_PREALLOCATE=false ipython`
"""
