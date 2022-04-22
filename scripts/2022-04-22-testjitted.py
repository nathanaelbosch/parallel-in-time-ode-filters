import timeit

import tornadox

import pof
from pof.ivp import logistic


ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e1)


ts, xs = pof.solve_ekf(ivp, dt=1e-2)

%timeit ts, xs = pof.solve_ekf(ivp, dt=1e-2); xs.mean.block_until_ready()


ts, xs = pof.solve_ieks(ivp, dt=1e-2)
