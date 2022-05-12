import jax
import tornadox

from pof.solve import get_solver_iterator

ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e0, tmax=12)
ylims = (-3, 3)
dt = 1e-2
order = 3

init_state, refine, project = get_solver_iterator(
    ivp, order=order, dt=dt, parallel=True
)
refine_jit = jax.jit(refine)

refine_jit(init_state).block_until_ready()
