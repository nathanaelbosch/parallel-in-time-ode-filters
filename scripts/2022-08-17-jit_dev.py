import jax
import jax.numpy as jnp

import pof.iterators
from pof.convenience import discretize_transitions, linearize_observation_model
from pof.initialization import taylor_mode_init, prior_init
from pof.ivp import logistic, lotkavolterra
from pof.parallel_filtsmooth import (
    linear_filtsmooth,
    linear_noiseless_filtering,
    smoothing,
)
from pof.solver import make_continuous_models

ivp = logistic()
f, y0 = ivp.f, ivp.y0

order = 2
dt = 2.0**-8
ts = jnp.arange(ivp.t0, ivp.tmax + dt, dt)

iwp, om = make_continuous_models(f, y0, order)
dtm = discretize_transitions(iwp, ts)
x0 = taylor_mode_init(f, y0, order)
traj = prior_init(x0=x0, dtm=dtm)
dom = linearize_observation_model(om, traj)

# %timeit with jax.disable_jit(): filt, _, _, _ = linear_noiseless_filtering(x0, dtm, dom)
# %timeit filt, _, _, _ = linear_noiseless_filtering(x0, dtm, dom)
# %timeit filt, _, _, _ = jax.jit(linear_noiseless_filtering)(x0, dtm, dom)

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    with jax.disable_jit():
        filt, _, _, _ = linear_noiseless_filtering(x0, dtm, dom)
    filt.mean.block_until_ready()

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    filt, _, _, _ = linear_noiseless_filtering(x0, dtm, dom)
    filt.mean.block_until_ready()

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    lnf = jax.jit(linear_noiseless_filtering)
    filt, _, _, _ = lnf(x0, dtm, dom)
    filt.mean.block_until_ready()
    filt, _, _, _ = lnf(x0, dtm, dom)
    filt.mean.block_until_ready()

with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    for _ in range(100):
        filt, _, _, _ = lnf(x0, dtm, dom)
        filt.mean.block_until_ready()
