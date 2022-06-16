"""
Contains both functions for computing the initial state x0,
and functions for computing the first linearization trajectory.
"""
import jax
import jax.numpy as jnp
import tornadox

from pof.utils import MVNSqrt
from pof.transitions import IWP
from pof.sequential_filtsmooth.filter import _sqrt_update, _sqrt_predict
from pof.observations import linearize


def taylor_mode_init(f, y0, num_derivatives):
    d = y0.shape[0]
    m0, P0 = tornadox.init.TaylorMode()(
        f=f, df=None, y0=y0, t0=0, num_derivatives=num_derivatives
    )
    m0, P0 = jnp.concatenate(m0.T), jnp.kron(jnp.eye(d), P0)
    x0 = MVNSqrt(m0, P0)
    return x0


def uncertain_init(f, y0, num_derivatives, var=1.0):
    d = y0.shape[0]
    q = num_derivatives

    y0 = y0.reshape(-1, 1)
    dy0 = f(None, y0).reshape(-1, 1)

    m0 = jnp.concatenate([y0, dy0, jnp.zeros((d, (q - 1)))], axis=1)
    m0 = m0.reshape(-1)
    P0 = jnp.eye(d * (q + 1)) * var
    for j in range(d):
        P0 = P0.at[(q + 1) * j, (q + 1) * j].set(0)
        P0 = P0.at[(q + 1) * j + 1, (q + 1) * j + 1].set(0)
    x0 = MVNSqrt(m0, jnp.sqrt(P0))
    return x0


def constant_init(*, y0, order, ts, f=True):
    d = y0.shape[-1]
    N = len(ts)
    if f is not None:
        dy0 = f(None, y0)
    else:
        dy0 = jnp.zeros_like(y0)
    x0 = jnp.concatenate(
        [y0[:, None], dy0[:, None], jnp.zeros((d, (order - 1)))], axis=1
    )
    x0 = x0.reshape(1, -1)
    traj = jnp.repeat(x0, N, axis=0)
    _, D = traj.shape
    cholcovs = jnp.zeros((N, D, D))
    return MVNSqrt(traj, cholcovs)


def classic_to_init(*, ys, order, f=None):
    N = ys.shape[0]
    d = ys.shape[-1]
    if f is not None:
        dys = jax.vmap(f, in_axes=(None, 0))(None, ys)
    else:
        dys = jnp.zeros_like(ys)
    traj = jnp.concatenate(
        [ys[:, :, None], dys[:, :, None], jnp.zeros((N, d, (order - 1)))], axis=2
    )
    traj = traj.reshape(N, -1)
    _, D = traj.shape
    cholcovs = jnp.zeros((N, D, D))
    return MVNSqrt(traj, cholcovs)


def prior_init(*, x0, dtm):
    states_raw = jax.vmap(lambda F, QL: _sqrt_predict(F, QL, x0))(dtm.F, dtm.QL)
    return states_raw


def updated_prior_init(*, x0, dtm, om):
    states = prior_init(x0=x0, dtm=dtm)

    def update(x, om):
        H, b, cholR = linearize(om, x)
        return _sqrt_update(H, cholR, b, x)[0]

    states = jax.vmap(update, in_axes=[0, None])(states, om)
    return states


def _get_coarse_dt(fine_ts):
    ts = fine_ts
    T = ts[-1] - ts[0]
    N = len(ts)
    N_coarse = jnp.ceil(jnp.log2(N))
    dt = T / N_coarse
    return dt


def coarse_ekf_init(*, f, y0, order, ts, fact=10):
    coarse_dt = _get_coarse_dt(ts) / fact

    tspan = (ts[0], ts[1])
    from pof.solver import sequential_eks_solve

    _out, _ts, _ = sequential_eks_solve(
        f, y0, tspan, dt=coarse_dt, order=order, return_full_states=True
    )

    idxs = jnp.floor(ts / coarse_dt).astype(int)
    interpolated_states = jax.vmap(lambda idx: jax.tree_map(lambda l: l[idx], _out))(
        idxs
    )
    # return jax.tree_map(lambda l: l[1:], interpolated_states)
    return interpolated_states


def coarse_rk_init(*, f, y0, order, ts, fact=10):
    coarse_dt = _get_coarse_dt(ts) / fact

    tspan = (ts[0], ts[1])
    from pof.diffrax import solve_diffrax

    sol_init = solve_diffrax(f, y0, tspan, dt=coarse_dt)
    ys = jax.vmap(sol_init.evaluate)(ts)
    traj = classic_to_init(ys=ys, f=f, order=order)
    return traj


def _interpolate(out, ts, t, idx, iwp):
    xprev = jax.tree_map(lambda l: l[idx], out)
    tprev = ts[idx]
    xnext = jax.tree_map(lambda l: l[idx + 1], out)
    tnext = ts[idx + 1]

    F, QL = preconditioned_discretize(iwp)

    # predict
    P, PI = nordsieck_preconditioner(iwp, t - tprev)
    xp = _sqrt_predict(F, QL, jax.tree_map(lambda l: PI @ l, xprev))
    xp = jax.tree_map(lambda l: P @ l, xp)

    # smooth
    P, PI = nordsieck_preconditioner(iwp, tnext - t)
    xnext = jax.tree_map(lambda l: PI @ l, xnext)
    xp = jax.tree_map(lambda l: PI @ l, xp)
    x = _sqrt_smooth(F, QL, xp, xnext)
    x = jax.tree_map(lambda l: P @ l, x)
    return x
