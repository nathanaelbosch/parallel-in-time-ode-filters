from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import parsmooth
import tornadox
import diffrax
from parsmooth.parallel import ekf
from parsmooth.sequential import ieks as seq_ieks
from parsmooth.utils import MVNormalParameters

seqekf = jax.jit(parsmooth.sequential.ekf, static_argnums=(2, 4, 7))
parieks = jax.jit(parsmooth.parallel.ieks, static_argnums=(2, 4, 7))


def diffrax_solve(ivp, ts, rtol=1e-3, atol=1e-3, max_steps=int(1e6)):
    vector_field = lambda t, y, args: ivp.f(t, y)
    term = diffrax.ODETerm(vector_field)
    solver = diffrax.Dopri5()
    # saveat = diffrax.SaveAt(steps=True)
    saveat = diffrax.SaveAt(t1=True, ts=ts)
    stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)

    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=ivp.t0,
        t1=ivp.tmax,
        dt0=None,
        y0=ivp.y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    idxs = jnp.isfinite(sol.ts)
    ts = sol.ts[idxs]
    ys = sol.ys[idxs]
    return ts, ys


def get_transition(
    iwp: tornadox.iwp.IntegratedWienerTransition, dt: float, diffusion: float = 1
):
    A, QS = iwp.non_preconditioned_discretize(dt)
    Q = diffusion * (QS @ QS.T)

    def trans(x, A=A):
        return A @ x

    return trans, Q


def get_observation(
    ivp: tornadox.ivp.InitialValueProblem, iwp: tornadox.iwp.IntegratedWienerTransition
):
    P0 = iwp.projection_matrix(0)
    P1 = iwp.projection_matrix(1)

    def obs(x, f=ivp.f, P0=P0, P1=P1):
        return P1 @ x - f(None, P0 @ x)

    R = 0 * jnp.eye(1)

    return obs, R


def get_initial_guess(ivp: tornadox.ivp.InitialValueProblem, order: int):
    m0, P0 = tornadox.init.TaylorMode()(
        f=ivp.f, df=None, y0=ivp.y0, t0=0, num_derivatives=order
    )
    m0, P0 = jnp.concatenate(m0.T), jnp.kron(jnp.eye(ivp.dimension), P0)
    initial_guess = MVNormalParameters(m0, P0)
    return initial_guess


def solve_ekf(
    ivp,
    dt,
    order=3,
    diffusion=0.1,
):

    d, q = ivp.dimension, order
    D = d * (q + 1)

    iwp = tornadox.iwp.IntegratedWienerTransition(
        num_derivatives=q, wiener_process_dimension=d
    )

    tfun, Q = get_transition(iwp, dt, diffusion=diffusion)
    ofun, R = get_observation(ivp, iwp)

    times = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    data = jnp.zeros((len(times) - 1, d))
    N = len(times)

    initial_guess = get_initial_guess(ivp, order)

    _, states = seqekf(initial_guess, data, tfun, Q, ofun, R)

    return times, states


def solve_ieks(
    ivp,
    dt,
    order=3,
    diffusion=0.1,
    linearize_at=None,
    n_iter=10,
):

    d, q = ivp.dimension, order
    D = d * (q + 1)

    iwp = tornadox.iwp.IntegratedWienerTransition(
        num_derivatives=q, wiener_process_dimension=d
    )

    tfun, Q = get_transition(iwp, dt, diffusion=diffusion)
    ofun, R = get_observation(ivp, iwp)

    times = jnp.arange(ivp.t0, ivp.tmax + dt, dt)
    data = jnp.zeros((len(times) - 1, d))
    N = len(times)

    initial_guess = get_initial_guess(ivp, order)

    if linearize_at == "auto":
        _, ys = diffrax_solve(ivp, ts=times)
        xs = jnp.concatenate((ys[0:-1], jnp.zeros((N, d * q))), 1)
        Ps = jnp.zeros((N, D, D))
        linearize_at = MVNormalParameters(xs, Ps)
    states = parieks(initial_guess, data, tfun, Q, ofun, R, linearize_at, n_iter=n_iter)

    return times, states
