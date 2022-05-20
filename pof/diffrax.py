import diffrax
import jax.numpy as jnp


def solve_diffrax(
    ivp,
    solver=diffrax.Dopri5,
    ts=None,
    rtol=1e-3,
    atol=1e-3,
    max_steps=int(1e6),
    dt=None,
):

    vector_field = lambda t, y, args: ivp.f(t, y)
    term = diffrax.ODETerm(vector_field)

    if ts is None:
        saveat = diffrax.SaveAt(steps=True, t0=True, dense=True)
    else:
        saveat = diffrax.SaveAt(ts=ts)

    if dt is None:
        stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
    else:
        stepsize_controller = diffrax.ConstantStepSize()

    sol = diffrax.diffeqsolve(
        term,
        solver(),
        t0=ivp.t0,
        t1=ivp.tmax,
        dt0=dt,
        y0=ivp.y0,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=max_steps,
    )
    return sol


def get_ts_ys(sol):
    idxs = jnp.isfinite(sol.ts)
    ts = sol.ts[idxs]
    ys = sol.ys[idxs]
    return ts, ys
