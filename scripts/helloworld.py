import timeit

import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox

import pof
from pof.ivp import logistic


def plot_results(times, states, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1)
    means = states.mean[:, 0]
    stds = states.cov[:, 0, 0]
    ax.plot(times, means, marker="o", markersize=1)
    ax.fill_between(times, means - 2 * stds, means + 2 * stds, alpha=0.2)
    return ax


# ivp = logistic()
ivp = tornadox.ivp.vanderpol_julia(stiffness_constant=1e1)

dt = 2e-2
ts = jnp.arange(ivp.t0, ivp.tmax, dt)
tts, tys = pof.diffrax_solve(ivp, ts=ts, rtol=1e-18, atol=1e-18)
# plt.plot(tts, tys[:, 0])

ts, xs = pof.solve_ekf(ivp, dt=dt)
# plot_results(ts, xs)
err = jnp.mean((tys[:, 0] - xs.mean[:, 0]) ** 2)
# print(err)

# ts, xs = pof.solve(ivp, dt=dt, )
# # plot_results(ts, xs)
# err = tys[-1] - xs.mean[-1, 0]
# print(err)


N = 5
logdts = jnp.arange(-1.5, -4, -0.25)

ekf_errs = []
ekf_times = []

pieks_errs = []
pieks_times = []

for logdt in logdts:
    dt = 10**logdt
    print(f"dt={dt}")
    ts = jnp.arange(ivp.t0, ivp.tmax, dt)
    tts, tys = pof.diffrax_solve(ivp, ts=ts, rtol=1e-18, atol=1e-18)

    ts, xs = pof.solve_ekf(ivp, dt=dt)
    err = jnp.mean((tys[:, 0] - xs.mean[:, 0]) ** 2)
    s = timeit.timeit(lambda: pof.solve_ekf(ivp, dt=dt), number=N) / N
    ekf_times.append(s)
    ekf_errs.append(err)

    # ts, xs = pof.solve(ivp, dt=dt, linearize_at=None)
    # err = jnp.mean((tys[:, 0] - xs.mean[:, 0]) ** 2)
    # s = timeit.timeit(lambda: pof.solve(ivp, dt=dt, ), number=N) / N
    # pieks_times.append(s)
    # pieks_errs.append(err)

plt.plot(ekf_times, ekf_errs, marker="o")
# plt.plot(pieks_times, pieks_errs, marker="o")
plt.xlabel("seconds")
plt.xscale("log")
plt.ylabel("MSE")
plt.yscale("log")
