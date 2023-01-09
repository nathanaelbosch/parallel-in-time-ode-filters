from functools import partial
from collections import defaultdict

# import diffrax
import pandas as pd
import jax
import jax.numpy as jnp
from tqdm import tqdm
import matplotlib.pyplot as plt
from celluloid import Camera

import pof
from pof.diffrax import solve_diffrax
import pof.initialization as init
from pof.ivp import *
from pof.observations import *
from pof.parallel_filtsmooth import linear_filtsmooth
from pof.transitions import *
from pof.iterators import *
from pof.solver import solve


def wrap_iterator(iterator, maxiter=1000):
    """Wrap the iterator to add a TQDM bar"""
    bar = trange(maxiter)
    for _, (out, nll, obj, *_) in zip(bar, iterator):
        bar.set_description(f"[OBJ={obj:.4e} NLL={nll:.4e} other={_}]")
        yield out, nll, obj


@partial(jax.vmap, in_axes=[0, None])
def project(states, setup):
    mean, cholcov = states
    pmean = jnp.dot(setup["E0"], mean.T).T
    pchol = setup["E0"] @ cholcov
    pcov = pchol @ pchol.T
    return pmean, pcov


tmaxs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
# tmaxs = [1, 2, 5]
DT = 1e-2
order = 2
results = defaultdict(list)

for tmax in tqdm(tmaxs):
    # ivp, ivpn, ylims = vanderpol(tmax=20, stiffness_constant=1e0), "vdp", (-2.5, 2.5)
    ivp, ivpn, ylims = fitzhughnagumo(tmax=tmax), "fhn", (-2.5, 2.5)
    sol_true = solve_diffrax(
        ivp.f, ivp.y0, ivp.t_span, rtol=1e-9, atol=1e-12, max_steps=int(1e6)
    )
    ts = jnp.arange(ivp.t0, ivp.tmax + DT, DT)
    N = len(ts)

    ys_true = jax.vmap(sol_true.evaluate)(ts)

    ys, info_dict = solve(f=ivp.f, y0=ivp.y0, ts=ts, order=order, init="constant")

    results["tmax"].append(tmax)
    results["iterations"].append(info_dict["iterations"])
    rmse = jnp.sqrt(jnp.mean(jnp.sum((ys.mean - ys_true) ** 2, axis=1)))
    results["rmse"].append(rmse)
    print(f"RMSE={rmse:.4e}")
    results["N"].append(N)
df = pd.DataFrame(results)
# df.to_csv(f"scripts/2022-10-05_step_size_comparison_{ivpn}_order2.csv", index=False)

from tueplots import axes, bundles
from tueplots.constants import markers

plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(axes.lines())
plt.rcParams.update(
    {
        # "lines.linewidth": 1.5,
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.2,
        # "lines.markersize": 4,
        "axes.grid": True,
        "axes.grid.which": "both",
    }
)
fig, axes = plt.subplots(2, 1)
axes[0].plot(df.tmax, df.iterations, marker="^")
axes[1].plot(df.tmax, df.rmse, marker="^")
axes[0].set_xscale("log")
axes[1].set_xscale("log")
# axes[0].set_yscale("log")
axes[1].set_yscale("log")
axes[1].set_xlabel("Time interval length")
axes[0].set_ylabel("Number of iterations")
axes[1].set_ylabel("RMSE")
# fig.savefig(
#     f"scripts/2022-10-06_tmax_comparison_{ivpn}_order{order}_dt{DT}.pdf",
#     bbox_inches="tight",
# )
