from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
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

fig, axes = plt.subplots(3, 2, sharex="col", sharey="row")

colored = (
    cycler("color", ["b", "r"])
    + cycler("marker", ["*", "^"])
    + cycler("linestyle", ["-", "-"])
)
monochrome = cycler("color", ["gray"]) * (
    cycler("linestyle", [(0, (3, 1, 1, 1)), "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))])
    + cycler("marker", ["p", "P", "X", "d", "s"])
)
cycle = colored.concat(monochrome)

# ivp_names = ("logistic", "lotkavolterra")
ivp_names = ("fitzhughnagumo", "vanderpol")
# titles = ("Logistic equation", "Lotka-Volterra")
titles = ("FitzHugh-Nagumo", "VanDerPol")
letters = ("a", "b")
other_solvers = (
    ("dp5", "Dopri5 (diffrax)"),
    # ("kv5", "Kvaerno5 (diffrax)"),
    # ("rk45", "RK45 (scipy)"),
    # ("lsoda", "LSODA (scipy)"),
)

for i, ivpname in enumerate(ivp_names):
    for j, yname in enumerate(("time", "err")):
        ax = axes[j, i]
        ax.set_prop_cycle(cycle)

        folder = Path("experiments/2_solver_scaling/")
        df = pd.read_csv(folder / f"{ivpname}.csv", index_col=0)
        x = "tmax"
        ax.plot(
            df[x],
            df[f"pEKS_{yname}"],
            label="parallel EKS",
            markersize=12,
            linewidth=4,
            zorder=3.2,
        )
        ax.plot(
            df[x],
            df[f"sEKS_{yname}"],
            label="sequential EKS",
            markersize=10,
            linewidth=4,
            zorder=3.1,
        )

        ref_alpha = 1
        for solver, solvername in other_solvers:
            ax.plot(df[x], df[f"{solver}_{yname}"], label=solvername, alpha=ref_alpha)

        if yname == "time":
            # ax.plot(
            #     df[x],
            #     df[x] / 8000 if i == 0 else df[x] / 5000,
            #     label="$\propto$ N",
            #     marker="",
            #     alpha=0.3,
            #     linewidth=10,
            #     linestyle="solid",
            #     color="r",
            # )

            # ax.plot(
            #     df[x],
            #     np.log(df[x]) / 1300 if i == 0 else np.log(df[x]) / 800,
            #     label="$\propto$ log(N)",
            #     marker="",
            #     alpha=0.3,
            #     linewidth=10,
            #     linestyle="solid",
            #     color="b",
            # )
            pass

        ax.set_xscale("log")
        ax.set_yscale("log")

for i, ivpname in enumerate(ivp_names):
    ax = axes[2, i]
    ax.set_prop_cycle(cycle)

    folder = Path("experiments/2_solver_scaling/")
    df = pd.read_csv(folder / f"{ivpname}.csv", index_col=0)
    ax.plot(df[x], df["pEKS_iters"], label="parallel EKS", markersize=12, linewidth=4)
    ax.set_yscale("log")

for i in range(2):
    axes[0, i].set_title(rf"$\bf {letters[i]})$ " + rf"{titles[i]}", loc="left")

# axes[1, 0].set_xlabel("Number of gridpoints")
# axes[1, 1].set_xlabel("Number of gridpoints")
axes[2, 0].set_xlabel("t_max")
axes[2, 1].set_xlabel("t_max")
axes[0, 0].set_ylabel("Runtime [s]")
axes[1, 0].set_ylabel("RMSE")
axes[2, 0].set_ylabel("IEKS Iterations")
axes[1, 0].legend()

filename = folder / f"plot.pdf"
fig.savefig(filename)
print(f"Saved plot to {filename}")
