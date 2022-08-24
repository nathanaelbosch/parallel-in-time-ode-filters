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

fig, axes = plt.subplots(1, 2, sharey=True)

colored = (
    cycler("color", ["r", "b"])
    + cycler("marker", ["*", "^"])
    + cycler("linestyle", ["-", "-"])
)
# colored2 = (
#     cycler("color", ["g", "y"])
#     + cycler("marker", ["*", "^"])
#     + cycler("linestyle", ["-", "-"])
# )
# colored = colored.concat(colored2)
monochrome = cycler("color", ["gray"]) * (
    cycler("linestyle", [(0, (3, 1, 1, 1)), "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))])
    + cycler("marker", ["p", "P", "X", "d", "s"])
)
cycle = colored.concat(monochrome)

filenames = ("logistic", "lotkavolterra")
titles = ("Logistic equation", "Lotka-Volterra")
letters = ("a", "b")

for i, ax in enumerate(axes):
    ax.set_prop_cycle(cycle)

    folder = Path("experiments/2_solver_scaling/")
    df = pd.read_csv(folder / f"{filenames[i]}.csv", index_col=0)
    x = "N"
    ax.plot(
        df[x], df.sEKS, label="sequential EKS", markersize=10, linewidth=4, zorder=100
    )
    ax.plot(
        df[x], df.pEKS, label="parallel EKS", markersize=12, linewidth=4, zorder=101
    )
    # ax.plot(df[x], df.pEKSq, label="parallel EKS QR", markersize=12, linewidth=4, zorder=101)
    # ax.plot(df[x], df.sEKSq, label="sequential EKS QR", markersize=10, linewidth=4, zorder=100,)

    ref_alpha = 1
    ax.plot(df[x], df.dp5, label="Dopri5 (diffrax)", alpha=ref_alpha)
    ax.plot(df[x], df.kv5, label="Kvaerno5 (diffrax)", alpha=ref_alpha)
    # ax.plot(df[x], df.rk45, label="RK45 (scipy)", alpha=ref_alpha)
    ax.plot(df[x], df.lsoda, label="LSODA (scipy)", alpha=ref_alpha)

    ax.plot(
        df[x],
        df[x] / 8000 if i == 0 else df[x] / 5000,
        label="$\propto$ N",
        marker="",
        alpha=0.3,
        linewidth=10,
        linestyle="solid",
        color="r",
    )

    ax.plot(
        df[x],
        np.log(df[x]) / 1300 if i == 0 else np.log(df[x]) / 800,
        label="$\propto$ log(N)",
        marker="",
        alpha=0.3,
        linewidth=10,
        linestyle="solid",
        color="b",
    )

    # if i == 0:
    #     ax.plot(df[x], df.probnumek0, label="EK0 (probnum)", alpha=ref_alpha)

    # ax.set_title("Runtimes on the logistic equation (1D)")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title(rf"$\bf {letters[i]})$ " + rf"{titles[i]}", loc="left")

axes[0].set_xlabel("Number of gridpoints")
axes[1].set_xlabel("Number of gridpoints")
axes[0].set_ylabel("Runtime [s]")
axes[0].legend()

filename = folder / f"plot.pdf"
fig.savefig(filename)
print(f"Saved plot to {filename}")
