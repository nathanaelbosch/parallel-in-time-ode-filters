from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from tueplots import axes, bundles
from tueplots.constants import markers

plt.rcParams.update(bundles.jmlr2001(nrows=3, ncols=3))
plt.rcParams.update(axes.lines())
plt.rcParams.update(
    {
        # "lines.linewidth": 1.5,
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.2,
        # "lines.markersize": 4,
        "axes.grid": True,
        "axes.grid.which": "both",
        "figure.titlesize": plt.rcParams["axes.labelsize"],
    }
)

_colors = ["r", "g", "b", "k", "dimgray", "gray", "darkgray", "lightgray"]
_markers = ["p", "P", "X", "d", "s", "*", "o", "v"]
cycle = cycler("color", _colors) + cycler("marker", _markers)


INITS = (
    ("prior", "Prior"),
    ("coarse_ekf", "Coarse EKS"),
)
OPTS = ("ieks", "qpm")
ORDERS = (
    2,
    3,
    5,
)
# IVP = "lotkavolterra"
IVP = "vanderpol"
DTS = (
    # "1e-1",
    "1e-2",
    "1e-3",
    # "1e-4"
)
FOLDER = Path("experiments/2_init_comparison/")
TO_PLOT = "mse"

fig, axes = plt.subplots(
    len(DTS),
    len(ORDERS),
    # sharey=True,
    sharex=True,
)

dt2tex = lambda dt: f"$10^{{{dt.split('e')[1]}}}$"

for k, order in enumerate(ORDERS):
    for i, dt in enumerate(DTS):
        ax = axes[i, k]
        ax.set_prop_cycle(cycle)
        for opt in OPTS:
            filename = f"prob={IVP}_dt={dt}_order={order}_opt={opt}.csv"
            df = pd.read_csv(FOLDER / "data" / filename, index_col=0)

            for j in range(len(INITS)):
                init_key, init_name = INITS[j]
                label = f"{opt}: {init_name}"
                # ms = 8 - 2 * j if j < 3 else 5
                ms = 5
                ax.plot(df[f"{TO_PLOT}_{init_key}"], label=label, markersize=ms)
                ax.set_yscale("log")

        ax.set_title(
            rf"$\bf {chr(97+i)})$ " + rf"dt={dt2tex(dt)}; order={order}", loc="left"
        )

        # best = df["mse_coarse_dopri5"].dropna()
        # minval = best[best.last_valid_index()]
        # ax.axhline(minval, color="black", linestyle="dashed", zorder=-1)
        # ax.set_xlim(-1, 41)
        # ax.set_ylim(1e-16, 1e6)

fig.supxlabel("Number of iterations")
fig.supylabel("Mean squared error")
axes[-1, -1].legend(loc="right")
# axes[0, 0].legend()
# fig.legend(*axes[0, 0].get_legend_handles_labels())

# fig.suptitle(f"Lotka-Volterra")

_p = FOLDER / f"{IVP}.pdf"
fig.savefig(_p)
print(f"saved figure to {_p}")
