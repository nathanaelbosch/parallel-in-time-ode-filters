from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from tueplots import axes, bundles
from tueplots.constants import markers

plt.rcParams.update(bundles.jmlr2001(nrows=1, ncols=2))
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

_colors = ["r", "g", "b", "y", "dimgray", "gray", "darkgray", "lightgray"]
_markers = ["P", "X", "p", "d", "s", "*", "o", "v"]
cycle = cycler("color", _colors) + cycler("marker", _markers)

FOLDER = Path("experiments/2_init_comparison/")
IVP = "lotkavolterra"
# IVP = "logistic"
TO_PLOT = "mse"

INITS = (
    ("prior", "Prior"),
    ("coarse_ekf", "Coarse"),
)
OPTS = (
    ("ieks", "GN"),
    ("qpm", "QPM"),
)

ORDER = 3
DT = "1e-3"

fig, ax = plt.subplots(1, 1)
ax.set_prop_cycle(cycle)

for (opt, opt_name) in OPTS:
    filename = f"prob={IVP}_dt={DT}_order={ORDER}_opt={opt}.csv"
    df = pd.read_csv(FOLDER / "data" / filename, index_col=0)
    for (init_key, init_name) in INITS:
        label = f"{opt_name} \& {init_name}"
        # ms = 8 - 2 * j if j < 3 else 5
        ms = 5
        ax.plot(
            df[f"{TO_PLOT}_{init_key}"],
            label=label,
            markersize=ms if not (opt == "qpm" and init_key == "coarse_ekf") else 3,
        )
        ax.set_yscale("log")


# ax.set_title(rf"$\bf {chr(97+i)})$ " + rf"dt={dt}; order={order}", loc="left")
ax.set_xlim(-0.5, 30.5)

fig.supxlabel("Number of iterations")
fig.supylabel("Mean squared error")
ax.legend()
# axes[0, 0].legend()
# fig.legend(*axes[0, 0].get_legend_handles_labels())

# fig.suptitle(f"Lotka-Volterra")

_p = FOLDER / f"plot.pdf"
fig.savefig(_p)
print(f"saved figure to {_p}")
