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

# _colors = ["r", "g", "b", "y", "dimgray", "gray", "darkgray", "lightgray"]
# _markers = ["p", "P", "X", "d", "s", "*", "o", "v"]
_colors = ["r", "g", "b", "c", "m", "y"]
_markers = ["p", "P", "X", "d", "s", "*"]
cycle = cycler("color", _colors) + cycler("marker", _markers)

# I want to plot:
# Coarse init
# Prior init + IEKS
# Prior init + QPM

inits = (
    ("constant", "Constant"),
    ("prior", "Prior"),
    # ("updated_prior", "Prior+Update"),
    ("coarse_ekf", "Coarse EKS"),
)
opts = (
    "ieks",
    # "qpm"
)
# opts = ("ieks",)

toplot = "mse"
order = 2
dt = "1e-3"
ivp = "lotkavolterra"
folder = Path("experiments/2_init_comparison/")

fig, ax = plt.subplots(
    1,
    1,
    sharey=True,
    sharex=True,
)

ax.set_prop_cycle(cycle)
for opt in opts:
    filename = f"prob={ivp}_dt={dt}_order={order}_opt={opt}.csv"
    df = pd.read_csv(folder / "data" / filename, index_col=0)

    for j in range(len(inits)):
        key, label = inits[j]
        if opt == "qpm":
            label += " + QPM"
        # ms = 8 - 2 * j if j < 3 else 5
        ms = 5
        ax.plot(df[f"{toplot}_{key}"], label=label, markersize=ms)

ax.set_yscale("log")
# ax.set_title(rf"$\bf {chr(97+i)})$ " + rf"dt={dt}; order={order}", loc="left")
best = df["mse_coarse_dopri5"].dropna()
minval = best[best.last_valid_index()]
ax.axhline(minval, color="black", linestyle="dashed", zorder=-1)
ax.set_xlim(-2, 102)
ax.set_ylim(1e-15, 1e5)
fig.supxlabel("Number of iterations")
fig.supylabel("Mean squared error")
ax.legend()

fig.suptitle(f"Lotka-Volterra")

_p = f"plot.pdf"
fig.savefig(_p)
print(f"saved figure to {_p}")
