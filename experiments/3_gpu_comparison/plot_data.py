from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
from tueplots import axes, bundles
from tueplots.constants import markers
from cuda_cores import CUDA_CORES

plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(axes.lines())
plt.rcParams.update(
    {
        # "lines.linewidth": 1.5,
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.2,
        # "lines.markersize": 4,
        "axes.grid": True,
        # "axes.grid.which": "both",
        "axes.grid.which": "major",
    }
)
colored = (
    cycler("color", ["r", "b"])
    + cycler("marker", ["*", "^"])
    + cycler("linestyle", ["-", "-"])
)
monochrome = cycler("color", ["gray"]) * (
    cycler("linestyle", [(0, (3, 1, 1, 1)), "--", ":", "-.", (0, (3, 1, 1, 1, 1, 1))])
    + cycler("marker", ["p", "P", "X", "d", "s"])
)
cycle = colored.concat(monochrome)


# filedir = Path(__file__).parent
filedir = Path("experiments/3_gpu_comparison")
gpus = ["1060", "2080ti", "v100"]
dfs = {gpu: pd.read_csv(filedir / f"{gpu}.csv") for gpu in gpus}

labels = {
    "pEKS": "parallel EKS",
    "sEKS": "sequential EKS",
    "dp5": "Dopri5 (diffrax)",
    "kv5": "Kvaerno5 (diffrax)",
}
gpu_labels = {
    "1060": "GTX 1060",
    "2080ti": "RTX 2080 Ti",
    "v100": "V100",
}

fig, axes = plt.subplots(1, len(gpus), sharey=True, sharex=True)

x = "N"
for ax, gpu in zip(axes, gpus):
    ax.set_prop_cycle(cycle)
    df = dfs[gpu]
    for i, m in enumerate(("sEKS", "pEKS")):
        z = 100 + i
        ax.plot(df[x], df[m], label=labels[m], markersize=10, linewidth=4, zorder=z)

    ref_alpha = 1
    for m in ("dp5", "kv5"):
        ax.plot(df[x], df[m], label=labels[m], alpha=ref_alpha)

    ax.set_xscale("log")
    ax.set_yscale("log")

axes[0].set_xlabel("Number of gridpoints")
axes[1].set_xlabel("Number of gridpoints")
axes[2].set_xlabel("Number of gridpoints")
axes[0].set_title("GeForce GTX 1060")
axes[1].set_title("GeForce RTX 2080 Ti")
axes[2].set_title("Tesla V100")
axes[0].set_ylabel("Runtime [s]")
axes[0].legend()

fig.savefig(filedir / "plot.pdf", bbox_inches="tight")
print(f"Saved plot to {filedir / 'plot.pdf'}")


DT = 2 ** -13
rs = defaultdict(lambda: [])
for gpu in gpus:
    rs["cores"].append(CUDA_CORES[gpu])
    df = dfs[gpu]
    row = df[df.dt == DT]
    for m in ("sEKS", "pEKS", "dp5", "kv5"):
        rs[m].append(row[m].values[0])
df = pd.DataFrame(rs)


fig, ax = plt.subplots(1, 1)
ax.set_prop_cycle(cycle)
for m in ("sEKS", "pEKS", "dp5", "kv5"):
    ax.plot(df["cores"], df[m], label=labels[m], markersize=10)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of CUDA cores")
ax.set_ylabel("Runtime [s]")
ax.legend()
fig.savefig(filedir / "plot_cores.pdf", bbox_inches="tight")
print(f"Saved plot to {filedir / 'plot_cores.pdf'}")


fig, ax = plt.subplots(1, 1)
cyc = cycler("color", ["r", "b"]) * cycler("linestyle", ["-", "--", ":"])
cyc = cycler("color", ["r", "b", "gray", "gray"]) * cycler(
    "linestyle", ["-", "--", ":"]
)
ax.set_prop_cycle(cyc)
for i, m in enumerate(("sEKS", "pEKS", "dp5", "kv5")):
    for gpu in gpus:
        df = dfs[gpu]
        ax.plot(df[x], df[m], linewidth=2, label=f"{labels[m]} ({gpu_labels[gpu]})")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of gridpoints")
ax.set_ylabel("Runtime [s]")
# ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

legend_elements = [
    matplotlib.patches.Patch(facecolor="r", label="Sequential EKS"),
    matplotlib.patches.Patch(facecolor="b", label="Parallel EKS"),
    matplotlib.patches.Patch(facecolor="gray", label="Dopri5 (diffrax)"),
    matplotlib.patches.Patch(facecolor="gray", label="Kvaerno5 (diffrax)"),
]
lg = ax.legend(handles=legend_elements, loc="upper left")
ax.add_artist(lg)
legend_elements = [
    matplotlib.lines.Line2D([0], [0], linestyle="-", color="k", label="GTX 1060"),
    matplotlib.lines.Line2D([0], [0], linestyle="--", color="k", label="RTX 2080 Ti"),
    matplotlib.lines.Line2D([0], [0], linestyle=":", color="k", label="V100"),
]
ax.legend(handles=legend_elements, loc="lower right")
# ax.legend(handles=legend_elements, loc="upper left")

for i, gpu in enumerate(gpus):
    s = ["-", "--", ":"][i]
    ax.axvline(CUDA_CORES[gpu], 0.0, 1.0, color="black", linewidth=1.0, linestyle=s)
fig.savefig(filedir / "gpu_comparison.pdf", bbox_inches="tight")
print(f"Saved plot to {filedir / 'gpu_comparison.pdf'}")
