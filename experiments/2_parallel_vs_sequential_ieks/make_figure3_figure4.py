import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_plot_stuff import *

import pandas as pd
import matplotlib.pyplot as plt
from tueplots import axes, bundles, figsizes
from tueplots.constants import markers
import diffrax

from pof.ivp import *
from pof.diffrax import solve_diffrax, get_ts_ys

plt.rcParams.update(bundles.jmlr2001())
plt.rcParams.update(axes.lines())
plt.rcParams.update(
    {
        # "lines.linewidth": 1.5,
        "lines.markeredgecolor": "black",
        "lines.markeredgewidth": 0.5,
        # "lines.markersize": 4,
        "axes.grid": True,
        # "axes.grid.which": "both",
        "axes.grid.which": "major",
    }
)
DIR = "./experiments/3_work_precision_diagram"


ORDERS = [1, 2]
ieks_keys = [f"IEKS({o})" for o in ORDERS]
sieks_keys = [f"sIEKS({o})" for o in ORDERS]
ORDER2MARKER = {
    1: "o",
    2: "s",
    3: "p",
}


# IVPNAMES = ["logistic", "fhn", "rigidbody"]
IVPNAMES = ["logistic", "rigidbody", "vdp0"]
PLOT_SETUPS = {
    "logistic": {
        1: (2**9, 2**20),
        2: (2**5, 2**12),
        3: (2**4, 2**10),
    },  # full grid: 2**4, 2**20
    "rigidbody": {
        1: (2**7, 2**20),
        2: (2**7, 2**17),
        3: (2**7, 2**14),
    },  # full grid: 2**7, 2**20
    "vdp0": {
        1: (2**7, 2**20),
        2: (2**7, 2**16),
        3: (2**7, 2**13),
    },  # full grid: 2**7, 2**20
}
devicename = "Tesla_V100-SXM2-32GB"


def threshold_df(df, order, ivpname):
    min_Ns, max_Ns = PLOT_SETUPS[ivpname][order]
    return df[(min_Ns <= df.Ns) & (df.Ns <= max_Ns)]


def plot_wpd(df, ax, ivpname, labels=False):
    replace_large_with_inf(df)
    for order in ORDERS:
        _df = threshold_df(df, order, ivpname)
        key = f"IEKS({order})"
        label = f"Parallel IEKS\n(IWP({order}))"
        ax.plot(
            _df[f"{key}_rmse_traj"],
            _df[f"{key}_runtime"],
            label=label if labels else "",
            marker=ORDER2MARKER[order],
            markersize=5,
            color=f"C{order}",
            zorder=100 + order,
            alpha=LINEALPHA,
            linewidth=LINEWIDTH,
        )
        key = f"sIEKS({order})"
        label = f"Sequential IEKS\n(IWP({order}))"
        ax.plot(
            _df[f"{key}_rmse_traj"],
            _df[f"{key}_runtime"],
            label=label if labels else "",
            marker=ORDER2MARKER[order],
            markersize=3,
            color=f"C{order}",
            linestyle="dashed",
            zorder=90 + order,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def plot_wpd_gridsize(df, ax, ivpname, labels=False):
    replace_large_with_inf(df)
    for order in ORDERS:
        _df = threshold_df(df, order, ivpname)
        key = f"IEKS({order})"
        label = f"Parallel IEKS\n(IWP({order}))"
        ax.plot(
            _df[f"{key}_rmse_traj"],
            _df["Ns"],
            label=label if labels else "",
            marker=ORDER2MARKER[order],
            markersize=5,
            color=f"C{order}",
            zorder=90 + order,
            alpha=LINEALPHA,
            linewidth=LINEWIDTH,
        )
        key = f"sIEKS({order})"
        label = f"Sequential IEKS\n(IWP({order}))"
        ax.plot(
            _df[f"{key}_rmse_traj"],
            _df["Ns"],
            label=label if labels else "",
            marker=ORDER2MARKER[order],
            markersize=3,
            color=f"C{order}",
            linestyle="dashed",
            zorder=100 + order,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    return ax


def plot_speedup(df, ax, ivpname, labels=True):
    replace_large_with_inf(df)
    for order in ORDERS:
        _df = threshold_df(df, order, ivpname)
        speedup = _df[f"sIEKS({order})_runtime"] / _df[f"IEKS({order})_runtime"]
        label = f"order = {order}"
        ax.plot(
            _df[f"IEKS({order})_rmse_traj"],
            speedup,
            label=label if labels else "",
            marker=ORDER2MARKER[order],
            markersize=5,
            color=f"C{order}",
            alpha=LINEALPHA,
            linewidth=LINEWIDTH,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=1)
    return ax


plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=1))
fig, axes = plt.subplots(2, len(IVPNAMES), sharey="row", sharex="col")
for i, ivpname in enumerate(IVPNAMES):
    ax = axes[0, i]

    df = get_df(ivpname, devicename)

    plot_wpd(df, ax, ivpname=ivpname, labels=i == 2)
    ax.set_title(rf"$\bf {chr(ord('a') + i)}.$ {IVPLABELS[ivpname]}", loc="left")

    ax1 = ax.inset_axes([0.65, 0.75, 0.33, 0.23])
    plot_solution(IVPS[ivpname], ax1)

    ax = axes[1, i]
    plot_speedup(df, ax, ivpname=ivpname, labels=False)
    axes[-1, i].set_xlabel("RMSE")

    # plot_wpd(df, axes[2, i], ivpname=ivpname, ykey=lambda k: "Ns")

axes[0, 0].set_ylabel("Runtime [s]")
axes[1, 0].set_ylabel("Parallel speed-up")
# axes[2, 0].set_ylabel("Grid size")

# fig.subplots_adjust(right=0.8)
leg = fig.legend(
    bbox_to_anchor=(1.01, 0.5),
    loc="center left",
    borderaxespad=0.0,
)

filepath = os.path.join("./experiments/2_parallel_vs_sequential_ieks/figure4.pdf")
fig.savefig(filepath, bbox_inches="tight")
print(f"Saved to {filepath}")


plt.rcParams.update(figsizes.jmlr2001(nrows=1.2, ncols=2))
fig, axes = plt.subplots(1, len(IVPNAMES), sharey="row", sharex="col")
for i, ivpname in enumerate(IVPNAMES):
    ax = axes[i]

    df = get_df(ivpname, devicename)

    ax.set_title(rf"$\bf {chr(ord('a') + i)}.$ {IVPLABELS[ivpname]}", loc="left")
    plot_wpd_gridsize(df, ax, ivpname=ivpname, labels=i == 2)
    ax.set_xlabel("RMSE")

    ax1 = ax.inset_axes([0.65, 0.75, 0.33, 0.23])
    plot_solution(IVPS[ivpname], ax1)


axes[0].set_ylabel("Grid size")

# fig.subplots_adjust(right=0.8)
leg = fig.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.0)

filepath = os.path.join("./experiments/2_parallel_vs_sequential_ieks/figure3.pdf")
fig.savefig(filepath, bbox_inches="tight")
print(f"Saved to {filepath}")
