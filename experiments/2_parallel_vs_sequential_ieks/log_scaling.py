import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_plot_stuff import *

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tueplots import axes, bundles, figsizes
from tueplots.constants import markers


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

classic_keys = [
    # "DP5",
    "ImplicitEuler",
    "KV3",
    "KV5",
    # "DP5",
]
eks_keys = [
    "EKS(1)",
    "EKS(2)",
    "sIEKS(1)",
    "sIEKS(2)",
    # "EKS(3)",
    # "EKS(5)",
]
ieks_keys = [
    "IEKS(1)",
    "IEKS(2)",
    # "IEKS(3)",
    # "IEKS(5)",
]

MAXITER = 1000

CLASSIC_MARKERS = ["D", "v", "^"]
# six different markers for PN
PN_MARKERS = ["o", "s", "p", "h", "P", "X", "d", "p", "h", "H", "8"]


def plot_xy(*, df, x, y, ax, labels=True):
    for (i, key) in enumerate(classic_keys):
        label = key
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            marker=CLASSIC_MARKERS[i],
            markersize=5,
            color="gray",
            alpha=LINEALPHA,
        )
    for i, key in enumerate(ieks_keys):
        label = key
        order = get_order(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            # marker="o",
            # markersize=3,
            marker=PN_MARKERS[i],
            markersize=5,
            color=f"C{order}",
            linewidth=LINEWIDTH,
            alpha=LINEALPHA,
        )
    for i, key in enumerate(eks_keys):
        label = key
        order = get_order(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            # marker="o",
            # markersize=3,
            marker=PN_MARKERS[i + len(ieks_keys)],
            markersize=5,
            color=f"C{order}",
            linestyle="dashed",
            alpha=LINEALPHA,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend()
    return ax


def plot_scaling(df, ax, labels=True):
    # replace_large_with_inf(df)
    plot_xy(
        df=df,
        y=lambda k: f"{k}_runtime",
        x=lambda k: f"Ns",
        ax=ax,
        labels=labels,
    )


def plot_iterations(df, ax, labels=True):
    # replace_large_with_inf(df)

    for (i, key) in enumerate(classic_keys):
        label = key
        ax.plot(
            df["Ns"],
            np.ones_like(df["Ns"]),
            label=label if labels else "",
            marker=CLASSIC_MARKERS[i],
            markersize=5,
            color="gray",
            alpha=LINEALPHA,
        )
    for i, key in enumerate(ieks_keys):
        label = key
        order = get_order(key)
        ax.plot(
            df["Ns"],
            df[f"{key}_iterations"],
            label=label if labels else "",
            # marker="o",
            # markersize=3,
            marker=PN_MARKERS[i],
            markersize=5,
            color=f"C{order}",
            linewidth=LINEWIDTH,
            alpha=LINEALPHA,
        )
    for i, key in enumerate(eks_keys):
        label = key
        order = get_order(key)
        ax.plot(
            df["Ns"],
            df[f"{key}_iterations"]
            if key.startswith("sIEKS")
            else np.ones_like(df["Ns"]),
            label=label if labels else "",
            # marker="o",
            # markersize=3,
            marker=PN_MARKERS[i + len(ieks_keys)],
            markersize=5,
            color=f"C{order}",
            linestyle="dashed",
            alpha=LINEALPHA,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(0, 100)
    # ax.legend()
    return ax


def make_individual_plot(ivpname, device):
    filename = os.path.join(DIR, "data", f"{ivpname}_{device}.csv")
    df = pd.read_csv(filename)
    # df = df[df.Ns <= 2 ** (4 + 11)]

    fig, ax = plt.subplots(1, 1)
    plot_scaling(df, ax)
    ax.set_ylabel("Runtime [s]")
    ax.set_xlabel("Number of grid points")

    # ax.axvline(
    #     [CUDA_CORES["v100"]], color="black", linestyle="--", linewidth=0.5, zorder=1
    # )

    ivp = IVPS[ivpname]

    ax1 = ax.inset_axes([1 - 0.71 - 0.27, 0.65, 0.27, 0.27])
    plot_solution(ivp, ax1)
    ax1.set_title(rf"$\bf b.$ {IVPLABELS[ivpname]}", loc="left")

    leg = fig.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.0)

    filepath = os.path.join(
        "./experiments/2_parallel_vs_sequential_ieks", f"scaling_{ivpname}.pdf"
    )
    fig.savefig(filepath)
    print(f"Saved to {filepath}")


device = DEVICE = "Tesla_V100-SXM2-32GB"
IVPNAMES = ["logistic", "rigidbody", "vdp0"]

# for ivpname in IVPNAMES:
#     make_individual_plot(ivpname, DEVICE)

fig, axes = plt.subplots(2, len(IVPNAMES), sharey="row", sharex="col")
for i, ivpname in enumerate(IVPNAMES):
    ax = axes[0, i]
    if i == 0:
        ax.set_ylabel("Runtime [s]")

    filename = os.path.join(DIR, "data", f"{ivpname}_{device}.csv")
    df = pd.read_csv(filename)

    plot_scaling(df, ax, labels=i == 0)
    ax.set_title(rf"$\bf {chr(ord('a') + i)}.$ {IVPLABELS[ivpname]}", loc="left")

    # ax.axvline(
    #     [CUDA_CORES["v100"]], color="black", linestyle="--", linewidth=0.5, zorder=1
    # )

    ivp = IVPS[ivpname]
    ax1 = ax.inset_axes([1 - 0.71 - 0.27, 0.65, 0.27, 0.27])
    plot_solution(ivp, ax1)
    # ax1.set_title(rf"$\bf b.$ {IVPLABELS[ivpname]}", loc="left")

    ax = axes[1, i]
    plot_iterations(df, ax, labels=False)
    if i == 0:
        ax.set_ylabel("Number of iterations")
    ax.set_xlabel("Number of grid points")

    leg = fig.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.0)

filepath = os.path.join("./experiments/2_parallel_vs_sequential_ieks", f"scaling.pdf")
fig.savefig(filepath)
print(f"Saved to {filepath}")
