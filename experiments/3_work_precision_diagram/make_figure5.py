import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_plot_stuff import *

import pandas as pd
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
    for i, key in enumerate(classic_keys):
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


# plt.rcParams.update(figsizes.jmlr2001(nrows=2, ncols=4))
def plot_wpd(df, ax, labels=True):
    # replace_large_with_inf(df)
    plot_xy(
        df=df,
        x=lambda k: f"{k}_rmse_traj",
        y=lambda k: f"{k}_runtime",
        ax=ax,
        labels=labels,
    )
    ax.set_title(rf"$\bf a.$ Work-precision diagram", loc="left")


def plot_wpd_gridsize(df, ax, labels=True):
    # replace_large_with_inf(df)
    plot_xy(
        df=df,
        x=lambda k: f"{k}_rmse_traj",
        y=lambda k: f"Ns",
        ax=ax,
        labels=labels,
    )


def make_individual_plot(ivpname, device):
    filename = os.path.join(DIR, "data", f"{ivpname}_{device}.csv")
    df = pd.read_csv(filename)
    # df = df[df.Ns <= 2 ** (4 + 11)]

    fig, ax = plt.subplots(1, 1)
    plot_wpd(df, ax)
    ax.set_ylabel("RMSE (trajectory)")
    ivp = IVPS[ivpname]

    ax1 = ax.inset_axes([0.71, 0.65, 0.27, 0.27])
    plot_solution(ivp, ax1)
    ax1.set_title(rf"$\bf b.$ {IVPLABELS[ivpname]}", loc="left")

    leg = fig.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.0)

    filepath = os.path.join(DIR, "figures", f"workprecision_{ivpname}.pdf")
    fig.savefig(filepath)
    print(f"Saved to {filepath}")


DEVICE = "Tesla_V100-SXM2-32GB"
IVPNAMES = ["logistic", "rigidbody", "vdp0"]
fig, axes = plt.subplots(2, len(IVPNAMES), sharey="row", sharex="col")
DEVICE = "Tesla_V100-SXM2-32GB"
for i, ivpname in enumerate(IVPNAMES):
    ax = axes[0, i]
    if i == 0:
        ax.set_ylabel("Runtime [s]")

    # filename = os.path.join(DIR, "data", f"data_{ivpname}.csv")
    filename = os.path.join(DIR, "data", f"{ivpname}_{DEVICE}.csv")
    df = pd.read_csv(filename)
    # df = df[df.Ns <= 2 ** (4 + 13)]

    plot_wpd(df, ax, labels=i == 0)
    ax.set_title(rf"$\bf {chr(ord('a') + i)}.$ {IVPLABELS[ivpname]}", loc="left")
    ivp = IVPS[ivpname]

    ax1 = ax.inset_axes([0.65, 0.75, 0.33, 0.23])
    plot_solution(ivp, ax1)
    # ax1.set_title(rf"$\bf b.$ {IVPLABELS[ivpname]}", loc="left")

    ax = axes[1, i]
    if i == 0:
        ax.set_ylabel("Grid size")
    plot_wpd_gridsize(df, ax, labels=False)

    axes[1, i].set_xlabel("RMSE")

leg = fig.legend(bbox_to_anchor=(1.01, 0.5), loc="center left", borderaxespad=0.0)

filepath = os.path.join(DIR, f"figure5.pdf")
fig.savefig(filepath)
print(f"Saved to {filepath}")
