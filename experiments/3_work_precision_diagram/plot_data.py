import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_plot_stuff import *

import pandas as pd
import matplotlib.pyplot as plt
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
        # "axes.grid.which": "both",
        "axes.grid.which": "major",
    }
)
# DIR = "./experiments/4_work_precision_diagram"
DIR = Path(__file__).parent

classic_keys = [
    "ImplicitEuler",
    "KV3",
    "KV5",
    # "DP5",
]
eks_keys = [
    "EKS(1)",
    "EKS(2)",
    "EKS(3)",
    # "EKS(5)",
]
ieks_keys = [
    "IEKS(1)",
    "IEKS(2)",
    "IEKS(3)",
    # "IEKS(5)",
    "sIEKS(1)",
    "sIEKS(2)",
    "sIEKS(3)",
]

CLASSIC_MARKERS = ["s", "v", "D", "P", "X", "d", "p", "h", "H", "8"]


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
        )
    for key in ieks_keys:
        label = key
        order = get_order(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            marker="o" if key.startswith("IEKS") else "d",
            markersize=3,
            color=f"C{order}",
        )
    for key in eks_keys:
        label = key
        order = get_order(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            marker="o",
            markersize=3,
            alpha=0.3,
            color=f"C{order}",
            linestyle="dashed",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend()
    return ax


def plot(df):
    fig, ax = plt.subplots(2, 2)

    # 1: error vs runtime
    # plot_xy(
    #     df=df,
    #     x=lambda k: f"{k}_runtime",
    #     y=lambda k: f"{k}_rmse_final",
    #     ax=ax[0, 0],
    # )
    # ax[0, 0].set_xlabel("Runtime [s]")
    # ax[0, 0].set_ylabel("RMSE (final)")

    plot_xy(
        df=df,
        x=lambda k: f"{k}_runtime",
        y=lambda k: f"{k}_rmse_traj",
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Runtime [s]")
    ax[0, 0].set_ylabel("RMSE (trajectory)")

    # 2: error vs N
    # plot_xy(
    #     df=df,
    #     x=lambda k: f"Ns",
    #     y=lambda k: f"{k}_rmse_final",
    #     ax=ax[0, 1],
    #     labels=False,
    # )
    # ax[0, 1].set_xlabel("Number of steps")
    # ax[0, 1].set_ylabel("RMSE (final)")

    plot_xy(
        df=df,
        x=lambda k: f"Ns",
        y=lambda k: f"{k}_rmse_traj",
        ax=ax[0, 1],
        labels=False,
    )
    ax[0, 1].set_xlabel("Number of steps")
    ax[0, 1].set_ylabel("RMSE (trajectory)")

    # 3: runtime vs N
    plot_xy(
        df=df,
        x=lambda k: f"Ns",
        y=lambda k: f"{k}_runtime",
        ax=ax[1, 0],
        labels=False,
    )
    ax[1, 0].set_xlabel("Number of steps")
    ax[1, 0].set_ylabel("Runtime [s]")

    # 4: ieks iterations
    for key in ieks_keys:
        label = key
        order = get_order(key)
        ax[1, 1].plot(
            df[f"Ns"],
            df[f"{key}_iterations"],
            # label=label,
            marker="o",
            markersize=3,
            color=f"C{order}",
        )
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("Number of steps")
    ax[1, 1].set_ylabel("IEKS Iterations")
    # ax[1,1].legend()

    leg = fig.legend(bbox_to_anchor=(1.02, 0.5), loc="center left", borderaxespad=0.0)
    # leg = fig.legend(loc="center right")
    # fig.tight_layout()

    return fig


PLOT_SETUPS = {
    "logistic": {"MinMaxGrid": (2**7, 2**20)},  # full grid: 2**4, 2**20
    "vdp0": {"MinMaxGrid": (2**7, 2**20)},  # full grid: 2**7, 2**20
    "rigidbody": {"MinMaxGrid": (2**7, 2**20)},  # full grid: 2**7, 2**20
}


for devicename in ["cpu", "Tesla_V100-SXM2-32GB", "NVIDIA_GeForce_RTX_2080_Ti"]:
    # for ivpname in ["logistic", "fhn", "fhn500", "vdp1", "seir", "rigidbody"]:
    for ivpname in ["logistic", "vdp0", "rigidbody"]:
        filename = os.path.join(DIR, "data", f"{ivpname}_{devicename}.csv")
        df = pd.read_csv(filename)
        min_Ns, max_Ns = PLOT_SETUPS[ivpname]["MinMaxGrid"]
        df = df[(min_Ns <= df.Ns) & (df.Ns <= max_Ns)]
        replace_large_with_inf(df)
        fig = plot(df)
        # save high resolution figure
        filename = os.path.join(DIR, "figures", f"wpd_{ivpname}_{devicename}.pdf")
        # fig.savefig(os.path.join(DIR, f"wpd_{ivpname}.png"), bbox_inches="tight", dpi=300)
        fig.savefig(filename, bbox_inches="tight")
        print(f"Saved to {filename}")
