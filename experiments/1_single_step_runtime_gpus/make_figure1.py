from pathlib import Path
from collections import defaultdict
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from common_plot_stuff import *

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
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


filedir = Path(__file__).parent
# filedir = Path("experiments/3_gpu_comparison")

GPUS = [
    "1060",
    "1080ti",
    "titanxp",
    "2080ti",
    "v100",
    # "3090",
]
dfs = {gpu: pd.read_csv(filedir / "results" / f"{gpu}.csv") for gpu in GPUS}

gpu_annotation_offset = {
    # "1060": (-20, 10),
    "1060": (-7, 10),
    "1080ti": (-65, -8),
    "2080ti": (-50, -15),
    # "2080ti": (5, 0),
    # "2080ti": (-8, 9),
    # "2080ti": (5, 0),
    "v100": (-13, -15),
    "titanxp": (-5, 10),
    "3090": (0, 0),
}


CLASSIC_SOLVERS = [
    "dp5",
    # "kv3",
    "kv5",
]
GPU_COMPARISON_DT = 2**-13
SOLVERS = ["sEKS", "pEKS", *CLASSIC_SOLVERS]

STYLES = {
    "pEKS": {"marker": "d", "color": "b"},
    "sEKS": {"marker": "o", "color": "r"},
    "dp5": {"marker": "^", "color": "gray"},
    "kv3": {"marker": ">", "color": "gray"},
    "kv5": {"marker": "v", "color": "gray"},
    "1060": {"linestyle": "-"},
    "1080ti": {"linestyle": "--"},
    "2080ti": {"linestyle": ":"},
    "v100": {"linestyle": (0, (3, 1, 1, 1))},
}

x = "N"
ALPHA = 0.2


def plot_runtimes_with_gpu_subplots():
    plt.rcParams.update(figsizes.jmlr2001(nrows=len(GPUS), ncols=1))
    fig, axes = plt.subplots(len(GPUS), 1, sharey=True, sharex=True)
    for ax, gpu in zip(axes, GPUS):
        df = dfs[gpu]
        for i, m in enumerate(("sEKS", "pEKS")):
            z = 100 + i
            ax.plot(
                df[x],
                df[m],
                label=LABELS[m],
                markersize=10,
                linewidth=4,
                zorder=z,
                **STYLES[m],
            )

        ref_alpha = 1
        for m in CLASSIC_SOLVERS:
            ax.plot(df[x], df[m], label=LABELS[m], alpha=ref_alpha, **STYLES[m])

        ax.set_xscale("log")
        ax.set_yscale("log")

    axes[0].set_xlabel("Number of gridpoints")
    axes[1].set_xlabel("Number of gridpoints")
    axes[2].set_xlabel("Number of gridpoints")

    for i, gpu in enumerate(GPUS):
        axes[i].set_title(LABELS[gpu])

    axes[0].set_ylabel("Runtime [s]")
    axes[0].legend()

    filename = filedir / "gpu_subplots.pdf"
    fig.savefig(filename, bbox_inches="tight")
    print(f"Saved plot to {filename}")


def plot_runtimes():
    plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=1))
    fig, ax = plt.subplots(1, 1)
    for m in SOLVERS:
        for gpu in GPUS:
            df = dfs[gpu]
            ax.plot(
                df[x],
                df[m],
                linewidth=2,
                label=f"{LABELS[m]} ({LABELS[gpu]})",
                markersize=5,
                **STYLES[m],
                **STYLES[gpu],
            )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of gridpoints")
    ax.set_ylabel("Runtime [s]")

    ax.plot(
        df[x],
        df[x] / 6500,
        label="$\propto$ N",
        marker="",
        alpha=ALPHA,
        linewidth=20,
        linestyle="solid",
        color="r",
    )

    ax.plot(
        df[x],
        np.log(df[x]) / 1100,
        label="$\propto$ log(N)",
        marker="",
        alpha=ALPHA,
        linewidth=20,
        linestyle="solid",
        color="b",
    )

    legend_elements = [
        matplotlib.lines.Line2D([0], [0], linestyle="-", label=LABELS[m], **STYLES[m])
        for m in SOLVERS
    ] + [
        matplotlib.patches.Patch(facecolor="r", label="$\propto$ N", alpha=ALPHA),
        matplotlib.patches.Patch(facecolor="b", label="$\propto$ log(N)", alpha=ALPHA),
    ]
    lg = ax.legend(handles=legend_elements, loc="upper left")
    ax.add_artist(lg)

    legend_elements = [
        matplotlib.lines.Line2D([0], [0], color="k", label=LABELS[gpu], **STYLES[gpu])
        for gpu in GPUS
    ]
    ax.legend(handles=legend_elements, loc="lower right")
    # ax.legend(handles=legend_elements, loc="upper left")

    ax.margins(x=0.025)

    filepath = filedir / "gpu_jointplot.pdf"
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved plot to {filepath}")


def plot_single_gpu_runtimes(ax, legend=True):
    gpu = "v100"

    ax.axvline(
        [CUDA_CORES[gpu]], color="black", linestyle="--", linewidth=0.5, zorder=1
    )
    # ax.annotate(
    #     "V100 \#CUDACORES",
    #     (CUDA_CORES[gpu], 1e-3),
    #     textcoords="offset points",
    #     xytext=(2, 5),
    #     fontsize=6,
    # )

    df = dfs[gpu]
    for m in SOLVERS:
        ax.plot(
            df[x],
            df[m],
            linewidth=2,
            label=f"{LABELS[m]} ({LABELS[gpu]})",
            markersize=5,
            **STYLES[m],
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of gridpoints")
    ax.set_ylabel("Runtime [s]")
    # ax.legend(bbox_to_anchor=(1.05, 0.5), loc="center left")

    ax.plot(
        df[x],
        df[x] / 8000,
        label="$\propto$ N",
        marker="",
        alpha=ALPHA,
        linewidth=10,
        linestyle="solid",
        color="r",
    )

    ax.plot(
        df[x],
        np.log(df[x]) / 1300,
        label="$\propto$ log(N)",
        marker="",
        alpha=ALPHA,
        linewidth=10,
        linestyle="solid",
        color="b",
    )

    if legend:
        legend_elements = [
            matplotlib.lines.Line2D(
                [0],
                [0],
                linestyle="-",
                color=STYLES["sEKS"]["color"],
                label="Sequential EKS",
                marker=STYLES["sEKS"]["marker"],
            ),
            matplotlib.lines.Line2D(
                [0],
                [0],
                linestyle="-",
                color=STYLES["pEKS"]["color"],
                label="Parallel EKS",
                marker=STYLES["pEKS"]["marker"],
            ),
            matplotlib.patches.Patch(facecolor="r", label="$\propto$ N", alpha=ALPHA),
            matplotlib.patches.Patch(
                facecolor="b", label="$\propto$ log(N)", alpha=ALPHA
            ),
        ]
        lg = ax.legend(handles=legend_elements, loc="upper left")
        ax.add_artist(lg)
        ax.margins(x=0.025)


def plot_cores(ax, legend=True, save=False):
    rs = defaultdict(lambda: [])
    for gpu in GPUS:
        rs["cores"].append(CUDA_CORES[gpu])
        df = dfs[gpu]
        row = df[df.dt == GPU_COMPARISON_DT]
        for m in ("sEKS", "pEKS", *CLASSIC_SOLVERS):
            rs[m].append(row[m].values[0])
    df = pd.DataFrame(rs)

    for m in SOLVERS:
        ax.plot(df["cores"], df[m], label=LABELS[m], markersize=10, **STYLES[m])
    # ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of CUDA cores")
    ax.set_ylabel("Runtime [s]")
    ax.set_xlim(1e3, 6e3)
    for i in range(len(GPUS)):
        ax.annotate(
            LABELS[GPUS[i]],
            (df["cores"][i], df["pEKS"][i]),
            textcoords="offset points",
            xytext=gpu_annotation_offset[GPUS[i]],
        )
    if legend:
        ax.legend()


def plot_things():
    plt.rcParams.update(figsizes.jmlr2001(nrows=1, ncols=1))
    fig, axes = plt.subplots(
        1, 2, sharey=True, gridspec_kw={"width_ratios": [1.61803, 1]}
    )

    plot_single_gpu_runtimes(ax=axes[0], legend=False)
    axes[0].set_title(rf"$\bf a.$ Single-step runtime benchmark", loc="left")

    plot_cores(ax=axes[1], legend=False)
    axes[1].set_title(rf"$\bf b.$ GPU comparison", loc="left")
    axes[1].set_ylabel(None)

    legend_elements = (
        [
            matplotlib.lines.Line2D(
                [0], [0], linestyle="-", label=LABELS[m], **STYLES[m]
            )
            for m in ("sEKS", "pEKS", *CLASSIC_SOLVERS)
        ]
        + [
            matplotlib.patches.Patch(facecolor="r", label="$\propto$ N", alpha=ALPHA),
            matplotlib.patches.Patch(
                facecolor="b", label="$\propto$ log(N)", alpha=ALPHA
            ),
        ]
        + [
            matplotlib.lines.Line2D(
                [0],
                [0],
                linestyle="--",
                label="V100 CUDA cores",
                color="black",
                linewidth=0.5,
            )
        ]
    )

    # leg = fig.legend(
    # leg = axes[1].legend(
    leg = axes[0].legend(
        handles=legend_elements,
        # bbox_to_anchor=(1.01, 0.5),
        # loc="center left",
        # borderaxespad=0.0,
        # loc="lower left",
        loc="upper left",
    )
    # ax.add_artist(lg)
    axes[1].margins(y=0.025)

    filepath = filedir / "figure1.pdf"
    fig.savefig(filepath, bbox_inches="tight")
    print(f"Saved plot to {filepath}")


plot_things()
# plot_runtimes_with_gpu_subplots()
# plot_runtimes()
