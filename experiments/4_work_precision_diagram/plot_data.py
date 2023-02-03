import os

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

current_dir = "./experiments/4_work_precision_diagram"
filename = os.path.join(current_dir, "data.csv")
df = pd.read_csv(filename)

MAXITER = 1000

CLASSIC_MARKERS = ["o", "s", "v", "x", "D", "P", "X", "d", "p", "h", "H", "8"]


def get_order_and_maxiter(k):
    if k.startswith("IEKS"):
        order = int(k[5])
    elif k.startswith("EKS"):
        order = int(k[4])
    if len(k) >= 8:
        maxiter = int(k[8:])
    else:
        maxiter = MAXITER
    return int(order), int(maxiter)


def replace_large_with_inf(df, large=1e8):
    df[df > large] = float("inf")
    return df


def plot_xy(*, df, keys, x, y, ax, labels=True):
    classic_keys = [c for c in keys if "IEKS" not in c and "EKS" not in c]
    eks_keys = [c for c in keys if "EKS" in c and "IEKS" not in c]
    ieks_keys = [c for c in keys if "IEKS" in c]
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
    for key in eks_keys:
        label = key
        order, _ = get_order_and_maxiter(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            marker="o",
            markersize=3,
            # alpha=(maxiter / MAXITER) ** (1 / 4),
            color=f"C{order}",
            linestyle="dashed",
        )
    for key in ieks_keys:
        label = key
        order, maxiter = get_order_and_maxiter(key)
        ax.plot(
            df[x(key)],
            df[y(key)],
            label=label if labels else "",
            marker="o",
            markersize=3,
            alpha=(maxiter / MAXITER) ** (1 / 4),
            color=f"C{order}",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.legend()
    return ax


def plot(df):
    fig, ax = plt.subplots(2, 2)

    keys = [c[:-5] for c in df.columns if "time" in c]

    classic_keys = [c for c in keys if "IEKS" not in c and "EKS" not in c]
    eks_keys = [c for c in keys if "EKS" in c and "IEKS" not in c]
    ieks_keys = [c for c in keys if "IEKS" in c]

    # 1: error vs runtime
    plot_xy(
        df=df,
        keys=keys,
        x=lambda k: f"{k}_time",
        y=lambda k: f"{k}_rmse",
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Runtime [s]")
    ax[0, 0].set_ylabel("RMSE")

    # 2: error vs N
    plot_xy(
        df=df,
        keys=keys,
        x=lambda k: f"Ns",
        y=lambda k: f"{k}_rmse",
        ax=ax[0, 1],
        labels=False,
    )
    ax[0, 1].set_xlabel("Number of steps")
    ax[0, 1].set_ylabel("RMSE")

    # 3: runtime vs N
    plot_xy(
        df=df,
        keys=keys,
        x=lambda k: f"Ns",
        y=lambda k: f"{k}_time",
        ax=ax[1, 0],
        labels=False,
    )
    ax[1, 0].set_xlabel("Number of steps")
    ax[1, 0].set_ylabel("Runtime [s]")

    # 4: ieks iterations
    for key in ieks_keys:
        label = key
        order, maxiter = get_order_and_maxiter(key)
        ax[1, 1].plot(
            df[f"Ns"],
            df[f"{key}_iterations"],
            # label=label,
            marker="o",
            markersize=3,
            alpha=(maxiter / 1000) ** (1 / 4),
            color=f"C{order}",
        )
    ax[1, 1].set_xscale("log")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_xlabel("Number of steps")
    ax[1, 1].set_ylabel("IEKS Iterations")
    # ax[1,1].legend()

    leg = fig.legend(bbox_to_anchor=(1.05, 0.5), loc="center left", borderaxespad=0.0)
    # leg = fig.legend(loc="center right")
    # fig.tight_layout()

    return fig


replace_large_with_inf(df)
fig = plot(df)
# save high resolution figure
fig.savefig(os.path.join(current_dir, "wpd.png"), bbox_inches="tight", dpi=300)
