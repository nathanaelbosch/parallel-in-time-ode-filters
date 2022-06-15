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


names = (
    # ("constant", "Constant"),
    ("prior", "Prior"),
    # ("updated_prior", "Prior+Update"),
    # ("coarse_solver_2p-0", "RK (dt=$2^{0}$)"),
    # ("coarse_solver_2p-1", "RK (dt=$2^{-1}$)"),
    # ("coarse_solver_2p-2", "RK (dt=$2^{-2}$)"),
    # ("coarse_solver_2p-3", "RK (dt=$2^{-3}$)"),
    # ("coarse_solver_2p-4", "RK (dt=$2^{-4}$)"),
    ("coarse_dopri5", "Coarse RK"),
    ("coarse_ekf", "Coarse EKS"),
)
_colors = ["r", "g", "b", "k", "dimgray", "gray", "darkgray", "lightgray"]
_markers = ["p", "P", "X", "d", "s", "*", "o", "v"]
cycle = cycler("color", _colors) + cycler("marker", _markers)

toplot = "mse"
# toplot = "nll"
# toplot = "obj"
orders = (2, 3, 5)
suffix = "qpm"


folder = Path("experiments/2_init_comparison/")
for ivp, dts in (
    # (
    #     "logistic",
    #     (
    #         "1e-0",
    #         "1e-1",
    #         "1e-2",
    #         # "1e-3",
    #     ),
    # ),
    (
        "lotkavolterra",
        (
            # "1e-0",
            "1e-1",
            "1e-2",
            "1e-3",
            # "1e-4",
        ),
    ),
):

    fig, axes = plt.subplots(
        len(orders),
        len(dts),
        sharey=True,
        sharex=True,
    )

    for k, order in enumerate(orders):
        for i, dt in enumerate(dts):
            ax = axes[k, i]
            filename = f"prob={ivp}_dt={dt}_order={order}_{suffix}.csv"
            df = pd.read_csv(folder / "data" / filename, index_col=0)

            ax.set_prop_cycle(cycle)
            for j in range(len(names)):
                key, label = names[j]
                # ms = 8 - 2 * j if j < 3 else 5
                ms = 5
                ax.plot(df[f"{toplot}_{key}"], label=label, markersize=ms)
            ax.set_yscale("log")

            ax.set_title(
                rf"$\bf {chr(97+i)})$ " + rf"dt={dt}; order={order}", loc="left"
            )

            best = df["mse_coarse_dopri5"].dropna()
            minval = best[best.last_valid_index()]
            ax.axhline(minval, color="black", linestyle="dashed", zorder=-1)
        ax.set_xlim(-1, 51)
        # ax.set_ylim(1e-16, 1e15)
    fig.supxlabel("Number of iterations")
    fig.supylabel("Mean squared error")
    axes[-1, -1].legend()
    # fig.legend(*axes[0, 0].get_legend_handles_labels())

    fig.suptitle(f"Lotka-Volterra")

    _p = folder / f"{ivp}_{suffix}.pdf"
    fig.savefig(_p)
    print(f"saved figure to {_p}")
