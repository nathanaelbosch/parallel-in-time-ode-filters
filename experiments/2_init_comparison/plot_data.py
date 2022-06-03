from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from cycler import cycler
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
        "axes.grid.which": "both",
        "figure.titlesize": plt.rcParams["axes.labelsize"],
    }
)


names = (
    ("constant", "Constant"),
    ("prior", "Prior"),
    ("updated_prior", "Prior+Update"),
    # ("coarse_solver_2p-0", "RK (dt=$2^{0}$)"),
    ("coarse_solver_2p-1", "RK (dt=$2^{-1}$)"),
    ("coarse_solver_2p-2", "RK (dt=$2^{-2}$)"),
    ("coarse_solver_2p-3", "RK (dt=$2^{-3}$)"),
    ("coarse_solver_2p-4", "RK (dt=$2^{-4}$)"),
)
_colors = ["r", "g", "b", "k", "dimgray", "gray", "darkgray", "lightgray"]
_markers = ["p", "P", "X", "d", "s", "*", "o", "v"]
cycle = cycler("color", _colors) + cycler("marker", _markers)

toplot = "mse"
# toplot = "nll"
# toplot = "obj"
order = 4


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
            "1e-0",
            "1e-1",
            "1e-2",
            "1e-3",
            # "1e-4",
        ),
    ),
):

    fig, axes = plt.subplots(
        1,
        len(dts),
        sharey=True,
        # sharex=True,
    )

    for i, (ax, dt) in enumerate(zip(axes, dts)):
        filename = f"prob={ivp}_dt={dt}_order={order}_dev.csv"
        df = pd.read_csv(folder / "data" / filename, index_col=0)

        ax.set_prop_cycle(cycle)
        for j in range(len(names)):
            key, label = names[j]
            ms = 8 - 2 * j if j < 3 else 5
            # print(df[f"{toplot}_{key}"].dropna())
            ax.plot(df[f"{toplot}_{key}"], label=label, markersize=ms)
            # ax.set_xscale("log")
        ax.set_yscale("log")

        ax.set_title(rf"$\bf {chr(97+i)})$ " + rf"dt={dt}", loc="left")

        best = df["mse_coarse_solver_2p-4"].dropna()
        minval = best[best.last_valid_index()]
        ax.axhline(minval, color="black", linestyle="dashed", zorder=-1)
        if ivp == "logistic":
            # ax.set_xlim(-3, 25)
            pass
        else:
            if order == 5:
                ax.set_ylim(1e-20, 1e16)
            # if order == 4:
            #     ax.set_ylim(1e-20, 1e15)

            # ax.set_xlim(-3, 100)
            pass
    # for ax in axes:
    #     ax.set_xlabel("Number of iterations")
    fig.supxlabel("Number of iterations")
    fig.supylabel("Mean squared error")
    axes[0].legend()

    fig.suptitle(f"Lotka-Volterra (order={order}; LM-regularized)")

    _p = folder / f"{ivp}_order{order}_reg.pdf"
    fig.savefig(_p)
    print(f"saved figure to {_p}")
