import os

import pandas as pd
import diffrax

from pof.ivp import *
from pof.diffrax import solve_diffrax

LINEALPHA = 0.8
LINEWIDTH = 3

IVPS = {
    "logistic": logistic(),
    "fhn": fitzhughnagumo(),
    "rigidbody": rigid_body(),
    "vdp0": vanderpol(stiffness_constant=1e0),
}
IVPLABELS = {
    "logistic": "Logistic",
    "fhn": "FitzHugh--Nagumo",
    "rigidbody": "Rigid Body",
    "vdp0": "Van der Pol",
}
LABELS = {
    "pEKS": "Parallel EKS",
    "sEKS": "Sequential EKS",
    "dp5": "Dopri5 (diffrax)",
    "kv5": "Kvaerno5 (diffrax)",
    "kv3": "Kvaerno3 (diffrax)",
    "1060": "GTX 1060",
    "1080ti": "GTX 1080 Ti",
    "titanxp": "TITAN Xp",
    "2080ti": "RTX 2080 Ti",
    "v100": "V100",
}


def get_order(key):
    if key.startswith("IEKS"):
        return int(key[5])
    elif key.startswith("EKS"):
        return int(key[4])
    elif key.startswith("sIEKS"):
        return int(key[6])
    else:
        raise ValueError(f"Can't parse order from the given key: {key}")


def get_df(ivpname, devicename):
    DIR = "./experiments/3_work_precision_diagram"
    filename = os.path.join(DIR, "data", f"{ivpname}_{devicename}.csv")
    df = pd.read_csv(filename)
    return df


def plot_solution(ivp, ax):
    sol = solve_diffrax(
        ivp.f,
        ivp.y0,
        ivp.t_span,
        # solver=diffrax.Kvaerno5,
        # atol=1e-6,
        # rtol=1e-3,
        max_steps=int(1e5),
    )
    ts = jnp.linspace(*ivp.t_span, 100)
    ys = jax.vmap(sol.evaluate)(ts)
    ax.plot(ts, ys, linewidth=1.0, color="black")
    # if ys.shape[1] != 2:
    #     ax.plot(ts, ys, linewidth=1.0, color="black")
    # else:
    #     ax.plot(ys[:, 0], ys[:, 1], linewidth=1.0, color="black")
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.tick_params(left=False, bottom=False)
    ax.grid(visible=False)


def replace_large_with_inf(df, large=1e8):
    df[df > large] = float("inf")
    return df
