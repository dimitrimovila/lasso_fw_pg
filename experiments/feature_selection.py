"""
Feature-selection comparison across FW, PFW, and PG.

Runs all three algorithms with the same settings as run_experiments.py and 
compares which features end up in the support of the final x. 
Since all three solve the same convex LASSO problem, the final supports 
should agree (or nearly).

Support is defined equally for all three: |x_i| > SUPPORT_TOL on the final x.

Outputs figure on the top features selected by each algorithm as a figure .png
file with name dataset_feature_selection.png in the \results\figures folder. 
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from algorithms.frank_wolfe import frank_wolfe
from algorithms.pairwise_fw import pairwise_fw
from algorithms.projected_gradient import projected_gradient

# We run the algorithms for the a larger num of iterations (20,000) without
# terminating at the stopping condition of FW/PWF's gap and PG's gradient 
# mapping norm (ensured by setting epsilon = -1) since these stopping
# criterions are not on the same scale ((PG's ||g_C|| carries a 1/L deflation).
# Running this ensures fair comparison and that the final solution we attain 
# is closer to x* for the final support comparison. 

EPSILON = -1
MAX_ITER = 20_000
STEP_SIZE_FW = "exact"
TAU = {
    "diabetes": 1.5,
    "riboflavin": 1.0,
    "communities": 2.5,
}
DATASETS = ["diabetes", "riboflavin", "communities"]

# threshold below which a coefficient counts getting zeroed out
SUPPORT_TOL = 1e-8

# cap on features shown per bar chart (top by max |coefficient|)
TOP_K = 20

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# load feature names
DIABETES_NAMES = list(load_diabetes().feature_names)

for name in DATASETS:
    tau = TAU[name]
    data = load_dataset(name)
    A, b = data["A_train"], data["b_train"]
    p = A.shape[1]

    # x0 = tau * e_1: same vertex start used in run_experiments.py
    x0 = np.zeros(p)
    x0[0] = tau

    res_fw = frank_wolfe(A, b, tau, x0=x0, step_size=STEP_SIZE_FW,
                          epsilon=EPSILON, max_iter=MAX_ITER)
    res_pfw = pairwise_fw(A, b, tau, x0=x0, epsilon=EPSILON, max_iter=MAX_ITER)
    res_pg = projected_gradient(A, b, tau, x0=x0, epsilon=EPSILON, max_iter=MAX_ITER)
    x_fw, x_pfw, x_pg = res_fw["x"], res_pfw["x"], res_pg["x"]

    # check criterion value of the alg after budget iterations
    print(f"  n_iter (of {MAX_ITER} budget): FW={res_fw['n_iter']}  "
          f"PFW={res_pfw['n_iter']}  PG={res_pg['n_iter']}")
    print(f"  final criterion:              FW gap={res_fw['gap_history'][-1]:.3e}  "
          f"PFW gap={res_pfw['gap_history'][-1]:.3e}  "
          f"PG ||g_C||={res_pg['gmap_norm_history'][-1]:.3e}")

    support_fw = np.abs(x_fw) > SUPPORT_TOL
    support_pfw = np.abs(x_pfw) > SUPPORT_TOL
    support_pg = np.abs(x_pg) > SUPPORT_TOL

    # pairwise Jaccard overlap of the three supports (1.0 = identical support)
    u1 = np.sum(support_fw | support_pfw)
    jac_fw_pfw = np.sum(support_fw & support_pfw) / u1 if u1 > 0 else 1.0
    u2 = np.sum(support_fw | support_pg)
    jac_fw_pg = np.sum(support_fw & support_pg) / u2 if u2 > 0 else 1.0
    u3 = np.sum(support_pfw | support_pg)
    jac_pfw_pg = np.sum(support_pfw & support_pg) / u3 if u3 > 0 else 1.0

    print(f"\n{name}: p={p}, tau={tau}")
    print(f"  |support| FW={support_fw.sum()}  PFW={support_pfw.sum()}  PG={support_pg.sum()}")
    print(f"  Jaccard(FW,PFW)={jac_fw_pfw:.4f}  Jaccard(FW,PG)={jac_fw_pg:.4f}  "
          f"Jaccard(PFW,PG)={jac_pfw_pg:.4f}")

    # features selected by at least one algorithm, ranked by max |coefficient|
    any_selected = support_fw | support_pfw | support_pg
    idx = np.where(any_selected)[0]
    max_abs = np.maximum(np.maximum(np.abs(x_fw[idx]), np.abs(x_pfw[idx])), np.abs(x_pg[idx]))
    order = idx[np.argsort(-max_abs)][:TOP_K]

    if name == "diabetes":
        labels = [DIABETES_NAMES[i] for i in order]
    else:
        labels = [f"x{i}" for i in order]

    xpos = np.arange(len(order))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(6, len(order) * 0.5), 4.5))
    ax.bar(xpos - width, x_fw[order], width, label="FW", color="tab:blue")
    ax.bar(xpos, x_pfw[order], width, label="PFW", color="tab:orange")
    ax.bar(xpos + width, x_pg[order], width, label="PG", color="tab:green")
    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("coefficient value")
    n_shown = len(order)
    n_total = int(any_selected.sum())
    ax.set_title(f"{name}: final coefficients (top {n_shown} of {n_total} selected, tau={tau})")
    ax.legend()
    fig.tight_layout()
    out_path = os.path.join(FIG_DIR, f"{name}_feature_selection.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  figure saved -> {os.path.normpath(out_path)}")
