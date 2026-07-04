"""
Run FW, PFW, PG on the three datasets and save the convergence plots + result tables.

Same setup for all three algorithms on a given dataset: same x0 (tau * e_1, a
vertex of the l1-ball), same tau, epsilon, max_iter.
"""

import os
import csv
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from lasso_problem import compute_L
from algorithms.frank_wolfe import frank_wolfe
from algorithms.pairwise_fw import pairwise_fw
from algorithms.projected_gradient import projected_gradient


# Experimental settings: 
EPSILON = 1e-4
MAX_ITER = 2000
STEP_SIZE_FW = "exact"

TAU = {
    "diabetes": 1.5,
    "riboflavin": 1.0,    
    "communities": 3.0,   
}

DATASETS = ["diabetes", "riboflavin", "communities"]
COLORS = {"FW": "tab:blue", "PFW": "tab:orange", "PG": "tab:green"}
STYLES = {"FW": "-", "PFW": "--", "PG": "-."}

BASE_DIR = os.path.dirname(__file__)
FIG_DIR = os.path.join(BASE_DIR, "..", "results", "figures")
TAB_DIR = os.path.join(BASE_DIR, "..", "results", "tables")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(TAB_DIR, exist_ok=True)

not_converged = []

for name in DATASETS:
    tau = TAU[name]
    data = load_dataset(name)
    A, b = data["A_train"], data["b_train"]
    p = A.shape[1]

    L = compute_L(A)
    print(f"\n{name}: n={data['n']}, p={p}, L={L:.6g} (step 1/L={1.0 / L:.3e})")

    # x0 = tau * e_1: a vertex of the l1-ball, feasible start for all three algos
    x0 = np.zeros(p)
    x0[0] = tau

    results = {
        "FW": frank_wolfe(A, b, tau, x0=x0, step_size=STEP_SIZE_FW, epsilon=EPSILON, 
                          max_iter=MAX_ITER),
        "PFW": pairwise_fw(A, b, tau, x0=x0, epsilon=EPSILON, max_iter=MAX_ITER),
        "PG": projected_gradient(A, b, tau, x0=x0, epsilon=EPSILON, max_iter=MAX_ITER),
    }

    # figure: objective | stopping criterion | active-set size
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    for algo, res in results.items():
        ax1.plot(res["f_history"], color=COLORS[algo], linestyle=STYLES[algo], label=algo)
    ax1.set_yscale("log")
    ax1.set_title("Objective f(x)")
    ax1.set_xlabel("iteration")
    ax1.legend()

    for algo, res in results.items():
        crit = res["gmap_norm_history"] if algo == "PG" else res["gap_history"]
        ax2.plot(crit, color=COLORS[algo], linestyle=STYLES[algo], label=algo)
    ax2.axhline(EPSILON, color="gray", linestyle=":", label="epsilon")
    ax2.set_yscale("log")
    ax2.set_title("Stopping criterion (FW gap / grad-map norm)")
    ax2.set_xlabel("iteration")
    ax2.legend()

    # PG has no active set; classic FW doesn't track one either, 
    # ||x_k||_0 is used as its proxy (PFW's |S^(t)| is the real value)
    for algo in ["FW", "PFW"]:
        hist_key = "active_set_size_history" if algo == "PFW" else "support_history"
        ax3.plot(results[algo][hist_key], color=COLORS[algo], linestyle=STYLES[algo], label=algo)
    ax3.set_title("Active-set size |S| (FW: ||x_k||_0)")
    ax3.set_xlabel("iteration")
    ax3.legend()

    fig.suptitle(f"{name}  (tau = {tau}, epsilon = {EPSILON:g})")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, f"{name}.png"), dpi=120)
    plt.close(fig)

    # table: one row per algorithm
    rows = []
    for algo, res in results.items():
        crit = res["gmap_norm_history"] if algo == "PG" else res["gap_history"]
        below_eps = np.where(np.asarray(crit) <= EPSILON)[0]
        converged = below_eps.size > 0
        iters = int(below_eps[0]) if converged else MAX_ITER

        if algo == "PFW":
            support = int(res["active_set_size_history"][-1])
        elif algo == "FW":
            support = int(res["support_history"][-1])
        else:  # PG: same count_nonzero(x) definition as FW, no separate threshold
            support = int(res["support_history"][-1])

        rows.append({
            "algorithm": algo,
            "final_f": float(res["f_history"][-1]),
            "iters_to_convergence": iters,
            "wall_clock_total_s": float(sum(res["time_history"])),
            "final_support_size": support,
            "converged": "yes" if converged else "did not converge",
        })
        if not converged:
            not_converged.append((name, algo))

    print(f"{name} results:")
    for r in rows:
        print(f"  {r['algorithm']:>4}: final_f={r['final_f']:.6e}  "
              f"iters={r['iters_to_convergence']}  "
              f"time={r['wall_clock_total_s']:.3f}s  "
              f"support={r['final_support_size']}  {r['converged']}")

    csv_path = os.path.join(TAB_DIR, f"{name}_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

print("\n" + "=" * 50)
if not_converged:
    print("Did not converge within max_iter:")
    for ds, algo in not_converged:
        print(f"  - {algo} on {ds}")
else:
    print("All (algorithm, dataset) pairs converged.")
