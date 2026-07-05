"""
Sweep tau over a range of values for one dataset, using PFW only (cheapest and
most reliable of the three per run_experiments.py's own results), to see the
accuracy/sparsity trade-off curve and where the currently chosen tau sits on it.
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from datasets import load_dataset
from algorithms.pairwise_fw import pairwise_fw

DATASET = "riboflavin"
TAUS = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
CURRENT_TAU = 1.0 
EPSILON = 1e-4
MAX_ITER = 10000

data = load_dataset(DATASET)
A, b = data["A_train"], data["b_train"]
p = A.shape[1]

rows = []
for tau in TAUS:
    x0 = np.zeros(p)
    x0[0] = tau
    res = pairwise_fw(A, b, tau, x0=x0, epsilon=EPSILON, max_iter=MAX_ITER)
    rows.append({
        "tau": tau,
        "final_f": res["f_history"][-1],
        "support": res["active_set_size_history"][-1],
        "n_iter": res["n_iter"],
        "converged": res["converged"],
    })
    print(f"tau={tau:>6}: final_f={res['f_history'][-1]:.4f}  "
          f"support={res['active_set_size_history'][-1]:>4}  "
          f"iters={res['n_iter']:>5}  converged={res['converged']}")

taus = [r["tau"] for r in rows]
fvals = [r["final_f"] for r in rows]
supports = [r["support"] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

ax1.plot(taus, fvals, "o-", color="tab:blue")
ax1.axvline(CURRENT_TAU, color="gray", linestyle=":", label=f"current tau={CURRENT_TAU}")
ax1.set_xscale("log")
ax1.set_xlabel("tau")
ax1.set_ylabel("final f(x)")
ax1.set_title(f"{DATASET}: objective vs tau (PFW)")
ax1.legend()

ax2.plot(taus, supports, "s-", color="tab:orange")
ax2.axvline(CURRENT_TAU, color="gray", linestyle=":", label=f"current tau={CURRENT_TAU}")
ax2.set_xscale("log")
ax2.set_xlabel("tau")
ax2.set_ylabel("final support size")
ax2.set_title(f"{DATASET}: sparsity vs tau (PFW)")
ax2.legend()

fig.tight_layout()
out_path = os.path.join(os.path.dirname(__file__), "..", "results", "figures", f"{DATASET}_tau_sweep.png")
fig.savefig(out_path, dpi=120)
plt.close(fig)
print(f"\nfigure saved -> {os.path.normpath(out_path)}")
