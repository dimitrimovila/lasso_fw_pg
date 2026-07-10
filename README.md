# LASSO: Frank-Wolfe, Pairwise Frank-Wolfe, and Projected Gradient

Course project for **Optimization for Data Science** (2025/2026), University of Padua.
Instructors: Francesco Rinaldi, Gennaro Auricchio.

Authors:  
Christina Caporale (2141881)  
Natalya Lavrenchuk, (2141882)  
Dumitru Movila (2144565)

Implements and compares three first-order algorithms: Frank-Wolfe (FW), Pairwise
Frank-Wolfe (PFW), and Projected Gradient (PG) on the ℓ1-constrained LASSO problem

```
min_x  ||Ax - b||_2^2   s.t.  ||x||_1 <= tau
```

across three datasets:  `diabetes` (p=10, well-conditioned), `communities`/`us_crime`
(p=101, ill-conditioned), and `riboflavin` (p=4088, rank-deficient, p >> n).

## Repo structure

```
src/
  lasso_problem.py       objective, gradient, Lipschitz constant, LMO, l1-ball projection
  line_search.py         diminishing and exact line-search step sizes
  stopping_criteria.py   FW gap, PG gradient mapping
  algorithms/
    frank_wolfe.py        classic FW
    pairwise_fw.py         PFW (active-set weight bookkeeping)
    projected_gradient.py  PG (fixed step s=1/L, full projection step alpha=1)

experiments/
  datasets.py             loads/splits/standardizes all three datasets
  tau_baseline.py         ||x_ols||_1 and conditioning (sigma_max, sigma_min, kappa) per dataset
  tau_sweep.py            tau sweep on riboflavin (PFW) to pick tau, train/test objective + sparsity
  run_experiments.py      main comparison: runs FW/PFW/PG on all 3 datasets, saves plots + tables
  feature_selection.py    compares final feature supports across algorithms (Jaccard overlap, bar charts)

results/
  figures/                convergence plots, tau-sweep plots, feature-selection bar charts (.png)
  tables/                 per-dataset results tables (.csv)

3 FW Lasso/               theory reference PDFs used for algorithm derivations and citations
```

## Setup

```
pip install -r requirements.txt
```

## Reproducing the results

Run from the repo root (each script resolves `src/` relative to its own file location):

```
python experiments/tau_baseline.py       # prints ||x_ols||_1, kappa(A) per dataset
python experiments/tau_sweep.py          # -> results/figures/riboflavin_tau_sweep.png
python experiments/run_experiments.py    # -> results/figures/*.png, results/tables/*.csv
python experiments/feature_selection.py  # -> results/figures/*_feature_selection.png
```

`tau` values used (fixed constants in `run_experiments.py`, justified by `tau_baseline.py`
and `tau_sweep.py`): `diabetes=1.5`, `riboflavin=1.0`, `communities=2.5`.

## Notes on the comparison

- All three algorithms are initialized from the same feasible vertex `x0 = tau * e_1`.
- `run_experiments.py` stops each algorithm at `epsilon=1e-4` (FW/PFW: FW gap; PG: norm
  of the gradient mapping). PG's
  criterion carries a `1/L` deflation, so iteration counts for PG are not directly
  comparable to FW/PFW's. See the report for the full discussion.
- `feature_selection.py` instead runs all algorithms to a shared large iteration budget
  (`epsilon=-1`, `max_iter=20000`) so that final supports are compared at a similar
  distance from the optimum rather than at each algorithm's own stopping
  point.

## References

Theory PDFs in `3 FW Lasso/` (Rinaldi and Auricchio lecture notes, Bomze/Rinaldi/Zeffiro survey,
Lacoste-Julien & Jaggi, Combettes & Pokutta) are the primary sources cited throughout
the code and report.
