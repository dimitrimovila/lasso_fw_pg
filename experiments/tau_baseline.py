"""Per-dataset ||x_ols||_1 and conditioning, to check the tau choices."""

import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datasets import load_dataset

for name in ["diabetes", "riboflavin", "communities"]:
    data = load_dataset(name)
    A, b = data["A_train"], data["b_train"]
    m, p = A.shape

    # singular values of A (descending); length is min(m, p)
    s = np.linalg.svd(A, compute_uv=False)
    sigma_max = s[0]
    L = 2.0 * sigma_max ** 2           # Lipschitz const of grad f = 2 A^T A

    # sigma_min relevant to strong convexity mu = 2 sigma_min^2.
    # If m < p, there are p - m columns spanning a null space, so sigma_min = 0 exactly 
    # (f is convex but NOT strongly convex).
    if m >= p:
        sigma_min = s[-1]
        kappa = sigma_max / sigma_min
    else:
        sigma_min = 0.0
        kappa = np.inf
    mu = 2.0 * sigma_min ** 2

    # ||x_ols||_1: exact binding threshold when A has full column rank (m >= p);
    # only an upper bound when m < p, where the least-squares solution is not unique.
    ols = LinearRegression(fit_intercept=False).fit(A, b)
    l1_ols = np.sum(np.abs(ols.coef_))

    print(f"{name}: n={m}, p={p}")
    print(f"  ||x_ols||_1 = {l1_ols:.4f}")
    print(f"  sigma_max = {sigma_max:.4f}   sigma_min = {sigma_min:.4f}")
    print(f"  kappa(A) = {kappa:.2f}   mu/L = {mu / L:.3e}")