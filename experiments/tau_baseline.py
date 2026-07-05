"""One-off diagnostic: ||x_ols||_1 per dataset, used to check the tau choices."""

import os
import sys

import numpy as np
from sklearn.linear_model import LinearRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from datasets import load_dataset

for name in ["diabetes", "riboflavin", "communities"]:
    data = load_dataset(name)
    ols = LinearRegression(fit_intercept=False).fit(data["A_train"], data["b_train"])
    print(f"{name}: ||x_ols||_1 = {np.sum(np.abs(ols.coef_)):.4f}")
