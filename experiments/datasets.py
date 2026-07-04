"""
Dataset loading and preprocessing for the LASSO study.

Each dataset provides a design matrix A and response vector b that feed into the
L1-constrained LASSO problem
   min_x ||A x - b||_2^2   s.t.   ||x||_1 <= tau

Three real-world datasets are used to cover different regimes so the FW/PFW
(sparsity-promoting, cheap LMO) vs. PG (cheap on small n, expensive projection at
scale) tradeoff can be studied in a diverse set of scenarios. The datasets are:

  1. "diabetes"     : low-dimensional dense baseline        (n=442,  p=10)
  2. "riboflavin"   : biotech/genomics, p >> n regime       (n=71,  p=4088)
  3. "communities"  : medium-dimensional, correlated features (n=1994, p=101)

Preprocessing (identical for all datasets):
  - Train/test split 80/20 with random_state=42.
  - Standardize FEATURES with StandardScaler fit on the TRAIN split only, applied to 
    both splits.
  - Standardize the TARGET b to zero mean / unit std using TRAIN statistics only.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Per-dataset raw loaders: each returns (A_raw, b_raw) as float64 arrays,
# before splitting/standardization.

def _load_diabetes():
    """Low-dimensional dense baseline (n=442, p=10)."""
    bunch = load_diabetes()
    A = np.asarray(bunch.data, dtype=float)
    b = np.asarray(bunch.target, dtype=float)
    return A, b


def _load_riboflavin():
    """
    Genomics p >> n instance (n=71, p=4088).  Target = log riboflavin production rate; 
    all 4088 gene-expression features are numeric with no missing values.
    """
    bunch = fetch_openml("riboflavin", version=1, as_frame=False)
    A = np.asarray(bunch.data, dtype=float)
    b = np.asarray(bunch.target, dtype=float)
    return A, b


def _load_communities():
    """
    Communities-and-Crime (medium-dim, correlated features).

    Cleaning steps: 
      - drop any non-numeric column,
      - drop any numeric column containing at least one missing value.
    The actual cleaned (n, p) is printed so it can be logged.
    """
    bunch = fetch_openml("us_crime", version=1, as_frame=True)
    X = bunch.data.copy()          # features as a DataFrame
    y = bunch.target               # ViolentCrimesPerPop (Series)

    # Coerce every column to numeric; a value that was present but fails to parse
    # marks a non-numeric column, which is dropped entirely.
    X_num = X.apply(pd.to_numeric, errors="coerce")
    became_nan = X_num.isna() & X.notna()
    nonnumeric_cols = [c for c in X.columns if bool(became_nan[c].any())]
    X_num = X_num.drop(columns=nonnumeric_cols)

    # Among the numeric columns, drop any that still contain a missing value.
    missing_cols = [c for c in X_num.columns if bool(X_num[c].isna().any())]
    X_clean = X_num.drop(columns=missing_cols)

    A = X_clean.to_numpy(dtype=float)
    b = np.asarray(pd.to_numeric(y, errors="coerce"), dtype=float)

    print(
        f"[communities/us_crime] cleaned (n, p) = {A.shape}  "
        f"(dropped {len(nonnumeric_cols)} non-numeric + {len(missing_cols)} "
        f"missing-value columns)"
    )
    return A, b


_LOADERS = {
    "diabetes": _load_diabetes,
    "riboflavin": _load_riboflavin,
    "communities": _load_communities,
}


# Shared split + standardization.

def _split_and_standardize(A_raw, b_raw, test_size=0.2, random_state=42):
    """
    80/20 split, then standardize features and target using TRAIN statistics only.

    StandardScaler is fit on the training split exclusively and applied to both
    splits (no information from the test split ever influences the scaling). 
    """
    A_tr, A_te, b_tr, b_te = train_test_split(
        A_raw, b_raw, test_size=test_size, random_state=random_state
    )

    x_scaler = StandardScaler().fit(A_tr)
    A_tr = x_scaler.transform(A_tr)
    A_te = x_scaler.transform(A_te)

    # Target standardized with the same fit-on-train-only principle.
    y_scaler = StandardScaler().fit(b_tr.reshape(-1, 1))
    b_tr = y_scaler.transform(b_tr.reshape(-1, 1)).ravel()
    b_te = y_scaler.transform(b_te.reshape(-1, 1)).ravel()

    return A_tr, b_tr, A_te, b_te


def load_dataset(name):
    """
    Load and preprocess one of the three study datasets.

    Parameters
    ----------
    name : {"diabetes", "riboflavin", "communities"}
        Dataset key.

    Returns
    -------
    dict with keys:
        "A_train" : ndarray (n_train, p)   standardized training design matrix.
        "b_train" : ndarray (n_train,)     standardized training response.
        "A_test"  : ndarray (n_test, p)    standardized test design matrix.
        "b_test"  : ndarray (n_test,)      standardized test response.
        "n"       : int   full cleaned dataset sample count (before split).
        "p"       : int   number of features.
        "name"    : str   the dataset key.
    """
    key = name.lower()
    if key not in _LOADERS:
        raise ValueError(
            f"unknown dataset {name!r}; expected one of {sorted(_LOADERS)}"
        )

    A_raw, b_raw = _LOADERS[key]()
    n, p = A_raw.shape        # full cleaned dataset dimensions (samples, features)

    A_train, b_train, A_test, b_test = _split_and_standardize(A_raw, b_raw)

    return {
        "A_train": A_train,
        "b_train": b_train,
        "A_test": A_test,
        "b_test": b_test,
        "n": int(n),
        "p": int(p),
        "name": key,
    }
