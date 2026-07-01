"""
Step-size (line-search) rules for Frank-Wolfe methods.

All rules are taken from FW_survey:
  - diminishing       : alpha_k = 2/(k+2)
  - exact_line_search : alpha_k = argmin_{alpha in [0, alpha_max]} f(x_k + alpha * d_k)

The exact rule is specialised in closed form to the LASSO quadratic objective f(x) = ||Ax - b||_2^2
"""
    
import numpy as np


def diminishing(k):
    """
    Diminishing step-size:
        alpha_k = 2 / (k + 2)

    Parameters
    ----------
    k : int
        Iteration index (k = 0, 1, 2, ...).

    Returns
    -------
    float
        Step-size 2/(k+2) in (0, 1].
    """
    return 2.0 / (k + 2.0)


def exact_line_search(x, d, A, b, alpha_max=1.0):
    """
    Closed-form exact line search for the LASSO quadratic along direction d.

    Exact line search:
        alpha_k = min argmin_{alpha in [0, alpha_max]} phi(alpha),
                  phi(alpha) = f(x_k + alpha * d_k).

    Derivation (calculus on the quadratic):
        Let r = A x - b  and  w = A d.  Then f(x + alpha * d) = ||A(x + alpha * d) - b||^2 gives
            phi(alpha) = || r + alpha w ||_2^2
                       = ||r||^2 + 2 alpha (r^T w) + alpha^2 ||w||^2.
        phi is a convex parabola in alpha (coefficient ||w||^2 >= 0). 
        Setting phi'(alpha) = 2 (r^T w) + 2 alpha ||w||^2 = 0 yields the unconstrained minimiser
            alpha* = - (r^T w) / ||w||^2.
        Because phi is convex in 1-D, the constrained minimiser over [0, alpha_max]
        is simply alpha* restricted to that interval.
        Degenerate case ||w||^2 = 0 (d in the null-space of A): phi is constant along d,
        so the objective cannot be decreased; we return 0 (no move).

    Parameters
    ----------
    x : ndarray, shape (n,)
        Current iterate.
    d : ndarray, shape (n,)
        Search direction (for FW: d = s_k - x_k).
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.
    alpha_max : float, optional
        Upper bound on the step (FW uses alpha_max = 1).

    Returns
    -------
    float
        Optimal step-size in [0, alpha_max].
    """
    w = A @ d
    ww = float(w @ w)
    if ww <= 0.0:
        # d lies in the null-space of A: objective is flat along d.
        return 0.0
    r = A @ x - b
    alpha_star = -float(r @ w) / ww
    # Restrict the parabola's vertex to the feasible step interval [0, alpha_max].
    return float(min(max(alpha_star, 0.0), alpha_max))
