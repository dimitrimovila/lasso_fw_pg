"""
Stopping / convergence criteria.

  - fw_gap           : Frank-Wolfe / duality gap 
  - gradient_mapping : Projected-Gradient mapping g_C(x)
"""

from lasso_problem import lmo, project_l1


def fw_gap(grad, x, tau, s=None):
    """
    Frank-Wolfe gap (duality gap) at the point x.

    G(x) = max_{s in C} -grad^T (s - x)

    Properties used as a stopping criterion:
      - G(x) >= 0, with equality only at a stationary point
      - For convex f it certifies optimality:
            G(x) >= f(x) - f*,
        so G(x) <= epsilon  ==>  x is optimal

    Parameters
    ----------
    grad : ndarray, shape (n,)
        Gradient of f at x.
    x : ndarray, shape (n,)
        Current (feasible) iterate.
    tau : float
        L1-ball radius defining C.
    s : ndarray, shape (n,), optional
        The LMO vertex argmin_{s in C} <grad, s>, 
        if already computed by the caller (FW computes it for the search direction).  
        Passing it avoids a redundant LMO evaluation.

    Returns
    -------
    float
        The Frank-Wolfe gap G(x) >= 0.
    """
    if s is None:
        s = lmo(grad, tau)
    return float(grad @ (x - s))


def gradient_mapping(x, grad, tau, L):
    """
    Projected-gradient mapping g_C(x) at x, the stopping quantity for PG.

    Using fixed s = 1/L:
        x_hat = rho_C(x - (1/L) * grad f(x)),
        g_C(x) = x - x_hat

    Parameters
    ----------
    x : ndarray, shape (n,)
        Current (feasible) iterate.
    grad : ndarray, shape (n,)
        Gradient of f at x.
    tau : float
        L1-ball radius defining C.
    L : float
        Lipschitz constant of grad f (sets the step s = 1/L).

    Returns
    -------
    x_hat : ndarray, shape (n,)
        The projected point rho_C(x - (1/L) grad).
    g_C : ndarray, shape (n,)
        The gradient mapping x - x_hat.  Its norm ||g_C|| is the stopping quantity.
    """
    x_hat = project_l1(x - grad / L, tau)
    return x_hat, x - x_hat
