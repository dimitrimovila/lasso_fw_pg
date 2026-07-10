"""
Projected Gradient (PG) for the L1-constrained LASSO problem.

    min_{x in R^n}  ||Ax - b||_2^2   s.t.   ||x||_1 <= tau

The constant step s = 1/L is used inside the projection while alpha = 1.

The projection rho_C = project_l1 and the Lipschitz constant L = compute_L(A)
are imported from src/lasso_problem.py and reused.  
The gradient mapping g_C(x) is imported from src/stopping_criteria.py.
"""

import time

import numpy as np

from lasso_problem import f, grad_f, compute_L
from stopping_criteria import gradient_mapping


def projected_gradient(A, b, tau, x0=None, epsilon=1e-6, max_iter=10_000, L=None):
    """
    Projected Gradient for L1-constrained LASSO.

    Core update rule
    ----------------
    1. Gradient Step & Projection (Target Point):
       Take an unconstrained gradient step with step size s = 1/L, and project 
       the result back onto the feasible set C:
        x_hat_k := rho_C( x_k - (1/L) * grad f(x_k) )
       
    2. Gradient Mapping:
       Define the gradient mapping as the residual between the current iterate 
       and the newly projected target point:
        g_C(x_k) := x_k - x_hat_k
       
    3. Primal Update:
       Take a full step to the projected target point (alpha = 1):
        x_{k+1} := x_hat_k
       
    4. Stopping Criterion:
       A point x* is optimal if and only if it is a fixed point of the projection 
       operator: x* = rho_C(x* - s * grad f(x*)) <=> g_C(x*) = 0. 
       Terminate if the norm of the gradient mapping satisfies:
        ||g_C(x_k)|| <= epsilon.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.
    tau : float
        L1-ball radius (tau > 0).
    x0 : ndarray, shape (n,), optional
        Feasible starting point.  Defaults to the origin (||0||_1 = 0 <= tau),
        matching classic FW's start for a fair comparison.
    epsilon : float, optional
        Stopping tolerance on the gradient-mapping norm ||g_C(x_k)|| 
    max_iter : int, optional
        Iteration budget.
    L : float, optional
        Lipschitz constant of grad f. If None, computed once via compute_L(A). (Reused)

    Returns
    -------
    result : dict
        Lists are aligned by iteration index k, evaluated AT x_k:
          - "x"               : ndarray, final iterate.
          - "f_history"       : list[float], f(x_k).
          - "gmap_norm_history": list[float], ||g_C(x_k)||.
          - "support_history" : list[int], ||x_k||_0 (sparsity tracking; PG has no
                                active set, so no |S^(t)| is reported).
          - "time_history"    : list[float], wall-clock seconds for iteration k.
          - "L"               : float, the Lipschitz constant used.
          - "n_iter", "converged", "step_size".
    """
    n = A.shape[1]
    if L is None:
        L = compute_L(A) 

    if x0 is None:
        x = np.zeros(n, dtype=float) 
    else:
        x = np.array(x0, dtype=float)
        if np.sum(np.abs(x)) > tau + 1e-12:
            raise ValueError("x0 is infeasible: ||x0||_1 > tau")

    f_history = []
    gmap_norm_history = []
    support_history = []
    time_history = []
    converged = False

    for k in range(max_iter):
        t0 = time.perf_counter()

        grad = grad_f(x, A, b)

        # Gradient mapping: 
        x_hat, g_C = gradient_mapping(x, grad, tau, L)
        gmap_norm = float(np.linalg.norm(g_C))

        # Quantities recorded at the current iterate x_k
        fval = f(x, A, b)
        nnz = int(np.count_nonzero(x))

        stop = gmap_norm <= epsilon
        if not stop:
            x = x_hat 

        dt = time.perf_counter() - t0

        f_history.append(fval)
        gmap_norm_history.append(gmap_norm)
        support_history.append(nnz)
        time_history.append(dt)

        if stop:
            converged = True
            break

    return {
        "x": x,
        "f_history": f_history,
        "gmap_norm_history": gmap_norm_history,
        "support_history": support_history,
        "time_history": time_history,
        "L": L,
        "n_iter": len(f_history),
        "converged": converged,
        "step_size": "fixed_1/L for s and alpha = 1",
    }
