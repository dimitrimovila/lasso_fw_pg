

import time

import numpy as np

from lasso_problem import f, grad_f, lmo
from line_search import diminishing, exact_line_search
from stopping_criteria import fw_gap


def frank_wolfe(A, b, tau, x0=None, step_size="diminishing",
                epsilon=1e-6, max_iter=10_000):
    
    """
    Classic Frank-Wolfe (FW) implementation for the LASSO problem.

    min_{x in R^n}  ||Ax - b||_2^2   s.t.   ||x||_1 <= tau

    LMO, objective, and gradient come from src/lasso_problem.py (not reimplemented).

    Core update rule
    ----------------
    1. Frank-Wolfe Atom (LMO):
       s_k := argmin_{s in C} <grad f(x_k), s>
    2. Frank-Wolfe Direction:
       d_k := s_k - x_k
    3. Primal Update (Convex Combination):
       Choose step size alpha_k in [0, 1].
       x_{k+1} := x_k + alpha_k * d_k
    4. Stopping Criterion:
       Evaluate the FW gap: g_k := <grad f(x_k), x_k - s_k>.
       Terminate if g_k <= epsilon (guaranteeing x_k is epsilon-optimal for convex f).

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.
    tau : float
        L1-ball radius (tau > 0).
    x0 : ndarray, shape (n,),
        Feasible starting point.  Defaults to the origin (||0||_1 = 0 <= tau),
        which is feasible.
    step_size : {"diminishing", "exact"}, optional
        Step-size rule:
          - "diminishing": alpha_k = 2/(k+2)        
          - "exact":       closed-form line search  
    epsilon : float, optional
        FW-gap stopping tolerance.
    max_iter : int, optional
        Maximum number of iterations (iteration budget).

    Returns
    -------
    result : dict
        Keys (all lists are aligned by iteration index k, evaluated at x_k):
          - "x"               : ndarray, final iterate.
          - "f_history"       : list[float], objective f(x_k).
          - "gap_history"     : list[float], FW gap g_k = <grad f(x_k), x_k - s_k>.
          - "support_history" : list[int], ||x_k||_0 (active-set / sparsity tracking.
          - "time_history"    : list[float], wall-clock seconds for iteration k.
          - "n_iter"          : int, number of iterations performed.
          - "converged"       : bool, whether the FW gap fell to <= epsilon.
          - "step_size"       : str, the step size rule used.
    """
    if step_size not in ("diminishing", "exact"):
        raise ValueError(
            f"unknown step_size {step_size!r}; expected 'diminishing' or 'exact'"
        )

    n = A.shape[1]
    if x0 is None:
        x = np.zeros(n, dtype=float)  # default starting point at origin (feasible)
    else:
        x = np.array(x0, dtype=float)
        if np.sum(np.abs(x)) > tau + 1e-12: # avoids ValueError incase x0 on tau border
            raise ValueError("x0 is infeasible: ||x0||_1 > tau")

    f_history = []
    gap_history = []
    support_history = []
    time_history = []
    converged = False

    for k in range(max_iter):
        t0 = time.perf_counter()

        # First-order information and the LMO vertex
        grad = grad_f(x, A, b)
        s = lmo(grad, tau)

        # FW gap (stopping criterion)
        gap = fw_gap(grad, x, tau, s=s)

        # Quantities recorded at the current iterate x_k.
        fval = f(x, A, b)
        nnz = int(np.count_nonzero(x))

        stop = gap <= epsilon
        if not stop:
            # FW direction and step-size, then update.
            d = s - x
            if step_size == "diminishing":
                alpha = diminishing(k)  # alpha_k = 2/(k+2)
            else:
                # closed form for the LASSO quadratic; alpha_max = 1
                alpha = exact_line_search(x, d, A, b, alpha_max=1.0)
            x = x + alpha * d   #update

        dt = time.perf_counter() - t0

        f_history.append(fval)
        gap_history.append(gap)
        support_history.append(nnz)
        time_history.append(dt)

        if stop:
            converged = True
            break

    return {
        "x": x,
        "f_history": f_history,
        "gap_history": gap_history,
        "support_history": support_history,
        "time_history": time_history,
        "n_iter": len(f_history),
        "converged": converged,
        "step_size": step_size,
    }
