"""
Pairwise Frank-Wolfe (PFW) implementation for the LASSO problem.

    min_{x in R^n}  ||Ax - b||_2^2   s.t.   ||x||_1 <= tau

Uses direction vector that removes weight from a "bad" away vertex (v_t) to a 
"good" FW vertex (s_t) with a gamma_max no more than the weight already in v_t.

    d_t^PFW = s_t - v_t ,        gamma_max = alpha_{v_t}.

Atoms of the scaled L1-ball are exactly the signed vertices  +/- tau * e_i,
so each atom is stored as a signed-index key (i, sign) meaning  sign * tau * e_i.

The active set S^(t) is held as a dict  {(i, sign): alpha_v}  with the invariant
    sum_v alpha_v = 1,   alpha_v >= 0,
and the iterate is reconstructed from the weights at every step,
    x_t = sum_{v in S^(t)} alpha_v * v,
rather than carried as a separately-drifting incremental variable 
(x_t+1 = x_t + gamma*d_t) that could compound floating-point errors in each iteration.

LMO, objective, and gradient come from src/lasso_problem.py (not reimplemented).
"""

import time

import numpy as np

from lasso_problem import f, grad_f, lmo
from line_search import exact_line_search
from stopping_criteria import fw_gap


# Atoms below drop_tol weight are removed from the active set (handles the
# exact-arithmetic "alpha_{v_t} -> 0" of a swap/drop step under floating point 
# conditions).
_DROP_TOL = 1e-12


def _atom_vector(key, tau, n):
    """ Constructs the dense vector v= sign * tau * e_i from the sparse active-set 
    atom key (i, sign)."""
    i, sign = key
    v = np.zeros(n, dtype=float)
    v[i] = sign * tau
    return v


def _reconstruct_x(weights, tau, n):
    """ Reconstructs x_t based on convex combination of weights in the active set:
    x = sum_{(i,sign) in S} alpha * sign * tau * e_i (source of truth for x_t)."""
    x = np.zeros(n, dtype=float)
    for (i, sign), alpha in weights.items():
        x[i] += alpha * sign * tau
    return x


def _lmo_key(s_vec):
    """ Recover the atom key (i, sign) from the (single-nonzero) LMO vertex vector."""
    i = int(np.argmax(np.abs(s_vec)))   # abs ensures that neg coord is selected over 0s
    return (i, int(np.sign(s_vec[i])))


def pairwise_fw(A, b, tau, x0=None, epsilon=1e-6, max_iter=10_000):
    """
    Pairwise Frank-Wolfe for L1-constrained LASSO.

    Core update rule
    ----------------
    1. Frank-Wolfe Atom (LMO):
       s_t := argmin_{s in C} <grad f(x_t), s> 
    2. Away Atom (Active Set):
       v_t := argmax_{v in S^(t)} <grad f(x_t), v> 
    3. Pairwise Direction & Step Bounds:
       d_t := s_t - v_t
       gamma_max := alpha_{v_t}
    4. Exact Line Search:
       gamma_t := argmin_{gamma in [0, gamma_max]} f(x_t + gamma * d_t)
    5. Pairwise Mass Transfer (Weight Update):
       alpha_{v_t} := alpha_{v_t} - gamma_t
       alpha_{s_t} := alpha_{s_t} + gamma_t
       (Swap step: If gamma_t == gamma_max, the away atom v_t is fully emptied 
       and dropped from the active set S^(t)).
    6. Stopping Criterion:
       The algorithm evaluates the standard FW gap: g_t := <grad f(x_t), x_t - s_t>.
       Terminate if g_t <= epsilon.

    Note on the Starting Point
    --------------------------
    PFW's active-set formulation requires the initial iterate x_0 to be a strictly 
    feasible vertex (an atom) of the L1-ball. Unlike classic FW, which initializes 
    at the interior origin, the default starting point here is the vertex 
    s_0 = LMO(grad f(0)). This guarantees the active set S^(0) is properly initialized 
    with a valid weight of 1.0.

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.
    tau : float
        L1-ball radius (tau > 0).
    x0 : ndarray, shape (n,), optional
        Optional starting vertex (must be a single +/- tau * e_i atom).  If None,
        starts at s_0 = LMO(grad f(0)).
    epsilon : float, optional
        FW-gap stopping tolerance.
    max_iter : int, optional
        Iteration budget.

    Returns
    -------
    result : dict
        Lists are aligned by iteration index t, evaluated at x_t:
          - "x"                      : ndarray, final iterate (reconstructed from weights).
          - "f_history"              : list[float], f(x_t).
          - "gap_history"            : list[float], FW gap g_t^FW = <grad f(x_t), x_t - s_t>.
          - "active_set_size_history": list[int], |S^(t)|.
          - "support_history"        : list[int], ||x_t||_0  (for parity with FW).
          - "time_history"           : list[float], wall-clock seconds for iteration t.
          - "weight_sum_history"     : list[float], sum_v alpha_v  (should be 1).
          - "min_weight_history"     : list[float], min_v alpha_v  (should be >= 0).
          - "active_set"             : dict, final {(i, sign): alpha_v}.
          - "n_iter", "converged", "step_size".
    """
    n = A.shape[1]  # number of features

    # --- Initial active set S^(0) = {x_0}, alpha_{x_0} = 1 (vertex start). ---
    if x0 is None:
        s0_vec = lmo(grad_f(np.zeros(n), A, b), tau)
        start_key = _lmo_key(s0_vec)
    else:
        x0 = np.asarray(x0, dtype=float)
        nnz = np.count_nonzero(x0)
        # ensure valid x0; one non-zero value closely equal to tau (with floating error).
        if nnz != 1 or not np.isclose(np.sum(np.abs(x0)), tau):
            raise ValueError(
                "x0 must be a vertex of the scaled L1-ball (a single +/- tau * e_i)"
            )
        start_key = _lmo_key(x0)
    weights = {start_key: 1.0}
    x = _reconstruct_x(weights, tau, n)

    f_history = []
    gap_history = []
    active_set_size_history = []
    support_history = []
    time_history = []
    weight_sum_history = []
    min_weight_history = []
    converged = False

    for t in range(max_iter):
        t0 = time.perf_counter()

        # First-order info and FW atom (LMO over the whole L1-ball).
        g = grad_f(x, A, b)
        s_vec = lmo(g, tau)
        s_key = _lmo_key(s_vec)

        # FW gap (native stopping criterion, reuses s_vec with no extra LMO call).
        gap = fw_gap(g, x, tau, s=s_vec)

        # Quantities recorded at the current iterate x_t.
        fval = f(x, A, b)
        size = len(weights)
        nnz = int(np.count_nonzero(x))

        stop = gap <= epsilon
        if not stop:
            # Away atom v_t = argmax over the active set of <grad, v>.
            # For atom (i, sign):  <grad, sign*tau*e_i> = grad[i] * sign * tau.
            v_key = max(weights, key=lambda key: g[key[0]] * key[1] * tau)
            v_vec = _atom_vector(v_key, tau, n)

            # Pairwise direction and step interval.
            d = s_vec - v_vec
            gamma_max = weights[v_key]            # gamma_max := alpha_{v_t}

            # Exact line search on the same LASSO quadratic, clipped to [0, gamma_max].
            gamma = exact_line_search(x, d, A, b, alpha_max=gamma_max)

            # --- Pairwise weight update: move gamma mass from v_t to s_t ---
            weights[v_key] -= gamma
            if weights[v_key] <= _DROP_TOL:        # swap/drop: v_t emptied
                del weights[v_key]
            weights[s_key] = weights.get(s_key, 0.0) + gamma

            # Reconstruct x_t+1 from the weights (source of truth).
            x = _reconstruct_x(weights, tau, n)

            # Invariant checks: simplex weights + no drift.
            w_sum = float(sum(weights.values()))
            w_min = float(min(weights.values()))
            assert np.isclose(w_sum, 1.0, atol=1e-9), f"weights sum {w_sum} != 1"
            assert w_min >= -1e-12, f"negative weight {w_min}"
        else:
            w_sum = float(sum(weights.values()))
            w_min = float(min(weights.values()))

        dt = time.perf_counter() - t0

        f_history.append(fval)
        gap_history.append(gap)
        active_set_size_history.append(size)
        support_history.append(nnz)
        time_history.append(dt)
        weight_sum_history.append(w_sum)
        min_weight_history.append(w_min)

        if stop:
            converged = True
            break

    return {
        "x": x,
        "f_history": f_history,
        "gap_history": gap_history,
        "active_set_size_history": active_set_size_history,
        "support_history": support_history,
        "time_history": time_history,
        "weight_sum_history": weight_sum_history,
        "min_weight_history": min_weight_history,
        "active_set": dict(weights),
        "n_iter": len(f_history),
        "converged": converged,
        "step_size": "exact",
    }
