"""
LASSO problem: objective, gradient, Lipschitz constant, LMO, and L1-ball projection.

Problem formulation:

    min_{x in R^n}  ||Ax - b||_2^2   s.t.   ||x||_1 <= tau

Feasible set: C = conv{ ±tau * e_i : i = 1, ..., n }  (scaled L1-ball).
"""

import numpy as np


def f(x, A, b):
    """
    Evaluate the LASSO objective function.

    f(x) = ||Ax - b||_2^2

    Parameters
    ----------
    x : ndarray, shape (n,)
        Current iterate (primal variable).
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.

    Returns
    -------
    float
        Objective value ||Ax - b||_2^2.
    """
    r = A @ x - b
    return float(r @ r)


def grad_f(x, A, b):
    """
    Gradient of the LASSO objective.

    grad_f(x) = 2 * A^T (Ax - b)

    Derivation: standard calculus applied to f(x) = ||Ax - b||_2^2.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Current iterate.
    A : ndarray, shape (m, n)
        Design matrix.
    b : ndarray, shape (m,)
        Response vector.

    Returns
    -------
    ndarray, shape (n,)
        Gradient vector 2 * A^T (Ax - b).
    """

    return 2.0 * (A.T @ (A @ x - b))


def compute_L(A):
    """
    Compute the Lipschitz constant of grad_f.

    L = 2 * sigma_max(A)^2

    Derivation: the Hessian of f(x) = ||Ax - b||_2^2 is H = 2 A^T A, whose
    spectral norm (largest eigenvalue) equals 2 * sigma_max(A)^2.  
    From the definition of an L-smooth gradient: 
    "f differentiable with L-Lipschitz gradient: ||∇f(x) - ∇f(y)|| ≤ L||x-y||" 
    L is the spectral norm of H.

    Used by:
    - PG fixed step-size s = 1/L 
    - Lipschitz-dependent FW step-size 

    Parameters
    ----------
    A : ndarray, shape (m, n)
        Design matrix.

    Returns
    -------
    float
        Lipschitz constant L = 2 * sigma_max(A)^2.
    """
    # sigma_max(A) is the largest singular value; numpy.linalg.svd with
    # compute_uv=False returns singular values in descending order.
    sigma_max = np.linalg.svd(A, compute_uv=False)[0]
    return 2.0 * sigma_max ** 2


def lmo(grad, tau):
    """
    Linear Minimization Oracle (LMO) over the scaled L1-ball C = {x : ||x||_1 <= tau}.

    Solves s* = argmin_{s in C} <grad, s> in closed form:

        i* = argmax_i |grad_i|
        s* = -sign(grad[i*]) * tau * e_{i*}

    This yields an extreme point (vertex) of the scaled L1-ball.

    LMO = sign(-∇_{i_k} f) * tau * e_{i_k},  i_k in argmax_i |∇_i f|
    closed-form LMO for the L1-ball: argmin_{||x||_1 ≤ 1} <x,y> = -sign([y]_{i_max}) * e_{i_max}
    Cost: O(n), single argmax over n coordinates, fully vectorized 

    Edge case: if grad is identically zero (already at the unconstrained minimum),
    np.sign(0) = 0 so the returned point is the origin (interior of C).  In that
    situation the FW gap is also zero and the algorithm would have already stopped.

    Parameters
    ----------
    grad : ndarray, shape (n,)
        Gradient of f at the current iterate.
    tau : float
        L1-ball radius (tau > 0).

    Returns
    -------
    s : ndarray, shape (n,)
        LMO solution: an extreme point of the scaled L1-ball.
    """
    # i* = argmax_i |grad_i|   (O(n) argmax, no coordinate loop)
    i_star = np.argmax(np.abs(grad))

    # s* = -sign(grad[i*]) * tau * e_{i*}
    s = np.zeros_like(grad, dtype=float)
    s[i_star] = -np.sign(grad[i_star]) * tau
    return s


def project_l1(x, tau):
    """
    Euclidean projection onto the scaled L1-ball {z : ||z||_1 <= tau}.

    Algorithm: sort-based reduction to simplex projection.
  
    # The algorithm implemented here is the standard O(n log n) sort-based threshold
    # method (descending sort + cumulative-sum threshold search), which is the
    # classical precursor cited alongside Condat.  Condat's strict O(n) variant
    # uses a pivot-selection step instead of a full sort; that refinement is not
    # needed for correctness, only for asymptotic speed at very large n.

    Parameters
    ----------
    x : ndarray, shape (n,)
        Point to project onto the scaled L1-ball.
    tau : float
        L1-ball radius (tau > 0).

    Returns
    -------
    p : ndarray, shape (n,)
        Euclidean projection of x onto {z : ||z||_1 <= tau}.
    """

    # If x is already feasible, projection is the identity.
    if np.sum(np.abs(x)) <= tau:
        return x.copy()

    # Work with absolute values (exploit sign symmetry of the L1-ball).
    u = np.abs(x)

    # Sort in descending order to find the soft-threshold value.
    u_sorted = np.sort(u)[::-1]                    # O(n log n)
    cumsum   = np.cumsum(u_sorted)                 # O(n)

    # Find rho = largest index j (0-based) such that
    #   u_sorted[j] > (cumsum[j] - tau) / (j + 1)
    # This is the number of non-zero components in the projected simplex point.
    j_arr = np.arange(1, len(u) + 1, dtype=float)  # j+1 in 1-based notation
    rho = int(np.max(np.where(u_sorted > (cumsum - tau) / j_arr)[0]))

    # Soft-threshold value theta.
    theta = (cumsum[rho] - tau) / (rho + 1)

    # Apply soft-thresholding and restore original signs.
    return np.sign(x) * np.maximum(u - theta, 0.0)
