import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy import linalg
from sklearn.preprocessing import normalize
from scipy.optimize import minimize, rosen, rosen_der
from typing import Dict, List, Optional, Set, Tuple, Union

Matrix = List[float]
Vector = List[float]


def create_random_weight_matrix(n: int, scale: float) -> Matrix:
    """Generates n x n matrix with elements drawn from gaussian of mean = 0 and sigma = scale/sqrt(n)

    Args:
        n (int): size of matrix W
        scale (float): scale of the gaussian

    Returns:
        W (Matrix): Random weight matrix
    """

    W = np.random.normal(0, scale / np.sqrt(n), size=(n, n))

    return W


def get_real_eigs(A: Matrix) -> List:
    """Computes eigenvalues of A and returns their real part.

    Args:
        A (Matrix): A square matrix.

    Returns:
        List: List containing the real part of the eigenvalues of A.
    """
    e, ev = np.linalg.eig(A)

    return np.real(e)


def get_local_lyapunov_metric(J: Matrix) -> Matrix:
    """Computes local metric, given Jacobian around fixed point.
    Can be used in constructing a quadratic Lyapunov function x.T @ M @ x

    Args:
        J (Matrix): Jacobian of system evaluated at fixed point.

    Returns:
        M (Matrix): Solution to Lyapunov equation M @ J + J.T @ M = -I
    """

    n = J.shape[0]

    # scipy sneakily transposes their A matrix
    M = linalg.solve_continuous_lyapunov(a=J.T, q=-np.eye(n))
    M_eigs, _ = np.linalg.eigh(M)

    assert all(
        M_eigs > 0
    ), "Lyapunov metric is not positive definite--fixed point is probably not stable."

    return M


def check_linear_stability(A: Matrix) -> str:
    """Checks linear stability (i.e., eigenvalue position) of square matrix A.

    Args:
        A (Matrix): Square matrix.

    Returns:
        str: Description of linear stability of A.
    """

    real_es = get_real_eigs(A)
    if any(real_es > 0):
        return "Matrix is linearly unstable."
    if any(real_es == 0):
        return "Matrix is marginally linearly stable."
    if all(real_es < 0):
        return "Matrix is linearly stable."


def RNN_update(x, W):
    return -x + W @ np.tanh(x)


def RNN_update_norm(x, W):
    return np.linalg.norm(RNN_update(x, W))


def RNN_update_jac(x, W):
    return -np.eye(x.shape[0]) + W @ np.diag(1 - np.tanh(x) ** 2)


def latent_RNN_update(y, W, Q):
    return -y + Q @ W @ np.tanh(Q.T @ y)


def V(M: Matrix, Q: Matrix, x: Matrix) -> float:
    y = Q @ x
    return np.diag(y.T @ (Q @ M @ Q.T) @ y)


def V_y(M: Matrix, Q: Matrix, Y, num_points) -> float:
    "y is the lower dimensional orthogonal projection."

    A = Q @ M @ Q.T
    f_x = np.dot(np.dot(Y.T, A), Y)
    f_x = np.diag(f_x).reshape(num_points, num_points)

    return f_x


def V_dot(M: Matrix, W: Matrix, Q: Matrix, x: Matrix) -> float:
    y = Q @ x
    f_x = RNN_update(x, W)
    f_y = Q @ f_x

    return 2 * np.diag(y.T @ (Q @ M @ Q.T) @ f_y)


def energy_barrier_on_plane(
    x1: Vector, x2: Vector, x3: Vector, lam_range: float = 100, num_lams: float = 500
) -> Matrix:
    # Gram–Schmidt process
    u = x2 - x1
    v = x3 - x1 - np.dot(x3 - x1, x2 - x1) * (x2 - x1) / np.linalg.norm(x2 - x1) ** 2

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    assert np.dot(u, v) <= 1e-15, print("Gram-Schmidt failed, vectors not orthonormal")

    # define grid of points to evaluate
    lams = np.linspace(-lam_range, lam_range, num_lams)

    # energies over the grid
    Es = np.zeros((num_lams, num_lams))

    for k_x, lam_x in enumerate(lams):
        for k_y, lam_y in enumerate(lams):
            # point in the plane containing the input vectors
            p = x1 + lam_x * u + lam_y * v

            # calculate ||f||^2 for each point
            E = RNN_update_norm(p, W)

            # store energy value
            Es[k_x, k_y] = E

    return Es, u, v
