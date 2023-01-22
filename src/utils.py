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


def RNN_update_jac(x, W):
    return -np.eye(x.shape[0]) + W @ np.diag(1 - np.tanh(x) ** 2)

def latent_RNN_update(y, W, Q):
    return -y + Q @ W @ np.tanh(Q.T @ y)


def V(M: Matrix, Q:Matrix, x: Matrix) -> float:
    y = Q @ x
    return np.diag(y.T @ (Q @ M @ Q.T) @ y)

def V_dot(M: Matrix, W:Matrix, Q:Matrix, x: Matrix) -> float:

    y = Q @ x
    f_x = RNN_update(x, W)
    f_y = Q @ f_x

    

    return 2*np.diag(y.T @ (Q @ M @ Q.T) @ f_y)







