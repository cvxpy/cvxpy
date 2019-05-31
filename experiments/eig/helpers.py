from typing import Tuple

import numpy as np
from scipy import sparse
import cvxpy as cvx


def generate_radiating_system_1d(n_domain_points: int, k: float) -> Tuple[sparse.csc_matrix, np.array]:
    """
    Generates a 1D system, with an incoming plane wave with wavenumber k from the LHS of the domain.
    :param n_domain_points: number of domain points.
    :param k: wavenumber of incoming wave.
    :return: tuple containing A, b for the system Ax = b
    """
    h = 1/n_domain_points
    h2 = 1/n_domain_points**2

    full_laplacian = sparse.diags(
        [np.ones(n_domain_points - 1)/h2,
         -2*np.ones(n_domain_points)/h2,
         np.ones(n_domain_points - 1)/h2],
        offsets=[-1, 0, 1], dtype='complex128'
    ).tocsc()

    # boundary conditions
    full_laplacian[0, 0] = -(1j*k + 1/h)/h
    full_laplacian[0, 1] = 1/h2

    full_laplacian[-1, -2] = 1/h2
    full_laplacian[-1, -1] = -(1/h + 1j*k)/h

    b = np.zeros(n_domain_points, dtype='complex128')
    b[0] = 1j*k/h

    return full_laplacian, b


def to_bar(complex_vec):
    """Stacks the real and imaginary parts
    """
    return np.hstack([np.real(complex_vec), np.imag(complex_vec)])


def to_bar_mat(A):
    return sparse.bmat([
        [np.real(A), -np.imag(A)],
        [np.imag(A), np.real(A)]
    ])


def generate_dirichlet_system_1d(n_domain_points: int, k: float) -> Tuple[sparse.csc_matrix, np.array]:
    """
    Generates a 1D system, with an incoming plane wave with wavenumber k from the LHS of the domain.
    :param n_domain_points: number of domain points.
    :param k: wavenumber of incoming wave.
    :return: tuple containing A, b for the system Ax = b
    """
    h2 = 1/n_domain_points**2

    full_laplacian = sparse.diags(
        [np.ones(n_domain_points - 1)/h2,
         -2*np.ones(n_domain_points)/h2,
         np.ones(n_domain_points - 1)/h2],
        offsets=[-1, 0, 1]
    ).tocsc()

    b = np.zeros(n_domain_points, dtype='complex128')

    return full_laplacian, b


def abs2(cons, expr):
    t = cvx.Variable()
    cons.append(cvx.SOC(t + 1, cvx.hstack([2*cvx.imag(expr), 2*cvx.real(expr), t - 1])))
    return t
