import numpy as np
import numpy.linalg as linalg
import cvxpy as cp
from scipy import sparse
from scipy.sparse import linalg as s_linalg
from scipy.linalg import norm

import cProfile
import time
import sys

import helpers

assert len(sys.argv) == 3

n_domain_points = int(sys.argv[1])
filename = sys.argv[2]

k = 20*np.pi

theta_min_all = 1
theta_max_all = 1.55

weights = np.zeros(n_domain_points)
weights[-1] = 1

weights2 = weights**2

z_hat = np.zeros(n_domain_points, dtype='complex128')

full_laplacian, b = helpers.generate_radiating_system_1d(n_domain_points, k)

boundary_indices = [0, n_domain_points-1]
interior_indices = [i for i in range(n_domain_points) if i not in boundary_indices]

A_boundary = full_laplacian[np.ix_(boundary_indices, boundary_indices)]
A_interior = full_laplacian[np.ix_(interior_indices, interior_indices)]  # This needs to be hermitian
B = full_laplacian[np.ix_(boundary_indices, interior_indices)]

reorganized_laplacian = sparse.bmat([
    [A_boundary, B],
    [B.H, A_interior]
])

A_boundary_hermitian = (A_boundary + A_boundary.H)/2
A_boundary_antihermitian = (A_boundary - A_boundary.H)/2

lambda_max_interior = s_linalg.eigsh(A_interior, k=1, which='LA')[0][0]

# Numerical problem with eigenvalues whenever N <= 2
if A_boundary_hermitian.shape[0] <= 2:
    lambda_min_hermitian_boundary = linalg.eigvalsh(A_boundary_hermitian.todense())[0]
    lambda_min_antihermitian_boundary = linalg.eigvalsh(1.j * A_boundary_antihermitian.todense())[0]
else:
    lambda_min_hermitian_boundary = s_linalg.eigsh(A_boundary_hermitian, k=1, which='SA')[0][0]
    lambda_min_antihermitian_boundary = s_linalg.eigsh(1.j * A_boundary_antihermitian, k=1, which='SA')[0][0]

# --- Begin the main program
print('Building system')
nu_boundary_real = cp.Variable(len(boundary_indices))
nu_boundary_imag = cp.Variable(len(boundary_indices))

nu_interior_real = cp.Variable(len(interior_indices))
nu_interior_imag = cp.Variable(len(interior_indices))

eta_real = cp.Variable(nonneg=True)
eta_imag = cp.Variable(nonneg=True)

# Convenience
nu_real = cp.hstack([
    nu_boundary_real,
    nu_interior_real
])

nu_imag = cp.hstack([
    nu_boundary_imag,
    nu_interior_imag
])

nu_bar = cp.hstack([
    nu_real,
    nu_imag
])

b_reordered = np.hstack([
    b[boundary_indices],
    b[interior_indices]
])

b_bar = np.hstack([
    np.real(b_reordered),
    np.imag(b_reordered),
])

b_bar_re = np.hstack([
    np.real(b[boundary_indices]),
    -np.real(b[interior_indices]),
    -np.imag(b[boundary_indices]),
    np.imag(b[interior_indices])
])
b_bar_im = np.hstack([
    -np.imag(b[boundary_indices]),
    -np.imag(b[interior_indices]),
    np.real(b[boundary_indices]),
    np.real(b[interior_indices])
])

z_tilde = np.hstack([
    np.real(z_hat[boundary_indices]),
    np.real(z_hat[interior_indices]),
    np.imag(z_hat[boundary_indices]),
    np.imag(z_hat[interior_indices])
])

expanded_laplacian = sparse.bmat([
    [np.real(reorganized_laplacian), -np.imag(reorganized_laplacian)],
    [np.imag(reorganized_laplacian), np.real(reorganized_laplacian)]
])

theta_min = theta_min_all*np.ones(n_domain_points)
theta_max = theta_max_all*np.ones(n_domain_points)
theta_max[0:len(boundary_indices)] = theta_min_all

theta_min_diag = sparse.diags([np.hstack([theta_min, theta_min])], [0])
theta_max_diag = sparse.diags([np.hstack([theta_max, theta_max])], [0])

# Generate sub-sums
dual_theta_min = (expanded_laplacian.T @ nu_bar + theta_min_diag @ nu_bar - np.hstack([weights2, weights2]) * z_tilde
                  - .5 * (eta_real * b_bar_re + eta_imag * b_bar_im))
dual_theta_max = (expanded_laplacian.T @ nu_bar + theta_max_diag @ nu_bar - np.hstack([weights2, weights2]) * z_tilde
                  - .5 * (eta_real * b_bar_re + eta_imag * b_bar_im))

P_re = np.hstack([
    lambda_min_hermitian_boundary * np.ones(len(boundary_indices)),
    - lambda_max_interior * np.ones(len(interior_indices)),
    lambda_min_hermitian_boundary * np.ones(len(boundary_indices)),
    - lambda_max_interior * np.ones(len(interior_indices))
])

P_im = np.hstack([
    lambda_min_antihermitian_boundary * np.ones(len(boundary_indices)),
    np.zeros(len(interior_indices)),
    lambda_min_antihermitian_boundary * np.ones(len(boundary_indices)),
    np.zeros(len(interior_indices))
])

dual_denominator_min = np.hstack([weights2, weights2]) + eta_real * (P_re -
        np.hstack([theta_min, theta_min])) + eta_imag * P_im
dual_denominator_max = np.hstack([weights2, weights2]) + eta_real * (P_re -
        np.hstack([theta_max, theta_max])) + eta_imag * P_im

all_terms_min = cp.hstack([
    cp.quad_over_lin(dual_theta_min[i], dual_denominator_min[i]) for i in range(2*n_domain_points)
])

all_terms_max = cp.hstack([
    cp.quad_over_lin(dual_theta_max[i], dual_denominator_max[i]) for i in range(2*n_domain_points)
])

objective = -.5 * cp.sum(
    cp.maximum(all_terms_min[0:n_domain_points] + all_terms_min[n_domain_points:],
                all_terms_max[0:n_domain_points] + all_terms_max[n_domain_points:])
) - nu_bar @ b_bar + .5 * norm(np.hstack([weights, weights]) * z_tilde)

print("Constructing problem.")
pr = cProfile.Profile()
pr.enable()
prob = cp.Problem(cp.Maximize(objective))
data = prob.get_problem_data(cp.SCS)
pr.disable()
pr.dump_stats(filename)
