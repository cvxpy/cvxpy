import numpy as np

import cvxpy as cp


# Define e_i in R^n
def e_i(n, i):
    e_i_vec = np.zeros((n, 1))
    e_i_vec[i] = 1
    return e_i_vec

# This function computes a⊙b = 1/2(ab^T + ba^T)
def odot(a, b):
    return (a @ b.T + b @ a.T) / 2

# Parameter values
N = 2  # Iteration budget
mu = 1/10  # Strong convexity parameter
L = 1  # Smoothness parameter
R = 1  # Initial condition distance
dim_Z = N + 2  # dimension of the matrix Z

# Define the index set I_N^star = {star, 0, 1, ..., N}
# where we use -1 as the index for star
I_N_star = list(range(-1, N + 1))

# Data generator function
def data_generator_function(N, alpha, mu, L):
    dim_bold_x = N + 2
    dim_bold_g = N + 2
    dim_bold_f = N + 1
    N_pts = N + 2  # number of points
    
    bold_x_0 = e_i(dim_bold_x, 0)  # Note: Python is 0-indexed
    bold_x_star = np.zeros((dim_bold_x, 1))
    
    # Initialize bold_g and bold_f as dictionaries for offset indexing
    bold_g = {}
    bold_f = {}
    
    # Construct bold_g vectors
    bold_g[-1] = np.zeros((dim_bold_g, 1))  # bold_g_star
    for k in range(0, N + 1):
        bold_g[k] = e_i(dim_bold_g, k + 1)  # Adjusted for 0-indexing
    
    # Construct bold_f vectors
    bold_f[-1] = np.zeros((dim_bold_f, 1))  # bold_f_star
    for k in range(0, N + 1):
        bold_f[k] = e_i(dim_bold_f, k)  # Adjusted for 0-indexing
    
    # Define bold_x vectors
    bold_x = {}
    bold_x[-1] = bold_x_star
    bold_x[0] = bold_x_0
    
    # Construct x_1, ..., x_N
    for i in range(1, N + 1):
        bold_x_i = bold_x_0 - (1/L) * sum(alpha[i-1, j] * bold_g[j] for j in range(i))
        bold_x[i] = bold_x_i
    
    return bold_x, bold_g, bold_f

# Matrix functions
def A_mat(i, j, alpha, bold_g, bold_x):
    return odot(bold_g[j], bold_x[i] - bold_x[j])

def B_mat(i, j, alpha, bold_x):
    return odot(bold_x[i] - bold_x[j], bold_x[i] - bold_x[j])

def C_mat(i, j, bold_g):
    return odot(bold_g[i] - bold_g[j], bold_g[i] - bold_g[j])

def D_mat(i, j, bold_g):
    return odot(bold_g[i], bold_g[j])

def E_mat(i, j, alpha, bold_g, bold_x):
    return odot(bold_g[i] - bold_g[j], bold_x[i] - bold_x[j])

def a_vec(i, j, bold_f):
    return bold_f[j] - bold_f[i]

# Create index set for lambda
class i_j_idx:
    def __init__(self, i, j):
        self.i = i
        self.j = j
    
    def __eq__(self, other):
        return self.i == other.i and self.j == other.j
    
    def __hash__(self):
        return hash((self.i, self.j))

idx_set_lambda = []
for i in I_N_star:
    for j in I_N_star:
        if i != j:
            idx_set_lambda.append(i_j_idx(i, j))

# Declare variables
# Lambda variables
lambda_vars = {}
for idx in idx_set_lambda:
    lambda_vars[idx] = cp.Variable(nonneg=True)

# Nu variable
nu = cp.Variable(nonneg=True)

# Z variable (symmetric matrix)
Z = cp.Variable((dim_Z, dim_Z), symmetric=True)

# P variable (for Cholesky decomposition)
P = cp.Variable((dim_Z, dim_Z))

# Alpha variables (stepsize matrix)
alpha = {}
for i in range(1, N + 1):
    for j in range(i):
        alpha[i-1, j] = cp.Variable(nonneg=True)

# Theta variables (symmetric matrices)
Theta = {}
for idx in idx_set_lambda:
    Theta[idx] = cp.Variable((N + 2, N + 2), symmetric=True)

# Create data using the generator function
dim_bold_x = N + 2
bold_x_0 = e_i(dim_bold_x, 0)
bold_x_star = np.zeros((dim_bold_x, 1))
bold_x, bold_g, bold_f = data_generator_function(N, alpha, mu, L)

# Objective: Minimize nu * R^2
objective = cp.Minimize(nu * R**2)

# Constraints
constraints = []

# Constraint 1: sum of lambda * a_vec == 0
sum_expr = sum(lambda_vars[idx] * a_vec(idx.i, idx.j, bold_f) for idx in idx_set_lambda)
constraints.append(sum_expr == 0)

# Helper function to find index
def index_finder_Theta(i, j, idx_set_lambda):
    target = i_j_idx(i, j)
    for k, idx in enumerate(idx_set_lambda):
        if idx == target:
            return idx
    return None

# Constraint 2: Main constraint
main_constraint_expr = (
    -C_mat(N, -1, bold_g) +
    nu * odot(bold_x_0 - bold_x_star, bold_x_0 - bold_x_star) +
    sum(lambda_vars[idx] * (
        A_mat(idx.i, idx.j, alpha, bold_g, bold_x) +
        (1 / (2 * (1 - (mu / L)))) * (
            (1 / L) * C_mat(idx.i, idx.j, bold_g) +
            mu * Theta[idx] -
            (2 * (mu / L) * E_mat(idx.i, idx.j, alpha, bold_g, bold_x))
        )
    ) for idx in idx_set_lambda) - Z
)
constraints.append(main_constraint_expr == 0)

# P is lower triangular
for i in range(dim_Z):
    for j in range(dim_Z):
        if i < j:
            constraints.append(P[i, j] == 0)

# Diagonal components of P are non-negative
for i in range(dim_Z):
    constraints.append(P[i, i] >= 0)

# Constraint: Z = P @ P.T
constraints.append(Z == P @ P.T)

# Constraint: Theta[i,j] = B[i,j] for all i,j in idx_set_lambda
for idx in idx_set_lambda:
    constraints.append(
        Theta[idx] == odot(bold_x[idx.i] - bold_x[idx.j], bold_x[idx.i] - bold_x[idx.j])
    )

# Create and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)

# Store optimal values
lambda_opt = {idx: lambda_vars[idx].value for idx in idx_set_lambda}
nu_opt = nu.value
alpha_opt = {key: alpha[key].value for key in alpha}
Z_opt = Z.value

print(f"Optimal objective value: {problem.value}")
print(f"nu_opt: {nu_opt}")
print(f"Status: {problem.status}")
