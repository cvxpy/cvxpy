import cvxpy as cp
import numpy as np

# Example setup: Define the size of the covariance matrix
n = 4  # Size of the covariance matrix
C = cp.Variable((n, n), symmetric=True)

# Original covariance matrix
C0 = np.array([[1.0, 0.2, 0.1, 0.0],
               [0.2, 1.0, 0.3, 0.1],
               [0.1, 0.3, 1.0, 0.4],
               [0.0, 0.1, 0.4, 1.0]])

# Distance tolerance
epsilon = 0.2

# Define the constraints for C
constraints = [
    C >> 0,  # C must be positive semi-definite
    cp.diag(C, 0) == np.ones(n),
    #cp.trace(C) == 1,  # Example constraint: Trace normalization
    cp.norm(C - C0, "fro") <= epsilon  # Frobenius norm constraint
]

# Objective: Maximize the smallest eigenvalue
objective = cp.Maximize(cp.lambda_min(C))

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

# Output the result
print("Value before (maximum smallest eigenvalue):", np.min(np.linalg.eigvals(C0)))
print("Optimal value (maximum smallest eigenvalue):", problem.value)
print("Optimal covariance matrix C:")
print(C.value)
