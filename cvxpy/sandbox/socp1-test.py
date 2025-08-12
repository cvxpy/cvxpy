import cvxpy as cp
import numpy as np

# Define variables
x = cp.Variable(3)
y = cp.Variable()

# Define objective function
objective = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])

# Define constraints
constraints = [
    cp.norm(x, 2) <= y,
    x[0] + x[1] + 3*x[2] >= 1.0,
    y <= 5
]

# Create and solve the problem
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.IPOPT, nlp=True)

# Print results
print("Status:", problem.status)
print("Optimal value:", problem.value)
print("Optimal x:", x.value)
print("Optimal y:", y.value)
