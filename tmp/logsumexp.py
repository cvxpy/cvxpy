import cvxpy as cp
import numpy as np


X = cp.Variable((2, 2))
array = np.array([[-5., -3.],
                  [ 2.,  1.]])
objective = cp.sum(X, axis=0)[1]
constraints = [X == array]
problem = cp.Problem(cp.Minimize(objective), constraints)
print("Problem value: ", problem.solve(cp.SCS))
print("Objective value: ", problem.objective.value)
print(X.value)
