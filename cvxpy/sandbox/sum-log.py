# write a log sum problem in cvxpy
import cvxpy as cp
import numpy as np
# Generate random data
np.random.seed(0)
m, n = 10, 50
A = np.random.randn(m, n)
b = np.random.randn(m)
# Define the variable
x = cp.Variable(n)
t = cp.Variable(m)
t.value = np.ones(m)
# set initial value for x
objective = cp.Minimize(-cp.sum(cp.log(t)))
problem = cp.Problem(objective, [t == A @ x - b])
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True)
print("Optimal value:", problem.value)
