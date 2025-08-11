# write a unconstrained least squares problem with l1-2 norm
import cvxpy as cp
import numpy as np

# Generate random data
np.random.seed(0)
m, n = 500, 100
A = np.random.randn(m, n)
b = np.random.randn(m)
# Define the variable
x = cp.Variable(n)
# set initial value for x
#xls = np.array([ 0.13023767,  0.09473619,  0.20023978,  0.129647  , -0.26661262,
 #      -0.18313258,  0.29880278,  0.10479523, -0.14954388,  0.32831736])
xls = np.linalg.lstsq(A, b, rcond=None)[0]
x.value = xls
gamma = 0.1
print(np.linalg.norm(A @ xls - b)**2 + gamma * np.sum(np.sqrt(np.abs(xls))))
# Define the objective function with l1-2 norm
objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma * cp.sum(cp.sqrt(cp.abs(x))))
problem = cp.Problem(objective)
# Solve the problem
problem.solve(solver=cp.IPOPT, nlp=True)
print(x.value)
print("Optimal value:", problem.value)
print(np.linalg.norm(-2* A.T@b))
print(2* A.T@(A @ x.value - b))
