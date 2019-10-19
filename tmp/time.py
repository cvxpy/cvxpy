import cvxpy as cp
import numpy
import matplotlib.pyplot as plt
import time

# Problem data.
n = 15
m = 10
numpy.random.seed(1)
A = numpy.random.randn(n, m)
b = numpy.random.randn(n)
# gamma must be nonnegative due to DCP rules.
gamma = cp.Parameter(nonneg=True)

# Construct the problem.
x = cp.Variable(m)
error = cp.sum_squares(A*x - b)
obj = cp.Minimize(error + gamma*cp.norm(x, 1))
prob = cp.Problem(obj)
assert prob.is_dpp()

# Construct a trade-off curve of ||Ax-b||^2 vs. ||x||_1
gamma_vals = numpy.logspace(-4, 1)
times = []
new_problem_times = []
for val in gamma_vals:
    gamma.value = val
    start = time.time()
    prob.solve(cp.SCS)
    end = time.time()
    times.append(end - start)
    new_problem = cp.Problem(obj)
    start = time.time()
    new_problem.solve(cp.SCS)
    end = time.time()
    new_problem_times.append(end - start)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.figure(figsize=(6, 6))
plt.plot(gamma_vals, times, label='Re-solving a DPP problem')
plt.plot(gamma_vals, new_problem_times, label='Solving a new problem')
plt.xlabel(r'$\gamma$', fontsize=16)
plt.ylabel(r'time (s)', fontsize=16)
plt.legend()
plt.show()
