import cvxpy as cp
import numpy as np
from time import time
from tqdm import tqdm

print("CVXPY version: ", cp.__version__)

# Problem data.
m = 300
n = 200
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
gamma = cp.Parameter(nonneg=True, name="gammix")
x = cp.Variable(n)
N = 100
solver = cp.OSQP  
# solver = cp.GUROBI


def solve_sequence(solver):

    print("Solver", solver)

    objective = cp.Minimize(cp.sum_squares(A @ x - b) + gamma * cp.norm(x, 1))
    constraints = [0 <= x, x <= 1]

    t_solve_full = 0
    t_start = time()
    for n in tqdm(range(N)):
        prob = cp.Problem(objective, constraints)
        gamma.value = 1
        prob.solve(solver=solver)
        t_solve_full += prob.solver_stats.solve_time/N
    t_full = time() - t_start


    t_solve_param = 0
    t_start = time()
    prob = cp.Problem(objective, constraints)
    for n in tqdm(range(N)):
        gamma.value = 1
        prob.solve(solver=solver)
        t_solve_param += prob.solver_stats.solve_time/N
    t_param = time() - t_start

    print("Time full: %.2e" % t_full)
    print("Avg solve time full: %2e" % t_solve_full)
    print("Time param: %.2e" % t_param)
    print("Avg solve time param: %2e" % t_solve_param)

solve_sequence(solver)
