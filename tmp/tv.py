import cProfile

import cvxpy as cp
import numpy as np


def benchmark(func, *func_args):
    vals = []
    print('Benchmarking ', func.__name__)
    func(*func_args)

N = 52
M = 52
Uorig = np.random.randn(N, M, 3)
rows, cols, colors = Uorig.shape
known = np.zeros((rows, cols, colors))
for i in range(rows):
    for j in range(cols):
        if np.random.random() > 0.7:
            for k in range(colors):
                known[i, j, k] = 1

def tv_inpainting():
    Ucorr = known*Uorig
    variables = []
    constraints = []
    for i in range(colors):
        U = cp.Variable(shape=(rows, cols))
        variables.append(U)
        constraints.append(cp.multiply(
            known[:, :, i], U) == cp.multiply(
            known[:, :, i], Ucorr[:, :, i]))
    problem = cp.Problem(cp.Minimize(cp.tv(*variables)), constraints)
    pf = cProfile.Profile()
    pf.enable()
    problem.get_problem_data(cp.SCS)
    pf.disable()
    pf.dump_stats('tv.prof')

benchmark(tv_inpainting)


