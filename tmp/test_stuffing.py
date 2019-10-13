import cProfile
import time

import cvxpy as cp

import numpy as np

def benchmark(func, *func_args):
    vals = []
    print('Benchmarking ', func.__name__)
    for _ in range(1):
#        start = time.time()
        func(*func_args)
#        vals.append(time.time() - start)
#    fastest_time = np.min(vals)
#    print("{:s}: avg={:.3e} s , std={:.3e} s best={:.3e} s "
#          "({:d} iterations)".format(
#          func.__name__, np.mean(vals), np.std(vals), fastest_time, 1))

m = 2000
n = 2000
A = np.random.randn(m, n)
C = np.random.rand(m // 2)
b = np.random.randn(m)

x = cp.Variable(n)
cost = cp.sum(A*x)

constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

p = cp.Problem(cp.Minimize(cost), constraints)

def stuff(mat):
    pf = cProfile.Profile()
    pf.enable()
    cp.reductions.dcp2cone.cone_matrix_stuffing.ConeMatrixStuffing().apply(mat)
    pf.disable()
    pf.dump_stats('stuff.prof')

benchmark(stuff, p)
