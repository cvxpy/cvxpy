import time
import random
import statistics

import cvxpy as cvx
import numpy as np

#SZ=8
SZ=(8,)

# 0.4.11
#def problem():
#    x = cvx.Variable(SZ)
#    mu = cvx.Parameter(SZ)

#    u = mu.T * x - cvx.sum_squares(x) - 0.1 * cvx.sum_entries(cvx.abs(x) ** (3/2))
#    c = [-0.2 <= x, x <= 0.2]

#    return {"problem": cvx.Problem(cvx.Maximize(u), c),
#            "variable": x,
#            "forecast": mu}

# 1.0
def problem():
    x = cvx.Variable(shape=SZ)
    mu = cvx.Parameter(shape=SZ)

    u = mu.T * x - cvx.sum_squares(x) - 0.1 * cvx.sum(cvx.abs(x) ** (3/2))
    c = [-0.2 <= x, x <= 0.2]

    return {"problem": cvx.Problem(cvx.Maximize(u), c),
            "variable": x,
            "forecast": mu}

def with_compile():
    p = problem()
    p["forecast"].value = np.random.normal(size=SZ[0])
    p["problem"].solve(solver=cvx.MOSEK)
    return p


p0 = problem()
def without_compile():
    p0["forecast"].value = np.random.normal(size=SZ[0])
    p0["problem"].solve(solver=cvx.MOSEK)
    return p0


functions = with_compile, without_compile
times = {f.__name__: [] for f in functions}

#import cProfile
#pr = cProfile.Profile()
#pr.enable()
for i in range(10):  # adjust accordingly so whole thing takes a few sec
    func = random.choice(functions)
    t0 = time.time()
    func()
    t1 = time.time()
    times[func.__name__].append((t1 - t0) * 1000)
#pr.disable()
#pr.dump_stats('cvxpy1compileconst.pf')

for name, numbers in times.items():
    print('FUNCTION:', name, 'Used', len(numbers), 'times (ms)')
    print('\tMEDIAN', statistics.median(numbers))
    print('\tMEAN  ', statistics.mean(numbers))
    print('\tSTDEV ', statistics.stdev(numbers))
