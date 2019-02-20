from __future__ import print_function
import time
import random

import cvxpy as cvx
import numpy as np

solver = "ECOS"

for N in [8, 800, 8000]:
    print("N =", N)

    def problem():
        x = cvx.Variable(shape=(N,))
        mu = cvx.Parameter(shape=(N,))

        u = mu.T * x - cvx.sum_squares(x) - 0.1 * cvx.sum(cvx.abs(x) ** (3./2.))
        c = [-0.2 <= x, x <= 0.2]

        return {"problem": cvx.Problem(cvx.Maximize(u), c),
                "variable": x,
                "forecast": mu}

    def with_compile():
        p = problem()
        p["forecast"].value = np.random.normal(size=N)
        p["problem"].solve(solver=solver)
        return p

    p0 = problem()

    def without_compile():
        p0["forecast"].value = np.random.normal(size=N)
        p0["problem"].solve(solver=solver)
        return p0

    functions = with_compile, without_compile
    times = {f.__name__: [] for f in functions}
    solver_times = {f.__name__: [] for f in functions}
    N_repeats = 100

    for i in range(N_repeats):
        func = random.choice(functions)
        t0 = time.time()
        p = func()
        t1 = time.time()
        times[func.__name__].append(t1 - t0)
        solver_times[func.__name__].append(
            p["problem"].solver_stats.solve_time
        )

    for f in functions:
        name = f.__name__
        print('FUNCTION:', name, 'Used', N_repeats, 'times',
              'with solver', solver)
        t = np.array(times[name])
        t_solver = np.array(solver_times[name])
        t_cvxpy = t - t_solver
        print('\tSolver time')
        print('\t\tMEDIAN', np.median(t_solver))
        print('\t\tMEAN  ', np.mean(t_solver))
        print('\t\tSTDEV ', np.std(t_solver))
        print('\tCVXPY time')
        print('\t\tMEDIAN', np.median(t_cvxpy))
        print('\t\tMEAN  ', np.mean(t_cvxpy))
        print('\t\tSTDEV ', np.std(t_cvxpy))
        print('\tTotal time')
        print('\t\tMEDIAN', np.median(t))
        print('\t\tMEAN  ', np.mean(t))
        print('\t\tSTDEV ', np.std(t))
