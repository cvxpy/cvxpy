import time

import numpy as np

import cvxpy as cp


def main():
    np.random.seed(1)

    timings = {}
    for m in np.logspace(1, 4, 4):
        m = int(m)
        print(m)
        n = int(m/10)
        A = np.random.randn(m, n)
        b = np.random.randn(m, 1)

        # Construct the problem.
        x = cp.Variable((n, 1))
        objective = cp.Minimize(cp.sum_squares(cp.reshape(A @ x, (m, 1)) - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        # The optimal objective is returned by prob.solve().
        # result = prob.solve(canon_backend=cp.RUST_CANON_BACKEND)

        timings = {}
        t1 = time.time()
        _F = prob.get_problem_data(cp.SCS, canon_backend=cp.RUST_CANON_BACKEND)
        t2 = time.time()
        _G = prob.get_problem_data(cp.SCS, canon_backend=cp.SCIPY_CANON_BACKEND)
        t3 = time.time()
        _H = prob.get_problem_data(cp.SCS, canon_backend=cp.CPP_CANON_BACKEND)
        t4 = time.time()

        timings[m] = {
            "Rust": t2 - t1,
            "SciPy": t3 - t2,
            "C++": t4 - t3
        }
        print(timings[m])

if __name__ == "__main__":
    main()
