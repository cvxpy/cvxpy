import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest
import numpy as np

import os
import time


def benchmark(func, *func_args, **bench_kwargs):
    iters = bench_kwargs['iters']
    vals = []
    for _ in range(iters):
        start = time.time()
        func(*func_args)
        vals.append(time.time() - start)
    print("{:s}: avg={:.3e} s , std={:.3e} s ({:d} iterations)".format(
        func.__name__, np.mean(vals), np.std(vals), iters))


class TestBenchmarks(BaseTest):
    def test_diffcp_sdp_example(self):
        def randn_symm(n):
            A = np.random.randn(n, n)
            return (A + A.T) / 2

        def randn_psd(n):
            A = 1. / 10 * np.random.randn(n, n)
            return np.matmul(A, A.T)

        n = 100
        p = 100
        C = randn_psd(n)
        As = [randn_symm(n) for _ in range(p)]
        Bs = np.random.randn(p)

        def diffcp_sdp():
            X = cp.Variable((n, n), PSD=True)
            objective = cp.trace(cp.matmul(C, X))
            constraints = [
                cp.trace(cp.matmul(As[i], X)) == Bs[i] for i in range(p)]
            problem = cp.Problem(cp.Minimize(objective), constraints)
            problem.get_problem_data(cp.SCS)
        benchmark(diffcp_sdp, iters=1)

    def test_tv_inpainting(self):
        if os.name == "nt":
            self.skipTest("Skipping test due to SciPy overflow issues.")
        Uorig = np.random.randn(512, 512, 3)
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
            problem.get_problem_data(cp.SCS)
        benchmark(tv_inpainting, iters=3)

    def test_least_squares(self):
        m = 20
        n = 15
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        def least_squares():
            x = cp.Variable(n)
            cost = cp.sum_squares(A*x - b)
            cp.Problem(cp.Minimize(cost)).get_problem_data(cp.OSQP)
        benchmark(least_squares, iters=1)

    def test_qp(self):
        m = 15
        n = 10
        p = 5
        P = np.random.randn(n, n)
        P = np.matmul(P.T, P)
        q = np.random.randn(n)
        G = np.random.randn(m, n)
        h = np.matmul(G, np.random.randn(n))
        A = np.random.randn(p, n)
        b = np.random.randn(p)

        def qp():
            x = cp.Variable(n)
            cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + cp.matmul(q.T, x)),
                       [cp.matmul(G, x) <= h,
                       cp.matmul(A, x) == b]).get_problem_data(cp.OSQP)
        benchmark(qp, iters=1)

    def test_stuffing_perf_many_constraints(self):
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
            ConeMatrixStuffing().apply(mat)

        benchmark(stuff, p, iters=1)
