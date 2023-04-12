import os
import time

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.tests.base_test import BaseTest


def benchmark(func, iters: int = 1, name=None) -> None:
    vals = []
    for _ in range(iters):
        start = time.time()
        func()
        vals.append(time.time() - start)
    name = func.__name__ if name is None else name
    print("{:s}: avg={:.3e} s , std={:.3e} s ({:d} iterations)".format(
        name, np.mean(vals), np.std(vals), iters))


class TestBenchmarks(BaseTest):
    def test_diffcp_sdp_example(self) -> None:

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

    def test_tv_inpainting(self) -> None:
        if os.name == "nt":
            self.skipTest("Skipping test due to overflow bug in SciPy < 1.2.0.")
        Uorig = np.random.randn(512, 512, 3)
        rows, cols, colors = Uorig.shape
        known = np.zeros((rows, cols, colors))
        for i in range(rows):
            for j in range(cols):
                if np.random.random() > 0.7:
                    for k in range(colors):
                        known[i, j, k] = 1

        def tv_inpainting():
            Ucorr = known * Uorig  # This is elementwise mult on numpy arrays.
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
        benchmark(tv_inpainting, iters=1)

    def test_least_squares(self) -> None:
        m = 20
        n = 15
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        def least_squares():
            x = cp.Variable(n)
            cost = cp.sum_squares(A @ x - b)
            cp.Problem(cp.Minimize(cost)).get_problem_data(cp.OSQP)
        benchmark(least_squares, iters=1)

    def test_qp(self) -> None:
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

    def test_cone_matrix_stuffing_with_many_constraints(self) -> None:
        m = 2000
        n = 2000
        A = np.random.randn(m, n)
        C = np.random.rand(m // 2)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)

        def cone_matrix_stuffing_with_many_constraints():
            ConeMatrixStuffing().apply(problem)

        benchmark(cone_matrix_stuffing_with_many_constraints, iters=1)

    def test_parameterized_cone_matrix_stuffing_with_many_constraints(self) -> None:
        self.skipTest("This benchmark takes too long.")
        m = 2000
        n = 2000
        A = cp.Parameter((m, n))
        C = cp.Parameter(m // 2)
        b = cp.Parameter(m)
        A.value = np.random.randn(m, n)
        C.value = np.random.rand(m // 2)
        b.value = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)

        def parameterized_cone_matrix_stuffing():
            ConeMatrixStuffing().apply(problem)

        benchmark(parameterized_cone_matrix_stuffing, iters=1)

    def test_small_cone_matrix_stuffing(self) -> None:
        m = 200
        n = 200
        A = np.random.randn(m, n)
        C = np.random.rand(m // 2)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)

        def small_cone_matrix_stuffing():
            ConeMatrixStuffing().apply(problem)

        benchmark(small_cone_matrix_stuffing, iters=10)

    @pytest.mark.skip(reason="Failing in Windows CI - potentially memory leak")
    def test_small_parameterized_cone_matrix_stuffing(self) -> None:
        m = 200
        n = 200
        A = cp.Parameter((m, n))
        C = cp.Parameter(m // 2)
        b = cp.Parameter(m)
        A.value = np.random.randn(m, n)
        C.value = np.random.rand(m // 2)
        b.value = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum(A @ x)

        constraints = [C[i] * x[i] <= b[i] for i in range(m // 2)]
        constraints.extend([C[i] * x[m // 2 + i] == b[m // 2 + i] for i in range(m // 2)])

        problem = cp.Problem(cp.Minimize(cost), constraints)

        def small_parameterized_cone_matrix_stuffing():
            ConeMatrixStuffing().apply(problem)

        benchmark(small_parameterized_cone_matrix_stuffing, iters=1)

    def test_small_lp(self) -> None:
        m = 200
        n = 200
        A = np.random.randn(m, n)
        b = np.random.randn(m)
        c = np.random.rand(n)

        x = cp.Variable(n)
        cost = cp.matmul(c, x)
        constraints = [A @ x <= b]
        problem = cp.Problem(cp.Minimize(cost), constraints)

        def small_lp():
            problem.get_problem_data(cp.SCS)

        benchmark(small_lp, iters=1)
        benchmark(small_lp, iters=1, name="small_lp_second_time")

    @pytest.mark.skip(reason="Failing in Windows CI - potentially memory leak")
    def test_small_parameterized_lp(self) -> None:
        m = 200
        n = 200
        A = cp.Parameter((m, n))
        b = cp.Parameter(m)
        c = cp.Parameter(n)
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        c.value = np.random.rand(n)

        x = cp.Variable(n)
        cost = cp.matmul(c, x)
        constraints = [A @ x <= b]
        problem = cp.Problem(cp.Minimize(cost), constraints)

        def small_parameterized_lp():
            problem.get_problem_data(cp.SCS)

        benchmark(small_parameterized_lp, iters=1)
        benchmark(small_parameterized_lp, iters=1,
                  name="small_parameterized_lp_second_time")

    def test_parameterized_qp(self) -> None:
        """Test speed of first solve with QP codepath and SOCP codepath.
        """
        m = 150
        n = 100
        np.random.seed(1)
        A = cp.Parameter((m, n))
        b = cp.Parameter((m,))

        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A@x - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        start = time.time()
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        prob.solve(solver=cp.ECOS)
        end = time.time()

        print('Conic canonicalization')
        print('(ECOS) solver time: ', prob.solver_stats.solve_time)
        print('cvxpy time: ', (end - start) - prob.solver_stats.solve_time)

        np.random.seed(1)
        A = cp.Parameter((m, n))
        b = cp.Parameter((m,))

        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum_squares(A@x - b))
        constraints = [0 <= x, x <= 1]
        prob = cp.Problem(objective, constraints)

        start = time.time()
        A.value = np.random.randn(m, n)
        b.value = np.random.randn(m)
        prob.solve(solver=cp.OSQP)
        end = time.time()

        print('Quadratic canonicalization')
        print('(OSQP) solver time: ', prob.solver_stats.solve_time)
        print('cvxpy time: ', (end - start) - prob.solver_stats.solve_time)

    def test_issue_1668_slow_pruning(self) -> None:
        """Regression test for https://github.com/cvxpy/cvxpy/issues/1668

        Pruning matrices caused order-of-magnitude slow downs in compilation times.
        """
        s = 2000
        t = 10
        x = np.linspace(-100.0, 100.0, s)
        rows = 50
        var = cp.Variable(shape=(rows, t))

        cost = cp.sum_squares(
            var @ np.tile(np.array([x]), t).reshape((t, x.shape[0]))
            - np.tile(x, rows).reshape((rows, s))
        )
        objective = cp.Minimize(cost)
        problem = cp.Problem(objective)

        start = time.time()
        problem.get_problem_data(cp.ECOS, verbose=True)
        end = time.time()

        print("Issue #1668 regression test")
        print("Compilation time: ", end - start)
