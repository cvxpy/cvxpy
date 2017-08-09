import unittest
from cvxpy import *
import numpy as np
from cvxpy.tests.base_test import *
import scipy.sparse as sp


class TestLinOps(BaseTest):
    def assertDataEqual(self, original_data, canon_data):
        for key in ['A', 'b', 'c', 'G', 'h']:
            M1, M2 = original_data[key], canon_data[key]
            if isinstance(M1, sp.csc.csc_matrix):
                self.assertTrue((M1 - M2).nnz == 0)
            else:
                self.assertTrue(np.allclose(M1, M2))

    def assertConstraintsMatch(self, constraints):
        settings.USE_CVXCANON = False
        p = Problem(Minimize(0), constraints)
        cvxpy_data = p.get_problem_data(ECOS)

        settings.USE_CVXCANON = True
        p = Problem(Minimize(0), constraints)
        cvx_canon_data = p.get_problem_data(ECOS)

        self.assertDataEqual(cvxpy_data, cvx_canon_data)

    def test_sum_dense(self):
        rows, cols = 3, 3
        x1 = Variable(rows, cols)
        x2 = Variable(rows, cols)
        A = np.random.randn(rows, cols)
        self.assertConstraintsMatch([x1 + x2 == A])

    def test_sum_sparse(self):
        rows, cols = 75, 100
        x1 = Variable(rows, cols)
        x2 = Variable(rows, cols)
        A_sp = sp.rand(rows, cols, density=0.01)
        self.assertConstraintsMatch([x1 + x2 == A_sp])

    def test_promote(self):
        rows = 10
        b = Variable()
        v1 = Variable(rows, 1)
        self.assertConstraintsMatch([v1 == b])

    def test_mul_dense(self):
        rows, cols = 15, 17

        A = np.random.randn(rows, cols)
        b = Variable(rows, 1)
        x = Variable(cols, 1)
        y = Variable(rows, 1)
        self.assertConstraintsMatch([A*x == b, A*x == y])

    def test_mul_sparse(self):
        rows, cols = 75, 100

        A_sp = Constant(sp.rand(rows, cols, density=0.01))
        b = Variable(rows, 1)
        x = Variable(cols, 1)
        y = Variable(rows, 1)
        self.assertConstraintsMatch([A_sp*x == b, A_sp*x == y])

    def test_rmul_dense(self):
        rows, cols = 15, 17
        A = np.random.randn(rows, cols)
        b = Variable(1, cols)
        x = Variable(1, rows)
        y = Variable(1, cols)
        self.assertConstraintsMatch([x * A == b, x * A == y])

    def test_rmul_sparse(self):
        rows, cols = 75, 100
        A_sp = Constant(sp.rand(rows, cols, density=0.01))
        b = Variable(1, cols)
        x = Variable(1, rows)
        y = Variable(1, cols)
        self.assertConstraintsMatch([x * A_sp == b, x * A_sp == y])

    def test_mul_elemwise_dense(self):
        rows, cols = 15, 17
        A = np.random.randn(rows, cols)
        x1 = Variable(rows, cols)
        M = Variable(rows, cols)
        self.assertConstraintsMatch([mul_elemwise(A, x1) == M])

    def test_mul_elemwise_sparse(self):
        rows, cols = 75, 100
        A_sp = Constant(sp.rand(rows, cols, density=0.01))
        x1 = Variable(rows, cols)
        M = Variable(rows, cols)
        self.assertConstraintsMatch([mul_elemwise(A_sp, x1) == M])

    def test_div(self):
        x = Variable()
        self.assertConstraintsMatch([x / 4 == 3])

    def test_neg_dense(self):
        rows, cols = 3, 3
        x1 = Variable(rows, cols)
        x2 = Variable(rows, cols)
        A = np.random.randn(rows, cols)
        self.assertConstraintsMatch([x1 - x2 == A])

    def test_neg_sparse(self):
        rows, cols = 3, 3
        x1 = Variable(rows, cols)
        x2 = Variable(rows, cols)
        A_sp = sp.rand(rows, cols, density=0.01)
        self.assertConstraintsMatch([x1 - x2 == A_sp])

    def test_index(self):
        rows, cols = 15, 17
        X = Variable(rows, cols)

        for trial in xrange(100):
            xi = np.random.choice(rows)
            xj = np.random.choice(rows)
            xk = np.random.choice(rows)
            ry = np.random.choice(cols)
            if xk == 0:
                xk = 1
            if np.random.rand() < 0.5:
                xk *= -1

            yi = np.random.choice(cols)
            yj = np.random.choice(cols)
            yk = np.random.choice(cols)
            rx = np.random.choice(rows)
            if yk == 0:
                yk = 1
            if np.random.rand() < 0.5:
                yk *= -1

            constr = [X[xi:xj:xk, ry] == 0, X[rx, yi:yj:yk] == 0]
            self.assertConstraintsMatch(constr)

    def test_transpose_dense(self):
        rows, cols = 15, 17
        X = Variable(rows, cols)
        A = np.random.randn(rows, cols)
        self.assertConstraintsMatch([X.T == A.T])

    def test_transpose_sparse(self):
        rows, cols = 75, 100
        X = Variable(rows, cols)
        A_sp = sp.rand(rows, cols, density=0.01)
        self.assertConstraintsMatch([X.T == A_sp.T])

    def test_sum_entries(self):
        rows, cols = 15, 17
        X = Variable(rows, cols)
        self.assertConstraintsMatch([sum_entries(X) == 4.5])

    def test_trace(self):
        n = 15
        X = Variable(n, n)
        self.assertConstraintsMatch([trace(X) == 3])

    def test_reshape_dense(self):
        rows, cols = 15, 17
        X = Variable(rows, cols)

        while True:
            n = np.random.choice(rows + cols)
            if n == 0:
                n = 1
            if rows * cols % n == 0:
                break
        m = (rows * cols) / n

        A = np.random.randn(m, n)

        constr = [reshape(X, m, n) == A]
        self.assertConstraintsMatch(constr)

    def test_reshape_sparse(self):
        rows, cols = 75, 100
        X = Variable(rows, cols)

        while True:
            n = np.random.choice(rows + cols)
            if n == 0:
                n = 1
            if rows * cols % n == 0:
                break
        m = (rows * cols) / n

        A = sp.rand(m, n, density=0.01)

        constr = [reshape(X, m, n) == A]
        self.assertConstraintsMatch(constr)

    def test_diag_vec_dense(self):
        n = 15
        x = Variable(n, 1)
        constr = [diag(x) == np.eye(n)]
        self.assertConstraintsMatch(constr)

    def test_diag_vec_sparse(self):
        n = 15
        x = Variable(n, 1)
        constr = [diag(x) == sp.eye(n)]
        self.assertConstraintsMatch(constr)

    def test_diag_mat_dense(self):
        n = 15
        X = Variable(n, n)
        v = np.random.randn(n)
        constr = [diag(X) == v]
        self.assertConstraintsMatch(constr)

    def test_diag_mat_sparse(self):
        n = 15
        X = Variable(n, n)
        v = sp.rand(n, 1, density=0.01)
        constr = [diag(X) == v]
        self.assertConstraintsMatch(constr)

    def test_upper_tri(self):
        n = 15
        X = Variable(n, n)
        constr = [upper_tri(X) == 0]
        self.assertConstraintsMatch(constr)

    def test_conv(self):
        n, m = 15, 17
        c = np.random.randn(m, 1)
        x = Variable(n, 1)
        y = np.random.randn(n + m - 1, 1)

        constr = [conv(c, x) == y]
        self.assertConstraintsMatch(constr)

    def test_hstack_dense(self):
        rows, cols = 15, 17
        X1 = Variable(rows, cols)
        X2 = Variable(rows, cols)
        X3 = Variable(rows, cols)
        X4 = Variable(rows, cols)
        X5 = Variable(rows, cols)

        A = np.random.randn(rows, cols * 5)
        constr = [hstack(X1, X2, X3, X4, X5) == A]
        self.assertConstraintsMatch(constr)

    def test_hstack_sparse(self):
        rows, cols = 15, 17
        X1 = Variable(rows, cols)
        X2 = Variable(rows, cols)
        X3 = Variable(rows, cols)
        X4 = Variable(rows, cols)
        X5 = Variable(rows, cols)

        A_sp = sp.rand(rows, cols * 5, density=0.01)
        constr = [hstack(X1, X2, X3, X4, X5) == A_sp]
        self.assertConstraintsMatch(constr)

    def test_vstack_dense(self):
        rows, cols = 15, 17
        X1 = Variable(rows, cols)
        X2 = Variable(rows, cols)
        X3 = Variable(rows, cols)
        X4 = Variable(rows, cols)
        X5 = Variable(rows, cols)

        A = np.random.randn(5 * rows, cols)
        constr = [vstack(X1, X2, X3, X4, X5) == A]
        self.assertConstraintsMatch(constr)

    def test_vstack_sparse(self):
        rows, cols = 75, 100
        X1 = Variable(rows, cols)
        X2 = Variable(rows, cols)
        X3 = Variable(rows, cols)
        X4 = Variable(rows, cols)
        X5 = Variable(rows, cols)

        A_sp = sp.rand(5 * rows, cols, density=0.01)
        constr = [vstack(X1, X2, X3, X4, X5) == A_sp]
        self.assertConstraintsMatch(constr)

    def test_kron_dense(self):
        m, n = 3, 5
        p, q = 7, 9
        X = Variable(m, n)
        C = np.random.randn(p, q)
        A = np.random.randn(p * m, q * n)
        constr = [kron(C, X) == A]
        self.assertConstraintsMatch(constr)

    def test_kron_sparse(self):
        m, n = 13, 15
        p, q = 17, 19
        X = Variable(m, n)
        C = sp.rand(p, q, density=0.1)
        A = sp.rand(p * m, q * n, density=0.1)
        constr = [kron(C, X) == A]
        self.assertConstraintsMatch(constr)

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLinOps)
    unittest.TextTestRunner(verbosity=2).run(suite)
