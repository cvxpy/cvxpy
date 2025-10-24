import numpy as np
import pytest

import cvxpy as cp


class TestJacMatmul():

    # X @ Y with X constant, Y variable
    def test_matmul_constant_test_one(self):
        m, n, p = 2, 3, 2
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = X @ Y
        result_dict = expr.jacobian()
        rows, cols, vals = result_dict[Y]
        computed_jacobian = np.zeros((m * p, n * p))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.kron(np.eye(p), X)
        assert np.allclose(computed_jacobian, correct_jacobian)

    # X @ Y with X variable, Y constant
    def test_matmul_constant_test_two(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = np.array([[1.0, 2.0],
                      [3.0, 4.0],
                      [5.0, 6.0]])
        expr = X @ Y
        result_dict = expr.jacobian()
        rows, cols, vals = result_dict[X]
        computed_jacobian = np.zeros((m * p, m * n))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.kron(Y.T, np.eye(m))
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_matmul_both_variables(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = X @ Y
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        correct_jacobian_y = np.kron(np.eye(p), X.value)
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        correct_jacobian_x = np.kron(Y.value.T, np.eye(m))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)

        assert len(result_dict) == 2

    def test_matmul_constant_test_one_m3_n1_p3(self):
        m, n, p = 3, 1, 3
        X = np.array([[1.0],
                      [2.0],
                      [3.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0, 3.0]])
        expr = X @ Y
        result_dict = expr.jacobian()
        rows, cols, vals = result_dict[Y]
        computed_jacobian = np.zeros((m * p, n * p))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.kron(np.eye(p), X)
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_matmul_constant_test_two_m3_n1_p3(self):
        m, n, p = 3, 1, 3
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0],
                            [2.0],
                            [3.0]])
        Y = np.array([[1.0, 2.0, 3.0]])
        expr = X @ Y
        result_dict = expr.jacobian()
        rows, cols, vals = result_dict[X]
        computed_jacobian = np.zeros((m * p, m * n))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.kron(Y.T, np.eye(m))
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_matmul_both_variables_m3_n1_p3(self):
        m, n, p = 3, 1, 3
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0],
                            [2.0],
                            [3.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0, 3.0]])
        expr = X @ Y
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        correct_jacobian_y = np.kron(np.eye(p), X.value)
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

    def test_matmul_X_logY(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = X @ cp.log(Y)
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        correct_jacobian_y = (np.kron(np.eye(p), X.value) @
                              np.diag((1.0 / Y.value).flatten(order='F')))
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        Z = np.log(Y.value)
        correct_jacobian_x = np.kron(Z.T, np.eye(m))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)

        assert len(result_dict) == 2
    
    def test_matmul_expX_Y(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = cp.exp(X) @ Y
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        Z = np.exp(X.value)
        correct_jacobian_y = np.kron(np.eye(p), Z)
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        correct_jacobian_x = np.kron(Y.value.T, np.eye(m)) @ np.diag(Z.flatten(order='F'))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)
        assert len(result_dict) == 2

    def test_matmul_expX_logY(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = cp.exp(X) @ cp.log(Y)
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        Z = np.exp(X.value)
        correct_jacobian_y = np.kron(np.eye(p), Z) @ np.diag((1.0 / Y.value).flatten(order='F'))
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        W = np.log(Y.value)
        correct_jacobian_x = np.kron(W.T, np.eye(m)) @ np.diag(Z.flatten(order='F'))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)
        assert len(result_dict) == 2
    
    def test_matmul_expX_sinY(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        expr = cp.exp(X) @ cp.sin(Y)
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        Z = np.exp(X.value)
        correct_jacobian_y = np.kron(np.eye(p), Z) @ np.diag(np.cos(Y.value).flatten(order='F'))
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        S = np.sin(Y.value)
        correct_jacobian_x = np.kron(S.T, np.eye(m)) @ np.diag(Z.flatten(order='F'))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)

        assert len(result_dict) == 2

    def test_matmul_expX_sinY_plus_cosZ(self):
        m, n, p = 2, 3, 2
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        Z = cp.Variable((n, p), name='Z')
        Z.value = np.array([[0.5, 1.5],
                            [2.5, 3.5],
                            [4.5, 5.5]])

        expr = cp.exp(X) @ (cp.sin(Y) + cp.cos(Z))
        result_dict = expr.jacobian()

        rows_y, cols_y, vals_y = result_dict[Y]
        computed_jacobian_y = np.zeros((m * p, n * p))
        computed_jacobian_y[rows_y, cols_y] = vals_y
        Zexp = np.exp(X.value)
        correct_jacobian_y = np.kron(np.eye(p), Zexp) @ np.diag(np.cos(Y.value).flatten(order='F'))
        assert np.allclose(computed_jacobian_y, correct_jacobian_y)

        rows_z, cols_z, vals_z = result_dict[Z]
        computed_jacobian_z = np.zeros((m * p, n * p))
        computed_jacobian_z[rows_z, cols_z] = vals_z
        correct_jacobian_z = (np.kron(np.eye(p), Zexp) @
                               np.diag((-np.sin(Z.value)).flatten(order='F')))
        assert np.allclose(computed_jacobian_z, correct_jacobian_z)

        rows_x, cols_x, vals_x = result_dict[X]
        computed_jacobian_x = np.zeros((m * p, m * n))
        computed_jacobian_x[rows_x, cols_x] = vals_x
        M = np.sin(Y.value) + np.cos(Z.value)
        correct_jacobian_x = np.kron(M.T, np.eye(m)) @ np.diag(Zexp.flatten(order='F'))
        assert np.allclose(computed_jacobian_x, correct_jacobian_x)

        assert len(result_dict) == 3


    def test_matmul_same_variables_one(self):
        m, n = 2, 3
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])

        expr = cp.exp(X) @ cp.log(X.T)
        with pytest.raises(ValueError):
            expr.jacobian()
            
    def test_matmul_same_variables_two(self):
        m, n, p = 2, 3, 2
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])

        expr = cp.exp(X) @ (cp.log(Y) + cp.sin(X.T))
        with pytest.raises(ValueError):
            expr.jacobian()
          
        
