import numpy as np
import pytest

import cvxpy as cp


class TestHessVecMatmul():

    # X @ Y with X constant, Y variable
    def test_matmul_constant_test_one(self):
        m, n, p = 2, 3, 2
        X = np.array([[1.0, 2.0, 3.0],
                      [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0],
                            [3.0, 4.0],
                            [5.0, 6.0]])
        lmbda = np.random.rand(m * p, 1)
        expr = X @ Y
        hess_dict = expr.hess_vec(lmbda)
        assert len(hess_dict) == 0
       

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
        lmbda = np.random.rand(m * p, 1)
        hess_dict = expr.hess_vec(lmbda)
        assert len(hess_dict) == 0
      

    def test_matmul_both_variables(self):
        m, n, p = 2, 3, 5
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [3.0, 4.0, 5.0, 6.0, 7.0],
                            [5.0, 6.0, 7.0, 8.0, 9.0]])
        expr = X @ Y
        lmbda = np.random.rand(m * p)
        hess_dict = expr.hess_vec(lmbda)

        rows, cols, vals = hess_dict[(X, Y)]
        computed_hess = np.zeros((m * n, n * p))
        computed_hess[rows, cols] = vals

        np.random.seed(0)
        correct_hess = np.zeros((m * n, n * p))
        for j in range(p):
            l1, l2 = lmbda[2*j : 2*j + 2]
            block = np.array([
                [l1, 0, 0],
                [l2, 0, 0],
                [0,  l1, 0],
                [0,  l2, 0],
                [0,  0,  l1],
                [0,  0,  l2]
            ])
            correct_hess[:, 3*j:3*(j+1)] = block

        assert np.allclose(computed_hess, correct_hess)

    def test_matmul_both_variables_sum(self):
        m, n, p = 2, 3, 5
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
                            [3.0, 4.0, 5.0, 6.0, 7.0],
                            [5.0, 6.0, 7.0, 8.0, 9.0]])
        expr = cp.sum(X @ Y)
        
        hess_dict = expr.hess_vec(np.array([1.0]))

        rows, cols, vals = hess_dict[(X, Y)]
        computed_hess = np.zeros((m * n, n * p))
        computed_hess[rows, cols] = vals

        ones = np.ones(m * p)
        np.random.seed(0)
        correct_hess = np.zeros((m * n, n * p))
        for j in range(p):
            l1, l2 = ones[2*j : 2*j + 2]
            block = np.array([
                [l1, 0, 0],
                [l2, 0, 0],
                [0,  l1, 0],
                [0,  l2, 0],
                [0,  0,  l1],
                [0,  0,  l2]
            ])
            correct_hess[:, 3*j:3*(j+1)] = block

        assert np.allclose(computed_hess, correct_hess)

    def test_matmul_both_variables_m3_n1_p3(self):
        m, n, p = 3, 1, 3
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0],
                            [2.0],
                            [3.0]])
        Y = cp.Variable((n, p), name='Y')
        Y.value = np.array([[1.0, 2.0, 3.0]])
        expr = X @ Y
        lmbda = np.random.rand(m * p)
        hess_dict = expr.hess_vec(lmbda)

        rows, cols, vals = hess_dict[(X, Y)]
        computed_hess = np.zeros((m * n, n * p))
        computed_hess[rows, cols] = vals

        HXY = np.array([
            [lmbda[0], lmbda[3], lmbda[6]],
            [lmbda[1], lmbda[4], lmbda[7]],
            [lmbda[2], lmbda[5], lmbda[8]],
        ])
       
        assert np.allclose(computed_hess, HXY)

    def test_matmul_same_variables_one(self):
        m, n = 3, 3
        X = cp.Variable((m, n), name='X')
        X.value = np.array([[1.0, 2.0, 3.0],
                            [4.0, 5.0, 6.0],
                            [7.0, 8.0, 9.0]])

        expr = X @ X
        lmbda = np.random.rand(m * m, 1)
        with pytest.raises(ValueError):
            expr.hess_vec(lmbda) 
