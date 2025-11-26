import numpy as np

import cvxpy as cp


class TestHessTranspose():


    def test_vec(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x).T
        result_dict = expr.hess_vec(np.ones(n))
        correct_hess = -np.diag(1/x.value)**2
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))
       
    def test_col_vec(self):
        n = 3 
        x = cp.Variable((n, 1), name='x')
        x.value = np.array([1.0, 2.0, 3.0]).reshape((n, 1), order='F')
        expr = cp.log(x).T
        result_dict = expr.hess_vec(np.ones(n))
        correct_hess = -np.diag(1/x.value.reshape((n)))**2

        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))

    def test_mat(self):
        n = 3
        m = 2
        x = cp.Variable((3, 2), name='x')
        x.value = np.array([[1.0, 2.0,], [3.0, 4.0], [5.0, 6.0]])
        expr = cp.log(x).T
        v = np.ones(n * m)
        v[2:4] = 2
        result_dict = expr.hess_vec(v)
        correct_hess = -np.array([[1., 0., 0., 0., 0., 0.,],
                                     [0., 0.22222222, 0, 0., 0., 0.,],
                                     [0., 0., 0.04, 0., 0., 0.,],
                                     [0., 0., 0., 0.25, 0., 0.,],
                                     [0., 0., 0., 0., 0.125, 0.,],
                                     [0., 0., 0., 0., 0., 0.027777778]]
        )

        
        computed_hess = np.zeros((n * m, n * m))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_hess))
