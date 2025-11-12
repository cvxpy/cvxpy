import numpy as np

import cvxpy as cp


class TestJacTranspose():


    def test_vec(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x).T
        result_dict = expr.jacobian()
        correct_jacobian = np.diag(1/x.value)
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
       
    def test_col_vec(self):
        n = 3 
        x = cp.Variable((n, 1), name='x')
        x.value = np.array([1.0, 2.0, 3.0]).reshape((n, 1), order='F')
        expr = cp.log(x).T
        result_dict = expr.jacobian()
        correct_jacobian_xx = np.diag(1/x.value.reshape((n)))

        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian_xx))
