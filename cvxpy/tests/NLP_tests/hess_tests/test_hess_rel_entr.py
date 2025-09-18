import numpy as np
import pytest

import cvxpy as cp


class TestHessVecRelativeEntropy():

    # x log (x/y) with x constant, y variable
    # This should raise an error I (DCED) think because 
    # we should have caught this in the canonicalization step
    def test_rel_entr_one_constant_test_one(self):
        x = np.array([1, 2, 3])
        y = cp.Variable(shape=(3,))
        vec = np.array([5, 4, 3])
        rel_entr = cp.rel_entr(x, y)
        with pytest.raises(ValueError):
            rel_entr.hess_vec(vec)

    # x log (x/y) with x variable, y constant
    # This should raise an error I (DCED) think because 
    # we should have caught this in the canonicalization step
    def test_rel_entr_one_constant_test_two(self):
        x = cp.Variable(shape=(3,))
        y = np.array([4, 5, 6])
        vec = np.array([5, 4, 3])
        rel_entr = cp.rel_entr(x, y)
        with pytest.raises(ValueError):
            rel_entr.hess_vec(vec)

    # x log (x/x) with x variable, should raise an error
    def test_rel_entr_same_variable(self):
        x = cp.Variable(shape=(3,))
        vec = np.array([5, 4, 3])
        x.value = np.array([1, 2, 3])
        rel_entr = cp.rel_entr(x, x)
        with pytest.raises(ValueError):
            rel_entr.hess_vec(vec)

    # x log (x/y) with x vector variable, y vector variable
    def test_rel_entr_two_vector_variables_same_dimension(self):
        n = 4
        x = cp.Variable(shape=(n,))
        y = cp.Variable(shape=(n,))
        vec = np.array([1, 2, 3, 4])

        x.value = np.array([1, 2, 3, 4])
        y.value = np.array([4, 3, 2, 1])

        rel_entr = cp.rel_entr(x, y)
        result_dict = rel_entr.hess_vec(vec)

        H_computed = np.block([
            [result_dict[(x, x)], result_dict[(x, y)]],
            [result_dict[(y, x)], result_dict[(y, y)]]
        ])

        # compute Hessian manually
        x = x.value 
        y = y.value

        H_analytic = np.zeros((2*n, 2*n))
        for i in range(n):
            H_analytic[i, i]       = vec[i] / x[i]
            H_analytic[n+i, n+i]   = vec[i] * x[i] / (y[i]**2)
            H_analytic[i, n+i]     = -vec[i] / y[i]
            H_analytic[n+i, i]     = -vec[i] / y[i]

        assert(np.allclose(H_computed, H_analytic))

    # x log (x/y) with x scalar variable, y scalar variable
    def test_rel_entr_two_scalar_variables_same_dimension(self):
        x = cp.Variable(shape=(1,))
        y = cp.Variable(shape=(1,))
        vec = np.array([3])
        x.value = np.array([2])
        y.value = np.array([4])

        rel_entr = cp.rel_entr(x, y)
        result_dict = rel_entr.hess_vec(vec)
        H_computed = np.block([
            [result_dict[(x, x)].reshape(-1, 1), result_dict[(x, y)].reshape(1, -1)],
            [result_dict[(y, x)].reshape(-1, 1), result_dict[(y, y)].reshape(-1, 1)]
        ])

        H_analytic = np.array([[vec[0] / x.value[0], - vec[0] / y.value[0]],
                               [- vec[0] / y.value[0], vec[0] * x.value[0] / (y.value[0]**2)]])
        assert(np.allclose(H_computed, H_analytic))


    # x log (x/y) with x vector variable, y scalar variable, 
    def test_rel_entr_two_variables_broadcasting_test_one(self):
        n = 4
        x = cp.Variable(shape=(n,))
        y = cp.Variable(shape=(1,))
        vec = np.array([1, 2, 3, 4])

        x.value = np.array([1, 2, 3, 4])
        y.value = np.array([4])

        rel_entr = cp.rel_entr(x, y)
        result_dict = rel_entr.hess_vec(vec)

        H_computed = np.block([
            [result_dict[(x, x)], result_dict[(x, y)].reshape(-1, 1)],
            [result_dict[(y, x)].reshape(1, -1), result_dict[(y, y)].reshape(-1, 1)]
        ])

        # compute Hessian manually
        x = x.value 
        y = y.value

        H_analytic = np.zeros((n+1, n+1))
        for i in range(n):
            H_analytic[i, i]   = vec[i] / x[i]
            H_analytic[i, -1]  = -vec[i] / y[0]
            H_analytic[-1, i]  = -vec[i] / y[0]
        H_analytic[-1, -1] = np.sum(vec * x / (y**2))

        assert(np.allclose(H_computed, H_analytic))

    # x log (x/y) with x scalar variable, y vector variable,
    def test_rel_entr_two_variables_broadcasting_test_two(self):
        n = 4
        x = cp.Variable(shape=(1,))
        y = cp.Variable(shape=(4,))
        vec = np.array([1, 2, 3, 4])

        x.value = np.array([1])
        y.value = np.array([4, 3, 2, 1])

        rel_entr = cp.rel_entr(x, y)
        result_dict = rel_entr.hess_vec(vec)

        H_computed = np.block([
            [result_dict[(x, x)].reshape(-1, 1), result_dict[(x, y)].reshape(1, -1)],
            [result_dict[(y, x)].reshape(-1, 1), result_dict[(y, y)]]
        ])

        # compute Hessian manually
        x = x.value 
        y = y.value

        H = np.zeros((n+1, n+1))
        H[0,0] = vec.sum() / x[0]
        H[0, 1:] = - vec / y
        H[1:, 0] = - vec / y
        H[1:, 1:] = np.diag(vec * x / (y**2))

        assert(np.allclose(H_computed, H))




