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

        computed_hess_xx = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_xx[rows, cols] = vals

        computed_hess_yy = np.zeros((n, n))
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_yy[rows, cols] = vals

        computed_hess_xy = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, y)]
        computed_hess_xy[rows, cols] = vals

        computed_hess_yx = np.zeros((n, n))
        rows, cols, vals = result_dict[(y, x)]
        computed_hess_yx[rows, cols] = vals

        H_computed = np.block([
            [computed_hess_xx, computed_hess_xy],
            [computed_hess_yx, computed_hess_yy]
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

        computed_hess_xx = np.zeros((1, 1))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_xx[rows, cols] = vals

        computed_hess_yy = np.zeros((1, 1))
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_yy[rows, cols] = vals

        computed_hess_xy = np.zeros((1, 1))
        rows, cols, vals = result_dict[(x, y)]
        computed_hess_xy[rows, cols] = vals

        computed_hess_yx = np.zeros((1, 1))
        rows, cols, vals = result_dict[(y, x)]
        computed_hess_yx[rows, cols] = vals

        H_computed = np.block([
            [computed_hess_xx, computed_hess_xy],
            [computed_hess_yx, computed_hess_yy]
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

        computed_hess_xx = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_xx[rows, cols] = vals

        computed_hess_yy = np.zeros((1, 1))
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_yy[rows, cols] = vals

        computed_hess_xy = np.zeros((n, 1))
        rows, cols, vals = result_dict[(x, y)]
        computed_hess_xy[rows, cols] = vals

        computed_hess_yx = np.zeros((1, n))
        rows, cols, vals = result_dict[(y, x)]
        computed_hess_yx[rows, cols] = vals

        H_computed = np.block([
            [computed_hess_xx, computed_hess_xy],
            [computed_hess_yx, computed_hess_yy]
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

        computed_hess_xx = np.zeros((1, 1))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_xx[rows, cols] = vals

        computed_hess_yy = np.zeros((n, n))
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_yy[rows, cols] = vals

        computed_hess_xy = np.zeros((1, n))
        rows, cols, vals = result_dict[(x, y)]
        computed_hess_xy[rows, cols] = vals

        computed_hess_yx = np.zeros((n, 1))
        rows, cols, vals = result_dict[(y, x)]
        computed_hess_yx[rows, cols] = vals

        H_computed = np.block([
            [computed_hess_xx, computed_hess_xy],
            [computed_hess_yx, computed_hess_yy]
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

class TestQuadOverLin_HessVec():

    def test_simple_quad_over_lin(self):
        """
        The Hessian of (x, y) -> sum(x^2)/y
        with values x = [1, 2, 3], y = 4
        and dual = [1] is as follows:
        array(
        [[ 0.5   ,  0.    ,  0.    , -0.125 ],
        [ 0.    ,  0.5   ,  0.    , -0.25  ],
        [ 0.    ,  0.    ,  0.5   , -0.375 ],
        [-0.125 , -0.25  , -0.375 ,  0.4375]])
       """
        x = cp.Variable((3,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((1,), name='y')
        y.value = np.array([4.0])
        dual = np.array([1.0])
        expr = cp.quad_over_lin(x, y)
        hess_xx = np.diag([2.0/4.0, 2.0/4.0, 2.0/4.0])
        hess_yy = np.array([[0.4375]])
        hess_xy = np.array([[-2.0/(4.0**2)],
                            [-4.0/(4.0**2)],
                            [-6.0/(4.0**2)]])
        hess_yx = hess_xy.T
        result_dict = expr.hess_vec(dual)
        H = np.block([[hess_xx, hess_xy],
                    [hess_yx, hess_yy]])
        computed_hess = np.zeros((4, 4))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals

        rows, cols, vals = result_dict[(y, y)]
        computed_hess[rows+3, cols+3] = vals

        rows, cols, vals = result_dict[(x, y)]
        computed_hess[rows, cols+3] = vals
        
        rows, cols, vals = result_dict[(y, x)]
        computed_hess[rows+3, cols] = vals
        assert(np.allclose(computed_hess, H))
