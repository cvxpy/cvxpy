import numpy as np

import cvxpy as cp


class TestJacSum():
    """
    This class adds tests for verifying the correctness of the Jacobian
    of the sum atom. We also support taking the sum of a matrix expression
    along a given axis. Taking the jacobian of N-dimensional expressions and
    arbitrary axes are not yet supported.
    """

    def test_sum_one(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(cp.log(x))
        result_dict = sum.jacobian()
        correct_jacobian = 1 / x.value
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_sum_two(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(4 * cp.log(x))
        result_dict = sum.jacobian()
        correct_jacobian = 4 / x.value
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
    
    def test_sum_three(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(cp.multiply(np.array([6, 4, 5]), cp.log(x)))
        result_dict = sum.jacobian()
        correct_jacobian = np.array([6, 4, 5]) / x.value
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_sum_axis_0(self):
        m, n = 3, 2
        x = cp.Variable((m, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sum = cp.sum(cp.log(x), axis=0)
        result_dict = sum.jacobian()
        correct_jacobian = np.array([[1, 1/3.0, 1/5.0, 0, 0, 0],
                                    [0, 0, 0, 1/2.0, 1/4.0, 1/6.0]])
        computed_jacobian = np.zeros((n, m*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_sum_axis_1(self):
        m, n = 3, 2
        x = cp.Variable((m, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        sum = cp.sum(cp.log(x), axis=1)
        result_dict = sum.jacobian()
        correct_jacobian = np.array([[1/1.0, 0, 0, 1/2.0, 0, 0],
                                     [0.0, 1/3.0, 0, 0, 1/4.0, 0],
                                     [0.0, 0.0, 1/5.0, 0.0, 0.0, 1/6.0]])
        computed_jacobian = np.zeros((m, m*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_sum_multiply_axis(self):
        pass
