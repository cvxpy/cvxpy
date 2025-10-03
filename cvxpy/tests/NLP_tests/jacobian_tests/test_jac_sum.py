import numpy as np

import cvxpy as cp


class TestJacSum():


    def test_sum_one(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(cp.log(x))
        result_dict = sum.jacobian()
        correct_jacobian = 1 / x.value
        computed_jacobian = np.zeros((1, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_sum_two(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(4 * cp.log(x))
        result_dict = sum.jacobian()
        correct_jacobian = 4 / x.value
        computed_jacobian = np.zeros((1, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
    
    def test_sum_three(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.sum(cp.multiply(np.array([6, 4, 5]), cp.log(x)))
        result_dict = sum.jacobian()
        correct_jacobian = np.array([6, 4, 5]) / x.value
        computed_jacobian = np.zeros((1, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))