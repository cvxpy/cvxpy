import numpy as np

import cvxpy as cp


class TestHessSum():


    def test_sum_one(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        dummy_vec = np.array([3.5])
        sum = cp.sum(cp.log(x))
        result_dict = sum.hess_vec(dummy_vec)
        correct_matrix = -3.5 * np.diag(1 / x.value ** 2)
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))

    def test_sum_two(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        dummy_vec = np.array([1])
        sum = cp.sum(4 * cp.log(x))
        result_dict = sum.hess_vec(dummy_vec)
        correct_matrix = -4 * np.diag(1 / x.value ** 2)
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))
    
    def test_sum_three(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        dummy_vec = np.array([4]) 
        sum = cp.sum(cp.multiply(np.array([1, 2, 3]), cp.log(x)))
        result_dict = sum.hess_vec(dummy_vec)
        correct_matrix = 4 * np.diag(-1 * np.array([1 / (1.0**2), 2 / (2.0**2), 3 / (3.0**2)]))
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))
