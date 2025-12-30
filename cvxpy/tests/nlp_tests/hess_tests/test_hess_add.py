import numpy as np

import cvxpy as cp


class TestHessAdd():


    def test_two_atoms_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([-1.0, 2, 4])
        sum = cp.log(x) + cp.exp(x) + 7 * cp.log(x)
        result_dict = sum.hess_vec(vec)
        correct_matrix = np.diag([-1 * (-8.0/(1.0**2) + np.exp(1.0)), 
                                    2 * (-8.0/(2.0**2) + np.exp(2.0)), 
                                    4 * (-8.0/(3.0**2) + np.exp(3.0))])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))
       
    def test_three_atoms_different_variables(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((n, ), name='y')
        y.value = np.array([4.0, 5.0, 6.0])
        z = cp.Variable((n, ), name='z')
        z.value = np.array([7.0, 8.0, 9.0])
        vec = np.array([-1.0, 2, 4])
        sum = cp.log(x) + cp.exp(y) + cp.square(z)
        result_dict = sum.hess_vec(vec)
        correct_matrix_xx = np.diag([1 / (1.0**2), -2 / (2.0**2), -4 / (3.0**2)])
        correct_matrix_yy = np.diag([-1 * np.exp(4.0), 2 * np.exp(5.0), 4 * np.exp(6.0)])
        correct_matrix_zz = np.diag([2 * -1, 2 * 2, 2 * 4])

        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix_xx))

        computed_hess.fill(0)
        rows, cols, vals = result_dict[(y, y)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix_yy))

        computed_hess.fill(0)
        rows, cols, vals = result_dict[(z, z)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix_zz))
        
    def test_negation_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        neg_log = -cp.log(x)
        result_dict = neg_log.hess_vec(vec)
        correct_matrix = -np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))
