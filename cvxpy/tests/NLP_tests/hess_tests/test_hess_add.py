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
        result_correct = np.diag([-1 * (-8.0/(1.0**2) + np.exp(1.0)), 
                                    2 * (-8.0/(2.0**2) + np.exp(2.0)), 
                                    4 * (-8.0/(3.0**2) + np.exp(3.0))])
        assert(np.allclose(result_dict[(x, x)], result_correct))

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
        result_correct_xx = np.diag([1 / (1.0**2), -2 / (2.0**2), -4 / (3.0**2)])
        result_correct_yy = np.diag([-1 * np.exp(4.0), 2 * np.exp(5.0), 4 * np.exp(6.0)])
        result_correct_zz = np.diag([2 * -1, 2 * 2, 2 * 4])
        assert(np.allclose(result_dict[(x, x)], result_correct_xx))
        assert(np.allclose(result_dict[(y, y)], result_correct_yy))
        assert(np.allclose(result_dict[(z, z)], result_correct_zz))
        
    def test_negation_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        neg_log = -cp.log(x)
        result_dict = neg_log.hess_vec(vec)
        correct_matrix = -np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))
