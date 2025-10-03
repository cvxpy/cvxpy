import numpy as np

import cvxpy as cp


class TestJacAdd():


    def test_two_atoms_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        sum = cp.log(x) + cp.exp(x) + 7 * cp.log(x)
        result_dict = sum.jacobian()
        correct_jacobian = np.diag(1/x.value) + np.diag(np.exp(x.value)) + np.diag(7/x.value)
        computed_jacobian = np.zeros((n, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
       
    def test_three_atoms_different_variables(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((n, ), name='y')
        y.value = np.array([4.0, 5.0, 6.0])
        z = cp.Variable((n, ), name='z')
        z.value = np.array([7.0, 8.0, 9.0])
        sum = cp.log(x) + cp.exp(y) + cp.square(z)
        result_dict = sum.jacobian()
        correct_jacobian_xx = np.diag(1/x.value)
        correct_jacobian_yy = np.diag(np.exp(y.value))
        correct_jacobian_zz = np.diag(2 * z.value)

        computed_jacobian = np.zeros((n, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian_xx))

        computed_jacobian.fill(0)
        rows = result_dict[y][0]
        cols = result_dict[y][1]
        vals = result_dict[y][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian_yy))

        computed_jacobian.fill(0)
        rows = result_dict[z][0]
        cols = result_dict[z][1]
        vals = result_dict[z][2]       
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian_zz))
        
    def test_negation_one_variable(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        neg_log = -cp.log(x)
        result_dict = neg_log.jacobian()
        correct_jacobian = -np.diag(1/x.value)
        computed_jacobian = np.zeros((n, n))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
    
    
