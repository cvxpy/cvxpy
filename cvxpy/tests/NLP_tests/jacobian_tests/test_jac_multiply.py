import numpy as np
import pytest

import cvxpy as cp

# TODO: should check what happens if we use * instead of cp.multiply

class TestJacMultiply():

    # x * y with x constant, y variable
    def test_jacobian_multiply_one_constant_test_one(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(5.0, x)
        result_dict = mult.jacobian()
        rows, cols, vals = result_dict[x]
        computed_jacobian = np.zeros((n, n))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.diag([5.0, 5.0, 5.0])
        assert(np.allclose(computed_jacobian, correct_jacobian))

    # x * y with x variable, y constant
    def test_jacobian_multiply_one_constant_test_two(self):
        n = 3 
        y = cp.Variable((n, ), name='y')
        y.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(y, 5.0)
        result_dict = mult.jacobian()
        rows, cols, vals = result_dict[y]
        computed_jacobian = np.zeros((n, n))
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.diag([5.0, 5.0, 5.0])
        assert(np.allclose(computed_jacobian, correct_jacobian))

    # x * y with x variable, should raise an error
    def test_jacobian_multiply_same_variable(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(x, x)
        with pytest.raises(ValueError):
            mult.jacobian()

    # x * y with x vector variable, y vector variable
    def test_jacobian_multiply_two_vector_variables_same_dimension(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((3, ), name='y')
        y.value = np.array([4.0, 5.0, 6.0])
        mult = cp.multiply(x, y)
        result_dict = mult.jacobian()
        correct_jacobian_x = np.diag(y.value)
        correct_jacobian_y = np.diag(x.value)

        computed_jacobian_x = np.zeros((3, 3))
        rows = result_dict[x][0]
        cols = result_dict[x][1]
        vals = result_dict[x][2]
        computed_jacobian_x[rows, cols] = vals
        assert(np.allclose(computed_jacobian_x, correct_jacobian_x))

        computed_jacobian_y = np.zeros((3, 3))
        rows = result_dict[y][0]
        cols = result_dict[y][1]
        vals = result_dict[y][2]
        computed_jacobian_y[rows, cols] = vals
        assert(np.allclose(computed_jacobian_y, correct_jacobian_y))

        assert(len(result_dict) == 2)

    # x * y with x scalar variable, y scalar variable
    def test_jacobian_multiply_two_scalar_variables_same_dimension(self):
        x = cp.Variable((1, ), name='x')
        x.value = np.array([3.0])
        y = cp.Variable((1, ), name='y')
        y.value = np.array([4.0])
        mult = cp.multiply(x, y)
        result_dict = mult.jacobian()
        correct_jacobian_x = np.array([[y.value]])
        correct_jacobian_y = np.array([[x.value]])

        computed_jacobian_x = np.zeros((1, 1))
        rows, cols, vals = result_dict[x]
        computed_jacobian_x[rows, cols] = vals
        assert(np.allclose(computed_jacobian_x, correct_jacobian_x))

        computed_jacobian_y = np.zeros((1, 1))
        rows, cols, vals = result_dict[y]
        computed_jacobian_y[rows, cols] = vals
        assert(np.allclose(computed_jacobian_y, correct_jacobian_y))

        assert(len(result_dict) == 2)

    # x * y with x vector variable, y scalar variable,
    def test_jacobian_multiply_two_variables_broadcasting_test_one(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        y = cp.Variable((1, ), name='y')
        y.value = np.array([4.0])
        mult = cp.multiply(x, y)
        result_dict = mult.jacobian()

        correct_jacobian_x = np.eye(3) * y.value
        computed_jacobian_x = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jacobian_x[rows, cols] = vals
        assert(np.allclose(computed_jacobian_x, correct_jacobian_x))

        correct_jacobian_y = x.value
        computed_jacobian_y = np.zeros((3, 1))
        rows, cols, vals = result_dict[y]
        computed_jacobian_y[rows, cols] = vals
        assert(np.allclose(computed_jacobian_y.flatten(), correct_jacobian_y))

        assert(len(result_dict) == 2)

    # x * y with x scalar variable, y vector variable,
    def test_jacobian_multiply_two_variables_broadcasting_test_two(self):
        x = cp.Variable((1, ), name='x')
        x.value = np.array([4.0])
        y = cp.Variable((3, ), name='y')
        y.value = np.array([1.0, 2.0, 3.0])
        mult = cp.multiply(x, y)
        result_dict = mult.jacobian()

        correct_jacobian_x = np.array([[1.0], [2.0], [3.0]])
        computed_jacobian_x = np.zeros((3, 1))
        rows, cols, vals = result_dict[x]
        computed_jacobian_x[rows, cols] = vals
        assert(np.allclose(computed_jacobian_x, correct_jacobian_x))

        correct_jacobian_y = np.eye(3) * x.value
        computed_jacobian_y = np.zeros((3, 3))
        rows, cols, vals = result_dict[y]
        computed_jacobian_y[rows, cols] = vals
        assert(np.allclose(computed_jacobian_y, correct_jacobian_y))

        assert(len(result_dict) == 2)

    # scalar constant * phi(x) 
    def test_jacobian_scalar_constant_and_atom_one(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([0.5, 1.0, 2.0])
        log = cp.log(x)
        mult = cp.multiply(3.0, log)
        result_dict = mult.jacobian()

        correct_jacobian = np.zeros((3, 3))
        log_result_dict = log.jacobian()
        rows, cols, vals = log_result_dict[x]
        correct_jacobian[rows, cols] = 3 * vals

        computed_jacobian = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
        assert(len(result_dict) == 1)


    # phi(x) * scalar constant
    def test_jacobian_scalar_constant_and_atom_two(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([0.5, 1.0, 2.0])
        log = cp.log(x)
        mult = cp.multiply(log, 3.0)
        result_dict = mult.jacobian()
        
        correct_jacobian = np.zeros((3, 3))
        log_result_dict = log.jacobian()
        rows, cols, vals = log_result_dict[x]
        correct_jacobian[rows, cols] = 3 * vals

        computed_jacobian = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
        assert(len(result_dict) == 1)

    # vector constant * phi(x) 
    def test_jacobian_vector_constant_and_atom_one(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([0.5, 1.0, 2.0])
        log = cp.log(x)
        constant_vector = np.array([7.0, 2.0, 3.0])
        mult = cp.multiply(constant_vector, log)
        result_dict = mult.jacobian()

        correct_jacobian = np.diag(constant_vector / x.value)
        computed_jacobian = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
        assert(len(result_dict) == 1)

    # phi(x) * vector constant
    def test_jacobian_vector_constant_and_atom_two(self):
        x = cp.Variable((3, ), name='x')
        x.value = np.array([0.5, 1.0, 2.0])
        log = cp.log(x)
        constant_vector = np.array([7.0, 2.0, 3.0])
        mult = cp.multiply(log, constant_vector)
        result_dict = mult.jacobian()
        
        correct_jacobian = np.diag(constant_vector / x.value)

        computed_jacobian = np.zeros((3, 3))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
        assert(len(result_dict) == 1)

