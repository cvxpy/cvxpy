import numpy as np

import cvxpy as cp


class TestJacVecElementwiseUnivariate():

    def test_log(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        log = cp.log(x)
        result_dict = log.jacobian()
        correct_jacobian = np.diag([1/(1.0), 1/(2.0), 1/(3.0)])
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_matrix_log(self):
        n = 2 
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        log = cp.log(x)
        result_dict = log.jacobian()
        correct_jacobian = np.diag([1/(1.0), 1/(3.0), 1/(2.0), 1/(4.0)])
        computed_jacobian = np.zeros((n*n, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_constant_log(self):
        x = np.array([1.0, 2.0, 3.0])
        log = cp.log(x)
        result_dict = log.jacobian()
        assert(result_dict == {})

    def test_exp(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        exp = cp.exp(x)
        result_dict = exp.jacobian()
        correct_jacobian = np.diag([np.exp(1.0), np.exp(2.0), np.exp(3.0)])
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_matrix_exp(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        exp = cp.exp(x)
        result_dict = exp.jacobian()
        correct_jacobian = np.diag([np.exp(1.0), np.exp(3.0), np.exp(2.0), np.exp(4.0)])
        computed_jacobian = np.zeros((n*n, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_constant_exp(self):
        x = np.array([1.0, 2.0, 3.0])
        exp = cp.exp(x)
        result_dict = exp.jacobian()
        assert(result_dict == {})

    def test_entr(self):
        n = 3
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        entr = cp.entr(x)
        result_dict = entr.jacobian()
        correct_jacobian = -np.diag(np.log(x.value) + 1)
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_constant_entr(self):
        x = np.array([1.0, 2.0, 3.0])
        entr = cp.entr(x)
        result_dict = entr.jacobian()
        assert(result_dict == {})

    def test_square(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        square = cp.square(x)
        result_dict = square.jacobian()
        correct_matrix = np.diag([ 2.0, 4.0, 6.0])
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_matrix))

    def test_matrix_square(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        square = cp.square(x)
        result_dict = square.jacobian()
        correct_jacobian = np.diag([2.0, 6.0, 4.0, 8.0])
        computed_jacobian = np.zeros((n*n, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_constant_square(self):
        x = np.array([1.0, 2.0, 3.0])
        square = cp.square(x)
        result_dict = square.jacobian()
        assert(result_dict == {})

    def test_power(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        power = cp.power(x, 3)
        result_dict = power.jacobian()
        correct_jacobian = np.diag([ 3.0*(1.0**2), 3.0*(2.0**2), 3.0*(3.0**2)])
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_matrix_power(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        power = cp.power(x, 3)
        result_dict = power.jacobian()
        correct_jacobian = np.diag([3.0*(1.0**2), 3.0*(3.0**2), 3.0*(2.0**2), 3.0*(4.0**2)])
        computed_jacobian = np.zeros((n*n, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_constant_power(self):
        x = np.array([1.0, 2.0, 3.0])
        power = cp.power(x, 3)
        result_dict = power.jacobian()
        assert(result_dict == {})
