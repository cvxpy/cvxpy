import numpy as np

import cvxpy as cp


class TestHessVecElementwiseUnivariate():

    def test_log(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        log = cp.log(x)
        result_dict = log.hess_vec(vec)
        correct_matrix = np.diag([ -5.0/(1.0**2), -4.0/(2.0**2), -3.0/(3.0**2)])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))

    def test_constant_log(self):
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        log = cp.log(x)
        result_dict = log.hess_vec(vec)
        assert(result_dict == {})

    def test_exp(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        exp = cp.exp(x)
        result_dict = exp.hess_vec(vec)
        correct_matrix = np.diag([ 5.0*np.exp(1.0), 4.0*np.exp(2.0), 3.0*np.exp(3.0)])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))

    def test_constant_exp(self):
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        exp = cp.exp(x)
        result_dict = exp.hess_vec(vec)
        assert(result_dict == {})

    def test_entr(self):
        n = 3
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        entr = cp.entr(x)
        result_dict = entr.hess_vec(vec)
        correct_matrix = np.diag([ -5.0/(1.0), -4.0/(2.0), -3.0/(3.0)])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))

    def test_constant_entr(self):
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        entr = cp.entr(x)
        result_dict = entr.hess_vec(vec)
        assert(result_dict == {})

    def test_square(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        square = cp.square(x)
        result_dict = square.hess_vec(vec)
        correct_matrix = np.diag([10.0, 8.0, 6.0])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))

    def test_constant_square(self):
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        square = cp.square(x)
        result_dict = square.hess_vec(vec)
        assert(result_dict == {})

    def test_power(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        power = cp.power(x, 3)
        result_dict = power.hess_vec(vec)
        correct_matrix = np.diag([30.0, 48.0, 54.0])
        assert(np.allclose(result_dict[(x, x)], correct_matrix))

    def test_constant_power(self):
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([5.0, 4.0, 3.0])
        power = cp.power(x, 3)
        result_dict = power.hess_vec(vec)
        assert(result_dict == {})


