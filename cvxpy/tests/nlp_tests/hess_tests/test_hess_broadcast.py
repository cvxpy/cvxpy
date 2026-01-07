import numpy as np

import cvxpy as cp


class TestHessianBroadcasting():

    def test_row_wise_stacking_one(self):
        x = cp.Variable(2, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 2))
        x.value = np.array([3.0, 2.0])
        lmbda = np.random.randn(4 * 2)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((2, 2))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda[0:4]) * (-1/9.0), 0.0],
                                        [0.0, np.sum(lmbda[4:8]) * (-1/4.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))

    def test_row_wise_stacking_two(self):
        x = cp.Variable((1, 2), name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 2))
        x.value = np.array([[3.0, 2.0]])
        lmbda = np.random.rand(4 * 2)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((2, 2))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda[0:4]) * (-1/9.0), 0.0],
                                        [0.0, np.sum(lmbda[4:8]) * (-1/4.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))
        
    def test_row_wise_stacking_three(self):
        x = cp.Variable((1, 2), name='x')
        expr1 = cp.broadcast_to(x, (4, 2))
        x.value = np.array([[3.0, 2.0]])
        lmbda = np.random.rand(4 * 2)
        result = expr1.hess_vec(lmbda)
        assert(result == {})

    def test_column_wise_stacking(self):
        x = cp.Variable((2, 1), name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (2, 4))
        x.value = np.array([[3.0], [2.0]])
        lmbda = np.random.rand(2 * 4)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((2, 2))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda[0::2]) * (-1/9.0), 0.0],
                                        [0.0, np.sum(lmbda[1::2]) * (-1/4.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))
        

    def test_scalar_to_matrix(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (2, 3))
        x.value = np.array([3.0])
        lmbda = np.random.rand(2 * 3)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((1, 1))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda) * ( -1/9.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))
                

    def test_scalar_to_row_vector(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (1, 4))
        x.value = np.array([3.0])
        lmbda = np.random.rand(1 * 4)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((1, 1))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda) * ( -1/9.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))

    def test_scalar_to_column_vector(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 1))
        x.value = np.array([3.0])
        lmbda = np.random.rand(4 * 1)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((1, 1))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda) * ( -1/9.0)]])
        assert(np.allclose(computed_hessian, correct_hessian))

    def test_row_minus_column(self):
        x = cp.Variable()
        logx = cp.log(x)
        expx = cp.exp(x)
        row = cp.broadcast_to(logx, (1, 4))
        col = cp.broadcast_to(expx, (3, 1))
        expr1 = row - col
        x.value = np.array(3.0)
        lmbda = np.random.rand(3 * 4)
        result = expr1.hess_vec(lmbda)
        computed_hessian = np.zeros((1, 1))
        rows, cols, vals = result[(x, x)]
        computed_hessian[rows, cols] = vals
        correct_hessian = np.array([[np.sum(lmbda) * ( -1/9.0 - np.exp(3.0))]])
        assert(np.allclose(computed_hessian, correct_hessian))
