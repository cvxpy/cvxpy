import numpy as np

import cvxpy as cp


class TestJacobianBroadcasting():

    def test_row_wise_stacking_one(self):
        x = cp.Variable(2, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 2))
        x.value = np.array([3.0, 2.0])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((4 * 2, 2))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.array([[1/3.0, 0.0],
                                     [1/3.0, 0.0],
                                     [1/3.0, 0.0],
                                     [1/3.0, 0.0],
                                     [0.0, 1/2.0],
                                     [0.0, 1/2.0],
                                     [0.0, 1/2.0],
                                     [0.0, 1/2.0]])
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_row_wise_stacking_two(self):
        x = cp.Variable((1, 2), name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 2))
        x.value = np.array([[3.0, 2.0]])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((4 * 2, 2))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.array([[1/3.0, 0.0],
                                    [1/3.0, 0.0],
                                    [1/3.0, 0.0],
                                    [1/3.0, 0.0],
                                    [0.0, 1/2.0],
                                    [0.0, 1/2.0],
                                    [0.0, 1/2.0],
                                    [0.0, 1/2.0]])
        assert(np.allclose(computed_jacobian, correct_jacobian))
        
    def test_row_wise_stacking_three(self):
        x = cp.Variable((1, 2), name='x')
        expr1 = cp.broadcast_to(x, (4, 2))
        x.value = np.array([[3.0, 2.0]])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((4 * 2, 2))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.array([[1, 0.0],
                                     [1, 0.0],
                                     [1, 0.0],
                                     [1, 0.0],
                                     [0.0, 1],
                                     [0.0, 1],
                                     [0.0, 1],
                                     [0.0, 1]])
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_column_wise_stacking(self):
        x = cp.Variable((2, 1), name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (2, 4))
        x.value = np.array([[3.0], [2.0]])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((2 * 4, 2))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.array([[1/3, 0.0],
                                     [0.0, 1/2],
                                     [1/3, 0.0],
                                     [0.0, 1/2],
                                     [1/3, 0.0],
                                     [0.0, 1/2],
                                     [1/3, 0.0],
                                     [0.0, 1/2]])
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_scalar_to_matrix(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (2, 3))
        x.value = np.array([3.0])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((2 * 3, 1))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = (1/3) * np.ones((2 * 3, 1))
        assert(np.allclose(computed_jacobian, correct_jacobian))
                

    def test_scalar_to_row_vector(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (1, 4))
        x.value = np.array([3.0])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((1 * 4, 1))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = (1/3) * np.ones((1 * 4, 1))
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_scalar_to_column_vector(self):
        x = cp.Variable(1, name='x')
        logx = cp.log(x)
        expr1 = cp.broadcast_to(logx, (4, 1))
        x.value = np.array([3.0])
        result = expr1.jacobian()
        computed_jacobian = np.zeros((4 * 1, 1))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = (1/3) * np.ones((1 * 4, 1))
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_row_minus_column(self):
        x = cp.Variable()
        logx = cp.log(x)
        expx = cp.exp(x)
        row = cp.broadcast_to(logx, (1, 4))
        col = cp.broadcast_to(expx, (3, 1))
        expr1 = row - col
        x.value = np.array(3.0)
        result = expr1.jacobian()
        computed_jacobian = np.zeros((3 * 4, 1))
        rows, cols, vals = result[x]
        computed_jacobian[rows, cols] = vals
        correct_jacobian = np.zeros((3 * 4, 1))
        correct_jacobian[:, 0] = (1/3.0) - np.exp(3.0)
        assert(np.allclose(computed_jacobian, correct_jacobian))
        
