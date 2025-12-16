import numpy as np

import cvxpy as cp


class TestJacTranspose():


    def test_vec(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x).T
        result_dict = expr.jacobian()
        correct_jacobian = np.diag(1/x.value)
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))
       
    def test_col_vec(self):
        n = 3 
        x = cp.Variable((n, 1), name='x')
        x.value = np.array([1.0, 2.0, 3.0]).reshape((n, 1), order='F')
        expr = cp.log(x).T
        result_dict = expr.jacobian()
        correct_jacobian_xx = np.diag(1/x.value.reshape((n)))

        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian_xx))

    def test_mat(self):
        n = 3
        m = 2
        x = cp.Variable((3, 2), name='x')
        x.value = np.array([[1.0, 2.0,], [3.0, 4.0], [5.0, 6.0]])
        expr = cp.log(x).T
        result_dict = expr.jacobian()
        correct_jacobian = np.array([[1., 0., 0., 0., 0., 0.,],
                                     [0., 0., 0., 0.5, 0., 0.,],
                                     [0., 0.33333333, 0., 0., 0., 0.,],
                                     [0., 0., 0., 0., 0.25, 0.,],
                                     [0., 0., 0.2, 0., 0., 0.,],
                                     [0., 0., 0., 0., 0., 0.16666667]])

        computed_jacobian = np.zeros((n * m, n * m))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        print(computed_jacobian)
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_nondiagonal(self):
        n = 3
        m = 2
        x = cp.Variable((3, 2), name='x')
        x.value = np.array([[1.0, 2.0,], [3.0, 4.0], [5.0, 6.0]])
        expr = cp.log(x).sum(axis=1).T
        result_dict = expr.jacobian()
        correct_jacobian = np.array([[1., 0., 0., 0.5, 0., 0.,],
                                     [0., 0.33333333, 0., 0., 0.25, 0.,],
                                     [0., 0., 0.2, 0., 0., 0.16666667,],
                                     ])

        computed_jacobian = np.zeros((expr.size, n * m))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_nondiagonal_v2(self):
        n = 3
        m = 2
        x = cp.Variable((3, 2), name='x')
        x.value = np.array([[1.0, 2.0,], [3.0, 4.0], [5.0, 6.0]])
        A = np.array([[1., 2., 3.],
                      [4., 5., 6.]])
        expr = (A @ cp.log(x)).T
        result_dict = expr.jacobian()
        correct_jacobian = np.array([
            [1.,        0.66666667,0.6,       0.,        0.,        0.,       ],
            [0.,        0.        ,0.,        0.5,       0.5,       0.5,      ],
            [4.,        1.66666667,1.2,       0.,        0.,        0.,       ],
            [0.,        0.        ,0.,        2.,        1.25,      1.,       ],

        ])

        computed_jacobian = np.zeros((expr.size, n * m))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        print(computed_jacobian)
        assert(np.allclose(computed_jacobian, correct_jacobian))
