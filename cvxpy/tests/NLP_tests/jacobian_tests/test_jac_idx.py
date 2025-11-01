import numpy as np
import pytest

import cvxpy as cp


class TestJacobianIndex():


    def test_jacobian_simple_idx(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)[1]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((1, n))
        correct_jacobian[0, 1] = 1/x.value[1]
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_jacobian_array_idx(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        idxs = np.array([0, 2])
        expr = cp.log(x)[idxs]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n))
        correct_jacobian[0, 0] = 1/x.value[0]
        correct_jacobian[1, 2] = 1/x.value[2]
        computed_jacobian = np.zeros((2, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_jacobian_slice_idx(self):
        n = 4
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0])
        expr = cp.log(x)[1:3]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n))
        correct_jacobian[0, 1] = 1/x.value[1]
        correct_jacobian[1, 2] = 1/x.value[2]
        computed_jacobian = np.zeros((2, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    @pytest.mark.xfail(reason="Boolean indexing not yet supported")
    def test_jacobian_special_idx(self):
        n = 4
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0])
        expr = cp.log(x)[[True, False, True, False]]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n))
        correct_jacobian[0, 0] = 1/x.value[0]
        correct_jacobian[1, 2] = 1/x.value[2]
        computed_jacobian = np.zeros((2, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_jacobian_matrix_slice(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        expr = cp.log(x)[0, :]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n*n))
        correct_jacobian[0, 0] = 1/x.value[0, 0]
        correct_jacobian[1, 2] = 1/x.value[0, 1]
        computed_jacobian = np.zeros((2, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    @pytest.mark.xfail(reason="Boolean indexing not yet supported")
    def test_jacobian_matrix_special_idx(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        expr = cp.log(x)[[True, False], :]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n*n))
        correct_jacobian[0, 0] = 1/x.value[0, 0]
        correct_jacobian[1, 2] = 1/x.value[0, 1]
        computed_jacobian = np.zeros((2, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    @pytest.mark.xfail(reason="List indexing not yet supported for matrices")
    def test_jacobian_matrix_list_idx(self):
        n = 2
        x = cp.Variable((n, n), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        expr = cp.log(x)[[0, 1], [1, 0]]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((2, n*n))
        correct_jacobian[0, 1] = 1/x.value[0, 1]
        correct_jacobian[1, 2] = 1/x.value[1, 0]
        computed_jacobian = np.zeros((2, n*n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    @pytest.mark.xfail(reason="None indexing not yet supported")
    def test_jacobian_none_indexing(self):
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)[:, None]
        result_dict = expr.jacobian()
        correct_jacobian = np.zeros((n, n))
        for i in range(n):
            correct_jacobian[i, i] = 1/x.value[i]
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))


class TestJacobianReshape():

    def test_jacobian_reshape(self):
        n = 6
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expr = cp.reshape(x, (2, 3), order='F')
        # check that expr.jacobian() is the same as x.jacobian()
        result_dict = expr.jacobian()
        correct_jacobian = np.eye(n)
        computed_jacobian = np.zeros((n, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert(np.allclose(computed_jacobian, correct_jacobian))

    def test_jacobian_reshape_orderC(self):
        n = 6
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        expr = cp.reshape(x, (2, 3), order='C')
        # should raise an assertion error
        with pytest.raises(AssertionError):
            expr.jacobian()
