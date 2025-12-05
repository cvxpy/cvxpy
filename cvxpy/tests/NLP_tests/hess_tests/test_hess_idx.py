import numpy as np
import pytest

import cvxpy as cp


class TestHessIndex():

    def test_single_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([4])
        log2 = cp.log(x)[2]
        result_dict = log2.hess_vec(vec)
        correct_matrix = 4 * (-np.diag(np.array([0, 0, 1/9])))
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))

    def test_slice_two_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([2, 4])
        idxs = np.array([1, 2])
        log12 = cp.log(x)[idxs]
        result_dict = log12.hess_vec(vec)
        correct_matrix = 2 * (-np.diag(np.array([0, 1/4, 0]))) + \
                         4 * (-np.diag(np.array([0, 0, 1/9]))) 
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))

    def test_slice_two_other_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.5, 2.0, 3.0])
        vec = np.array([2, 4])
        idxs = np.array([0, 2])
        log12 = cp.log(x)[idxs]
        result_dict = log12.hess_vec(vec)
        correct_matrix = 2 * (-np.diag(np.array([1 / 2.25, 0, 0]))) + \
                             4 * (-np.diag(np.array([0, 0, 1/9])))
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))

    def test_special_index_matrix(self):
        """
        This test was failing because hess_vec was not properly handling
        matrix indexing.
        """
        x = cp.Variable((2, 2), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec = np.array([5, 6])
        rows, cols = np.array([0, 0]), np.array([0, 1])
        logx = cp.log(x)[rows, cols]
        result_dict = logx.hess_vec(vec)
        correct_matrix = 5 * (-np.diag(np.array([1.0, 0, 0, 0]))) + \
                            6 * (-np.diag(np.array([0, 0, 1/4, 0])))
        computed_hess = np.zeros((4, 4))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))

    @pytest.mark.skip(reason="TODO fix this test for duplicate indices")
    def test_special_index_duplicate_matrix(self):
        """
        TODO fix this test
        """
        x = cp.Variable((2, 2), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec = np.array([5, 6])
        rows, cols = np.array([0, 0]), np.array([0, 0])
        logx = cp.log(x)[rows, cols]
        result_dict = logx.hess_vec(vec)
        correct_matrix = 5 * (-np.diag(np.array([1.0, 0, 0, 0]))) + \
                            6 * (-np.diag(np.array([1.0, 0, 0, 0])))
        computed_hess = np.zeros((4, 4))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert(np.allclose(computed_hess, correct_matrix))
