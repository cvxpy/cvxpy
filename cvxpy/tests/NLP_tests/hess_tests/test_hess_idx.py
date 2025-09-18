import numpy as np

import cvxpy as cp


class TestHessAdd():


    def test_single_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([4])
        log2 = cp.log(x)[2]
        result_dict = log2.hess_vec(vec)
        result_correct = 4 * (-np.diag(np.array([0, 0, 1/9])))
        assert(np.allclose(result_dict[(x, x)], result_correct))

    def test_slice_two_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.0, 2.0, 3.0])
        vec = np.array([2, 4])
        idxs = np.array([1, 2])
        log12 = cp.log(x)[idxs]
        result_dict = log12.hess_vec(vec)
        result_correct = 2 * (-np.diag(np.array([0, 1/4, 0]))) + \
                         4 * (-np.diag(np.array([0, 0, 1/9]))) 
        assert(np.allclose(result_dict[(x, x)], result_correct))

    
    def test_slice_two_other_idx(self):
        n = 3 
        x = cp.Variable((n, ), name='x')
        x.value = np.array([1.5, 2.0, 3.0])
        vec = np.array([2, 4])
        idxs = np.array([0, 2])
        log12 = cp.log(x)[idxs]
        result_dict = log12.hess_vec(vec)
        result_correct = 2 * (-np.diag(np.array([1 / 2.25, 0, 0]))) + \
                             4 * (-np.diag(np.array([0, 0, 1/9])))
        assert(np.allclose(result_dict[(x, x)], result_correct))