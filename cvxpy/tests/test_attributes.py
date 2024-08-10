import numpy as np
import scipy.sparse as sp

import cvxpy as cp


class TestAttributes():
    def test_sparsity_pattern(self):
        X = cp.Variable((3, 3), sparsity=((0, 1), (0, 2)))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X.value, z)
    
    def test_sparsity_input(self):
        pass

    def test_sparsity_incorrect_dim(self):
        pass

    def test_sparsity_out_of_bounds(self):
        pass

    def test_sparsity_reduces_problem_dim(self):
        X = cp.Variable((3, 3), sparsity=((0, 1), (0, 2)))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL) == 0
        


    def test_diag_value(self):
        X = cp.Variable((3, 3), diag=True)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve(verbose=True)
        z = -np.eye(3)
        assert type(X.value) is sp._dia.dia_matrix
        assert np.allclose(X.value.toarray(), z)
