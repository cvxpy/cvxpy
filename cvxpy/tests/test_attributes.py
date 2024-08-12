import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp


class TestAttributes():
    def test_sparsity_pattern(self):
        X = cp.Variable((3, 3), sparsity=[(0, 1), (0, 2)])
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X.value, z)
    
    def test_sparsity_invalid_input(self):
        with pytest.raises(ValueError, match="Sparsity should have 2 dimensions."):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1), (0, 1)])
    
    def test_sparsity_incorrect_dim(self):
        with pytest.raises(
            ValueError, match="All index tuples in sparsity must have the same length."
        ):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1, 2)])

    def test_sparsity_out_of_bounds(self):
        with pytest.raises(
            ValueError, match="Sparsity is out of bounds for expression with shape \\(3, 3\\)."
        ):
            sparsity = [(0, 1, 2), (3, 4, 5)]
            cp.Variable((3, 3), sparsity=sparsity)

    def test_sparsity_reduces_num_var(self):
        X = cp.Variable((3, 3), sparsity=[(0, 1), (0, 2)])
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 2

        X = cp.Variable((3, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 9

    def test_diag_value_sparse(self):
        X = cp.Variable((3, 3), diag=True)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = -np.eye(3)
        assert type(X.value) is sp._dia.dia_matrix
        assert np.allclose(X.value.toarray(), z)
