import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp


class TestAttributes():
    @pytest.mark.parametrize("sparsity", [[np.array([0, 0]), np.array([0, 1])], [(0, 1), (0, 2)]])
    def test_sparsity_pattern(self, sparsity):
        X = cp.Variable((3, 3), sparsity=sparsity)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X.value, z)

    def test_sparsity_invalid_input(self):
        with pytest.raises(ValueError, match="Indices should have 2 dimensions."):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1), (0, 1)])
    
    def test_sparsity_incorrect_dim(self):
        with pytest.raises(
            ValueError, match="All index tuples in indices must have the same length."
        ):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1, 2)])

    def test_sparsity_out_of_bounds(self):
        with pytest.raises(
            ValueError, match="Indices is out of bounds for expression with shape \\(3, 3\\)."
        ):
            cp.Variable((3, 3), sparsity=[(0, 1, 2), (3, 4, 5)])

    def test_sparsity_0D_variable(self):
        with pytest.raises(ValueError, match="Indices should have 0 dimensions."):
            cp.Variable(sparsity=[(0, 1)])

    def test_parameter_sparsity(self):
        """
        sparsity = [(0, 1), (1, 2)]
        X = cp.Parameter((3, 3), sparsity=sparsity)
        values = np.array([1, 1])
        X.value = values
        
        # Define an optimization problem
        Y = cp.Variable((3, 3))
        objective = cp.Minimize(cp.sum_squares(Y - X))
        constraints = [Y >= -1, Y <= 1]
        prob = cp.Problem(objective, constraints)
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = 1
        assert np.allclose(X.value, z)
        """

    def test_sparsity_reduces_num_var(self):
        X = cp.Variable((3, 3), sparsity=[(0, 1), (0, 2)])
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 2

        X = cp.Variable((3, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 9

    def test_sparsity_sparse_value(self):
        pass

    def test_non_sparse_value(self):
        pass

    def test_parameter_sparse_value(self):
        pass

    def test_integer_indices(self):
        pass

    def test_diag_value_sparse(self):
        X = cp.Variable((3, 3), diag=True)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = -np.eye(3)
        assert type(X.value) is sp.dia_matrix
        assert np.allclose(X.value.toarray(), z)
