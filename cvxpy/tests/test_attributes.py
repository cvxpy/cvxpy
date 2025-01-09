import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp


class TestAttributes:
    @pytest.mark.parametrize("sparsity", [[np.array([0, 0]), np.array([0, 1])], [(0, 1), (0, 2)]])
    def test_sparsity_pattern(self, sparsity):
        X = cp.Variable((3, 3), sparsity=sparsity)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X.value, z)

    def test_sparsity_condition(self):
        data = np.arange(8).reshape((2,2,2))
        mask = np.where(data % 2 == 0)
        X = cp.Variable((2,2,2), sparsity=mask)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((2,2,2))
        z[mask] = -1
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
            ValueError, match="Indices are out of bounds for expression with shape \\(3, 3\\)."
        ):
            cp.Variable((3, 3), sparsity=[(0, 1, 2), (3, 4, 5)])

    def test_sparsity_0D_variable(self):
        with pytest.raises(ValueError, match="Indices should have 0 dimensions."):
            cp.Variable(sparsity=[(0, 1)])

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
        assert type(X.value) is sp.dia_matrix
        assert np.allclose(X.value.toarray(), z)

    def test_variable_bounds(self):
        # Valid bounds: Scalars promoted to arrays
        x = cp.Variable((2, 2), name="x", bounds=[0, 10])
        assert np.array_equal(x.bounds[0], np.zeros((2, 2)))
        assert np.array_equal(x.bounds[1], np.full((2, 2), 10))

        # Valid bounds: Arrays with matching shape
        bounds = [np.zeros((2, 2)), np.ones((2, 2)) * 5]
        x = cp.Variable((2, 2), name="x", bounds=bounds)
        assert np.array_equal(x.bounds[0], np.zeros((2, 2)))
        assert np.array_equal(x.bounds[1], np.ones((2, 2)) * 5)

        # Valid bounds: One bound is None
        bounds = [None, 5]
        x = cp.Variable((2, 2), name="x", bounds=bounds)
        assert np.array_equal(x.bounds[0], np.full((2, 2), -np.inf))
        assert np.array_equal(x.bounds[1], np.full((2, 2), 5))

        # Invalid bounds: Length not equal to 2
        bounds = [0]  # Only one item
        with pytest.raises(ValueError, match="Bounds should be a list of two items."):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Non-iterable type
        bounds = 10  # Not iterable
        with pytest.raises(ValueError, match="Bounds should be a list of two items."):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Arrays with non-matching shape
        bounds = [np.zeros((3, 3)), np.ones((3, 3))]
        with pytest.raises(
            ValueError,
            match="Bounds should be None, scalars, or arrays with the same dimensions "
                "as the variable/parameter.",
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Lower bound > Upper bound
        bounds = [5, 0]
        with pytest.raises(
            ValueError,
            match="Invalid bounds: some upper bounds are less than "
                "corresponding lower bounds.",
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: NaN in bounds
        bounds = [np.nan, 10]
        with pytest.raises(
            ValueError, match="np.nan is not feasible as lower or upper bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Upper bound is -inf
        bounds = [0, -np.inf]
        with pytest.raises(
            ValueError, match="-np.inf is not feasible as an upper bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Lower bound is inf
        bounds = [np.inf, 10]
        with pytest.raises(
            ValueError, match="np.inf is not feasible as a lower bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)