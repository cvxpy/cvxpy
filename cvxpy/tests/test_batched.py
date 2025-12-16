"""
Copyright 2025, the CVXPY developers

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import numpy as np
import pytest

import cvxpy as cp


class TestBatchedParameterAPI:
    """Tests for the batched parameter API."""

    def test_set_value_batch_1d_param(self):
        """Test set_value with batch=True for 1D parameter."""
        param = cp.Parameter(5)
        batch_data = np.random.randn(10, 5)

        param.set_value(batch_data, batch=True)

        assert param.batch_shape == (10,)
        assert param.is_batched
        assert param.value.shape == (10, 5)
        np.testing.assert_array_equal(param.value, batch_data)

    def test_set_value_batch_2d_param(self):
        """Test set_value with batch=True for 2D parameter."""
        param = cp.Parameter((3, 4))
        batch_data = np.random.randn(10, 3, 4)

        param.set_value(batch_data, batch=True)

        assert param.batch_shape == (10,)
        assert param.is_batched
        assert param.value.shape == (10, 3, 4)
        np.testing.assert_array_equal(param.value, batch_data)

    def test_set_value_batch_scalar_param(self):
        """Test set_value with batch=True for scalar parameter."""
        param = cp.Parameter()
        batch_data = np.random.randn(10)

        param.set_value(batch_data, batch=True)

        assert param.batch_shape == (10,)
        assert param.is_batched
        assert param.value.shape == (10,)
        np.testing.assert_array_equal(param.value, batch_data)

    def test_set_value_batch_multiple_batch_dims(self):
        """Test set_value with multiple batch dimensions."""
        param = cp.Parameter((3, 4))
        batch_data = np.random.randn(5, 10, 3, 4)

        param.set_value(batch_data, batch=True)

        assert param.batch_shape == (5, 10)
        assert param.is_batched
        assert param.value.shape == (5, 10, 3, 4)

    def test_set_value_non_batch_clears_batch_shape(self):
        """Test that setting value without batch=True clears batch_shape."""
        param = cp.Parameter((3, 4))

        # First set batched value
        batch_data = np.random.randn(10, 3, 4)
        param.set_value(batch_data, batch=True)
        assert param.is_batched

        # Then set non-batched value
        single_data = np.random.randn(3, 4)
        param.value = single_data
        assert not param.is_batched
        assert param.batch_shape == ()

    def test_set_value_batch_invalid_shape(self):
        """Test that set_value raises error for invalid shape."""
        param = cp.Parameter((3, 4))
        # Wrong problem dimensions
        bad_data = np.random.randn(10, 4, 3)

        with pytest.raises(ValueError, match="problem shape"):
            param.set_value(bad_data, batch=True)

    def test_set_value_batch_insufficient_dims(self):
        """Test error when value has fewer dims than parameter."""
        param = cp.Parameter((3, 4))
        # Only 1D but parameter is 2D
        bad_data = np.random.randn(10)

        with pytest.raises(ValueError, match="dimensions"):
            param.set_value(bad_data, batch=True)

    def test_set_value_none_clears_batch(self):
        """Test that setting value to None clears batch_shape."""
        param = cp.Parameter((3, 4))
        batch_data = np.random.randn(10, 3, 4)
        param.set_value(batch_data, batch=True)

        param.set_value(None)

        assert param.value is None
        assert param.batch_shape == ()
        assert not param.is_batched


class TestBatchedLeafProperties:
    """Tests for batch_shape and is_batched on Leaf classes."""

    def test_variable_batch_shape_default(self):
        """Test that Variable has empty batch_shape by default."""
        var = cp.Variable((3, 4))
        assert var.batch_shape == ()
        assert not var.is_batched

    def test_constant_batch_shape_default(self):
        """Test that Constant has empty batch_shape by default."""
        const = cp.Constant(np.random.randn(3, 4))
        assert const.batch_shape == ()
        assert not const.is_batched

    def test_parameter_batch_shape_default(self):
        """Test that Parameter has empty batch_shape by default."""
        param = cp.Parameter((3, 4))
        param.value = np.random.randn(3, 4)
        assert param.batch_shape == ()
        assert not param.is_batched


class TestProblemBatchShape:
    """Tests for Problem batch shape computation."""

    def test_problem_no_batched_params(self):
        """Test _compute_batch_shape with no batched parameters."""
        x = cp.Variable(3)
        param = cp.Parameter(3)
        param.value = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(param @ x)))

        assert prob._compute_batch_shape() == ()

    def test_problem_single_batched_param(self):
        """Test _compute_batch_shape with one batched parameter."""
        x = cp.Variable(3)
        param = cp.Parameter(3)
        param.set_value(np.random.randn(10, 3), batch=True)

        prob = cp.Problem(cp.Minimize(cp.sum(param @ x)))

        assert prob._compute_batch_shape() == (10,)

    def test_problem_multiple_batched_params_same_shape(self):
        """Test _compute_batch_shape with multiple batched params, same batch shape."""
        x = cp.Variable(3)
        A = cp.Parameter((3, 3))
        b = cp.Parameter(3)

        A.set_value(np.random.randn(10, 3, 3), batch=True)
        b.set_value(np.random.randn(10, 3), batch=True)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        assert prob._compute_batch_shape() == (10,)

    def test_problem_batched_params_broadcasting(self):
        """Test _compute_batch_shape with broadcastable batch shapes."""
        x = cp.Variable(3)
        A = cp.Parameter((3, 3))
        b = cp.Parameter(3)

        # A has shape (5, 1, 3, 3), b has shape (1, 10, 3)
        # Should broadcast to (5, 10)
        A.set_value(np.random.randn(5, 1, 3, 3), batch=True)
        b.set_value(np.random.randn(1, 10, 3), batch=True)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        assert prob._compute_batch_shape() == (5, 10)

    def test_problem_batched_params_incompatible(self):
        """Test _compute_batch_shape raises error for incompatible shapes."""
        x = cp.Variable(3)
        A = cp.Parameter((3, 3))
        b = cp.Parameter(3)

        # A has shape (5, 3, 3), b has shape (7, 3)
        # These are not broadcastable
        A.set_value(np.random.randn(5, 3, 3), batch=True)
        b.set_value(np.random.randn(7, 3), batch=True)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        with pytest.raises(ValueError, match="Incompatible batch shapes"):
            prob._compute_batch_shape()

    def test_problem_mixed_batched_non_batched(self):
        """Test _compute_batch_shape with mixed batched and non-batched params."""
        x = cp.Variable(3)
        A = cp.Parameter((3, 3))
        b = cp.Parameter(3)
        c = cp.Parameter()

        # A is batched, b and c are not
        A.set_value(np.random.randn(10, 3, 3), batch=True)
        b.value = np.random.randn(3)
        c.value = 1.0

        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b) + c))

        # Only A is batched, so batch shape is (10,)
        assert prob._compute_batch_shape() == (10,)


class TestSolverBatchCapable:
    """Tests for the BATCH_CAPABLE solver flag."""

    def test_solver_batch_capable_default_false(self):
        """Test that Solver.BATCH_CAPABLE is False by default."""
        from cvxpy.reductions.solvers.solver import Solver
        assert Solver.BATCH_CAPABLE is False


class TestBatchedCanonInterface:
    """Tests for batched canonInterface functions."""

    def test_get_parameter_vector_batched_1d(self):
        """Test get_parameter_vector with 1D batch shape."""
        from cvxpy.cvxcore.python import canonInterface
        from cvxpy.lin_ops.lin_op import CONSTANT_ID

        batch_shape = (10,)
        param_size = 6  # e.g., a (2, 3) parameter

        # Mock param_id_to_col: one param at col 0, constant at col 6
        param_id_to_col = {1: 0, CONSTANT_ID: 6}
        param_id_to_size = {1: 6}

        # Batched values: shape (10, 2, 3)
        batched_values = {1: np.random.randn(10, 2, 3)}

        def param_value_fn(idx):
            return batched_values[idx]

        param_vec = canonInterface.get_parameter_vector(
            param_size, param_id_to_col, param_id_to_size,
            param_value_fn, batch_shape=batch_shape
        )

        # Should be shape (param_size + 1, *batch_shape) = (7, 10)
        assert param_vec.shape == (7, 10)
        # Constant offset should be 1
        assert np.all(param_vec[6, :] == 1)

    def test_get_parameter_vector_batched_2d(self):
        """Test get_parameter_vector with 2D batch shape."""
        from cvxpy.cvxcore.python import canonInterface
        from cvxpy.lin_ops.lin_op import CONSTANT_ID

        batch_shape = (5, 4)
        param_size = 3

        param_id_to_col = {1: 0, CONSTANT_ID: 3}
        param_id_to_size = {1: 3}

        # Batched values: shape (5, 4, 3)
        batched_values = {1: np.random.randn(5, 4, 3)}

        def param_value_fn(idx):
            return batched_values[idx]

        param_vec = canonInterface.get_parameter_vector(
            param_size, param_id_to_col, param_id_to_size,
            param_value_fn, batch_shape=batch_shape
        )

        # Should be shape (param_size + 1, *batch_shape) = (4, 5, 4)
        assert param_vec.shape == (4, 5, 4)
        # Constant offset should be 1
        assert np.all(param_vec[3, :, :] == 1)


class TestBatchedApplyParameters:
    """Tests for ParamConeProg.apply_parameters with batch_shape."""

    def test_apply_parameters_batched_matches_single(self):
        """Test that batched results match individual apply_parameters calls."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0])

        # First set non-batched values to get problem data
        c_vals = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        c.value = c_vals[0]

        # Get problem data (this triggers canonicalization and caches ParamConeProg)
        prob.get_problem_data(solver=cp.SCS)

        # ParamConeProg is cached in prob._cache.param_prog
        param_cone_prog = prob._cache.param_prog
        if param_cone_prog is None:
            pytest.skip("Could not find ParamConeProg")

        # Now set batched values and test apply_parameters with batch_shape
        batch_shape = (3,)
        c.set_value(c_vals, batch=True)

        # Get batched result
        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        # Compare with individual calls
        for i in range(3):
            c.value = c_vals[i]
            q_single, d_single, A_single, b_single = param_cone_prog.apply_parameters()

            np.testing.assert_allclose(q_batch[i], q_single, rtol=1e-10)
            np.testing.assert_allclose(d_batch[i], d_single, rtol=1e-10)
            np.testing.assert_allclose(A_batch[i], A_single.toarray(), rtol=1e-10)
            np.testing.assert_allclose(b_batch[i], b_single, rtol=1e-10)

    def test_apply_parameters_batched_2d_batch(self):
        """Test apply_parameters with 2D batch shape."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0])

        # First set non-batched values
        c.value = np.array([1.0, 2.0])

        # Get problem data (triggers canonicalization)
        prob.get_problem_data(solver=cp.SCS)

        param_cone_prog = prob._cache.param_prog
        if param_cone_prog is None:
            pytest.skip("Could not find ParamConeProg")

        # Set 2D batched values
        batch_shape = (3, 4)
        c_vals = np.random.randn(3, 4, n)
        c.set_value(c_vals, batch=True)

        # Get batched result
        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        # Check shapes
        assert q_batch.shape == (3, 4, param_cone_prog.x.size)
        assert d_batch.shape == (3, 4)
        assert A_batch.shape == (3, 4, param_cone_prog.constr_size, param_cone_prog.x.size)
        assert b_batch.shape == (3, 4, param_cone_prog.constr_size)
