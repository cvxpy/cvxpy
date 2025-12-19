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
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import (
    CLARABEL,
    dims_to_solver_cones,
)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


class BatchSolution:
    """Container for batched solution results from BATCH_CLARABEL."""

    def __init__(self, statuses, obj_vals, xs, zs, solve_times, iterations):
        self.statuses = statuses
        self.obj_vals = obj_vals
        self.xs = xs
        self.zs = zs
        self.solve_times = solve_times
        self.iterations = iterations

    @property
    def status(self):
        return self.statuses

    @property
    def obj_val(self):
        return self.obj_vals

    @property
    def x(self):
        return self.xs

    @property
    def z(self):
        return self.zs

    @property
    def solve_time(self):
        return np.sum(self.solve_times)


class BATCH_CLARABEL(CLARABEL):
    """Test-only batch-capable solver that wraps CLARABEL in a loop."""

    BATCH_CAPABLE = True

    def name(self):
        return 'BATCH_CLARABEL'

    def solve_via_data(self, data, warm_start: bool, verbose: bool,
                       solver_opts, solver_cache=None):
        """Solve batched problem by looping over instances."""
        import clarabel

        batch_shape = data.get(s.BATCH_SHAPE)

        if batch_shape is None or len(batch_shape) == 0:
            # Non-batch: use parent implementation
            return super().solve_via_data(
                data, warm_start, verbose, solver_opts, solver_cache
            )

        # Batch mode
        q_batch = data[s.C]
        b_batch = data[s.B]
        A_batch = data[s.A]
        P_batch = data.get(s.P)

        cone_dims = data[ConicSolver.DIMS]
        cones = dims_to_solver_cones(cone_dims)

        n_batch = int(np.prod(batch_shape))
        n_var = q_batch.shape[-1]
        n_constr = b_batch.shape[-1]

        q_flat = q_batch.reshape(n_batch, n_var)
        b_flat = b_batch.reshape(n_batch, n_constr)
        A_flat = A_batch.reshape(n_batch, n_constr, n_var)
        P_flat = P_batch.reshape(n_batch, n_var, n_var) if P_batch is not None else None

        settings = self.parse_solver_opts(verbose, solver_opts)

        statuses = []
        obj_vals = np.full(n_batch, np.nan)
        xs = np.full((n_batch, n_var), np.nan)
        zs = np.full((n_batch, n_constr), np.nan)
        solve_times = np.zeros(n_batch)
        iterations = np.zeros(n_batch, dtype=int)

        for i in range(n_batch):
            q_i = q_flat[i]
            b_i = b_flat[i]
            A_i = sp.csc_array(A_flat[i])
            P_i = sp.triu(sp.csc_array(P_flat[i])).tocsc() if P_flat is not None \
                else sp.csc_array((n_var, n_var))

            solver = clarabel.DefaultSolver(P_i, q_i, A_i, b_i, cones, settings)
            result = solver.solve()

            statuses.append(str(result.status))
            obj_vals[i] = result.obj_val
            if result.x is not None:
                xs[i] = result.x
            if result.z is not None:
                zs[i] = result.z
            solve_times[i] = result.solve_time
            iterations[i] = result.iterations

        statuses = np.array(statuses).reshape(batch_shape)
        obj_vals = obj_vals.reshape(batch_shape)
        xs = xs.reshape(batch_shape + (n_var,))
        zs = zs.reshape(batch_shape + (n_constr,))
        solve_times = solve_times.reshape(batch_shape)
        iterations = iterations.reshape(batch_shape)

        return BatchSolution(statuses, obj_vals, xs, zs, solve_times, iterations)

    def invert(self, solution, inverse_data):
        """Invert batched solution."""
        batch_shape = inverse_data.get(s.BATCH_SHAPE)

        if batch_shape is None or len(batch_shape) == 0:
            return super().invert(solution, inverse_data)

        status_map = self.STATUS_MAP.copy()
        statuses = np.vectorize(lambda st: status_map.get(st, s.SOLVER_ERROR))(
            solution.statuses
        )

        opt_vals = solution.obj_vals + inverse_data[s.OFFSET]

        primal_vars = {inverse_data[self.VAR_ID]: solution.xs}

        # Construct batched dual_vars
        dual_vars = {}
        zs = solution.zs  # shape: batch_shape + (n_constr,)
        cone_dims = inverse_data[ConicSolver.DIMS]

        # Handle equality constraints (zero cone)
        eq_constrs = inverse_data.get(self.EQ_CONSTR, [])
        offset = 0
        for constr in eq_constrs:
            size = constr.size
            dual_vars[constr.id] = zs[..., offset:offset+size]
            offset += size

        # Handle inequality constraints (nonneg cone + other cones)
        neq_constrs = inverse_data.get(self.NEQ_CONSTR, [])
        # offset continues from zero cone size
        offset = cone_dims.zero
        for constr in neq_constrs:
            size = constr.size
            dual_vars[constr.id] = zs[..., offset:offset+size]
            offset += size

        attr = {
            s.SOLVE_TIME: solution.solve_time,
            s.NUM_ITERS: solution.iterations,
            s.BATCH_SHAPE: batch_shape,
        }

        return Solution(statuses, opt_vals, primal_vars, dual_vars, attr)


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
        assert param_cone_prog is not None, "ParamConeProg should be cached"

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
        assert param_cone_prog is not None, "ParamConeProg should be cached"

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


class TestBatchedSolveEndToEnd:
    """End-to-end tests for problem.solve() with batched parameters."""

    def test_solve_with_batched_params(self):
        """Test that solve() auto-detects batch mode."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        # Set batched values
        batch_size = 5
        np.random.seed(42)
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        # Solve with BATCH_CLARABEL (pass instance)
        prob.solve(solver=BATCH_CLARABEL())

        # Check results
        assert prob.batch_shape == (batch_size,)
        assert x.value.shape == (batch_size, n)
        assert prob.value.shape == (batch_size,)

    def test_batched_solve_matches_individual(self):
        """Test batched results match individual solves."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        np.random.seed(42)
        c_vals = np.random.randn(batch_size, n)

        # Individual solves
        individual_values = []
        individual_x = []
        for i in range(batch_size):
            c.value = c_vals[i]
            prob.solve(solver=cp.CLARABEL)
            individual_values.append(prob.value)
            individual_x.append(x.value.copy())

        # Batched solve
        c.set_value(c_vals, batch=True)
        prob.solve(solver=BATCH_CLARABEL())

        # Compare
        np.testing.assert_allclose(prob.value, individual_values, rtol=1e-5)
        np.testing.assert_allclose(x.value, np.array(individual_x), rtol=1e-5)

    def test_non_batch_solver_raises_error(self):
        """Test that non-batch-capable solver raises SolverError."""
        x = cp.Variable(3)
        c = cp.Parameter(3)
        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0])

        c.set_value(np.random.randn(5, 3), batch=True)

        with pytest.raises(cp.error.SolverError, match="batch mode"):
            prob.solve(solver=cp.CLARABEL)

    def test_batched_qp_matches_individual(self):
        """Test batched QP (quadratic objective) matches individual solves."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        # Quadratic objective using sum_squares (DPP-compliant)
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(x) + c @ x),
            [x >= -1, x <= 1]
        )

        batch_size = 4
        np.random.seed(123)
        c_vals = np.random.randn(batch_size, n)

        # Individual solves
        individual_values = []
        individual_x = []
        for i in range(batch_size):
            c.value = c_vals[i]
            prob.solve(solver=cp.CLARABEL, use_quad_obj=True)
            individual_values.append(prob.value)
            individual_x.append(x.value.copy())

        # Batched solve
        c.set_value(c_vals, batch=True)
        prob.solve(solver=BATCH_CLARABEL(), use_quad_obj=True)

        # Compare
        np.testing.assert_allclose(prob.value, individual_values, rtol=1e-4)
        np.testing.assert_allclose(x.value, np.array(individual_x), rtol=1e-4)


class TestBatchedValueAccess:
    """Tests for .value behavior with batched variables/parameters."""

    def test_variable_value_batched_shape(self):
        """Variable.value returns batched array after batched solve."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        prob.solve(solver=BATCH_CLARABEL())

        # Variable should have batched values
        assert x.value.shape == (batch_size, n)
        assert x.is_batched
        assert x.batch_shape == (batch_size,)

    def test_problem_value_batched(self):
        """Problem.value returns batched optimal values."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        prob.solve(solver=BATCH_CLARABEL())

        # Problem.value should be batched
        assert prob.value.shape == (batch_size,)

    def test_objective_value_raises_when_batched(self):
        """Objective.value raises BatchedValueError when variables are batched."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        prob.solve(solver=BATCH_CLARABEL())

        # Objective.value should raise
        with pytest.raises(cp.error.BatchedValueError):
            _ = prob.objective.value

    def test_elementwise_expression_value_batched(self):
        """Elementwise expressions return batched values correctly."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        prob.solve(solver=BATCH_CLARABEL())

        # Elementwise expressions should return batched values
        expr = x + 1
        val = expr.value
        assert val.shape == (batch_size, n)
        np.testing.assert_allclose(val, x.value + 1)

        # Test other elementwise operations
        expr2 = cp.square(x)
        val2 = expr2.value
        assert val2.shape == (batch_size, n)
        np.testing.assert_allclose(val2, np.square(x.value))

        expr3 = cp.abs(x) + cp.square(x)
        val3 = expr3.value
        assert val3.shape == (batch_size, n)
        np.testing.assert_allclose(val3, np.abs(x.value) + np.square(x.value))

    def test_parameter_value_batched(self):
        """Parameter.value returns batched array when set with batch=True."""
        n = 3
        c = cp.Parameter(n)

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        assert c.value.shape == (batch_size, n)
        assert c.is_batched
        assert c.batch_shape == (batch_size,)

    def test_dual_variable_value_batched(self):
        """Dual variable values are batched after batched solve."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        constraint = x >= 0
        prob = cp.Problem(cp.Minimize(c @ x), [constraint, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)
        c.set_value(c_vals, batch=True)

        prob.solve(solver=BATCH_CLARABEL())

        # Dual variable should have batched values
        dual_val = constraint.dual_value
        assert dual_val.shape == (batch_size, n)

    def test_non_batched_value_still_works(self):
        """Verify non-batched .value access still works normally."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        # Non-batched solve
        c.value = np.array([1.0, 2.0, 3.0])
        prob.solve(solver=cp.CLARABEL)

        # All .value accesses should work
        assert x.value.shape == (n,)
        assert not x.is_batched
        assert isinstance(prob.value, (int, float, np.floating))
        assert isinstance(prob.objective.value, (int, float, np.floating))

        # Expression value should also work
        expr = cp.sum(x)
        assert isinstance(expr.value, (int, float, np.floating))


class TestBatchedAtomEvaluation:
    """Comprehensive tests for atom .value evaluation with batched variables."""

    @pytest.fixture
    def batched_var_1d(self):
        """1D variable with batch dimension."""
        x = cp.Variable(4)
        x._value = np.arange(20).reshape(5, 4).astype(float)
        x._batch_shape = (5,)
        return x

    @pytest.fixture
    def batched_var_2d(self):
        """2D variable with batch dimension."""
        y = cp.Variable((3, 4))
        y._value = np.arange(60).reshape(5, 3, 4).astype(float)
        y._batch_shape = (5,)
        return y

    # === Elementwise atoms (should preserve shape, operate element-by-element) ===
    @pytest.mark.parametrize("atom_fn,np_fn", [
        (cp.abs, np.abs),
        (cp.square, np.square),
        (lambda x: cp.power(cp.abs(x) + 1, 2), lambda x: np.power(np.abs(x) + 1, 2)),
        (lambda x: cp.sqrt(cp.abs(x)), lambda x: np.sqrt(np.abs(x))),
        (cp.exp, np.exp),
        (lambda x: cp.log(cp.abs(x) + 1), lambda x: np.log(np.abs(x) + 1)),
        (cp.pos, lambda x: np.maximum(x, 0)),
        (cp.neg, lambda x: np.maximum(-x, 0)),
        (lambda x: cp.maximum(x, 0), lambda x: np.maximum(x, 0)),
        (lambda x: cp.minimum(x, 0), lambda x: np.minimum(x, 0)),
    ])
    def test_elementwise_atoms(self, batched_var_1d, atom_fn, np_fn):
        """Elementwise atoms preserve batch dimensions."""
        x = batched_var_1d
        expr = atom_fn(x)
        result = expr.value
        expected = np_fn(x.value)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # === Axis-based atoms with axis=None (reduce to batch shape) ===
    @pytest.mark.parametrize("atom_fn,np_fn", [
        (cp.sum, lambda x: np.sum(x, axis=-1)),
        (cp.max, lambda x: np.max(x, axis=-1)),
        (cp.min, lambda x: np.min(x, axis=-1)),
        (cp.norm, lambda x: np.linalg.norm(x, axis=-1)),
    ])
    def test_axis_atoms_reduce_all(self, batched_var_1d, atom_fn, np_fn):
        """Axis atoms with axis=None reduce over all problem dims."""
        x = batched_var_1d
        expr = atom_fn(x)
        result = expr.value
        expected = np_fn(x.value)
        assert result.shape == x.batch_shape, f"Expected {x.batch_shape}, got {result.shape}"
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # === Axis-based atoms with specific axis ===
    @pytest.mark.parametrize("axis,expected_shape", [
        (0, (5, 4)),  # Reduce over problem axis 0 -> batch + remaining
        (1, (5, 3)),  # Reduce over problem axis 1 -> batch + remaining
    ])
    def test_axis_atoms_specific_axis(self, batched_var_2d, axis, expected_shape):
        """Axis atoms with specific axis preserve batch dims."""
        y = batched_var_2d
        for atom_fn in [cp.sum, cp.max]:
            expr = atom_fn(y, axis=axis)
            result = expr.value
            assert result.shape == expected_shape, f"{atom_fn.__name__} axis={axis}"

    # === Affine operations ===
    @pytest.mark.parametrize("expr_fn,np_fn", [
        (lambda x: x + 1, lambda x: x + 1),
        (lambda x: 2 * x, lambda x: 2 * x),
        (lambda x: x + x, lambda x: x + x),
        (lambda x: -x, lambda x: -x),
        (lambda x: x / 2, lambda x: x / 2),
    ])
    def test_affine_operations(self, batched_var_1d, expr_fn, np_fn):
        """Affine operations preserve batch dimensions."""
        x = batched_var_1d
        result = expr_fn(x).value
        expected = np_fn(x.value)
        assert result.shape == expected.shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # === Compound expressions ===
    def test_compound_expression(self, batched_var_1d):
        """Compound expressions work with batched values."""
        x = batched_var_1d
        expr = cp.sum(cp.square(x)) + cp.max(cp.abs(x))
        result = expr.value
        expected = np.sum(np.square(x.value), axis=-1) + np.max(np.abs(x.value), axis=-1)
        assert result.shape == x.batch_shape
        np.testing.assert_allclose(result, expected, rtol=1e-10)

    # === Multi-dimensional batch ===
    def test_multi_batch_dims(self):
        """Test with multiple batch dimensions."""
        x = cp.Variable(3)
        x._value = np.arange(30).reshape(2, 5, 3).astype(float)
        x._batch_shape = (2, 5)

        result = cp.sum(x).value
        expected = np.sum(x.value, axis=-1)
        assert result.shape == (2, 5)
        np.testing.assert_allclose(result, expected)

        result2 = cp.square(x).value
        assert result2.shape == (2, 5, 3)

    # === Reshape/Transpose ===
    def test_reshape(self):
        """Reshape preserves batch dimensions."""
        x = cp.Variable(6)
        x._value = np.arange(30).reshape(5, 6).astype(float)
        x._batch_shape = (5,)

        result = cp.reshape(x, (2, 3), order='F').value
        assert result.shape == (5, 2, 3)
        # Verify each batch is reshaped correctly
        for i in range(5):
            expected = x.value[i].reshape(2, 3, order='F')
            np.testing.assert_allclose(result[i], expected)

    def test_transpose(self, batched_var_2d):
        """Transpose preserves batch dimensions."""
        y = batched_var_2d  # shape (5, 3, 4)
        result = y.T.value
        assert result.shape == (5, 4, 3)
        np.testing.assert_allclose(result, np.transpose(y.value, axes=(0, 2, 1)))

    def test_trace(self):
        """Trace preserves batch dimensions."""
        y = cp.Variable((3, 3))
        y._value = np.arange(45).reshape(5, 3, 3).astype(float)
        y._batch_shape = (5,)

        result = cp.trace(y).value
        assert result.shape == (5,)
        expected = np.trace(y.value, axis1=1, axis2=2)
        np.testing.assert_allclose(result, expected)

    # === Indexing/slicing ===
    def test_indexing_1d(self, batched_var_1d):
        """Indexing preserves batch dimensions for 1D variables."""
        x = batched_var_1d
        # Single index
        result = x[0].value
        assert result.shape == (5,)
        np.testing.assert_allclose(result, x.value[:, 0])
        # Slice
        result = x[1:3].value
        assert result.shape == (5, 2)
        np.testing.assert_allclose(result, x.value[:, 1:3])

    def test_indexing_2d(self, batched_var_2d):
        """Indexing preserves batch dimensions for 2D variables."""
        y = batched_var_2d
        # Row index
        result = y[0, :].value
        assert result.shape == (5, 4)
        np.testing.assert_allclose(result, y.value[:, 0, :])
        # Column index
        result = y[:, 0].value
        assert result.shape == (5, 3)
        np.testing.assert_allclose(result, y.value[:, :, 0])
        # Slice both
        result = y[1:3, 2:4].value
        assert result.shape == (5, 2, 2)
        np.testing.assert_allclose(result, y.value[:, 1:3, 2:4])

    # === Non-batched still works ===
    def test_non_batched_unchanged(self):
        """Non-batched evaluation still works correctly."""
        x = cp.Variable(3)
        x._value = np.array([1.0, 2.0, 3.0])
        x._batch_shape = ()

        assert cp.sum(x).value == 6.0
        assert cp.max(x).value == 3.0
        np.testing.assert_array_equal(cp.square(x).value, [1.0, 4.0, 9.0])

    # === Matrix atoms: diag, upper_tri ===
    def test_diag_mat(self):
        """diag(matrix) extracts diagonal with batch dimensions preserved."""
        y = cp.Variable((3, 3))
        y._value = np.arange(45).reshape(5, 3, 3).astype(float)
        y._batch_shape = (5,)

        # Extract main diagonal
        result = cp.diag(y).value
        assert result.shape == (5, 3), f"Expected (5, 3), got {result.shape}"
        expected = np.diagonal(y.value, axis1=-2, axis2=-1)
        np.testing.assert_allclose(result, expected)

    def test_diag_mat_offset(self):
        """diag(matrix, k) extracts k-th diagonal with batch dimensions."""
        y = cp.Variable((4, 4))
        y._value = np.arange(80).reshape(5, 4, 4).astype(float)
        y._batch_shape = (5,)

        # Superdiagonal (k=1)
        result = cp.diag(y, k=1).value
        assert result.shape == (5, 3)
        expected = np.diagonal(y.value, offset=1, axis1=-2, axis2=-1)
        np.testing.assert_allclose(result, expected)

        # Subdiagonal (k=-1)
        result = cp.diag(y, k=-1).value
        assert result.shape == (5, 3)
        expected = np.diagonal(y.value, offset=-1, axis1=-2, axis2=-1)
        np.testing.assert_allclose(result, expected)

    def test_diag_vec(self):
        """diag(vector) creates diagonal matrix with batch dimensions."""
        x = cp.Variable(3)
        x._value = np.arange(15).reshape(5, 3).astype(float)
        x._batch_shape = (5,)

        result = cp.diag(x).value
        assert result.shape == (5, 3, 3), f"Expected (5, 3, 3), got {result.shape}"
        # Verify each batch element
        for i in range(5):
            expected = np.diag(x.value[i])
            np.testing.assert_allclose(result[i], expected)

    def test_diag_vec_offset(self):
        """diag(vector, k) creates k-th diagonal matrix with batch dimensions."""
        x = cp.Variable(3)
        x._value = np.arange(15).reshape(5, 3).astype(float)
        x._batch_shape = (5,)

        # Superdiagonal
        result = cp.diag(x, k=1).value
        assert result.shape == (5, 4, 4)
        for i in range(5):
            expected = np.diag(x.value[i], k=1)
            np.testing.assert_allclose(result[i], expected)

        # Subdiagonal
        result = cp.diag(x, k=-1).value
        assert result.shape == (5, 4, 4)
        for i in range(5):
            expected = np.diag(x.value[i], k=-1)
            np.testing.assert_allclose(result[i], expected)

    def test_upper_tri(self):
        """upper_tri preserves batch dimensions."""
        y = cp.Variable((4, 4))
        y._value = np.arange(80).reshape(5, 4, 4).astype(float)
        y._batch_shape = (5,)

        result = cp.upper_tri(y).value
        # Upper triangular entries: n*(n-1)/2 = 4*3/2 = 6
        # Shape from upper_tri is (6, 1), so batched is (5, 6, 1)
        assert result.shape == (5, 6, 1) or result.shape == (5, 6), \
            f"Expected (5, 6, 1) or (5, 6), got {result.shape}"
        # Verify values
        upper_idx = np.triu_indices(n=4, k=1)
        expected = y.value[..., upper_idx[0], upper_idx[1]]
        np.testing.assert_allclose(result.squeeze(), expected)

    # === Additional axis-based atoms ===
    def test_cumsum(self):
        """cumsum preserves batch dimensions."""
        y = cp.Variable((3, 4))
        y._value = np.arange(60).reshape(5, 3, 4).astype(float)
        y._batch_shape = (5,)

        # cumsum along axis 0 (rows)
        result = cp.cumsum(y, axis=0).value
        assert result.shape == (5, 3, 4)
        expected = np.cumsum(y.value, axis=1)  # axis offset by batch dim
        np.testing.assert_allclose(result, expected)

        # cumsum along axis 1 (cols)
        result = cp.cumsum(y, axis=1).value
        assert result.shape == (5, 3, 4)
        expected = np.cumsum(y.value, axis=2)
        np.testing.assert_allclose(result, expected)

    def test_cummax(self):
        """cummax preserves batch dimensions."""
        x = cp.Variable(4)
        x._value = np.array([[3, 1, 4, 1], [2, 7, 1, 8], [5, 9, 2, 6]]).astype(float)
        x._batch_shape = (3,)

        result = cp.atoms.cummax(x).value
        assert result.shape == (3, 4)
        expected = np.maximum.accumulate(x.value, axis=1)
        np.testing.assert_allclose(result, expected)

    def test_norm1(self):
        """norm1 preserves batch dimensions."""
        x = cp.Variable(4)
        x._value = np.arange(20).reshape(5, 4).astype(float)
        x._batch_shape = (5,)

        # axis=None reduces to batch shape
        result = cp.norm1(x).value
        assert result.shape == (5,)
        expected = np.linalg.norm(x.value, 1, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_norm_inf(self):
        """norm_inf preserves batch dimensions."""
        x = cp.Variable(4)
        x._value = np.arange(20).reshape(5, 4).astype(float)
        x._batch_shape = (5,)

        result = cp.norm_inf(x).value
        assert result.shape == (5,)
        expected = np.linalg.norm(x.value, np.inf, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_log_sum_exp(self):
        """log_sum_exp preserves batch dimensions."""
        x = cp.Variable(4)
        x._value = np.arange(20).reshape(5, 4).astype(float) / 10  # Scale down
        x._batch_shape = (5,)

        from scipy.special import logsumexp
        result = cp.log_sum_exp(x).value
        assert result.shape == (5,)
        expected = logsumexp(x.value, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_prod(self):
        """prod preserves batch dimensions."""
        x = cp.Variable(3)
        x._value = np.array([[1, 2, 3], [2, 3, 4], [1, 1, 2]]).astype(float)
        x._batch_shape = (3,)

        result = cp.prod(x).value
        assert result.shape == (3,)
        expected = np.prod(x.value, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_cumprod(self):
        """cumprod preserves batch dimensions."""
        x = cp.Variable(4)
        x._value = np.array([[1, 2, 3, 4], [2, 2, 2, 2]]).astype(float)
        x._batch_shape = (2,)

        result = cp.atoms.cumprod(x).value
        assert result.shape == (2, 4)
        expected = np.cumprod(x.value, axis=-1)
        np.testing.assert_allclose(result, expected)

    def test_lambda_max_batched(self):
        """lambda_max preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, n))
        # Create batch of symmetric positive definite matrices
        vals = np.random.randn(batch, n, n)
        vals = vals @ np.swapaxes(vals, -2, -1) + np.eye(n)  # Make symmetric PD
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.lambda_max(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            expected_i = np.linalg.eigvalsh(vals[i])[-1]
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_lambda_min_batched(self):
        """lambda_min preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, n))
        # Create batch of symmetric positive definite matrices
        vals = np.random.randn(batch, n, n)
        vals = vals @ np.swapaxes(vals, -2, -1) + np.eye(n)  # Make symmetric PD
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.lambda_min(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            expected_i = np.linalg.eigvalsh(vals[i])[0]
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_log_det_batched(self):
        """log_det preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, n))
        # Create batch of symmetric positive definite matrices
        vals = np.random.randn(batch, n, n)
        vals = vals @ np.swapaxes(vals, -2, -1) + 2 * np.eye(n)  # Make symmetric PD
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.log_det(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            _, expected_i = np.linalg.slogdet(vals[i])
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_quad_form_batched(self):
        """quad_form preserves batch dimensions."""
        batch = 5
        n = 4

        x = cp.Variable(n)
        P = np.random.randn(n, n)
        P = P @ P.T + np.eye(n)  # Make symmetric PD

        x_vals = np.random.randn(batch, n)
        x._value = x_vals
        x._batch_shape = (batch,)

        result = cp.quad_form(x, P).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            expected_i = x_vals[i] @ P @ x_vals[i]
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_sum_largest_batched(self):
        """sum_largest preserves batch dimensions."""
        batch = 5
        n = 6

        x = cp.Variable(n)
        x._value = np.random.randn(batch, n)
        x._batch_shape = (batch,)

        k = 3
        result = cp.sum_largest(x, k).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            sorted_vals = np.sort(x.value[i])[::-1]
            expected_i = np.sum(sorted_vals[:k])
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_sum_smallest_batched(self):
        """sum_smallest preserves batch dimensions."""
        batch = 5
        n = 6

        x = cp.Variable(n)
        x._value = np.random.randn(batch, n)
        x._batch_shape = (batch,)

        k = 3
        result = cp.sum_smallest(x, k).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            sorted_vals = np.sort(x.value[i])
            expected_i = np.sum(sorted_vals[:k])
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_quad_over_lin_batched(self):
        """quad_over_lin preserves batch dimensions."""
        batch = 5
        n = 4

        x = cp.Variable(n)
        y = cp.Variable()

        x._value = np.random.randn(batch, n)
        y._value = np.abs(np.random.randn(batch)) + 1  # positive values
        x._batch_shape = (batch,)
        y._batch_shape = (batch,)

        result = cp.quad_over_lin(x, y).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            expected_i = np.sum(np.square(x.value[i])) / y.value[i]
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_tr_inv_batched(self):
        """tr_inv preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, n))
        # Create batch of symmetric positive definite matrices
        vals = np.random.randn(batch, n, n)
        vals = vals @ np.swapaxes(vals, -2, -1) + 2 * np.eye(n)  # Make symmetric PD
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.tr_inv(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            eigs = np.linalg.eigvalsh(vals[i])
            expected_i = np.sum(1.0 / eigs)
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_matrix_frac_batched(self):
        """matrix_frac preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, 1))
        P = cp.Variable((n, n))

        # Create batch values
        x_vals = np.random.randn(batch, n, 1)
        p_vals = np.random.randn(batch, n, n)
        p_vals = p_vals @ np.swapaxes(p_vals, -2, -1) + 2 * np.eye(n)  # Make symmetric PD

        X._value = x_vals
        P._value = p_vals
        X._batch_shape = (batch,)
        P._batch_shape = (batch,)

        result = cp.matrix_frac(X, P).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            P_inv = np.linalg.inv(p_vals[i])
            expected_i = (x_vals[i].T @ P_inv @ x_vals[i]).item()
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-9)

    def test_sigma_max_batched(self):
        """sigma_max preserves batch dimensions."""
        batch = 5
        m, n = 4, 3

        X = cp.Variable((m, n))
        vals = np.random.randn(batch, m, n)
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.sigma_max(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            s = np.linalg.svd(vals[i], compute_uv=False)
            expected_i = s[0]  # Largest singular value
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_norm_nuc_batched(self):
        """norm_nuc preserves batch dimensions."""
        batch = 5
        m, n = 4, 3

        X = cp.Variable((m, n))
        vals = np.random.randn(batch, m, n)
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.normNuc(X).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            s = np.linalg.svd(vals[i], compute_uv=False)
            expected_i = np.sum(s)
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_eye_minus_inv_batched(self):
        """eye_minus_inv preserves batch dimensions."""
        batch = 5
        n = 4

        X = cp.Variable((n, n))
        # Create batch of matrices with spectral radius < 1
        vals = np.random.randn(batch, n, n) * 0.1  # Small values for convergence
        X._value = vals
        X._batch_shape = (batch,)

        result = cp.eye_minus_inv(X).value
        assert result.shape == (batch, n, n)

        # Verify against loop over batches
        for i in range(batch):
            expected_i = np.linalg.inv(np.eye(n) - vals[i])
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)

    def test_dotsort_batched(self):
        """dotsort preserves batch dimensions."""
        batch = 5
        n = 6

        x = cp.Variable(n)
        w = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # Sum of 3 largest

        x._value = np.random.randn(batch, n)
        x._batch_shape = (batch,)

        result = cp.dotsort(x, w).value
        assert result.shape == (batch,)

        # Verify against loop over batches
        for i in range(batch):
            x_sorted = np.sort(x.value[i])
            w_sorted = np.sort(w)
            expected_i = x_sorted @ w_sorted
            np.testing.assert_allclose(result[i], expected_i, rtol=1e-10)
