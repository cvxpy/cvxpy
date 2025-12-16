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

        batch_shape = data.get('batch_shape')

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
        from cvxpy.reductions.solution import Solution

        batch_shape = inverse_data.get('batch_shape')

        if batch_shape is None or len(batch_shape) == 0:
            return super().invert(solution, inverse_data)

        status_map = self.STATUS_MAP.copy()
        statuses = np.vectorize(lambda st: status_map.get(st, s.SOLVER_ERROR))(
            solution.statuses
        )

        opt_vals = solution.obj_vals + inverse_data[s.OFFSET]

        primal_vars = {inverse_data[self.VAR_ID]: solution.xs}

        attr = {
            s.SOLVE_TIME: solution.solve_time,
            s.NUM_ITERS: solution.iterations,
            'batch_shape': batch_shape,
        }

        return Solution(statuses, opt_vals, primal_vars, {}, attr)


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


class TestBatchClarabelSolver:
    """End-to-end tests for BATCH_CLARABEL solver."""

    def test_batch_clarabel_lp_matches_individual_solves(self):
        """Test that BATCH_CLARABEL results match individual CLARABEL solves."""
        n = 3
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        # Generate batch of cost vectors
        batch_size = 5
        np.random.seed(42)
        c_vals = np.random.randn(batch_size, n)

        # Get param_cone_prog first
        c.value = c_vals[0]
        prob.get_problem_data(solver=cp.CLARABEL)
        param_cone_prog = prob._cache.param_prog
        assert param_cone_prog is not None

        # Individual solves using CLARABEL directly on param_cone_prog data
        clarabel_solver = CLARABEL()
        individual_obj_vals = []
        individual_x_vals = []
        for i in range(batch_size):
            c.value = c_vals[i]
            q, d, A, b = param_cone_prog.apply_parameters()
            data = {
                s.C: q,
                s.B: b,
                s.A: A,
                ConicSolver.DIMS: param_cone_prog.cone_dims,
            }
            result = clarabel_solver.solve_via_data(data, False, False, {})
            individual_obj_vals.append(result.obj_val + d)
            individual_x_vals.append(result.x.copy())

        individual_obj_vals = np.array(individual_obj_vals)
        individual_x_vals = np.array(individual_x_vals)

        # Batched solve using BATCH_CLARABEL
        batch_shape = (batch_size,)
        c.set_value(c_vals, batch=True)

        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        data = {
            s.C: q_batch,
            s.B: b_batch,
            s.A: A_batch,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            'batch_shape': batch_shape,
        }

        batch_solver = BATCH_CLARABEL()
        batch_solution = batch_solver.solve_via_data(data, False, False, {})

        # Verify results match
        np.testing.assert_allclose(
            batch_solution.obj_vals + d_batch,
            individual_obj_vals,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            batch_solution.xs,
            individual_x_vals,
            rtol=1e-5
        )

    def test_batch_clarabel_qp_with_use_quad_obj(self):
        """Test BATCH_CLARABEL with quadratic objective (use_quad_obj=True)."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        # Use sum_squares which is DPP-compliant
        prob = cp.Problem(
            cp.Minimize(0.5 * cp.sum_squares(x) + c @ x),
            [x >= -1, x <= 1]
        )

        # Generate batch of parameters
        batch_size = 4
        np.random.seed(123)
        c_vals = np.random.randn(batch_size, n)

        # Get problem data first
        c.value = c_vals[0]
        prob.get_problem_data(solver=cp.CLARABEL)
        param_cone_prog = prob._cache.param_prog
        assert param_cone_prog is not None

        # Individual solves using CLARABEL directly
        clarabel_solver = CLARABEL()
        individual_obj_vals = []
        individual_x_vals = []
        for i in range(batch_size):
            c.value = c_vals[i]
            P, q, d, A, b = param_cone_prog.apply_parameters(quad_obj=True)
            data = {
                s.P: P,
                s.C: q,
                s.B: b,
                s.A: A,
                ConicSolver.DIMS: param_cone_prog.cone_dims,
            }
            result = clarabel_solver.solve_via_data(data, False, False, {})
            individual_obj_vals.append(result.obj_val + d)
            individual_x_vals.append(result.x.copy())

        individual_obj_vals = np.array(individual_obj_vals)
        individual_x_vals = np.array(individual_x_vals)

        # Batched solve
        batch_shape = (batch_size,)
        c.set_value(c_vals, batch=True)

        P_batch, q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape, quad_obj=True
        )

        data = {
            s.P: P_batch,
            s.C: q_batch,
            s.B: b_batch,
            s.A: A_batch,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            'batch_shape': batch_shape,
        }

        batch_solver = BATCH_CLARABEL()
        batch_solution = batch_solver.solve_via_data(data, False, False, {})

        # Verify results match
        np.testing.assert_allclose(
            batch_solution.obj_vals + d_batch,
            individual_obj_vals,
            rtol=1e-4
        )
        np.testing.assert_allclose(
            batch_solution.xs,
            individual_x_vals,
            rtol=1e-4
        )

    def test_batch_clarabel_2d_batch_shape(self):
        """Test BATCH_CLARABEL with 2D batch shape."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        # 2D batch shape
        batch_shape = (3, 4)
        np.random.seed(456)
        c_vals = np.random.randn(*batch_shape, n)

        # Get problem data first
        c.value = c_vals[0, 0]
        prob.get_problem_data(solver=cp.CLARABEL)
        param_cone_prog = prob._cache.param_prog
        assert param_cone_prog is not None

        # Individual solves using CLARABEL directly
        clarabel_solver = CLARABEL()
        individual_obj_vals = np.zeros(batch_shape)
        individual_x_vals = np.zeros(batch_shape + (param_cone_prog.x.size,))
        for i in range(batch_shape[0]):
            for j in range(batch_shape[1]):
                c.value = c_vals[i, j]
                q, d, A, b = param_cone_prog.apply_parameters()
                data = {
                    s.C: q,
                    s.B: b,
                    s.A: A,
                    ConicSolver.DIMS: param_cone_prog.cone_dims,
                }
                result = clarabel_solver.solve_via_data(data, False, False, {})
                individual_obj_vals[i, j] = result.obj_val + d
                individual_x_vals[i, j] = result.x

        # Batched solve
        c.set_value(c_vals, batch=True)

        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        data = {
            s.C: q_batch,
            s.B: b_batch,
            s.A: A_batch,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            'batch_shape': batch_shape,
        }

        batch_solver = BATCH_CLARABEL()
        batch_solution = batch_solver.solve_via_data(data, False, False, {})

        # Verify shapes
        assert batch_solution.obj_vals.shape == batch_shape
        assert batch_solution.xs.shape == batch_shape + (param_cone_prog.x.size,)

        # Verify results match
        np.testing.assert_allclose(
            batch_solution.obj_vals + d_batch,
            individual_obj_vals,
            rtol=1e-5
        )
        np.testing.assert_allclose(
            batch_solution.xs,
            individual_x_vals,
            rtol=1e-5
        )

    def test_batch_clarabel_all_optimal(self):
        """Test that all batch instances return optimal status."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 5
        c_vals = np.random.randn(batch_size, n)

        c.value = c_vals[0]
        prob.get_problem_data(solver=cp.CLARABEL)

        param_cone_prog = prob._cache.param_prog
        assert param_cone_prog is not None

        batch_shape = (batch_size,)
        c.set_value(c_vals, batch=True)

        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        data = {
            s.C: q_batch,
            s.B: b_batch,
            s.A: A_batch,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            'batch_shape': batch_shape,
        }

        solver = BATCH_CLARABEL()
        batch_solution = solver.solve_via_data(data, False, False, {})

        # All should be "Solved"
        assert all(status == "Solved" for status in batch_solution.statuses)

    def test_batch_clarabel_invert(self):
        """Test BATCH_CLARABEL invert method."""
        n = 2
        x = cp.Variable(n)
        c = cp.Parameter(n)

        prob = cp.Problem(cp.Minimize(c @ x), [x >= 0, cp.sum(x) == 1])

        batch_size = 3
        c_vals = np.random.randn(batch_size, n)

        c.value = c_vals[0]
        prob.get_problem_data(solver=cp.CLARABEL)

        param_cone_prog = prob._cache.param_prog
        assert param_cone_prog is not None

        batch_shape = (batch_size,)
        c.set_value(c_vals, batch=True)

        q_batch, d_batch, A_batch, b_batch = param_cone_prog.apply_parameters(
            batch_shape=batch_shape
        )

        data = {
            s.C: q_batch,
            s.B: b_batch,
            s.A: A_batch,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            'batch_shape': batch_shape,
        }

        solver = BATCH_CLARABEL()
        batch_solution = solver.solve_via_data(data, False, False, {})

        # Create inverse_data
        inverse_data = {
            solver.VAR_ID: x.id,
            ConicSolver.DIMS: param_cone_prog.cone_dims,
            s.OFFSET: d_batch,
            'batch_shape': batch_shape,
        }

        # Invert the solution
        solution = solver.invert(batch_solution, inverse_data)

        # Check that solution has batched values
        assert solution.opt_val.shape == batch_shape
        assert x.id in solution.primal_vars
        assert solution.primal_vars[x.id].shape == batch_shape + (n,)
