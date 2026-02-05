"""
Copyright 2025 CVXPY developers

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

import cvxpy as cp
from cvxpy.utilities.solver_context import SolverInfo
from cvxpy.utilities.values import (
    get_expr_value_if_supported,
    propagate_dual_values_to_constraints,
)

# ──────────────────────────────────────────────────────────────
#  Unit tests for get_expr_value_if_supported
# ──────────────────────────────────────────────────────────────

class TestGetExprValueIfSupported:
    """Unit tests for the get_expr_value_if_supported helper."""

    def _warm_start_context(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=True)

    def _no_warm_start_context(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=False)

    def test_returns_none_when_no_context(self):
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.exp(x)
        assert get_expr_value_if_supported(expr, None) is None

    def test_returns_none_when_not_supported(self):
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.exp(x)
        assert get_expr_value_if_supported(expr, self._no_warm_start_context()) is None

    def test_returns_value_when_supported(self):
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.exp(x)
        result = get_expr_value_if_supported(expr, self._warm_start_context())
        assert result is not None
        np.testing.assert_allclose(result, np.exp([1.0, 2.0, 3.0]))

    def test_returns_none_when_arg_has_no_value(self):
        x = cp.Variable(3)
        expr = cp.exp(x)
        assert get_expr_value_if_supported(expr, self._warm_start_context()) is None

    def test_correct_shape(self):
        x = cp.Variable((2, 3))
        x.value = np.arange(6.0).reshape(2, 3)
        expr = cp.abs(x)
        result = get_expr_value_if_supported(expr, self._warm_start_context())
        assert result is not None
        assert result.shape == (2, 3)
        np.testing.assert_allclose(result, np.abs(np.arange(6.0).reshape(2, 3)))

    def test_scalar_expr(self):
        x = cp.Variable()
        x.value = -5.0
        expr = cp.abs(x)
        result = get_expr_value_if_supported(expr, self._warm_start_context())
        assert result is not None
        np.testing.assert_allclose(result, np.array([5.0]))


# ──────────────────────────────────────────────────────────────
#  Integration tests: canonicalize with warm-start context
# ──────────────────────────────────────────────────────────────

class TestValuePropagation:
    """Integration tests verifying that value propagation sets aux var values
    during canonicalization when a warm-start context is provided."""

    def _warm_start_context(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=True)

    def test_exp_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
        x = cp.Variable(3)
        x.value = np.array([0.0, 1.0, -1.0])
        expr = cp.exp(x)
        t, _ = exp_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.exp([0.0, 1.0, -1.0]))

    def test_log_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.log_canon import log_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)
        t, _ = log_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.log([1.0, 2.0, 3.0]))

    def test_abs_canon_propagates_value(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
        x = cp.Variable(3)
        x.value = np.array([-1.0, 0.0, 2.0])
        expr = cp.abs(x)
        t, _ = abs_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([1.0, 0.0, 2.0]))

    def test_max_canon_propagates_value(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon
        x = cp.Variable((2, 3))
        x.value = np.array([[1.0, 5.0, 3.0], [4.0, 2.0, 6.0]])
        expr = cp.max(x)
        t, _ = max_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([6.0]))

    def test_maximum_canon_propagates_value(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.maximum_canon import maximum_canon
        x = cp.Variable(3)
        y = cp.Variable(3)
        x.value = np.array([1.0, 5.0, 3.0])
        y.value = np.array([4.0, 2.0, 6.0])
        expr = cp.maximum(x, y)
        t, _ = maximum_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([4.0, 5.0, 6.0]))

    def test_norm_inf_canon_propagates_value(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.norm_inf_canon import norm_inf_canon
        x = cp.Variable(3)
        x.value = np.array([-1.0, 3.0, -2.0])
        expr = cp.norm_inf(x)
        t, _ = norm_inf_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([3.0]))

    def test_power_exact_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.power_canon import power_exact_canon
        x = cp.Variable(3, pos=True)
        x.value = np.array([1.0, 4.0, 9.0])
        expr = cp.power(x, 0.5)
        t, _ = power_exact_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.sqrt([1.0, 4.0, 9.0]))

    def test_pnorm_exact_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.pnorm_canon import pnorm_exact_canon
        x = cp.Variable(3)
        x.value = np.array([3.0, 4.0, 0.0])
        expr = cp.pnorm(x, 2)
        t, _ = pnorm_exact_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array(5.0))


# ──────────────────────────────────────────────────────────────
#  Negative tests: no warm-start context
# ──────────────────────────────────────────────────────────────

class TestValuePropagationNegative:
    """Verify that aux vars have None values without warm-start context."""

    def test_exp_canon_no_value_without_context(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
        x = cp.Variable(3)
        x.value = np.array([0.0, 1.0, -1.0])
        expr = cp.exp(x)
        t, _ = exp_canon(expr, expr.args, solver_context=None)
        assert t.value is None

    def test_abs_canon_no_value_without_context(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
        x = cp.Variable(3)
        x.value = np.array([-1.0, 0.0, 2.0])
        expr = cp.abs(x)
        t, _ = abs_canon(expr, expr.args, solver_context=None)
        assert t.value is None

    def test_max_canon_no_value_without_warm_start(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 5.0, 3.0])
        expr = cp.max(x)
        ctx = SolverInfo(solver="TEST", supported_constraints=[],
                         supports_warm_start=False)
        t, _ = max_canon(expr, expr.args, solver_context=ctx)
        assert t.value is None

    def test_log_canon_no_value_without_context(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.log_canon import log_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)
        t, _ = log_canon(expr, expr.args, solver_context=None)
        assert t.value is None


# ──────────────────────────────────────────────────────────────
#  Unit tests for propagate_dual_values_to_constraints
# ──────────────────────────────────────────────────────────────

class TestPropagateDualValuesToConstraints:
    """Unit tests for the propagate_dual_values_to_constraints helper."""

    def _warm_ctx(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=True)

    def _no_warm_ctx(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=False)

    def test_no_context(self):
        """No-op when solver_context is None."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        con1 = (x >= 0)
        con2 = (x >= 0)
        # Should not raise, just cache new constraints.
        propagate_dual_values_to_constraints(expr, [con1], None)
        assert expr._cached_aux_constraints == [con1]
        propagate_dual_values_to_constraints(expr, [con2], None)
        assert expr._cached_aux_constraints == [con2]

    def test_no_cached_constraints(self):
        """First call: nothing to copy, just caches."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        con = (x >= 0)
        propagate_dual_values_to_constraints(expr, [con], self._warm_ctx())
        assert expr._cached_aux_constraints == [con]
        # dual variables should still be None
        assert con.dual_variables[0].value is None

    def test_mismatched_lengths(self):
        """Different constraint counts between old and new: skip copy, re-cache."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        old_con = (x >= 0)
        old_con.dual_variables[0].save_value(np.array([1.0, 2.0]))
        expr._cached_aux_constraints = [old_con]

        new_con1 = (x >= 0)
        new_con2 = (x <= 1)
        propagate_dual_values_to_constraints(expr, [new_con1, new_con2], self._warm_ctx())
        # Should not have copied since lengths differ.
        assert new_con1.dual_variables[0].value is None
        assert new_con2.dual_variables[0].value is None
        # But should have cached the new constraints.
        assert expr._cached_aux_constraints == [new_con1, new_con2]

    def test_successful_copy(self):
        """Duals are copied from old to new constraints."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        old_con1 = (x >= 0)
        old_con2 = (x <= 1)
        old_con1.dual_variables[0].save_value(np.array([1.0, 2.0]))
        old_con2.dual_variables[0].save_value(np.array([3.0, 4.0]))
        expr._cached_aux_constraints = [old_con1, old_con2]

        new_con1 = (x >= 0)
        new_con2 = (x <= 1)
        propagate_dual_values_to_constraints(expr, [new_con1, new_con2], self._warm_ctx())
        np.testing.assert_allclose(new_con1.dual_variables[0].value, [1.0, 2.0])
        np.testing.assert_allclose(new_con2.dual_variables[0].value, [3.0, 4.0])

    def test_no_copy_without_warm_start(self):
        """Duals are NOT copied when solver doesn't support warm start."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        old_con = (x >= 0)
        old_con.dual_variables[0].save_value(np.array([1.0, 2.0]))
        expr._cached_aux_constraints = [old_con]

        new_con = (x >= 0)
        propagate_dual_values_to_constraints(expr, [new_con], self._no_warm_ctx())
        assert new_con.dual_variables[0].value is None
        # But new constraints should still be cached.
        assert expr._cached_aux_constraints == [new_con]

    def test_partial_dual_values(self):
        """Only constraints with non-None duals are copied."""
        x = cp.Variable(2)
        expr = cp.abs(x)
        old_con1 = (x >= 0)
        old_con2 = (x <= 1)
        old_con1.dual_variables[0].save_value(np.array([1.0, 2.0]))
        # old_con2 has no dual value
        expr._cached_aux_constraints = [old_con1, old_con2]

        new_con1 = (x >= 0)
        new_con2 = (x <= 1)
        propagate_dual_values_to_constraints(expr, [new_con1, new_con2], self._warm_ctx())
        np.testing.assert_allclose(new_con1.dual_variables[0].value, [1.0, 2.0])
        assert new_con2.dual_variables[0].value is None


# ──────────────────────────────────────────────────────────────
#  Integration tests: invert() saves duals on aux constraints
# ──────────────────────────────────────────────────────────────

class TestInvertSavesDuals:
    """Verify that Canonicalization.invert() saves dual values on aux constraints."""

    def test_invert_saves_aux_duals(self):
        from cvxpy.reductions.inverse_data import InverseData

        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        inverse_data = InverseData(prob)
        inverse_data.cons_id_map = {}

        # Create some aux constraints and pretend they were generated.
        aux_con = (x >= 0)
        inverse_data.aux_constraints = [aux_con]

        # Build a fake solution with dual values for the aux constraint.
        from cvxpy.reductions.solution import Solution
        dual_val = np.array([5.0, 6.0])
        solution = Solution("optimal", 0.0, {}, {aux_con.id: dual_val}, {})

        from cvxpy.reductions.canonicalization import Canonicalization
        canon = Canonicalization(canon_methods={})
        result = canon.invert(solution, inverse_data)

        # The aux constraint should now have the dual saved.
        np.testing.assert_allclose(aux_con.dual_variables[0].value, [5.0, 6.0])
        assert result.status == "optimal"


# ──────────────────────────────────────────────────────────────
#  Integration tests: dual propagation in canonicalizers
# ──────────────────────────────────────────────────────────────

class TestDualPropagationInCanonicalizers:
    """Test that re-canonicalization copies cached duals to new constraints."""

    def _warm_ctx(self):
        return SolverInfo(solver="TEST", supported_constraints=[],
                          supports_warm_start=True)

    def test_exp_canon_dual_propagation(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
        x = cp.Variable(3)
        x.value = np.array([0.0, 1.0, -1.0])
        expr = cp.exp(x)
        ctx = self._warm_ctx()

        # First canonicalization: caches constraints.
        _, constraints1 = exp_canon(expr, expr.args, solver_context=ctx)
        assert len(constraints1) == 1
        # Simulate solver saving duals on the cached constraints.
        for dv in constraints1[0].dual_variables:
            dv.save_value(np.ones(dv.shape))

        # Second canonicalization: should copy duals.
        _, constraints2 = exp_canon(expr, expr.args, solver_context=ctx)
        for dv in constraints2[0].dual_variables:
            assert dv.value is not None
            np.testing.assert_allclose(dv.value, np.ones(dv.shape))

    def test_log_canon_dual_propagation(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.log_canon import log_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log(x)
        ctx = self._warm_ctx()

        _, constraints1 = log_canon(expr, expr.args, solver_context=ctx)
        for dv in constraints1[0].dual_variables:
            dv.save_value(np.ones(dv.shape))

        _, constraints2 = log_canon(expr, expr.args, solver_context=ctx)
        for dv in constraints2[0].dual_variables:
            assert dv.value is not None

    def test_abs_canon_dual_propagation(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.abs_canon import abs_canon
        x = cp.Variable(3)
        x.value = np.array([-1.0, 0.0, 2.0])
        expr = cp.abs(x)
        ctx = self._warm_ctx()

        _, constraints1 = abs_canon(expr, expr.args, solver_context=ctx)
        assert len(constraints1) == 2
        for con in constraints1:
            con.dual_variables[0].save_value(np.array([1.0, 2.0, 3.0]))

        _, constraints2 = abs_canon(expr, expr.args, solver_context=ctx)
        for con in constraints2:
            np.testing.assert_allclose(con.dual_variables[0].value, [1.0, 2.0, 3.0])

    def test_max_canon_dual_propagation(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.max_canon import max_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 5.0, 3.0])
        expr = cp.max(x)
        ctx = self._warm_ctx()

        _, constraints1 = max_canon(expr, expr.args, solver_context=ctx)
        assert len(constraints1) == 1
        constraints1[0].dual_variables[0].save_value(np.array([0.2, 0.5, 0.3]))

        _, constraints2 = max_canon(expr, expr.args, solver_context=ctx)
        np.testing.assert_allclose(constraints2[0].dual_variables[0].value, [0.2, 0.5, 0.3])

    def test_maximum_canon_dual_propagation(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.maximum_canon import maximum_canon
        x = cp.Variable(3)
        y = cp.Variable(3)
        x.value = np.array([1.0, 5.0, 3.0])
        y.value = np.array([4.0, 2.0, 6.0])
        expr = cp.maximum(x, y)
        ctx = self._warm_ctx()

        _, constraints1 = maximum_canon(expr, expr.args, solver_context=ctx)
        assert len(constraints1) == 2
        for i, con in enumerate(constraints1):
            con.dual_variables[0].save_value(np.array([0.1 * (i + 1)] * 3))

        _, constraints2 = maximum_canon(expr, expr.args, solver_context=ctx)
        for i, con in enumerate(constraints2):
            np.testing.assert_allclose(con.dual_variables[0].value,
                                       [0.1 * (i + 1)] * 3)

    def test_norm_inf_canon_dual_propagation(self):
        from cvxpy.reductions.eliminate_pwl.canonicalizers.norm_inf_canon import norm_inf_canon
        x = cp.Variable(3)
        x.value = np.array([-1.0, 3.0, -2.0])
        expr = cp.norm_inf(x)
        ctx = self._warm_ctx()

        _, constraints1 = norm_inf_canon(expr, expr.args, solver_context=ctx)
        assert len(constraints1) == 2
        for con in constraints1:
            con.dual_variables[0].save_value(np.array([1.0, 2.0, 3.0]))

        _, constraints2 = norm_inf_canon(expr, expr.args, solver_context=ctx)
        for con in constraints2:
            np.testing.assert_allclose(con.dual_variables[0].value, [1.0, 2.0, 3.0])

    def test_pnorm_exact_p2_dual_propagation(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.pnorm_canon import pnorm_exact_canon
        x = cp.Variable(3)
        x.value = np.array([3.0, 4.0, 0.0])
        expr = cp.pnorm(x, 2)
        ctx = self._warm_ctx()

        _, constraints1 = pnorm_exact_canon(expr, expr.args, solver_context=ctx)
        for con in constraints1:
            for dv in con.dual_variables:
                dv.save_value(np.ones(dv.shape))

        _, constraints2 = pnorm_exact_canon(expr, expr.args, solver_context=ctx)
        for con in constraints2:
            for dv in con.dual_variables:
                assert dv.value is not None

    def test_power_exact_canon_dual_propagation(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.power_canon import power_exact_canon
        x = cp.Variable(3, pos=True)
        x.value = np.array([1.0, 4.0, 9.0])
        expr = cp.power(x, 0.5)
        ctx = self._warm_ctx()

        _, constraints1 = power_exact_canon(expr, expr.args, solver_context=ctx)
        for con in constraints1:
            for dv in con.dual_variables:
                dv.save_value(np.ones(dv.shape))

        _, constraints2 = power_exact_canon(expr, expr.args, solver_context=ctx)
        for con in constraints2:
            for dv in con.dual_variables:
                assert dv.value is not None

    def test_no_dual_propagation_without_warm_start(self):
        """No duals copied when warm start not supported."""
        from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
        x = cp.Variable(3)
        x.value = np.array([0.0, 1.0, -1.0])
        expr = cp.exp(x)

        # First call with warm start to cache constraints.
        ctx_warm = self._warm_ctx()
        _, constraints1 = exp_canon(expr, expr.args, solver_context=ctx_warm)
        for dv in constraints1[0].dual_variables:
            dv.save_value(np.ones(dv.shape))

        # Second call WITHOUT warm start — should not copy.
        ctx_cold = SolverInfo(solver="TEST", supported_constraints=[],
                              supports_warm_start=False)
        _, constraints2 = exp_canon(expr, expr.args, solver_context=ctx_cold)
        for dv in constraints2[0].dual_variables:
            assert dv.value is None
