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
from cvxpy.utilities.values import get_expr_value_if_supported

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
