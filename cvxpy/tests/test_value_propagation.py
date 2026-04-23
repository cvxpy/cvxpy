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
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
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

    def test_entr_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.entr_canon import entr_canon
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 0.5])
        expr = cp.entr(x)
        t, _ = entr_canon(expr, expr.args, solver_context=self._warm_start_context())
        assert t.value is not None
        expected = -np.array([1.0, 2.0, 0.5]) * np.log([1.0, 2.0, 0.5])
        np.testing.assert_allclose(t.value, expected)

    def test_log_sum_exp_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.log_sum_exp_canon import (
            log_sum_exp_canon,
        )
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.log_sum_exp(x)
        t, _ = log_sum_exp_canon(expr, expr.args,
                                 solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value,
                                   np.log(np.sum(np.exp([1.0, 2.0, 3.0]))),
                                   rtol=1e-5)

    def test_logistic_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.logistic_canon import logistic_canon
        x = cp.Variable(3)
        x.value = np.array([0.0, 1.0, -1.0])
        expr = cp.logistic(x)
        t, _ = logistic_canon(expr, expr.args,
                              solver_context=self._warm_start_context())
        assert t.value is not None
        expected = np.log(1 + np.exp([0.0, 1.0, -1.0]))
        np.testing.assert_allclose(t.value, expected, rtol=1e-5)

    def test_sigma_max_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.sigma_max_canon import sigma_max_canon
        A = cp.Variable((2, 3))
        A.value = np.array([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        expr = cp.sigma_max(A)
        t, _ = sigma_max_canon(expr, expr.args,
                               solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([3.0]), rtol=1e-5)

    def test_rel_entr_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.rel_entr_canon import rel_entr_canon
        x = cp.Variable(3)
        y = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        y.value = np.array([2.0, 2.0, 6.0])
        expr = cp.rel_entr(x, y)
        obj, _ = rel_entr_canon(expr, expr.args,
                                solver_context=self._warm_start_context())
        # obj is -t, and t should have value = -expr_value
        # So obj.value should equal expr.value
        assert obj.value is not None
        expected = x.value * np.log(x.value / y.value)
        np.testing.assert_allclose(obj.value, expected, rtol=1e-5)

    def test_quad_over_lin_canon_propagates_value(self):
        from cvxpy.reductions.dcp2cone.canonicalizers.quad_over_lin_canon import (
            quad_over_lin_canon,
        )
        x = cp.Variable(3)
        x.value = np.array([1.0, 2.0, 3.0])
        expr = cp.quad_over_lin(x, 1)
        t, _ = quad_over_lin_canon(expr, expr.args,
                                   solver_context=self._warm_start_context())
        assert t.value is not None
        np.testing.assert_allclose(t.value, np.array([14.0]), rtol=1e-5)


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
#  End-to-end integration tests: solve with warm starting
# ──────────────────────────────────────────────────────────────

def _solve_and_seed(atom_fn, constraints_fn, n, solver=cp.SCS):
    """Solve a problem cold, then re-solve warm-seeded at the optimum.

    Returns (cold_iters, warm_iters, cold_val, warm_val).
    """
    x = cp.Variable(n)
    prob = cp.Problem(atom_fn(x), constraints_fn(x))
    prob.solve(solver=solver)
    cold_iters = prob.solver_stats.num_iters
    cold_val = prob.value
    opt_x = x.value.copy()

    # Fresh problem seeded at the optimum.
    x2 = cp.Variable(n)
    x2.value = opt_x
    prob2 = cp.Problem(atom_fn(x2), constraints_fn(x2))
    prob2.solve(solver=solver, warm_start=True)
    warm_iters = prob2.solver_stats.num_iters
    warm_val = prob2.value
    return cold_iters, warm_iters, cold_val, warm_val


@pytest.mark.skipif('SCS' not in INSTALLED_SOLVERS, reason='SCS not installed.')
class TestWarmStartEndToEnd:
    """End-to-end: warm-seeded SCS solves should use fewer iterations."""

    @pytest.mark.parametrize("atom_fn,constr_fn,n", [
        (lambda x: cp.Minimize(cp.sum(cp.exp(x))),
         lambda x: [cp.sum(x) == 1], 10),
        (lambda x: cp.Maximize(cp.sum(cp.log(x))),
         lambda x: [cp.sum(x) == 1], 10),
        (lambda x: cp.Maximize(cp.sum(cp.entr(x))),
         lambda x: [cp.sum(x) == 1], 10),
        (lambda x: cp.Minimize(cp.sum(cp.logistic(x))),
         lambda x: [cp.sum(x) == 0], 10),
        (lambda x: cp.Minimize(cp.norm(x, 2)),
         lambda x: [x >= 1], 10),
    ], ids=["exp", "log", "entr", "logistic", "norm2"])
    def test_fewer_iters_when_seeded(self, atom_fn, constr_fn, n):
        cold, warm, cold_v, warm_v = _solve_and_seed(atom_fn, constr_fn, n)
        np.testing.assert_allclose(warm_v, cold_v, rtol=1e-2)
        assert warm <= cold, f"warm {warm} > cold {cold}"

    def test_no_values_set_still_works(self):
        """warm_start=True with no values degrades gracefully."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.norm(x, 1)), [cp.sum(x) == 1, x >= 0])
        prob.solve(solver=cp.SCS, warm_start=True)
        assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    def test_partial_values_still_works(self):
        """Only some variables have values; should not crash."""
        x = cp.Variable(3)
        y = cp.Variable(3)
        x.value = np.ones(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(x) + cp.sum_squares(y)),
                          [x + y == 2])
        prob.solve(solver=cp.SCS, warm_start=True)
        assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}
