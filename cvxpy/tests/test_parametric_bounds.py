"""Tests for parametric (expression) bounds on variables."""

import numpy as np
import pytest

import cvxpy as cp
import cvxpy.error as error
import cvxpy.reductions.eval_params

CONIC_SOLVERS = [cp.SCS, cp.CLARABEL]
BOUNDED_SOLVERS = ["HIGHS", "DAQP"]


def _solver_available(solver_name):
    try:
        cp.Problem(cp.Minimize(0)).solve(solver=solver_name)
        return True
    except Exception:
        return False


def _skip_if_unavailable(solver_name):
    if not _solver_available(solver_name):
        pytest.skip(f"{solver_name} not available")


class TestParametricBoundsCreation:
    """Tests for creating variables with parametric bounds."""

    def test_basic_creation(self):
        lb, ub = cp.Parameter(), cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])
        assert isinstance(x.bounds[0], cp.Expression)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_expression_bounds(self):
        p = cp.Parameter()
        x = cp.Variable(bounds=[p + 1, 2 * p])
        assert x.bounds is not None

    def test_mixed_none_and_param(self):
        ub = cp.Parameter()
        x = cp.Variable(bounds=[None, ub])
        assert isinstance(x.bounds[0], np.ndarray)
        assert np.all(x.bounds[0] == -np.inf)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_mixed_numeric_and_param(self):
        ub = cp.Parameter()
        x = cp.Variable(bounds=[0, ub])
        assert isinstance(x.bounds[0], np.ndarray)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_scalar_param_broadcast(self):
        x = cp.Variable((3,), bounds=[cp.Parameter(), cp.Parameter()])
        assert x.bounds is not None

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            cp.Variable((2, 2), bounds=[cp.Parameter((3,)), 10])

    def test_parameter_rejects_expression_bounds(self):
        with pytest.raises(ValueError, match="Parametric bounds"):
            cp.Parameter(bounds=[cp.Parameter(), 10])

    def test_has_lower_upper_bounds(self):
        assert cp.Variable(bounds=[cp.Parameter(), 10])._has_lower_bounds()
        assert cp.Variable(bounds=[0, cp.Parameter()])._has_upper_bounds()

    def test_parameters_discovered(self):
        lb, ub = cp.Parameter(name="lb"), cp.Parameter(name="ub")
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        ids = {p.id for p in prob.parameters()}
        assert lb.id in ids and ub.id in ids


class TestParametricBoundsSolving:
    """Tests that parametric bounds work across solver paths."""

    def test_minimize_maximize(self):
        lb, ub = cp.Parameter(value=2), cp.Parameter(value=5)
        x = cp.Variable(bounds=[lb, ub])

        prob_min = cp.Problem(cp.Minimize(x))
        prob_min.solve(solver=cp.SCS)
        assert np.isclose(x.value, 2, atol=1e-4)

        prob_max = cp.Problem(cp.Maximize(x))
        prob_max.solve(solver=cp.SCS)
        assert np.isclose(x.value, 5, atol=1e-4)

    def test_re_solve(self):
        lb, ub = cp.Parameter(value=2), cp.Parameter(value=5)
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 2, atol=1e-4)

        lb.value = 3
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 3, atol=1e-4)

    def test_expression_bounds(self):
        scale = cp.Parameter(nonneg=True, value=10)
        x = cp.Variable(bounds=[-scale, scale])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, -10, atol=1e-4)

    def test_mixed_numeric_and_param(self):
        ub = cp.Parameter(value=5)
        x = cp.Variable(bounds=[0, ub])
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 5, atol=1e-4)

    def test_one_sided_param_bound(self):
        ub = cp.Parameter(value=3)
        x = cp.Variable(bounds=[None, ub])
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 3, atol=1e-4)

    def test_vector_variable_scalar_param(self):
        lb = cp.Parameter(value=-2)
        x = cp.Variable(3, bounds=[lb, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 10])
        prob.solve(solver=cp.SCS)
        assert np.allclose(x.value, -2 * np.ones(3), atol=1e-4)

    def test_multiple_variables(self):
        lb1, lb2 = cp.Parameter(value=1), cp.Parameter(value=2)
        x = cp.Variable(bounds=[lb1, None])
        y = cp.Variable(bounds=[lb2, None])
        prob = cp.Problem(cp.Minimize(x + y))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 1, atol=1e-4)
        assert np.isclose(y.value, 2, atol=1e-4)

    def test_matrix_variable(self):
        lb = cp.Parameter((2, 3), value=np.ones((2, 3)))
        ub = cp.Parameter((2, 3), value=5 * np.ones((2, 3)))
        X = cp.Variable((2, 3), bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(cp.sum(X)))
        prob.solve(solver=cp.CLARABEL)
        np.testing.assert_allclose(X.value, np.ones((2, 3)), atol=1e-4)

    @pytest.mark.parametrize("solver_name", BOUNDED_SOLVERS)
    def test_bounded_solver_basic(self, solver_name):
        _skip_if_unavailable(solver_name)
        lb, ub = cp.Parameter(value=2), cp.Parameter(value=5)
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 2, atol=1e-6)

    @pytest.mark.parametrize("solver_name", BOUNDED_SOLVERS)
    def test_bounded_solver_re_solve(self, solver_name):
        _skip_if_unavailable(solver_name)
        lb, ub = cp.Parameter(value=1), cp.Parameter(value=5)
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 1, atol=1e-6)

        lb.value = 3
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 3, atol=1e-6)


class TestParametricBoundsEdgeCases:
    """Edge cases for parametric bounds."""

    def test_infinite_bounds_unbounded(self):
        ub = cp.Parameter(value=5)
        x = cp.Variable(bounds=[-np.inf, ub])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.CLARABEL)
        assert prob.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)

    def test_equal_bounds_pins_value(self):
        b = cp.Parameter(value=7)
        x = cp.Variable(bounds=[b, b])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.CLARABEL)
        assert np.isclose(x.value, 7, atol=1e-4)

    def test_infeasible_bounds(self):
        lb, ub = cp.Parameter(value=10), cp.Parameter(value=5)
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.CLARABEL)
        assert prob.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)

    def test_unset_parameter_raises(self):
        x = cp.Variable(bounds=[cp.Parameter(), 10])
        prob = cp.Problem(cp.Minimize(x))
        with pytest.raises(Exception):
            prob.solve(solver=cp.CLARABEL)


class TestParametricBoundsDPP:
    """DPP tests for parametric bounds."""

    def _make_prob(self):
        lb, ub = cp.Parameter(), cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        return lb, ub, x, prob

    def test_is_dpp_and_chain(self):
        _, _, _, prob = self._make_prob()
        assert prob.is_dpp()
        _, chain, _ = prob.get_problem_data(cp.SCS)
        reduction_types = [type(r) for r in chain.reductions]
        assert cvxpy.reductions.eval_params.EvalParams not in reduction_types

    def test_get_problem_data_without_param_values(self):
        _, _, _, prob = self._make_prob()
        data, _, _ = prob.get_problem_data(cp.SCS)
        assert data is not None

    def test_enforce_dpp_re_solve(self):
        lb, ub, x, prob = self._make_prob()
        lb.value, ub.value = 1, 10
        prob.solve(solver=cp.SCS, enforce_dpp=True)
        assert np.isclose(x.value, 1, atol=1e-4)

        lb.value = 5
        prob.solve(solver=cp.SCS, enforce_dpp=True)
        assert np.isclose(x.value, 5, atol=1e-4)

    def test_enforce_dpp_expression_bounds(self):
        scale = cp.Parameter(nonneg=True, value=10)
        x = cp.Variable(bounds=[-scale, scale])
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS, enforce_dpp=True)
        assert np.isclose(x.value, -10, atol=1e-4)

    @pytest.mark.parametrize("solver_name", BOUNDED_SOLVERS)
    def test_enforce_dpp_bounded_solver(self, solver_name):
        _skip_if_unavailable(solver_name)
        lb, ub, x, prob = self._make_prob()
        lb.value, ub.value = 1, 10
        prob.solve(solver=solver_name, enforce_dpp=True)
        assert np.isclose(x.value, 1, atol=1e-6)

        lb.value = 5
        prob.solve(solver=solver_name, enforce_dpp=True)
        assert np.isclose(x.value, 5, atol=1e-6)

    @pytest.mark.parametrize("bound_expr", [
        lambda: cp.Parameter() * cp.Parameter(),
        lambda: cp.Parameter((2, 2)) @ cp.Parameter(2),
    ])
    def test_non_dpp_bound_raises(self, bound_expr):
        expr = bound_expr()
        n = 2 if expr.ndim > 0 else 1
        x = cp.Variable(n, bounds=[expr, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 10])
        assert not prob.is_dpp()
        for p in prob.parameters():
            p.value = np.ones(p.shape) if p.ndim > 0 else 1
        with pytest.raises(error.DPPError):
            prob.solve(solver=cp.SCS, enforce_dpp=True)
