"""Tests for parametric (expression) bounds on variables."""

import numpy as np
import pytest

import cvxpy as cp


def _param_in_list(param, param_list):
    """Check if a parameter is in a list by id (avoids Expression __eq__)."""
    return any(p.id == param.id for p in param_list)


def _solver_available(solver_name):
    """Check if a solver is available and supports BOUNDED_VARIABLES."""
    try:
        prob = cp.Problem(cp.Minimize(0), [])
        prob.solve(solver=solver_name)
        return True
    except (cp.error.SolverError, Exception):
        return False


class TestParametricBoundsCreation:
    """Tests for creating variables with parametric bounds."""

    def test_parameter_bounds(self):
        """Variable with Parameter bounds."""
        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])
        assert x.bounds is not None
        assert isinstance(x.bounds[0], cp.Expression)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_expression_bounds(self):
        """Variable with expression bounds."""
        p = cp.Parameter()
        x = cp.Variable(bounds=[p + 1, 2 * p])
        assert x.bounds is not None

    def test_mixed_bounds_param_and_numeric(self):
        """One bound is parametric, the other is numeric."""
        ub = cp.Parameter()
        x = cp.Variable(bounds=[0, ub])
        assert isinstance(x.bounds[0], np.ndarray)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_mixed_bounds_none_and_param(self):
        """One bound is None, the other is parametric."""
        ub = cp.Parameter()
        x = cp.Variable(bounds=[None, ub])
        # None should be promoted to -inf ndarray
        assert isinstance(x.bounds[0], np.ndarray)
        assert np.all(x.bounds[0] == -np.inf)
        assert isinstance(x.bounds[1], cp.Expression)

    def test_scalar_param_broadcast_to_array(self):
        """Scalar Parameter bound should be accepted for array variable."""
        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable((3,), bounds=[lb, ub])
        assert x.bounds is not None

    def test_array_param_matching_shape(self):
        """Array Parameter bound with matching shape."""
        lb = cp.Parameter((2, 2))
        ub = cp.Parameter((2, 2))
        x = cp.Variable((2, 2), bounds=[lb, ub])
        assert x.bounds is not None

    def test_invalid_param_shape(self):
        """Parameter bound with non-matching shape should raise."""
        lb = cp.Parameter((3,))
        with pytest.raises(ValueError):
            cp.Variable((2, 2), bounds=[lb, 10])

    def test_has_lower_bounds_param(self):
        """_has_lower_bounds returns True for parametric lower bound."""
        lb = cp.Parameter()
        x = cp.Variable(bounds=[lb, 10])
        assert x._has_lower_bounds()

    def test_has_upper_bounds_param(self):
        """_has_upper_bounds returns True for parametric upper bound."""
        ub = cp.Parameter()
        x = cp.Variable(bounds=[0, ub])
        assert x._has_upper_bounds()

    def test_parameters_discovered(self):
        """Parameters in bounds are discoverable via Variable.parameters()."""
        lb = cp.Parameter(name="lb")
        ub = cp.Parameter(name="ub")
        x = cp.Variable(bounds=[lb, ub])
        params = x.parameters()
        assert _param_in_list(lb, params)
        assert _param_in_list(ub, params)

    def test_problem_parameters_include_bounds(self):
        """Problem.parameters() includes bounds parameters."""
        lb = cp.Parameter(name="lb")
        ub = cp.Parameter(name="ub")
        x = cp.Variable(bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(x))
        params = prob.parameters()
        assert _param_in_list(lb, params)
        assert _param_in_list(ub, params)


class TestParametricBoundsReduceTrue:
    """Tests for reduce_bounds=True path (bounds -> constraints).

    This is the default path for solvers that don't support BOUNDED_VARIABLES
    (e.g. SCS, Clarabel).
    """

    def test_basic_param_bounds_minimize(self):
        """Basic parametric bounds with minimization."""
        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])

        lb.value = 2
        ub.value = 5
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(x.value, 2, atol=1e-4)

    def test_basic_param_bounds_maximize(self):
        """Basic parametric bounds with maximization."""
        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])

        lb.value = 2
        ub.value = 5
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.SCS)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(x.value, 5, atol=1e-4)

    def test_re_solve_changed_bounds(self):
        """Re-solve with changed parameter values."""
        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])

        lb.value = 2
        ub.value = 5
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 2, atol=1e-4)

        # Change bounds
        lb.value = 3
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 3, atol=1e-4)

    def test_expression_bounds(self):
        """Expression (not just Parameter) as bounds."""
        scale = cp.Parameter(nonneg=True)
        x = cp.Variable(bounds=[-scale, scale])

        scale.value = 10
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, -10, atol=1e-4)

    def test_mixed_numeric_and_param(self):
        """One bound numeric, other parametric."""
        ub = cp.Parameter()
        x = cp.Variable(bounds=[0, ub])

        ub.value = 5
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 5, atol=1e-4)

    def test_one_sided_param_bound(self):
        """Only upper bound is parametric, lower is None."""
        ub = cp.Parameter()
        x = cp.Variable(bounds=[None, ub])

        ub.value = 3
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 3, atol=1e-4)

    def test_vector_variable_scalar_param(self):
        """Scalar parameter broadcast to vector variable."""
        lb = cp.Parameter()
        x = cp.Variable(3, bounds=[lb, None])

        lb.value = -2
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 10])
        prob.solve(solver=cp.SCS)
        assert np.allclose(x.value, -2 * np.ones(3), atol=1e-4)


class TestParametricBoundsReduceFalse:
    """Tests for reduce_bounds=False path (BOUNDED_VARIABLES).

    This uses solvers that natively support variable bounds (e.g. HIGHS, DAQP).
    """

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_basic_param_bounds(self, solver_name):
        """Basic parametric bounds with BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])

        lb.value = 2
        ub.value = 5
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=solver_name)
        assert prob.status == cp.OPTIMAL
        assert np.isclose(x.value, 2, atol=1e-6)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_re_solve_changed_bounds(self, solver_name):
        """Re-solve with changed bounds using BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        lb = cp.Parameter()
        ub = cp.Parameter()
        x = cp.Variable(bounds=[lb, ub])

        lb.value = 1
        ub.value = 5
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 1, atol=1e-6)

        lb.value = 2
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 2, atol=1e-6)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_expression_bounds(self, solver_name):
        """Expression bounds with BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        scale = cp.Parameter(nonneg=True)
        x = cp.Variable(bounds=[-scale, scale])

        scale.value = 10
        prob = cp.Problem(cp.Minimize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, -10, atol=1e-6)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_mixed_bounds(self, solver_name):
        """Mixed numeric/parametric bounds with BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        ub = cp.Parameter()
        x = cp.Variable(bounds=[0, ub])

        ub.value = 5
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 5, atol=1e-6)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_vector_variable(self, solver_name):
        """Vector variable with parametric bounds on BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        lb = cp.Parameter()
        x = cp.Variable(3, bounds=[lb, None])

        lb.value = -2
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x <= 10])
        prob.solve(solver=solver_name)
        assert np.allclose(x.value, -2 * np.ones(3), atol=1e-6)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_one_sided_param_bound(self, solver_name):
        """One-sided parametric bound with BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        ub = cp.Parameter()
        x = cp.Variable(bounds=[None, ub])

        ub.value = 3
        prob = cp.Problem(cp.Maximize(x))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 3, atol=1e-6)


class TestParametricBoundsMultiVariable:
    """Test problems with multiple variables, some with parametric bounds."""

    def test_two_variables_one_parametric(self):
        """Two variables, one with parametric bounds."""
        lb = cp.Parameter()
        x = cp.Variable(bounds=[lb, None])
        y = cp.Variable()

        lb.value = 1
        prob = cp.Problem(cp.Minimize(x + y), [y >= 2])
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 1, atol=1e-4)
        assert np.isclose(y.value, 2, atol=1e-4)

    def test_two_variables_both_parametric(self):
        """Two variables, both with parametric bounds."""
        lb1 = cp.Parameter()
        lb2 = cp.Parameter()
        x = cp.Variable(bounds=[lb1, None])
        y = cp.Variable(bounds=[lb2, None])

        lb1.value = 1
        lb2.value = 2
        prob = cp.Problem(cp.Minimize(x + y))
        prob.solve(solver=cp.SCS)
        assert np.isclose(x.value, 1, atol=1e-4)
        assert np.isclose(y.value, 2, atol=1e-4)

    @pytest.mark.parametrize("solver_name", ["HIGHS", "DAQP"])
    def test_multi_variable_bounded_solver(self, solver_name):
        """Multiple variables with parametric bounds on BOUNDED_VARIABLES solver."""
        if not _solver_available(solver_name):
            pytest.skip(f"{solver_name} not available")

        lb = cp.Parameter()
        x = cp.Variable(bounds=[lb, None])
        y = cp.Variable(bounds=[0, None])

        lb.value = 1
        prob = cp.Problem(cp.Minimize(x + y))
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 1, atol=1e-6)
        assert np.isclose(y.value, 0, atol=1e-6)

        lb.value = 3
        prob.solve(solver=solver_name)
        assert np.isclose(x.value, 3, atol=1e-6)
