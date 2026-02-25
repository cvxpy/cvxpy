
import numpy as np
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.nlp_solving_chain import _set_nlp_initial_point


class TestProblem():
    """Tests for internal NLP functions in the DNLP extension."""

    @pytest.mark.parametrize("bounds, expected", [
        (None, 0.0),
        ([None, None], 0.0),
        ([-np.inf, np.inf], 0.0),
        ([None, np.inf], 0.0),
        ([-np.inf, None], 0.0),
        ([None, 3.5], 2.5),
        ([-np.inf, 3.5], 2.5),
        ([3.5, None], 4.5),
        ([3.5, np.inf], 4.5),
        ([3.5, 4.5], 4.0),
    ])
    def test_set_initial_point_scalar_bounds(self, bounds, expected):
        x = cp.Variable((3, ), bounds=bounds)
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        _set_nlp_initial_point(prob)
        assert (x.value == expected * np.ones((3, ))).all()

    def test_set_initial_point_mixed_inf_and_finite(self):
        lb = np.array([-np.inf, 3.5, -np.inf, -1.5, 2, 2.5])
        ub = np.array([-4, 4.5, np.inf, 4.5, np.inf, 4.5])
        x = cp.Variable((6, ), bounds=[lb, ub])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        _set_nlp_initial_point(prob)
        expected = np.array([-5, 4.0, 0.0, 1.5, 3, 3.5])
        assert (x.value == expected).all()

    def test_set_initial_point_two_variables(self):
        x = cp.Variable((2, ), bounds=[-np.inf, np.inf])
        y = cp.Variable((2, ), bounds=[-3, np.inf])
        prob = cp.Problem(cp.Minimize(cp.sum(x) + cp.sum(y)))
        _set_nlp_initial_point(prob)
        assert (x.value == np.zeros((2, ))).all()
        assert (y.value == -2 * np.ones((2, ))).all()

    def test_set_initial_point_nonnegative_attributes(self):
        x = cp.Variable((2, ), nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        _set_nlp_initial_point(prob)
        assert (x.value == np.ones((2, ))).all()

    def test_set_initial_point_nonnegative_attributes_and_bounds(self):
        x = cp.Variable((2, ), nonneg=True, bounds=[1, None])
        prob = cp.Problem(cp.Minimize(cp.sum(x)))
        _set_nlp_initial_point(prob)
        assert (x.value == 2 * np.ones((2, ))).all()
