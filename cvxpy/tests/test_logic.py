"""
Copyright, the CVXPY authors

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
from cvxpy.atoms.elementwise.logic import And, Not, Or, Xor


class TestLogicExpressions:
    """Tests for LogicExpression atom classes."""

    def test_expression_types(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        assert isinstance(Not(x), Not)
        assert isinstance(And(x, y), And)
        assert isinstance(Or(x, y), Or)
        assert isinstance(Xor(x, y), Xor)

    def test_composition(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        expr = Or(And(x1, x2), Not(x3))
        assert isinstance(expr, Or)

    def test_dcp_compliance(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        assert And(x, y).is_dcp()
        assert Or(x, y).is_dcp()
        assert Not(x).is_dcp()
        assert Xor(x, y).is_dcp()

    def test_namespace(self):
        """Test that cp.logic namespace works."""
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        assert isinstance(cp.logic.Not(x), Not)
        assert isinstance(cp.logic.And(x, y), And)
        assert isinstance(cp.logic.Or(x, y), Or)
        assert isinstance(cp.logic.Xor(x, y), Xor)


class TestLogicSolve:
    """Solve-based truth table tests for logic atoms."""

    @staticmethod
    def _solve_with_fixed(expr, var_vals):
        """Solve minimizing expr with variables fixed to given values."""
        constraints = [v == val for v, val in var_vals.items()]
        prob = cp.Problem(cp.Minimize(0), constraints + [expr >= 0])
        prob.solve(solver=cp.HIGHS)
        return prob

    @pytest.mark.parametrize("val,expected", [(0, 1), (1, 0)])
    def test_not_truth_table(self, val, expected):
        x = cp.Variable(boolean=True)
        expr = Not(x)
        prob = cp.Problem(cp.Minimize(0), [x == val])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, expected), \
            f"Not({val}) = {expr.value}, expected {expected}"

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1),
    ])
    def test_and_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = And(x, y)
        prob = cp.Problem(cp.Minimize(0), [x == a, y == b])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, expected), \
            f"And({a}, {b}) = {expr.value}, expected {expected}"

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
    ])
    def test_or_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = Or(x, y)
        prob = cp.Problem(cp.Minimize(0), [x == a, y == b])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, expected), \
            f"Or({a}, {b}) = {expr.value}, expected {expected}"

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
    ])
    def test_xor_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = Xor(x, y)
        prob = cp.Problem(cp.Minimize(0), [x == a, y == b])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, expected), \
            f"Xor({a}, {b}) = {expr.value}, expected {expected}"


class TestLogicNary:
    """N-ary logic operation tests."""

    def test_and_3(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        expr = And(x1, x2, x3)
        # All true -> true
        prob = cp.Problem(cp.Minimize(0), [x1 == 1, x2 == 1, x3 == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)
        # One false -> false
        prob = cp.Problem(cp.Minimize(0), [x1 == 1, x2 == 0, x3 == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_or_3(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        expr = Or(x1, x2, x3)
        # All false -> false
        prob = cp.Problem(cp.Minimize(0), [x1 == 0, x2 == 0, x3 == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)
        # One true -> true
        prob = cp.Problem(cp.Minimize(0), [x1 == 0, x2 == 1, x3 == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_xor_3_parity(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        expr = Xor(x1, x2, x3)
        # 0 true (even) -> 0
        prob = cp.Problem(cp.Minimize(0), [x1 == 0, x2 == 0, x3 == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)
        # 1 true (odd) -> 1
        prob = cp.Problem(cp.Minimize(0), [x1 == 1, x2 == 0, x3 == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)
        # 2 true (even) -> 0
        prob = cp.Problem(cp.Minimize(0), [x1 == 1, x2 == 1, x3 == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)
        # 3 true (odd) -> 1
        prob = cp.Problem(cp.Minimize(0), [x1 == 1, x2 == 1, x3 == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)


class TestLogicVector:
    """Element-wise vector logic operations."""

    def test_vector_and(self):
        x = cp.Variable(3, boolean=True)
        y = cp.Variable(3, boolean=True)
        expr = And(x, y)
        assert expr.shape == (3,)
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 0, 1]), y == np.array([1, 1, 0])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(expr.value, [1, 0, 0])

    def test_vector_or(self):
        x = cp.Variable(3, boolean=True)
        y = cp.Variable(3, boolean=True)
        expr = Or(x, y)
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 0, 0]), y == np.array([0, 0, 1])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(expr.value, [1, 0, 1])

    def test_vector_not(self):
        x = cp.Variable(3, boolean=True)
        expr = Not(x)
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 0, 1])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(expr.value, [0, 1, 0])


class TestLogicComposition:
    """Test composed logic expressions solve correctly."""

    def test_or_and_not(self):
        """Or(And(x1, x2), Not(x3)) truth table."""
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        expr = Or(And(x1, x2), Not(x3))
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    expected = (a and b) or (not c)
                    prob = cp.Problem(
                        cp.Minimize(0),
                        [x1 == a, x2 == b, x3 == c]
                    )
                    prob.solve(solver=cp.HIGHS)
                    assert np.isclose(expr.value, int(expected)), \
                        f"Or(And({a},{b}), Not({c})) = {expr.value}, expected {int(expected)}"


class TestLogicValidation:
    """Test validation of arguments."""

    def test_non_boolean_raises(self):
        x = cp.Variable()
        y = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, y)

    def test_non_boolean_not_raises(self):
        x = cp.Variable()
        with pytest.raises(ValueError, match="boolean"):
            Not(x)

    def test_too_few_args_and(self):
        x = cp.Variable(boolean=True)
        with pytest.raises(TypeError):
            And(x)

    def test_too_few_args_or(self):
        x = cp.Variable(boolean=True)
        with pytest.raises(TypeError):
            Or(x)

    def test_too_few_args_xor(self):
        x = cp.Variable(boolean=True)
        with pytest.raises(TypeError):
            Xor(x)

    def test_constant_raises(self):
        """Constants (even 0/1) are not boolean variables."""
        x = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, 1)


class TestLogicInConstraint:
    """Test logic expressions used in constraints."""

    def test_xor_constraint(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        prob = cp.Problem(
            cp.Minimize(0),
            [Xor(x1, x2) == 1, x1 == 1]
        )
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(x2.value, 0)

    def test_and_maximize(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        prob = cp.Problem(cp.Maximize(And(x, y)))
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(x.value, 1)
        assert np.isclose(y.value, 1)

    def test_or_minimize(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        prob = cp.Problem(cp.Minimize(Or(x, y)))
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(x.value, 0)
        assert np.isclose(y.value, 0)
