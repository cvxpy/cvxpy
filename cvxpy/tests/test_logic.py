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
from cvxpy.atoms.elementwise.logic import And, Not, Or, Xor, iff, implies


def _eval(expr, constraints=None):
    """Solve a feasibility problem and return the value of expr via y == expr."""
    y = cp.Variable(expr.shape)
    prob = cp.Problem(cp.Minimize(0), [y == expr] + (constraints or []))
    prob.solve(solver=cp.HIGHS)
    return y.value


class TestLogicName:
    """Tests for name() and format_labeled() pretty printing."""

    def test_basic_names(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        assert Not(x).name() == "~x"
        assert And(x, y).name() == "x & y"
        assert Or(x, y).name() == "x | y"
        assert Xor(x, y).name() == "x ^ y"

    def test_nary_names(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        z = cp.Variable(boolean=True, name='z')
        assert And(x, y, z).name() == "x & y & z"
        assert Or(x, y, z).name() == "x | y | z"
        assert Xor(x, y, z).name() == "x ^ y ^ z"

    def test_not_parenthesizes_binary(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        assert Not(And(x, y)).name() == "~(x & y)"
        assert Not(Or(x, y)).name() == "~(x | y)"
        assert Not(Xor(x, y)).name() == "~(x ^ y)"

    def test_not_no_parens_for_leaf(self):
        x = cp.Variable(boolean=True, name='x')
        assert Not(x).name() == "~x"
        assert Not(Not(x)).name() == "~~x"

    def test_precedence(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        z = cp.Variable(boolean=True, name='z')
        # & is higher precedence than | and ^
        assert And(Or(x, y), z).name() == "(x | y) & z"
        assert And(x, Xor(y, z)).name() == "x & (y ^ z)"
        assert And(And(x, y), z).name() == "x & y & z"
        # ^ parenthesizes | but not &
        assert Xor(Or(x, y), z).name() == "(x | y) ^ z"
        assert Xor(And(x, y), z).name() == "x & y ^ z"
        # | is lowest: never parenthesizes
        assert Or(And(x, y), z).name() == "x & y | z"
        assert Or(Xor(x, y), z).name() == "x ^ y | z"

    def test_complex_expression(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        assert Or(And(x, y), And(Not(x), Not(y))).name() == \
            "x & y | ~x & ~y"

    def test_format_labeled(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        assert And(x, y).set_label("my_and").format_labeled() == "my_and"
        inner = And(x, y).set_label("both")
        assert Or(inner, Not(x)).format_labeled() == "both | ~x"


class TestLogicProperties:
    """Tests for types, DCP, monotonicity, and log-log properties."""

    def test_expression_types(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        assert isinstance(Not(x), Not)
        assert isinstance(And(x, y), And)
        assert isinstance(Or(x, y), Or)
        assert isinstance(Xor(x, y), Xor)

    def test_dcp_compliance(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        for expr in [And(x, y), Or(x, y), Not(x), Xor(x, y)]:
            assert expr.is_dcp()

    def test_namespace(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        assert isinstance(cp.logic.Not(x), Not)
        assert isinstance(cp.logic.And(x, y), And)
        assert isinstance(cp.logic.Or(x, y), Or)
        assert isinstance(cp.logic.Xor(x, y), Xor)
        assert isinstance(cp.logic.implies(x, y), Or)
        assert isinstance(cp.logic.iff(x, y), Not)

    def test_monotonicity(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        # Not is decreasing
        assert Not(x).is_decr(0) and not Not(x).is_incr(0)
        # And/Or are increasing
        for cls in (And, Or):
            expr = cls(x, y)
            for i in range(2):
                assert expr.is_incr(i) and not expr.is_decr(i)
        # Xor is neither
        xor = Xor(x, y)
        for i in range(2):
            assert not xor.is_incr(i) and not xor.is_decr(i)

    def test_not_log_log(self):
        """Logic atoms are not log-log convex or concave (domain includes 0)."""
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        for expr in [Not(x), And(x, y), Or(x, y), Xor(x, y)]:
            assert not expr.is_atom_log_log_convex()
            assert not expr.is_atom_log_log_concave()


class TestLogicSolve:
    """Solve-based truth table tests for logic atoms."""

    @pytest.mark.parametrize("val,expected", [(0, 1), (1, 0)])
    def test_not_truth_table(self, val, expected):
        x = cp.Variable(boolean=True)
        assert np.isclose(_eval(Not(x), [x == val]), expected)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 1),
    ])
    def test_and_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert np.isclose(_eval(And(x, z), [x == a, z == b]), expected)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 1),
    ])
    def test_or_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert np.isclose(_eval(Or(x, z), [x == a, z == b]), expected)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0),
    ])
    def test_xor_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert np.isclose(_eval(Xor(x, z), [x == a, z == b]), expected)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 1), (0, 1, 1), (1, 0, 0), (1, 1, 1),
    ])
    def test_implies_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert np.isclose(_eval(implies(x, z), [x == a, z == b]), expected)

    @pytest.mark.parametrize("a,b,expected", [
        (0, 0, 1), (0, 1, 0), (1, 0, 0), (1, 1, 1),
    ])
    def test_iff_truth_table(self, a, b, expected):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert np.isclose(_eval(iff(x, z), [x == a, z == b]), expected)

    def test_nary(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        def fix(a, b, c):
            return [x1 == a, x2 == b, x3 == c]
        # And: all-true -> 1, one-false -> 0
        assert np.isclose(_eval(And(x1, x2, x3), fix(1, 1, 1)), 1)
        assert np.isclose(_eval(And(x1, x2, x3), fix(1, 0, 1)), 0)
        # Or: all-false -> 0, one-true -> 1
        assert np.isclose(_eval(Or(x1, x2, x3), fix(0, 0, 0)), 0)
        assert np.isclose(_eval(Or(x1, x2, x3), fix(0, 1, 0)), 1)
        # Xor parity: 0->0, 1->1, 2->0, 3->1
        assert np.isclose(_eval(Xor(x1, x2, x3), fix(0, 0, 0)), 0)
        assert np.isclose(_eval(Xor(x1, x2, x3), fix(1, 0, 0)), 1)
        assert np.isclose(_eval(Xor(x1, x2, x3), fix(1, 1, 0)), 0)
        assert np.isclose(_eval(Xor(x1, x2, x3), fix(1, 1, 1)), 1)

    def test_vector(self):
        x = cp.Variable(3, boolean=True)
        z = cp.Variable(3, boolean=True)
        fix = [x == np.array([1, 0, 1]), z == np.array([1, 1, 0])]
        np.testing.assert_array_almost_equal(_eval(And(x, z), fix), [1, 0, 0])
        np.testing.assert_array_almost_equal(_eval(Or(x, z), fix), [1, 1, 1])
        np.testing.assert_array_almost_equal(_eval(Not(x), fix), [0, 1, 0])

    def test_composition(self):
        """Or(And(x1, x2), Not(x3)) truth table."""
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        x3 = cp.Variable(boolean=True)
        for a in [0, 1]:
            for b in [0, 1]:
                for c in [0, 1]:
                    expected = int((a and b) or (not c))
                    got = _eval(Or(And(x1, x2), Not(x3)),
                                [x1 == a, x2 == b, x3 == c])
                    assert np.isclose(got, expected)


class TestLogicOperators:
    """Test ~, &, |, ^ operator syntax on boolean variables."""

    def test_operators(self):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        assert isinstance(~x, Not)
        assert isinstance(x & z, And)
        assert isinstance(x | z, Or)
        assert isinstance(x ^ z, Xor)
        assert np.isclose(_eval(~x, [x == 1]), 0)
        assert np.isclose(_eval(x & z, [x == 1, z == 0]), 0)
        assert np.isclose(_eval(x | z, [x == 0, z == 1]), 1)
        assert np.isclose(_eval(x ^ z, [x == 1, z == 1]), 0)

    def test_composed_operators(self):
        """(x & z) | (~x & ~z) is XNOR."""
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        expr = (x & z) | (~x & ~z)
        for a in [0, 1]:
            for b in [0, 1]:
                assert np.isclose(_eval(expr, [x == a, z == b]), int(a == b))

    def test_invert_on_logic_expr(self):
        x = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        expr = ~(x & z)
        assert isinstance(expr, Not)
        assert np.isclose(_eval(expr, [x == 1, z == 1]), 0)
        assert np.isclose(_eval(expr, [x == 1, z == 0]), 1)

    def test_non_boolean_raises(self):
        x = cp.Variable()
        y = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            x & y
        with pytest.raises(ValueError, match="boolean"):
            x | y
        with pytest.raises(ValueError, match="boolean"):
            x ^ y
        with pytest.raises(ValueError, match="boolean"):
            ~x

    def test_vector_operators(self):
        x = cp.Variable(3, boolean=True)
        z = cp.Variable(3, boolean=True)
        fix = [x == np.array([1, 0, 1]), z == np.array([1, 1, 0])]
        np.testing.assert_array_almost_equal(_eval(x & z, fix), [1, 0, 0])
        np.testing.assert_array_almost_equal(_eval(x | z, fix), [1, 1, 1])
        np.testing.assert_array_almost_equal(_eval(x ^ z, fix), [0, 1, 1])
        np.testing.assert_array_almost_equal(_eval(~x, fix), [0, 1, 0])


class TestLogicValidation:
    """Test validation of arguments."""

    def test_non_boolean_raises(self):
        x = cp.Variable()
        y = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, y)
        with pytest.raises(ValueError, match="boolean"):
            Not(x)

    def test_too_few_args(self):
        x = cp.Variable(boolean=True)
        with pytest.raises(TypeError):
            And(x)
        with pytest.raises(TypeError):
            Or(x)
        with pytest.raises(TypeError):
            Xor(x)

    def test_numeric_constant_raises(self):
        """Integer/float constants (even 0/1) are not boolean-typed."""
        x = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, 1)
        with pytest.raises(ValueError, match="boolean"):
            And(x, 1.0)


class TestLogicBoolConstant:
    """Test that Python bool and np.bool_ constants are accepted."""

    def test_scalar_bool_constants(self):
        x = cp.Variable(boolean=True)
        assert np.isclose(_eval(And(x, True), [x == 1]), 1)
        assert np.isclose(_eval(And(x, False), [x == 1]), 0)
        assert np.isclose(_eval(Or(x, np.bool_(True)), [x == 0]), 1)
        assert np.isclose(_eval(Not(True)), 0)
        assert np.isclose(_eval(Xor(x, True), [x == 1]), 0)
        assert np.isclose(_eval(Xor(x, True), [x == 0]), 1)

    def test_bool_array_constant(self):
        x = cp.Variable(3, boolean=True)
        mask = np.array([True, False, True])
        fix = [x == np.array([1, 1, 0])]
        np.testing.assert_array_almost_equal(_eval(And(x, mask), fix), [1, 0, 0])
        np.testing.assert_array_almost_equal(_eval(x & mask, fix), [1, 0, 0])

    def test_operator_with_bool_constant(self):
        x = cp.Variable(boolean=True)
        fix = [x == 1]
        assert np.isclose(_eval(x & True, fix), 1)
        assert np.isclose(_eval(x | False, fix), 1)
        assert np.isclose(_eval(x ^ True, fix), 0)

    def test_sparse_bool_constant(self):
        import scipy.sparse as sp
        x = cp.Variable((2, 2), boolean=True)
        mask = sp.csc_matrix(np.array([[True, False], [False, True]]))
        np.testing.assert_array_almost_equal(
            _eval(And(x, mask), [x == np.ones((2, 2))]),
            np.array([[1, 0], [0, 1]])
        )


class TestLogicInConstraint:
    """Test logic expressions used in constraints."""

    def test_xor_constraint(self):
        x1 = cp.Variable(boolean=True)
        x2 = cp.Variable(boolean=True)
        prob = cp.Problem(cp.Minimize(0), [Xor(x1, x2) == 1, x1 == 1])
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
