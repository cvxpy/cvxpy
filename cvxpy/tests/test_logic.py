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
        # Not(Not(x)) should not parenthesize inner Not
        assert Not(Not(x)).name() == "~~x"

    def test_and_parenthesizes_lower_precedence(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        z = cp.Variable(boolean=True, name='z')
        # & is higher precedence than | and ^
        assert And(Or(x, y), z).name() == "(x | y) & z"
        assert And(x, Xor(y, z)).name() == "x & (y ^ z)"
        # & with & children: no parens needed
        assert And(And(x, y), z).name() == "x & y & z"

    def test_xor_parenthesizes_or(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        z = cp.Variable(boolean=True, name='z')
        # ^ is higher precedence than |
        assert Xor(Or(x, y), z).name() == "(x | y) ^ z"
        # ^ with & child: no parens needed
        assert Xor(And(x, y), z).name() == "x & y ^ z"

    def test_or_no_parens(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        z = cp.Variable(boolean=True, name='z')
        # | is lowest precedence: no children need parens
        assert Or(And(x, y), z).name() == "x & y | z"
        assert Or(Xor(x, y), z).name() == "x ^ y | z"

    def test_complex_expression(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        # XNOR: (x & y) | (~x & ~y)
        assert Or(And(x, y), And(Not(x), Not(y))).name() == \
            "x & y | ~x & ~y"

    def test_format_labeled_uses_labels(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        expr = And(x, y).set_label("my_and")
        assert expr.format_labeled() == "my_and"

    def test_format_labeled_recurses(self):
        x = cp.Variable(boolean=True, name='x')
        y = cp.Variable(boolean=True, name='y')
        inner = And(x, y).set_label("both")
        outer = Or(inner, Not(x))
        assert outer.format_labeled() == "both | ~x"


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


class TestLogicMonotonicity:
    """Tests for monotonicity of logic atoms."""

    def test_not_is_decreasing(self):
        x = cp.Variable(boolean=True)
        expr = Not(x)
        assert expr.is_decr(0)
        assert not expr.is_incr(0)

    def test_and_is_increasing(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = And(x, y)
        assert expr.is_incr(0)
        assert expr.is_incr(1)
        assert not expr.is_decr(0)
        assert not expr.is_decr(1)

    def test_or_is_increasing(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = Or(x, y)
        assert expr.is_incr(0)
        assert expr.is_incr(1)
        assert not expr.is_decr(0)
        assert not expr.is_decr(1)

    def test_xor_is_neither(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = Xor(x, y)
        assert not expr.is_incr(0)
        assert not expr.is_incr(1)
        assert not expr.is_decr(0)
        assert not expr.is_decr(1)

    def test_nary_and_is_increasing(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        expr = And(x, y, z)
        for i in range(3):
            assert expr.is_incr(i)
            assert not expr.is_decr(i)

    def test_nary_or_is_increasing(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        z = cp.Variable(boolean=True)
        expr = Or(x, y, z)
        for i in range(3):
            assert expr.is_incr(i)
            assert not expr.is_decr(i)


class TestLogicSolve:
    """Solve-based truth table tests for logic atoms."""

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


class TestLogicOperators:
    """Test ~, &, |, ^ operator syntax on boolean variables."""

    def test_invert_operator(self):
        x = cp.Variable(boolean=True)
        expr = ~x
        assert isinstance(expr, Not)
        prob = cp.Problem(cp.Minimize(0), [x == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_and_operator(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = x & y
        assert isinstance(expr, And)
        prob = cp.Problem(cp.Minimize(0), [x == 1, y == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_or_operator(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = x | y
        assert isinstance(expr, Or)
        prob = cp.Problem(cp.Minimize(0), [x == 0, y == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_xor_operator(self):
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = x ^ y
        assert isinstance(expr, Xor)
        prob = cp.Problem(cp.Minimize(0), [x == 1, y == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_composed_operators(self):
        """Test (x & y) | (~x & ~y) which is XNOR."""
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = (x & y) | (~x & ~y)
        for a in [0, 1]:
            for b in [0, 1]:
                expected = int(a == b)
                prob = cp.Problem(cp.Minimize(0), [x == a, y == b])
                prob.solve(solver=cp.HIGHS)
                assert np.isclose(expr.value, expected), \
                    f"XNOR({a},{b}) = {expr.value}, expected {expected}"

    def test_invert_on_logic_expr(self):
        """~(x & y) should produce Not(And(x, y))."""
        x = cp.Variable(boolean=True)
        y = cp.Variable(boolean=True)
        expr = ~(x & y)
        assert isinstance(expr, Not)
        prob = cp.Problem(cp.Minimize(0), [x == 1, y == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)
        prob = cp.Problem(cp.Minimize(0), [x == 1, y == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_operator_non_boolean_raises(self):
        """Operators should raise on non-boolean variables."""
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
        """Operators work elementwise on vectors."""
        x = cp.Variable(3, boolean=True)
        y = cp.Variable(3, boolean=True)
        and_expr = x & y
        or_expr = x | y
        xor_expr = x ^ y
        not_expr = ~x
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 0, 1]), y == np.array([1, 1, 0])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(and_expr.value, [1, 0, 0])
        np.testing.assert_array_almost_equal(or_expr.value, [1, 1, 1])
        np.testing.assert_array_almost_equal(xor_expr.value, [0, 1, 1])
        np.testing.assert_array_almost_equal(not_expr.value, [0, 1, 0])


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

    def test_int_constant_raises(self):
        """Integer constants (even 0/1) are not boolean-typed."""
        x = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, 1)

    def test_float_constant_raises(self):
        """Float constants are not boolean-typed."""
        x = cp.Variable(boolean=True)
        with pytest.raises(ValueError, match="boolean"):
            And(x, 1.0)


class TestLogicBoolConstant:
    """Test that Python bool and np.bool_ constants are accepted."""

    def test_scalar_bool_constant(self):
        x = cp.Variable(boolean=True)
        expr = And(x, True)
        assert isinstance(expr, And)
        prob = cp.Problem(cp.Minimize(0), [x == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_scalar_bool_false(self):
        x = cp.Variable(boolean=True)
        expr = And(x, False)
        prob = cp.Problem(cp.Minimize(0), [x == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_np_bool_constant(self):
        x = cp.Variable(boolean=True)
        expr = Or(x, np.bool_(True))
        prob = cp.Problem(cp.Minimize(0), [x == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_bool_array_constant(self):
        x = cp.Variable(3, boolean=True)
        mask = np.array([True, False, True])
        expr = And(x, mask)
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 1, 0])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(expr.value, [1, 0, 0])

    def test_not_bool_constant(self):
        expr = Not(True)
        prob = cp.Problem(cp.Minimize(0))
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)

    def test_xor_with_bool_constant(self):
        x = cp.Variable(boolean=True)
        expr = Xor(x, True)
        prob = cp.Problem(cp.Minimize(0), [x == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 0)
        prob = cp.Problem(cp.Minimize(0), [x == 0])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(expr.value, 1)

    def test_operator_with_bool_constant(self):
        """Operators should work with bool constants via & | ^."""
        x = cp.Variable(boolean=True)
        and_expr = x & True
        or_expr = x | False
        xor_expr = x ^ True
        prob = cp.Problem(cp.Minimize(0), [x == 1])
        prob.solve(solver=cp.HIGHS)
        assert np.isclose(and_expr.value, 1)
        assert np.isclose(or_expr.value, 1)
        assert np.isclose(xor_expr.value, 0)

    def test_bool_array_operator(self):
        x = cp.Variable(3, boolean=True)
        mask = np.array([True, False, True])
        expr = x & mask
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.array([1, 1, 0])]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(expr.value, [1, 0, 0])

    def test_sparse_bool_constant(self):
        import scipy.sparse as sp
        x = cp.Variable((2, 2), boolean=True)
        mask = sp.csc_matrix(np.array([[True, False], [False, True]]))
        expr = And(x, mask)
        prob = cp.Problem(
            cp.Minimize(0),
            [x == np.ones((2, 2))]
        )
        prob.solve(solver=cp.HIGHS)
        np.testing.assert_array_almost_equal(
            expr.value, np.array([[1, 0], [0, 1]])
        )


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
