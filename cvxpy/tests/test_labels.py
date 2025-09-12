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

Tests for the label feature that allows custom names for expressions and constraints.
"""

import numpy as np

import cvxpy as cp

# ============================================================================
# CORE TESTS - Essential functionality
# ============================================================================


def test_expression_label_basics():
    """Test basic label property and setter."""
    x = cp.Variable(3)
    expr = cp.sum(x)

    # Initially no label
    assert expr.label is None

    # Set and retrieve label
    expr.set_label("total")
    assert expr.label == "total"

    # Method chaining
    expr2 = cp.norm(x).set_label("magnitude")
    assert expr2.label == "magnitude"

    # Clear label
    expr.set_label(None)
    assert expr.label is None
    
    # Non-string values should be converted to string
    expr.label = 42
    assert expr.label == "42"
    assert isinstance(expr.label, str)
    
    expr.set_label(3.14)
    assert expr.label == "3.14"
    assert isinstance(expr.label, str)
    
    # Property setter can also clear label
    expr.label = None
    assert expr.label is None
    
    # Test with Path-like object (if available)
    try:
        from pathlib import Path
        expr.label = Path("my/path")
        assert expr.label == "my/path"
        assert isinstance(expr.label, str)
    except ImportError:
        pass  # pathlib not available in older Python


def test_expression_format_labeled_simple():
    """Test format_labeled on simple expressions."""
    x = cp.Variable(3, name="x")

    # Without label, format_labeled returns mathematical form
    expr = cp.sum(x)
    assert "Sum(x" in expr.format_labeled()  # Don't test exact format

    # With label, format_labeled returns the label
    expr.set_label("total")
    assert expr.format_labeled() == "total"


def test_expression_format_labeled_recursive():
    """Test that format_labeled recursively substitutes labels in compound expressions."""
    x = cp.Variable(3, name="x")

    # Create labeled sub-expressions
    cost = cp.sum_squares(x).set_label("cost")
    penalty = cp.norm(x).set_label("penalty")

    # Compound expression without its own label
    objective = cost + 2 * penalty

    # format_labeled should recursively use labels
    formatted = objective.format_labeled()
    assert "cost" in formatted
    assert "penalty" in formatted

    # Regular str/name should NOT show labels
    assert "cost" not in str(objective)
    assert "penalty" not in str(objective)


def test_constraint_label_shows_in_str():
    """Test that constraint labels appear in str() output."""
    x = cp.Variable(3, name="x")

    # Constraint without label
    con = x >= 0
    str_without_label = str(con)
    assert ":" not in str_without_label  # No label separator

    # Constraint with label
    con.set_label("non_negative")
    assert str(con) == f"non_negative: {str_without_label}"

    # format_labeled should be same as str for constraints
    assert con.format_labeled() == str(con)
    
    # Non-string values should be converted to string
    con.label = 123
    assert con.label == "123"
    assert isinstance(con.label, str)
    assert "123:" in str(con)


def test_problem_format_labeled():
    """Test Problem.format_labeled with labeled objectives and constraints."""
    x = cp.Variable(3, name="x")

    # Create labeled objective
    cost = cp.sum_squares(x).set_label("cost")
    penalty = cp.norm(x).set_label("penalty")
    objective = cp.Minimize(cost + penalty)

    # Create labeled constraints
    constraints = [(x >= 0).set_label("non_negative"), (cp.sum(x) == 1).set_label("budget")]

    problem = cp.Problem(objective, constraints)

    # format_labeled should show all labels
    formatted = problem.format_labeled()
    assert "cost" in formatted
    assert "penalty" in formatted
    assert "non_negative:" in formatted
    assert "budget:" in formatted

    # Regular str should show labels only for constraints
    regular = str(problem)
    assert "cost" not in regular  # Objective labels don't show
    assert "non_negative:" in regular  # Constraint labels do show


# ============================================================================
# EDGE CASES - Important but not critical
# ============================================================================


def test_label_termination():
    """Test that format_labeled terminates at the first label it finds."""
    x = cp.Variable(3, name="x")

    # Create nested labeled expressions
    base = cp.sum_squares(x).set_label("base")
    scaled = 2 * base
    total = scaled + cp.norm(x)

    # Give the top-level expression a label
    total.set_label("objective")

    # Should return just "objective", not recurse deeper
    assert total.format_labeled() == "objective"

    # Without top-level label, should recurse
    total.set_label(None)
    formatted = total.format_labeled()
    assert "base" in formatted


def test_mixed_labeled_unlabeled():
    """Test expressions with mix of labeled and unlabeled sub-expressions."""
    x = cp.Variable(3, name="x")
    y = cp.Variable(3, name="y")

    # Some expressions labeled, some not
    expr1 = cp.sum(x).set_label("x_sum")
    expr2 = cp.sum(y)  # No label
    expr3 = cp.norm(x).set_label("x_norm")

    # Build compound expression
    result = expr1 + expr2 - expr3

    # Should show labels where available, math where not
    formatted = result.format_labeled()
    assert "x_sum" in formatted
    assert "Sum(y" in formatted  # Unlabeled shows math


# ============================================================================
# OPTIONAL/ADVANCED TESTS - Can be skipped for initial review
# ============================================================================


def test_division_multiplication_precedence():
    """
    Test correct parenthesization of division and multiplication.
    This tests a bug fix where a / (b * c) was incorrectly formatted as a / b @ c.
    """
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")
    z = cp.Variable(name="z")
    
    # Test division with multiplication on right needs parentheses
    expr = x / (y * z)
    assert str(expr) == "x / (y @ z)"  # Should have parentheses
    
    # Test with labels
    a = x.set_label("a")
    b = y.set_label("b")
    c = z.set_label("c")
    
    expr_labeled = a / (b * c)
    assert expr_labeled.format_labeled() == "a / (b @ c)"
    
    # Test that multiplication followed by division doesn't add extra parens
    expr2 = x * y / z
    assert str(expr2) == "x @ y / z"  # No parentheses needed
    
    # Test with labeled version
    expr2_labeled = a * b / c
    assert expr2_labeled.format_labeled() == "a @ b / c"
    
    # Additional precedence tests
    # a + b * c should not have parens (multiplication has higher precedence)
    expr3 = a + b * c
    assert "a + b @ c" in expr3.format_labeled()
    
    # a * (b + c) should have parens
    expr4 = a * (b + c)
    assert "(b + c)" in expr4.format_labeled()
    
    # (a / b) * c vs a / (b * c)
    expr5 = (a / b) * c
    # Division gets parens when used in multiplication
    assert expr5.format_labeled() == "(a / b) @ c"
    
    expr6 = a / (b * c)
    assert expr6.format_labeled() == "a / (b @ c)"  # Needs parens


def test_matrix_expressions_with_labels():
    """
    OPTIONAL: Test labels on matrix operations.
    Tests that labels work with matrix-specific operations like trace.
    """
    X = cp.Variable((3, 3), name="X")

    # Matrix operation with label
    trace_X = cp.trace(X).set_label("trace_X")

    # Should preserve label
    assert trace_X.format_labeled() == "trace_X"

    # In compound expression
    expr = trace_X + cp.norm(X, "fro")
    assert "trace_X" in expr.format_labeled()


def test_parameterized_expressions():
    """
    OPTIONAL: Test labels with parameters.
    Ensures labels work with parameterized problems.
    """
    x = cp.Variable(3, name="x")
    a = cp.Parameter(3, name="a", value=np.array([1, 2, 3]))

    # Parameterized expression with label
    param_expr = (a @ x).set_label("weighted_sum")

    # Should preserve label
    assert param_expr.format_labeled() == "weighted_sum"


def test_deeply_nested_labels():
    """
    OPTIONAL: Test labels in deeply nested expressions.
    Edge case for complex expression trees.
    """
    x = cp.Variable(3, name="x")

    # Build nested expression with labels at various levels
    a = cp.sum(x).set_label("a")
    b = cp.norm(x).set_label("b")
    c = (a + b).set_label("a_plus_b")
    d = cp.sum_squares(x)  # No label
    e = c * 2 + d
    f = -e  # Negation

    # Check format before labeling e
    pre_label_format = f.format_labeled()
    assert "a_plus_b" in pre_label_format  # Shows the nested label
    # Note: Can't check for absence of "e" because "quad_over_lin" contains "e"

    # Now label e
    e.set_label("total_expr")

    # The negation should now show the label
    post_label_format = f.format_labeled()
    assert post_label_format == "-(total_expr)"  # Label with negation and parens
    assert "a_plus_b" not in post_label_format  # Nested label no longer shown


# ============================================================================
# KNOWN LIMITATIONS - Document expected behavior
# ============================================================================


def test_label_flattening_limitation():
    """
    Documents a known limitation where labels may be lost during expression flattening.
    This is expected behavior due to CVXPY's internal optimizations.
    """
    x = cp.Variable(3, name="x")
    y = cp.Variable(3, name="y")

    # Create a labeled compound expression
    expr1 = cp.sum(x).set_label("x_sum")
    expr2 = cp.sum(y)
    left = (expr1 + expr2).set_label("left_total")

    # When used in further operations, CVXPY may flatten the expression
    # and the "left_total" label might be lost
    result = left - cp.norm(x)

    # Document what actually happens
    formatted = result.format_labeled()
    # LIMITATION: Labeled compound (left_total) gets flattened
    # Expected might be: 'left_total - Pnorm(x, 2)'
    # Actually get: 'x_sum + Sum(y) + -(Pnorm(x, 2))'
    
    # The "left_total" label is lost because CVXPY flattens (a+b)-c into a+b+(-c)
    # This is expected behavior, not a bug
    assert "x_sum" in formatted  # Individual labels are preserved
    assert "left_total" not in formatted  # But the compound label is lost
