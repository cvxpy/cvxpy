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

# Basic Behavior


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

    # Clear label using set_label
    expr.set_label(None)
    assert expr.label is None
    
    # Test deleter
    expr.label = "test"
    assert expr.label == "test"
    del expr.label
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
    
    # Test with Path-like object
    from pathlib import Path
    expr.label = Path("my/path")
    # Path will use OS-appropriate separator when converted to string
    assert expr.label in ["my/path", "my\\path"]  # Unix or Windows
    assert isinstance(expr.label, str)


def test_expression_format_labeled_simple():
    """Test format_labeled on simple expressions."""
    x = cp.Variable(3, name="x")

    # Without label, format_labeled returns exact mathematical form
    expr = cp.sum(x)
    assert expr.format_labeled() == "Sum(x, None, False)"

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
    """Constraint labels appear in str() output and format_labeled()."""
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
    
    # Test deleter for constraints
    del con.label
    assert con.label is None
    assert ":" not in str(con)


def test_problem_format_labeled():
    """Problem.format_labeled shows labels for objective expressions and constraints."""
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


# Expression Composition


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


def test_various_operations_with_labels():
    """Labels propagate through common operations and atoms (concise checks)."""
    x = cp.Variable(3, name="x")
    y = cp.Variable(3, name="y")
    A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])  # Diagonal matrix for quad_form
    
    # Set labels
    x.set_label("x_vec")
    y.set_label("y_vec")
    
    # Test various operations inherit format_labeled correctly
    # These use default implementation from Expression
    
    # Multi-arg operations
    vstack_expr = cp.vstack([x, y])
    assert vstack_expr.format_labeled() == "Vstack(x_vec, y_vec)"
    
    hstack_expr = cp.hstack([x, y])
    assert hstack_expr.format_labeled() == "Hstack(x_vec, y_vec)"
    
    # Norms with custom name() methods
    norm_expr = cp.norm(x)
    assert norm_expr.format_labeled() == "Pnorm(x_vec, 2)"
    
    norm_inf_expr = cp.norm_inf(x)
    assert norm_inf_expr.format_labeled() == "norm_inf(x_vec)"
    
    norm1_expr = cp.norm1(x)
    assert norm1_expr.format_labeled() == "norm1(x_vec)"
    
    # Other atoms with custom name() methods
    sum_expr = cp.sum(x)
    # Sum shows axis=None and keepdims=False explicitly
    assert sum_expr.format_labeled() == "Sum(x_vec, None, False)"
    
    transpose_expr = cp.transpose(x)
    assert transpose_expr.format_labeled() == "x_vec.T"
    
    quad_form_expr = cp.quad_form(x, A)
    assert quad_form_expr.format_labeled() == (
        "QuadForm(x_vec, [[1.00 0.00 0.00]\n"
        " [0.00 2.00 0.00]\n"
        " [0.00 0.00 3.00]])"
    )
    
    power_expr = cp.power(cp.sum(x), 2)
    assert power_expr.format_labeled() == "power(Sum(x_vec, None, False), 2.0)"
    
    # Test indexing (has custom name method)
    index_expr = x[0]
    assert index_expr.format_labeled() == "x_vec[0]"
    
    # Test geo_mean (weights sum to 1 after normalization)
    weights = np.array([0.3, 0.3, 0.4])
    geo_mean_expr = cp.geo_mean(x, weights)
    assert geo_mean_expr.format_labeled() == "geo_mean(x_vec[True, True, True], (3/10, 3/10, 2/5))"
    
    # Test that operations can themselves be labeled
    norm_expr.set_label("x_magnitude")
    assert norm_expr.format_labeled() == "x_magnitude"
    
    norm1_expr.set_label("l1_norm")
    assert norm1_expr.format_labeled() == "l1_norm"
    
    # Test in a compound expression
    objective = norm_expr + sum_expr.set_label("x_total")
    formatted = objective.format_labeled()
    assert "x_magnitude" in formatted
    assert "x_total" in formatted
    
    # Test compound with multiple labeled norms
    compound = norm1_expr + 2 * norm_expr
    formatted = compound.format_labeled()
    assert "l1_norm" in formatted
    assert "x_magnitude" in formatted


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
    assert formatted == "x_sum + Sum(y, None, False) + -(x_norm)"


# Precedence and Operators


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
    assert expr3.format_labeled() == "a + b @ c"
    
    # a * (b + c) should have parens
    expr4 = a * (b + c)
    assert expr4.format_labeled() == "a @ (b + c)"
    
    # (a / b) * c vs a / (b * c)
    expr5 = (a / b) * c
    # Division gets parens when used in multiplication
    assert expr5.format_labeled() == "(a / b) @ c"
    
    expr6 = a / (b * c)
    assert expr6.format_labeled() == "a / (b @ c)"  # Needs parens


def test_matrix_expressions_with_labels():
    """Labels on matrix operations (e.g., trace) and in compound expressions."""
    X = cp.Variable((3, 3), name="X")

    # Matrix operation with label
    trace_X = cp.trace(X).set_label("trace_X")

    # Should preserve label
    assert trace_X.format_labeled() == "trace_X"

    # In compound expression
    expr = trace_X + cp.norm(X, "fro")
    assert expr.format_labeled() == "trace_X + Pnorm(reshape(X, (9,), F), 2)"


def test_parameterized_expressions():
    """Labels on parameterized expressions are preserved."""
    x = cp.Variable(3, name="x")
    a = cp.Parameter(3, name="a", value=np.array([1, 2, 3]))

    # Parameterized expression with label
    param_expr = (a @ x).set_label("weighted_sum")

    # Should preserve label
    assert param_expr.format_labeled() == "weighted_sum"


def test_deeply_nested_labels():
    """Labels in deeply nested expressions with negation wrapping."""
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


# Limitations


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


def test_label_display_catalog_exact():
    """Catalog-style, exact-string checks for many atoms.

    Each assertion documents the precise display with labels propagated.
    """
    import numpy as _np

    # Scalars
    a = cp.Variable(name="a").set_label("aL")
    b = cp.Variable(name="b").set_label("bL")
    c = cp.Variable(name="c").set_label("cL")

    # Vectors / matrices
    x = cp.Variable(3, name="x").set_label("xL")
    y = cp.Variable(3, name="y").set_label("yL")
    X = cp.Variable((2, 2), name="X").set_label("XL")

    # Addition / negation
    assert (x + y).format_labeled() == "xL + yL"
    assert (-x).format_labeled() == "-xL"

    # Division / multiplication precedence
    assert (a / (b * c)).format_labeled() == "aL / (bL @ cL)"
    assert ((a / b) * c).format_labeled() == "(aL / bL) @ cL"

    # Transpose and indexing
    assert cp.transpose(x).format_labeled() == "xL.T"
    assert x[0].format_labeled() == "xL[0]"

    # Norms and power
    assert cp.norm(x + y).format_labeled() == "Pnorm(xL + yL, 2)"
    assert cp.norm1(x).format_labeled() == "norm1(xL)"
    assert cp.norm_inf(x).format_labeled() == "norm_inf(xL)"
    assert cp.power(cp.sum(x), 2).format_labeled() == "power(Sum(xL, None, False), 2.0)"

    # Quad form
    A = _np.diag([1, 2, 3])
    qf = cp.quad_form(x, A).format_labeled()
    assert qf == (
        "QuadForm(xL, [[1.00 0.00 0.00]\n"
        " [0.00 2.00 0.00]\n"
        " [0.00 0.00 3.00]])"
    )

    # Geometric mean (weights show exactly)
    g = cp.geo_mean(x, [1, 2, 1])
    assert g.format_labeled() == "geo_mean(xL[True, True, True], (1/4, 1/2, 1/4))"

    # Perron-Frobenius eigenvalue
    assert cp.pf_eigenvalue(X).format_labeled() == "pf_eigenvalue(XL)"

    # eye_minus_inv
    assert cp.eye_minus_inv(X).format_labeled() == "eye_minus_inv(XL)"

    # gmatmul (geometric matmul)
    A2 = _np.array([[1.0, 2.0], [0.0, 1.0]])
    Xpos = cp.Variable((2, 2), pos=True, name="Xp").set_label("XpL")
    gm = cp.gmatmul(A2, Xpos).format_labeled()
    assert gm == "gmatmul([[1.00 2.00]\n [0.00 1.00]], XpL)"


def test_format_labeled_parity_unlabeled():
    """For unlabeled expressions, format_labeled() equals name().

    This guards the default behavior and ensures overrides mirror name().
    """
    import numpy as _np

    # Variables/parameters/constants without labels
    x = cp.Variable(3, name="x")
    y = cp.Variable(3, name="y")
    X = cp.Variable((2, 2), name="X")

    cases = []
    # Simple
    cases.append(cp.sum(x))
    cases.append(cp.norm(x))
    cases.append(cp.norm1(x))
    cases.append(cp.norm_inf(x))
    cases.append(cp.power(cp.sum(x), 2))
    cases.append(cp.transpose(x))
    cases.append(x[0])
    # Stacks
    cases.append(cp.vstack([x, y]))
    cases.append(cp.hstack([x, y]))
    # Operators
    cases.append(x + y)
    a = cp.Variable(name="a")
    b = cp.Variable(name="b")
    c = cp.Variable(name="c")
    cases.append(a / (b * c))
    cases.append((a / b) * c)
    cases.append(-x)
    # Atoms with data or matrices
    cases.append(cp.quad_form(x, _np.diag([1, 2, 3])))
    cases.append(cp.geo_mean(x, [1, 2, 1]))
    cases.append(cp.pf_eigenvalue(X))
    cases.append(cp.eye_minus_inv(X))
    Xpos = cp.Variable((2, 2), pos=True, name="Xp")
    cases.append(cp.gmatmul(_np.array([[1.0, 2.0], [0.0, 1.0]]), Xpos))

    for expr in cases:
        assert expr.format_labeled() == expr.name()
