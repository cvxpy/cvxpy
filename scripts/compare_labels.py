"""
Quick comparison of name()/str() vs format_labeled() for the labels feature.

This mirrors the usage example from the PR description and prints both
the regular representation and the labeled representation.
"""

import numpy as np
import cvxpy as cp


def main():
    # Create variables
    weights = cp.Variable(3, name="weights")

    # Create constraints with custom labels
    constraints = [
        (weights >= 0).set_label("non_negative_weights"),
        (cp.sum(weights) == 1).set_label("budget_constraint"),
        (weights <= 0.4).set_label("concentration_limits"),
    ]

    # Create expressions with custom labels
    rng = np.random.default_rng(0)
    data = rng.standard_normal(3)
    data_fit = cp.sum_squares(weights - data).set_label("data_fit")
    l2_reg = cp.norm(weights, 2).set_label("l2_regularization")
    l1_reg = cp.norm(weights, 1).set_label("l1_regularization")

    # Compound expression with label
    regularization = (l2_reg + 0.1 * l1_reg).set_label("total_regularization")

    # Build objective
    objective_expr = data_fit + 0.5 * regularization
    objective = cp.Minimize(objective_expr)

    # Create the problem
    problem = cp.Problem(objective, constraints)

    print("=== str(problem) (uses name()/str()) ===")
    print(problem)
    print()

    print("=== problem.format_labeled() ===")
    print(problem.format_labeled())
    print()

    # Also show direct comparisons for the objective expression
    print("=== objective expression: name() vs format_labeled() ===")
    print("name():", objective_expr.name())
    print("format_labeled():", objective_expr.format_labeled())
    print()

    # Demonstrate operator precedence case with labels
    print("=== precedence demo: division and multiplication ===")
    x = cp.Variable(name="x").set_label("a")
    y = cp.Variable(name="y").set_label("b")
    z = cp.Variable(name="z").set_label("c")
    expr = x / (y * z)
    print("name():", expr.name())
    print("format_labeled():", expr.format_labeled())


if __name__ == "__main__":
    main()

