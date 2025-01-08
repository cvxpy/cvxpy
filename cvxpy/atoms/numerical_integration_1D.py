"""
CVXPY Atom for Numerical Integration
"""

from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.expressions.expression import Expression
import numpy as np


def numerical_integration_1D(f_callable, w, x_start, x_end, num_points=100, method="trapezoidal"):
    """
    Numerical integration using CVXPY framework.
    
    Parameters:
    - f_callable: Function f(x, w) constructed using CVXPY atoms.
    - w: CVXPY variable or expression the function depends on.
    - x_start: Start of the integration range.
    - x_end: End of the integration range.
    - num_points: Number of points to discretize the integration range.
    - method: Numerical integration method ("trapezoidal" or "simpson").
    
    Returns:
    - CVXPY expression for the integral.
    """
    # Discretize the x range
    x_values = np.linspace(x_start, x_end, num_points)
    dx = (x_end - x_start) / (num_points - 1)

    # Evaluate the function at each discretized x
    f_values = [f_callable(x, w) for x in x_values]

    # Compute the integral using the specified method
    if method == "trapezoidal":
        # Apply the trapezoidal rule
        integral = 0.5 * dx * (f_values[0] + f_values[-1])  # First and last points
        integral += dx * cvx_sum(f_values[1:-1])  # Interior points summed
    elif method == "simpson":
        if len(x_values) % 2 == 0:
            raise ValueError("Simpson's rule requires an odd number of points.")
        # Apply Simpson's rule
        integral = dx / 3 * (
            f_values[0]
            + f_values[-1]
            + 4 * cvx_sum(f_values[1:-1:2])  # Odd-indexed points
            + 2 * cvx_sum(f_values[2:-2:2])  # Even-indexed points
        )
    else:
        raise ValueError(f"Unknown integration method: {method}")

    return integral




