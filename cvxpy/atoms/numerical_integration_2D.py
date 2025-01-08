"""
2D Numerical Integration with CVXPY (Vectorized)
"""

import numpy as np
from cvxpy.atoms.affine.sum import sum as cvx_sum


def numerical_integration_2D(f_callable, w, x_range, y_range, g_list, num_points=(100, 100),
                             method="trapezoidal", sampling_mode="grid", grid_points=10, monte_carlo_samples=1000):
    """
    Numerical integration for 2D domains using CVXPY framework (Vectorized).

    Parameters:
    - f_callable: Function f(x, y, w) constructed using CVXPY atoms.
    - w: CVXPY variable or expression the function depends on.
    - x_range: Tuple (x_start, x_end) defining the range of x.
    - y_range: Tuple (y_start, y_end) defining the range of y.
    - g_list: List of boundary functions g_i(x, y) <= 0 defining the domain.
    - num_points: Tuple (num_x, num_y) defining discretization points for x and y.
    - method: Numerical integration method ("trapezoidal", "simpson", "monte_carlo").
    - sampling_mode: "grid" or "monte_carlo" for fractional area computation.
    - grid_points: Number of subdivisions for grid sampling.
    - monte_carlo_samples: Number of samples for Monte Carlo sampling.

    Returns:
    - CVXPY expression for the integral.
    """
    # Discretize the x and y ranges
    x_values = np.linspace(*x_range, num_points[0])
    y_values = np.linspace(*y_range, num_points[1])
    dx = (x_range[1] - x_range[0]) / (num_points[0] - 1)
    dy = (y_range[1] - y_range[0]) / (num_points[1] - 1)

    # Create grid for x and y values
    X, Y = np.meshgrid(x_values, y_values, indexing='ij')

    # Evaluate boundary conditions
    domain_mask = np.all([g(X, Y) <= 0 for g in g_list], axis=0)

    # Compute fractional tile areas
    tile_areas = np.where(domain_mask, dx * dy, 0)
    if not np.all(domain_mask):  # Handle boundary tiles
        if sampling_mode == "grid":
            fractional_areas = _compute_fractional_area_grid_vectorized(X, Y, dx, dy, g_list, grid_points)
        elif sampling_mode == "monte_carlo":
            fractional_areas = _compute_fractional_area_monte_carlo_vectorized(X, Y, dx, dy, g_list, monte_carlo_samples)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        tile_areas = np.where(domain_mask, tile_areas, fractional_areas)

    # Evaluate the integrand for valid points
    f_values = f_callable(X, Y, w) * tile_areas

    # Compute the integral using the specified method
    if method == "trapezoidal":
        return cvx_sum(f_values)
    elif method == "simpson":
        return _simpsons_integral_vectorized(f_values, dx, dy)
    elif method == "monte_carlo":
        return _monte_carlo_integral_vectorized(f_values, x_range, y_range, monte_carlo_samples)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _compute_fractional_area_grid_vectorized(X, Y, dx, dy, g_list, grid_points):
    """Vectorized fractional area computation using grid sampling."""
    sub_dx = dx / grid_points
    sub_dy = dy / grid_points

    # Create subgrid for each tile
    sub_x_offsets = np.linspace(0, dx, grid_points + 1)
    sub_y_offsets = np.linspace(0, dy, grid_points + 1)
    sub_x, sub_y = np.meshgrid(sub_x_offsets, sub_y_offsets, indexing='ij')

    sub_x = X[..., None, None] + sub_x  # Offset by X
    sub_y = Y[..., None, None] + sub_y  # Offset by Y

    # Evaluate boundary conditions for subgrid
    domain_mask = np.all([g(sub_x, sub_y) <= 0 for g in g_list], axis=0)

    # Fractional areas
    inside_count = np.sum(domain_mask, axis=(-2, -1))
    total_count = (grid_points + 1) ** 2
    fractional_area = (inside_count / total_count) * (dx * dy)

    return fractional_area


def _compute_fractional_area_monte_carlo_vectorized(X, Y, dx, dy, g_list, monte_carlo_samples):
    """Vectorized fractional area computation using Monte Carlo sampling."""
    # Generate random offsets within each tile
    random_offsets_x = np.random.uniform(0, dx, (monte_carlo_samples,))
    random_offsets_y = np.random.uniform(0, dy, (monte_carlo_samples,))

    sampled_x = X[..., None] + random_offsets_x
    sampled_y = Y[..., None] + random_offsets_y

    # Evaluate boundary conditions for sampled points
    domain_mask = np.all([g(sampled_x, sampled_y) <= 0 for g in g_list], axis=0)

    # Fractional areas
    inside_count = np.sum(domain_mask, axis=-1)
    fractional_area = (inside_count / monte_carlo_samples) * (dx * dy)

    return fractional_area


def _simpsons_integral_vectorized(f_values, dx, dy):
    """Apply Simpson's rule for integration (vectorized)."""
    weights_x = np.ones(f_values.shape[0])
    weights_x[1:-1:2] = 4
    weights_x[2:-2:2] = 2

    weights_y = np.ones(f_values.shape[1])
    weights_y[1:-1:2] = 4
    weights_y[2:-2:2] = 2

    weight_matrix = np.outer(weights_x, weights_y)
    weighted_sum = cvx_sum(f_values * weight_matrix)

    return weighted_sum * dx * dy / 9


def _monte_carlo_integral_vectorized(f_values, x_range, y_range, monte_carlo_samples):
    """Monte Carlo integration using cvx.sum (vectorized)."""
    integral = cvx_sum(f_values)
    bounding_area = (x_range[1] - x_range[0]) * (y_range[1] - y_range[0])
    return integral * bounding_area / monte_carlo_samples

