"""
ND Numerical Integration with CVXPY (Vectorized)
"""

import numpy as np
from cvxpy.atoms.affine.sum import sum as cvx_sum


def numerical_integration_ND(f_callable, w, ranges, g_list, num_points=None,
                             method="trapezoidal", sampling_mode="grid", grid_points=10, monte_carlo_samples=1000):
    """
    Numerical integration for ND domains using CVXPY framework (Vectorized).

    Parameters:
    - f_callable: Function f(*coords, w) constructed using CVXPY atoms.
    - w: CVXPY variable or expression the function depends on.
    - ranges: List of tuples [(x_start, x_end), (y_start, y_end), ...] defining the range for each dimension.
    - g_list: List of boundary functions g(coords) <= 0 defining the domain.
    - num_points: List of integers defining discretization points for each dimension.
    - method: Numerical integration method ("trapezoidal", "simpson", "monte_carlo").
    - sampling_mode: "grid" or "monte_carlo" for fractional area computation.
    - grid_points: Number of subdivisions for grid sampling.
    - monte_carlo_samples: Number of samples for Monte Carlo sampling.

    Returns:
    - CVXPY expression for the integral.
    """
    num_dimensions = len(ranges)

    if num_points is None:
        num_points = [100] * num_dimensions

    # Discretize the ranges for each dimension
    coords = [np.linspace(start, end, points) for (start, end), points in zip(ranges, num_points)]
    d_volumes = [(end - start) / (points - 1) for (start, end), points in zip(ranges, num_points)]
    d_volume = np.prod(d_volumes)

    # Create grid of points
    grids = np.meshgrid(*coords, indexing='ij')
    points = np.stack([grid.ravel() for grid in grids], axis=-1)  # Shape (num_points_total, num_dimensions)

    # Evaluate boundary conditions
    domain_mask = np.all(np.stack([g(*points.T) <= 0 for g in g_list], axis=0), axis=0)

    # Compute fractional tile volumes
    tile_volumes = d_volume * np.ones(points.shape[0], dtype=float)
    if not np.all(domain_mask):  # Handle boundary tiles
        boundary_points = points[~domain_mask]
        if sampling_mode == "grid":
            fractional_volumes = _compute_fractional_volume_grid_ND(boundary_points, d_volume, g_list, grid_points)
        elif sampling_mode == "monte_carlo":
            fractional_volumes = _compute_fractional_volume_monte_carlo_ND(boundary_points, d_volumes, g_list, monte_carlo_samples)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        tile_volumes[~domain_mask] = fractional_volumes

    # Evaluate the integrand at valid points
    valid_points = points[domain_mask]
    tile_volumes = tile_volumes[domain_mask]
    f_values = f_callable(*valid_points.T, w) * tile_volumes

    # Compute the integral using the specified method
    if method == "trapezoidal":
        return cvx_sum(f_values)
    elif method == "simpson":
        return _simpsons_integral_ND(f_values, d_volumes)
    elif method == "monte_carlo":
        return _monte_carlo_integral_ND(f_values, ranges, monte_carlo_samples)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _compute_fractional_volume_grid_ND(boundary_points, d_volume, g_list, grid_points):
    """Vectorized fractional volume computation using grid sampling for ND integration."""
    num_dimensions = boundary_points.shape[1]
    sub_deltas = d_volume ** (1 / num_dimensions) / grid_points

    sub_offsets = np.stack(
        np.meshgrid(*[np.linspace(0, sub_deltas * grid_points, grid_points + 1)] * num_dimensions, indexing='ij'),
        axis=-1
    ).reshape(-1, num_dimensions)  # Shape: (subgrid_points, num_dimensions)
    shifted_points = boundary_points[:, None, :] + sub_offsets[None, :, :]  # Shape: (num_boundary_points, subgrid_points, num_dimensions)
    inside_mask = np.all(
        np.stack([g(*shifted_points.T) <= 0 for g in g_list], axis=0),
        axis=0
    )  # Shape: (num_boundary_points, subgrid_points)

    inside_count = np.sum(inside_mask, axis=0)  # Shape: (num_boundary_points,)
    total_count = (grid_points + 1) ** num_dimensions
    fractional_volumes = (inside_count / total_count) * d_volume  # Shape: (num_boundary_points,)
    return fractional_volumes








def _compute_fractional_volume_monte_carlo_ND(boundary_points, d_volumes, g_list, monte_carlo_samples):
    """Vectorized fractional volume computation using Monte Carlo sampling for ND integration."""
    random_offsets = np.random.uniform(0, 1, size=(monte_carlo_samples, len(d_volumes))) * d_volumes
    shifted_points = boundary_points[:, None, :] + random_offsets[None, :, :]  # Shape (num_boundary_points, monte_carlo_samples, num_dimensions)

    inside_mask = np.all(np.stack([g(*shifted_points.T) <= 0 for g in g_list], axis=0), axis=0)
    inside_count = np.sum(inside_mask, axis=0)
    return (inside_count / monte_carlo_samples) * np.prod(d_volumes)


def _simpsons_integral_ND(f_values, d_volumes):
    """
    Apply Simpson's rule for ND integration (Vectorized).
    This function handles multi-dimensional integration with CVXPY expressions.
    """
    f_shape = f_values.shape
    num_dimensions = len(f_shape)


    weights = np.ones(f_shape)

    for dim in range(num_dimensions):
        index = [slice(None)] * num_dimensions  

        index[dim] = slice(1, -1, 2)  
        weights[tuple(index)] *= 4


        index[dim] = slice(2, -2, 2)  
        weights[tuple(index)] *= 2
        index[dim] = 0  
        weights[tuple(index)] = 1
        index[dim] = -1  
        weights[tuple(index)] = 1

    weighted_sum = cvx_sum(weights * f_values)
    return weighted_sum * np.prod(d_volumes) / 3




def _monte_carlo_integral_ND(f_values, ranges, monte_carlo_samples):
    """Monte Carlo integration for ND (Vectorized)."""
    integral = cvx_sum(f_values)
    bounding_volume = np.prod([end - start for start, end in ranges])
    return integral * bounding_volume / monte_carlo_samples
