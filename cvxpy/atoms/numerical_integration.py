"""
Copyright, the CVXPY Ashok Viswanathan

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
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from cvxpy.atoms.affine.binary_operators import MulExpression as cvx_multiply
from cvxpy.atoms.affine.sum import sum as cvx_sum
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


def numerical_integration(
    f_callable: Callable[..., Expression], 
    w: Union[Variable, Expression], 
    ranges: Union[Tuple[float, float], List[Tuple[float, float]]], 
    g_list: Optional[List[Callable[..., Expression]]] = None, 
    num_points: Union[int, List[int]] = 100,
    method: str = "trapezoidal",
    sampling_mode: str = "grid",
    grid_points: int = 10,
    monte_carlo_samples: int = 1000
) -> Expression:
    """
    numerical integration function using CVXPY framework.

    Parameters:
    - f_callable: Function f(*coords, w) constructed using CVXPY atoms.
    - w: CVXPY variable or expression the function depends on.
    - ranges: Integration range(s). For 1D: (x_start, x_end). For ND: 
      [(x1_start, x1_end), (x2_start, x2_end), ...].
    - g_list: For ND, list of boundary functions g(coords) <= 0 defining the domain. 
      (Ignored for 1D.)
    - num_points: Number of points to discretize the integration range. For ND, a list of 
      integers per dimension.
    - method: Numerical integration method ("trapezoidal", "simpson", "monte_carlo").
    - sampling_mode: "grid" or "monte_carlo" for fractional area computation (ND only).
    - grid_points: Number of subdivisions for grid sampling (ND only).
    - monte_carlo_samples: Number of samples for Monte Carlo sampling (ND only).


    Returns:
    - CVXPY expression for the integral.
    """
    # Check if the input is 1D or ND
    if isinstance(ranges, tuple) and len(ranges) == 2:
        # 1D case: ranges is a tuple (x_start, x_end)
        x_start, x_end = ranges
        return numerical_integration_1D(f_callable, w, x_start, x_end, num_points=num_points,
                                         method=method)

    elif isinstance(ranges, list) and all(isinstance(r, tuple) and len(r) == 2 for r in ranges):
        return numerical_integration_ND(f_callable, w, ranges, g_list=g_list, num_points=num_points,
                                        method=method,sampling_mode=sampling_mode,
                                        grid_points=grid_points,
                                        monte_carlo_samples=monte_carlo_samples)
    else:
        raise ValueError(
        "Invalid ranges. Provide a tuple for 1D "
        "or a list of tuples for ND integration."
    )





def numerical_integration_1D(
    f_callable: Callable[[np.ndarray, Union[Variable, Expression]], Expression], 
    w: Union[Variable, Expression], 
    x_start: float, 
    x_end: float, 
    num_points: int = 100, 
    method: str = "trapezoidal"
) -> Expression:
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
    x_values = np.linspace(x_start, x_end, num_points)  # Shape: (num_points,)
    dx = (x_end - x_start) / (num_points - 1)

    # Evaluate f_callable for all x values at once
    f_values = f_callable(x_values, w)  # Assume f_callable handles vectorized x_values

    # Compute the integral using the specified method
    if method == "trapezoidal":
        # Apply the trapezoidal rule
        integral = 0.5 * dx * (f_values[0] + f_values[-1])  # First and last points
        integral += dx * cvx_sum(f_values[1:-1])  # Interior points summed
    elif method == "simpson":
        if num_points % 2 == 0:
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



def numerical_integration_ND(
    f_callable: Callable[..., Expression], 
    w: Union[Variable, Expression], 
    ranges: List[Tuple[float, float]], 
    g_list: Optional[List[Callable[..., Expression]]] = None, 
    num_points: Optional[List[int]] = None,
    method: str = "trapezoidal", 
    sampling_mode: str = "grid", 
    grid_points: int = 10,
    monte_carlo_samples: int = 1000
) -> Expression:
    """
    Numerical integration for ND domains using CVXPY framework .

    Parameters:
    - f_callable: Function f(*coords, w) constructed using CVXPY atoms.
    - w: CVXPY variable or expression the function depends on.
    - ranges: List of tuples [(x_start, x_end), (y_start, y_end), ...] 
    - defining the range for each dimension.
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
    points = np.stack([grid.ravel() for grid in grids], axis=-1)#Shape(num_points, num_dims)

    # Evaluate boundary conditions
    # domain_mask = np.all(np.stack([g(*points.T) <= 0 for g in g_list], axis=0), axis=0)
    if g_list is not None:
        domain_mask = np.all(np.stack([g(*points.T) <= 0 for g in g_list], axis=0), axis=0)
    else:
        # If no boundary conditions are provided, consider the entire domain
        domain_mask = np.ones(points.shape[0], dtype=bool)

    # Compute fractional tile volumes
    tile_volumes = d_volume * np.ones(points.shape[0], dtype=float)
    if not np.all(domain_mask):  # Handle boundary tiles
        boundary_points = points[~domain_mask]
        if sampling_mode == "grid":
            fractional_volumes = _compute_fractional_volume_grid_ND(boundary_points, d_volume, 
                                                                    g_list, grid_points)
        elif sampling_mode == "monte_carlo":
            fractional_volumes = _compute_fractional_volume_monte_carlo_ND(boundary_points, 
                                                                    d_volumes,
                                                                    g_list, monte_carlo_samples)
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        tile_volumes[~domain_mask] = fractional_volumes

    # Evaluate the integrand at valid points
    valid_points = points[domain_mask]
    tile_volumes = tile_volumes[domain_mask]
    #f_values = f_callable(*valid_points.T, w) * tile_volumes
    f_values = cvx_multiply(f_callable(*valid_points.T, w), tile_volumes)

    # Compute the integral using the specified method
    if method == "trapezoidal":
        return cvx_sum(f_values)
    elif method == "simpson":
        return _simpsons_integral_ND(f_values, d_volumes)
    elif method == "monte_carlo":
        return _monte_carlo_integral_ND(f_values, ranges, monte_carlo_samples)
    else:
        raise ValueError(f"Unknown integration method: {method}")


def _compute_fractional_volume_grid_ND(
    boundary_points: np.ndarray, 
    d_volume: float, 
    g_list: List[Callable[..., Union[bool, np.ndarray]]], 
    grid_points: int
) -> np.ndarray:
    """Vectorized fractional volume computation using grid sampling for ND integration."""
    num_dimensions = boundary_points.shape[1]
    sub_deltas = d_volume ** (1 / num_dimensions) / grid_points

    sub_offsets = np.stack(
        np.meshgrid(*[np.linspace(0, sub_deltas * grid_points, grid_points + 1)] * num_dimensions,
                     indexing='ij'),
        axis=-1
    ).reshape(-1, num_dimensions)  # Shape: (subgrid_points, num_dimensions)
    shifted_points = boundary_points[:, None, :] + sub_offsets[None, :, :] 
    # Shape: (num_boundary_points, subgrid_points, num_dimensions)
    inside_mask = np.all(
        np.stack([g(*shifted_points.T) <= 0 for g in g_list], axis=0),
        axis=0
    )  # Shape: (num_boundary_points, subgrid_points)

    inside_count = np.sum(inside_mask, axis=0)  # Shape: (num_boundary_points,)
    total_count = (grid_points + 1) ** num_dimensions
    fractional_volumes = (inside_count / total_count) * d_volume  # Shape: (num_boundary_points,)
    return fractional_volumes








def _compute_fractional_volume_monte_carlo_ND(
    boundary_points: np.ndarray, 
    d_volumes: List[float], 
    g_list: List[Callable[..., Union[bool, np.ndarray]]], 
    monte_carlo_samples: int
) -> np.ndarray:
    """Vectorized fractional volume computation using Monte Carlo sampling for ND integration."""
    random_offsets = np.random.uniform(0, 1, size=(monte_carlo_samples, len(d_volumes))) * d_volumes
    shifted_points = boundary_points[:, None, :] + random_offsets[None, :, :] 
    # Shape (num_boundary_points, monte_carlo_samples, num_dimensions)

    inside_mask = np.all(np.stack([g(*shifted_points.T) <= 0 for g in g_list], axis=0), axis=0)
    inside_count = np.sum(inside_mask, axis=0)
    return (inside_count / monte_carlo_samples) * np.prod(d_volumes)


def _simpsons_integral_ND(
    f_values: Expression, 
    d_volumes: List[float], 
    num_points: List[int]
) -> Expression:
    """
    Apply Simpson's rule for ND integration (Vectorized).
    Ensures odd number of points per dimension for Simpson's rule.
    
    Parameters:
    - f_values: CVXPY expression for the function evaluated at the grid points.
    - d_volumes: List of the differential volumes for each dimension.
    - num_points: List of the number of points in each dimension.
    
    Returns:
    - CVXPY expression for the integral using Simpson's rule.
    """
    # Check if all dimensions have an odd number of points
    if any(points % 2 == 0 for points in num_points):
        raise ValueError("Simpson's rule requires an odd number of points in all dimensions.")

    # Initialize the weights array with ones
    weights = np.ones_like(f_values)
    num_dimensions = len(num_points)

    # Compute weights for each dimension
    for dim in range(num_dimensions):
        index = [slice(None)] * num_dimensions  # Multi-dimensional indexing

        # Odd indices get weight 4
        index[dim] = slice(1, -1, 2)
        weights[tuple(index)] *= 4

        # Even indices (except boundaries) get weight 2
        index[dim] = slice(2, -2, 2)
        weights[tuple(index)] *= 2

        # Boundary indices get weight 1 (already initialized as ones)

    # Compute the weighted sum
    #weighted_sum = cvx_sum(weights * f_values)
    weighted_sum = cvx_sum(cvx_multiply(weights, f_values))
    # Apply the Simpson's rule formula
    integral = weighted_sum * np.prod(d_volumes) / 3

    return integral





def _monte_carlo_integral_ND(
    f_values: Expression, 
    ranges: List[Tuple[float, float]], 
    monte_carlo_samples: int
) -> Expression:
    """Monte Carlo integration for ND (Vectorized)."""
    integral = cvx_sum(f_values)
    bounding_volume = np.prod([end - start for start, end in ranges])
    return integral * bounding_volume / monte_carlo_samples
