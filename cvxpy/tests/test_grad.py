"""
Copyright 2013 Steven Diamond, 2025 CVXPY authors.

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
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.transforms import linearize
from cvxpy.transforms.partial_optimize import partial_optimize

"""
Systematic gradient validation for CVXPY expressions.

Validates expression-level gradients (expr.grad[var]) against numerical
finite differences for all CVXPY atoms.
"""

# =============================================================================
# Constants
# =============================================================================

# Default finite difference step size for numerical gradient computation
DEFAULT_FD_EPS = 1e-5

# Default tolerances for gradient comparison (np.allclose)
DEFAULT_RTOL = 1e-4
DEFAULT_ATOL = 1e-4

# =============================================================================
# Core Gradcheck Utilities
# =============================================================================


def _compute_numerical_jacobian(
    eval_func: Callable[[np.ndarray], np.ndarray],
    var_value: np.ndarray,
    var_shape: Tuple[int, ...],
    output_size: int,
    eps: float = DEFAULT_FD_EPS,
) -> np.ndarray:
    """
    Compute numerical Jacobian via central finite differences.

    Parameters
    ----------
    eval_func : callable
        Function that takes a flat variable value and returns flat output
    var_value : ndarray
        Value to evaluate Jacobian at
    var_shape : tuple
        Shape of the variable (for reshaping perturbations)
    output_size : int
        Size of the output
    eps : float
        Finite difference step size

    Returns
    -------
    jacobian : ndarray
        Jacobian matrix with shape (output_size, input_size)
        jacobian[i, j] = d(output[i]) / d(input[j])
    """
    flat_value = var_value.flatten(order='F')  # CVXPY uses Fortran order
    input_size = flat_value.size
    jacobian = np.zeros((output_size, input_size))

    for idx in range(input_size):
        # Perturb in positive direction
        perturbed_plus = flat_value.copy()
        perturbed_plus[idx] += eps
        result_plus = eval_func(perturbed_plus.reshape(var_shape, order='F'))

        # Perturb in negative direction
        perturbed_minus = flat_value.copy()
        perturbed_minus[idx] -= eps
        result_minus = eval_func(perturbed_minus.reshape(var_shape, order='F'))

        # Central difference
        jacobian[:, idx] = (result_plus - result_minus) / (2 * eps)

    return jacobian


def expression_gradcheck(
    expr_factory: Callable[[cp.Variable], cp.Expression],
    var_shape: Tuple[int, ...],
    var_value: np.ndarray,
    eps: float = DEFAULT_FD_EPS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Tuple[bool, Optional[str]]:
    """
    Validate expression gradient against numerical finite differences.

    Uses central finite differences: (f(x+eps) - f(x-eps)) / (2*eps)

    Parameters
    ----------
    expr_factory : callable
        Function that takes a Variable and returns an Expression
    var_shape : tuple
        Shape of the variable
    var_value : ndarray
        Value to evaluate gradient at (must be in domain)
    eps : float
        Finite difference step size
    rtol, atol : float
        Tolerances for np.allclose comparison

    Returns
    -------
    passed : bool
        Whether gradient check passed
    message : str or None
        Error message if failed, None if passed
    """
    var = cp.Variable(var_shape)
    var.value = var_value.copy()
    expr = expr_factory(var)

    # Get analytic gradient
    analytic_grad = expr.grad.get(var)

    # Handle None gradient (outside domain or undefined)
    if analytic_grad is None:
        return True, "Gradient is None (outside domain)"

    # Convert to dense array for comparison
    if hasattr(analytic_grad, 'toarray'):
        analytic_grad = analytic_grad.toarray()
    analytic_grad = np.asarray(analytic_grad)

    # Compute numerical Jacobian via central differences
    # CVXPY gradient convention: grad[i, j] = d(output[j]) / d(input[i])
    # This is the TRANSPOSE of the standard Jacobian
    def eval_expr(val):
        var.value = val
        return np.asarray(expr.value).flatten(order='F')

    jacobian = _compute_numerical_jacobian(
        eval_expr, var_value, var_shape, expr.size, eps
    )

    # Restore original value
    var.value = var_value

    # CVXPY stores grad as transpose of Jacobian
    numerical_grad = jacobian.T

    # Compare gradients
    if not np.allclose(analytic_grad, numerical_grad, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(analytic_grad - numerical_grad))
        max_idx = np.unravel_index(
            np.argmax(np.abs(analytic_grad - numerical_grad)),
            analytic_grad.shape
        )
        return False, (
            f"Max difference: {max_diff:.2e} at index {max_idx}. "
            f"Analytic: {analytic_grad[max_idx]:.6f}, "
            f"Numerical: {numerical_grad[max_idx]:.6f}"
        )

    return True, None


def expression_gradcheck_symmetric(
    expr_factory: Callable[[cp.Variable], cp.Expression],
    n: int,
    var_value: np.ndarray,
    eps: float = DEFAULT_FD_EPS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Tuple[bool, Optional[str]]:
    """
    Validate expression gradient for symmetric matrix inputs.

    For symmetric matrices, we perturb in directions that maintain symmetry.
    This means perturbing (i,j) and (j,i) together for off-diagonal elements.

    Parameters
    ----------
    expr_factory : callable
        Function that takes a symmetric Variable and returns an Expression
    n : int
        Size of the n x n symmetric matrix
    var_value : ndarray
        Symmetric matrix value to evaluate gradient at
    eps : float
        Finite difference step size
    rtol, atol : float
        Tolerances for comparison

    Returns
    -------
    passed : bool
        Whether gradient check passed
    message : str or None
        Error message if failed
    """
    var = cp.Variable((n, n), symmetric=True)
    var.value = var_value.copy()
    expr = expr_factory(var)

    # Get analytic gradient
    analytic_grad = expr.grad.get(var)

    if analytic_grad is None:
        return True, "Gradient is None (outside domain)"

    if hasattr(analytic_grad, 'toarray'):
        analytic_grad = analytic_grad.toarray()
    analytic_grad = np.asarray(analytic_grad)

    # Compute numerical gradient via central differences
    # For symmetric matrices, we perturb in the symmetric subspace
    input_size = n * n
    output_size = expr.size

    # Build standard Jacobian first
    jacobian = np.zeros((output_size, input_size))

    for i in range(n):
        for j in range(n):
            # Create symmetric perturbation
            perturbation = np.zeros((n, n))
            if i == j:
                perturbation[i, j] = eps
            else:
                perturbation[i, j] = eps
                perturbation[j, i] = eps

            # Perturb in positive direction
            var.value = var_value + perturbation
            result_plus = np.asarray(expr.value).flatten(order='F')

            # Perturb in negative direction
            var.value = var_value - perturbation
            result_minus = np.asarray(expr.value).flatten(order='F')

            # Central difference
            # For diagonal: gradient is just d/d(A[i,i])
            # For off-diagonal: we perturbed both (i,j) and (j,i)
            # so the gradient contribution is d/d(A[i,j]) + d/d(A[j,i])
            idx = i + j * n  # F-order index
            diff = (result_plus - result_minus) / (2 * eps)

            if i == j:
                jacobian[:, idx] = diff
            else:
                # For symmetric matrix, grad[i,j] should equal grad[j,i]
                # The numerical diff gives sum of both, so divide by 2
                # But CVXPY stores full n x n gradient even for symmetric
                jacobian[:, idx] = diff / 2
                jacobian[:, j + i * n] = diff / 2

    # Restore original value
    var.value = var_value

    # CVXPY stores grad as transpose of Jacobian
    numerical_grad = jacobian.T

    # Compare gradients
    if not np.allclose(analytic_grad, numerical_grad, rtol=rtol, atol=atol):
        max_diff = np.max(np.abs(analytic_grad - numerical_grad))
        max_idx = np.unravel_index(
            np.argmax(np.abs(analytic_grad - numerical_grad)),
            analytic_grad.shape
        )
        return False, (
            f"Max difference: {max_diff:.2e} at index {max_idx}. "
            f"Analytic: {analytic_grad[max_idx]:.6f}, "
            f"Numerical: {numerical_grad[max_idx]:.6f}"
        )

    return True, None


def expression_gradcheck_multi(
    expr_factory: Callable[..., cp.Expression],
    var_shapes: List[Tuple[int, ...]],
    var_values: List[np.ndarray],
    eps: float = DEFAULT_FD_EPS,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> Tuple[bool, Optional[str]]:
    """
    Validate gradient for expressions with multiple variable arguments.

    Parameters
    ----------
    expr_factory : callable
        Function that takes multiple Variables and returns an Expression
    var_shapes : list of tuples
        Shapes of the variables
    var_values : list of ndarrays
        Values to evaluate gradients at

    Returns
    -------
    passed : bool
        Whether gradient check passed for all variables
    message : str or None
        Error message if failed
    """
    variables = [cp.Variable(shape) for shape in var_shapes]
    for var, val in zip(variables, var_values):
        var.value = val.copy()

    expr = expr_factory(*variables)

    # Check gradient for each variable
    for var_idx, (var, var_value) in enumerate(zip(variables, var_values)):
        analytic_grad = expr.grad.get(var)

        if analytic_grad is None:
            continue

        if hasattr(analytic_grad, 'toarray'):
            analytic_grad = analytic_grad.toarray()
        analytic_grad = np.asarray(analytic_grad)

        # Compute numerical Jacobian via central differences
        def eval_expr(val):
            var.value = val
            return np.asarray(expr.value).flatten(order='F')

        jacobian = _compute_numerical_jacobian(
            eval_expr, var_value, var.shape, expr.size, eps
        )

        # Restore value
        var.value = var_value

        # CVXPY stores grad as transpose of Jacobian
        numerical_grad = jacobian.T

        if not np.allclose(analytic_grad, numerical_grad, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(analytic_grad - numerical_grad))
            return False, f"Variable {var_idx}: max difference {max_diff:.2e}"

    return True, None


# =============================================================================
# Input Generators
# =============================================================================


class AtomInputGenerator:
    """Generate valid test inputs for different atom domain types."""

    @staticmethod
    def unrestricted(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
        """Generate unrestricted real inputs."""
        rng = np.random.default_rng(seed)
        return rng.standard_normal(shape)

    @staticmethod
    def positive(shape: Tuple[int, ...], seed: int = 42,
                 margin: float = 0.5) -> np.ndarray:
        """Generate strictly positive inputs (for log, sqrt, etc.)."""
        rng = np.random.default_rng(seed)
        return np.abs(rng.standard_normal(shape)) + margin

    @staticmethod
    def nonnegative(shape: Tuple[int, ...], seed: int = 42,
                    margin: float = 0.1) -> np.ndarray:
        """Generate non-negative inputs with small margin from zero."""
        rng = np.random.default_rng(seed)
        return np.abs(rng.standard_normal(shape)) + margin

    @staticmethod
    def psd_matrix(n: int, seed: int = 42) -> np.ndarray:
        """Generate positive definite matrix."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        return A @ A.T + np.eye(n)

    @staticmethod
    def symmetric(shape: Tuple[int, int], seed: int = 42) -> np.ndarray:
        """Generate symmetric matrix."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal(shape)
        return (A + A.T) / 2

    # === Domain violation generators (for testing grad returns None) ===
    @staticmethod
    def negative(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
        """Generate negative values (violates positive domain)."""
        rng = np.random.default_rng(seed)
        return -np.abs(rng.standard_normal(shape)) - 0.5

    @staticmethod
    def with_zero(shape: Tuple[int, ...], seed: int = 42) -> np.ndarray:
        """Generate values with at least one zero (violates strictly positive)."""
        arr = AtomInputGenerator.positive(shape, seed)
        arr.flat[0] = 0.0
        return arr

    @staticmethod
    def non_psd(n: int, seed: int = 42) -> np.ndarray:
        """Generate non-PSD symmetric matrix (has negative eigenvalue)."""
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        A = (A + A.T) / 2
        # Shift to ensure negative eigenvalue
        A[0, 0] = -abs(A[0, 0]) - 1.0
        return A

    @staticmethod
    def generate(input_type: str, shape: Tuple[int, ...],
                 seed: int = 42) -> np.ndarray:
        """Generate input based on type string."""
        generators = {
            "unrestricted": lambda: AtomInputGenerator.unrestricted(shape, seed),
            "positive": lambda: AtomInputGenerator.positive(shape, seed),
            "nonnegative": lambda: AtomInputGenerator.nonnegative(shape, seed),
            "psd": lambda: AtomInputGenerator.psd_matrix(shape[0], seed),
            "symmetric": lambda: AtomInputGenerator.symmetric(shape, seed),
            # Domain violation generators
            "negative": lambda: AtomInputGenerator.negative(shape, seed),
            "with_zero": lambda: AtomInputGenerator.with_zero(shape, seed),
            "non_psd": lambda: AtomInputGenerator.non_psd(shape[0], seed),
        }
        if input_type not in generators:
            raise ValueError(f"Unknown input type: {input_type}")
        return generators[input_type]()


# =============================================================================
# Atom Test Configuration
# =============================================================================


@dataclass
class AtomTestConfig:
    """Configuration for testing a single atom."""
    name: str
    atom_factory: Callable
    var_shapes: List[Tuple[int, ...]]
    input_generator: str
    rtol: float = 1e-4
    atol: float = 1e-4
    skip_reason: Optional[str] = None
    symmetric: bool = False  # For PSD/symmetric matrix atoms
    test_domain: bool = True  # Auto-test domain violation based on input_generator


# Map from valid input generator to its domain-violating counterpart
DOMAIN_VIOLATION_MAP = {
    "positive": "negative",
    "nonnegative": "negative",
    "psd": "non_psd",
}


@dataclass
class MultiVarAtomConfig:
    """Configuration for testing atoms with multiple variable arguments."""
    name: str
    atom_factory: Callable
    var_specs: List[Tuple[str, Tuple[int, ...]]]  # [(input_type, shape), ...]
    rtol: float = 1e-4
    atol: float = 1e-4
    skip_reason: Optional[str] = None


# =============================================================================
# Atom Registry - Single Variable Atoms
# =============================================================================

SINGLE_VAR_ATOM_CONFIGS = [
    # === Elementwise atoms (unrestricted domain) ===
    AtomTestConfig("exp", lambda x: cp.exp(x), [(2,), (2, 2), (2, 3, 4)],
                   "unrestricted"),
    AtomTestConfig("logistic", lambda x: cp.logistic(x), [(2,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("huber", lambda x: cp.huber(x), [(2,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("huber_M2", lambda x: cp.huber(x, M=2), [(2,)],
                   "unrestricted"),
    AtomTestConfig("abs", lambda x: cp.abs(x), [(3,), (2, 2)], "unrestricted"),
    AtomTestConfig("pos", lambda x: cp.pos(x), [(3,)], "unrestricted"),
    AtomTestConfig("neg", lambda x: cp.neg(x), [(3,)], "unrestricted"),
    AtomTestConfig("square", lambda x: cp.square(x), [(2,), (2, 2), (2, 3, 4)],
                   "unrestricted"),

    # === Scalar variable tests ===
    AtomTestConfig("square_scalar", lambda x: cp.square(x), [()], "unrestricted"),
    AtomTestConfig("exp_scalar", lambda x: cp.exp(x), [()], "unrestricted"),
    AtomTestConfig("log_scalar", lambda x: cp.log(x), [()], "positive"),

    # === Elementwise atoms (positive domain) ===
    AtomTestConfig("log", lambda x: cp.log(x), [(2,), (2, 2), (2, 3, 4)], "positive"),
    AtomTestConfig("log1p", lambda x: cp.log1p(x), [(2,), (2, 2)], "positive"),
    AtomTestConfig("sqrt", lambda x: cp.sqrt(x), [(2,), (2, 2), (2, 3, 4)], "positive"),
    AtomTestConfig("inv_pos", lambda x: cp.inv_pos(x), [(2,)], "positive"),
    AtomTestConfig("entr", lambda x: cp.entr(x), [(2,), (2, 2)], "positive"),
    AtomTestConfig("xexp", lambda x: cp.xexp(x), [(2,)], "positive"),

    # === Power atoms ===
    AtomTestConfig("power_2", lambda x: cp.power(x, 2), [(2,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("power_3", lambda x: cp.power(x, 3), [(2,)], "positive"),
    AtomTestConfig("power_0.5", lambda x: cp.power(x, 0.5), [(2,)], "positive"),
    AtomTestConfig("power_neg1", lambda x: cp.power(x, -1), [(2,)], "positive"),

    # === Norm atoms ===
    AtomTestConfig("pnorm_2", lambda x: cp.pnorm(x, 2), [(3,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("pnorm_1", lambda x: cp.pnorm(x, 1), [(3,)], "unrestricted"),
    AtomTestConfig("pnorm_inf", lambda x: cp.pnorm(x, 'inf'), [(3,)],
                   "unrestricted", skip_reason="Gradient not implemented"),
    AtomTestConfig("pnorm_3", lambda x: cp.pnorm(x, 3), [(3,)], "unrestricted"),
    AtomTestConfig("pnorm_axis0", lambda x: cp.pnorm(x, 2, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("pnorm_axis1", lambda x: cp.pnorm(x, 2, axis=1), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("norm1", lambda x: cp.norm1(x), [(3,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("norm_inf", lambda x: cp.norm_inf(x), [(3,), (2, 2)],
                   "unrestricted", skip_reason="Gradient not implemented"),
    AtomTestConfig("norm_nuc", lambda x: cp.normNuc(x), [(3, 3)],
                   "unrestricted"),
    AtomTestConfig("sigma_max", lambda x: cp.sigma_max(x), [(3, 3)],
                   "unrestricted"),
    AtomTestConfig("mixed_norm_21", lambda x: cp.mixed_norm(x, 2, 1), [(3, 2)],
                   "unrestricted"),
    AtomTestConfig("sum_squares", lambda x: cp.sum_squares(x), [(3,), (2, 2)],
                   "unrestricted"),

    # === Reduction atoms ===
    AtomTestConfig("sum", lambda x: cp.sum(x), [(2, 3), (2, 3, 4)], "unrestricted"),
    AtomTestConfig("sum_scalar", lambda x: cp.sum(x), [()], "unrestricted"),
    AtomTestConfig("sum_axis0", lambda x: cp.sum(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("sum_axis1", lambda x: cp.sum(x, axis=1), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("max", lambda x: cp.max(x), [(5,), (2, 3, 4)], "unrestricted"),
    AtomTestConfig("max_axis0", lambda x: cp.max(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("max_3d_axis1", lambda x: cp.max(x, axis=1), [(2, 3, 4)],
                   "unrestricted", skip_reason="_axis_grad doesn't support 3D"),
    AtomTestConfig("min", lambda x: cp.min(x), [(5,), (2, 3, 4)], "unrestricted"),
    AtomTestConfig("min_axis0", lambda x: cp.min(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("min_3d_axis2", lambda x: cp.min(x, axis=2), [(2, 3, 4)],
                   "unrestricted", skip_reason="_axis_grad doesn't support 3D"),
    AtomTestConfig("geo_mean", lambda x: cp.geo_mean(x), [(3,)], "positive"),
    AtomTestConfig("harmonic_mean", lambda x: cp.harmonic_mean(x), [(3,)],
                   "positive"),
    AtomTestConfig("log_sum_exp", lambda x: cp.log_sum_exp(x),
                   [(3,), (2, 2), (2, 3, 4)], "unrestricted"),
    AtomTestConfig("log_sum_exp_axis0",
                   lambda x: cp.log_sum_exp(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("log_sum_exp_3d_axis1",
                   lambda x: cp.log_sum_exp(x, axis=1), [(2, 3, 4)],
                   "unrestricted", skip_reason="_axis_grad doesn't support 3D"),
    AtomTestConfig("prod", lambda x: cp.prod(x), [(3,)], "unrestricted"),

    # === Affine atoms ===
    AtomTestConfig("trace", lambda x: cp.trace(x), [(3, 3)], "unrestricted"),
    AtomTestConfig("diag_extract", lambda x: cp.diag(x), [(3, 3)], "unrestricted"),
    AtomTestConfig("diag_create", lambda x: cp.diag(x), [(3,)], "unrestricted"),
    AtomTestConfig("reshape", lambda x: cp.reshape(x, (6,), order='F'),
                   [(2, 3)], "unrestricted"),
    AtomTestConfig("vec", lambda x: cp.vec(x), [(2, 3)], "unrestricted"),
    AtomTestConfig("transpose", lambda x: cp.transpose(x), [(2, 3)], "unrestricted"),
    AtomTestConfig("hstack", lambda x: cp.hstack([x, x]), [(3,)],
                   "unrestricted"),
    AtomTestConfig("vstack", lambda x: cp.vstack([x, x]), [(3,)],
                   "unrestricted"),
    AtomTestConfig("cumsum", lambda x: cp.cumsum(x), [(4,)], "unrestricted"),
    AtomTestConfig("cumsum_2d", lambda x: cp.cumsum(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("cumsum_3d_ax0", lambda x: cp.cumsum(x, axis=0), [(2, 3, 4)],
                   "unrestricted"),
    AtomTestConfig("cumsum_3d_ax1", lambda x: cp.cumsum(x, axis=1), [(2, 3, 4)],
                   "unrestricted"),
    AtomTestConfig("cumsum_3d_ax2", lambda x: cp.cumsum(x, axis=2), [(2, 3, 4)],
                   "unrestricted"),
    AtomTestConfig("cumsum_axis_none", lambda x: cp.cumsum(x, axis=None), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("cummax", lambda x: cp.cummax(x), [(4,)], "unrestricted",
                   skip_reason="cummax gradient has subdifferential issues"),
    AtomTestConfig("diff", lambda x: cp.diff(x), [(4,)], "unrestricted"),
    AtomTestConfig("diff_3d_axis0", lambda x: cp.diff(x, axis=0), [(3, 4, 5)],
                   "unrestricted"),
    AtomTestConfig("upper_tri", lambda x: cp.upper_tri(x), [(3, 3)], "unrestricted"),
    AtomTestConfig("index_single", lambda x: x[0], [(5,)], "unrestricted"),
    AtomTestConfig("index_slice", lambda x: x[1:4], [(5,)], "unrestricted"),

    # === Matrix atoms requiring symmetric input (not necessarily PSD) ===
    AtomTestConfig("lambda_max", lambda x: cp.lambda_max(x), [(3, 3)], "symmetric",
                   symmetric=True),
    AtomTestConfig("lambda_min", lambda x: cp.lambda_min(x), [(3, 3)], "symmetric",
                   symmetric=True),
    AtomTestConfig("lambda_sum_largest", lambda x: cp.lambda_sum_largest(x, 2),
                   [(3, 3)], "psd",
                   skip_reason="_grad raises NotImplementedError"),
    AtomTestConfig("log_det", lambda x: cp.log_det(x), [(3, 3)], "psd",
                   symmetric=True),
    AtomTestConfig("tr_inv", lambda x: cp.tr_inv(x), [(3, 3)], "psd",
                   symmetric=True),

    # === Quadratic atoms with constant parameters ===
    AtomTestConfig("quad_form", lambda x: cp.quad_form(x, np.eye(3)), [(3,)],
                   "unrestricted"),

    # === Other atoms ===
    AtomTestConfig("dotsort", lambda x: cp.dotsort(x, [3, 2, 1]), [(3,)],
                   "unrestricted"),
    AtomTestConfig("ptp", lambda x: cp.ptp(x), [(3,)], "unrestricted"),
    AtomTestConfig("sum_largest", lambda x: cp.sum_largest(x, 2), [(5,)],
                   "unrestricted",
                   skip_reason="subdifferential at ties - needs special handling"),
    AtomTestConfig("sum_smallest", lambda x: cp.sum_smallest(x, 2), [(5,)],
                   "unrestricted",
                   skip_reason="subdifferential at ties - needs special handling"),

    # === Atoms with unimplemented or trivial gradients ===
    AtomTestConfig("loggamma", lambda x: cp.loggamma(x), [(2,)], "positive",
                   skip_reason="_grad not implemented"),
    AtomTestConfig("log_normcdf", lambda x: cp.log_normcdf(x), [(2,)],
                   "unrestricted", skip_reason="_grad not implemented"),
    AtomTestConfig("ceil", lambda x: cp.ceil(x), [(2,)], "unrestricted",
                   skip_reason="zero gradient everywhere"),
    AtomTestConfig("floor", lambda x: cp.floor(x), [(2,)], "unrestricted",
                   skip_reason="zero gradient everywhere"),
    AtomTestConfig("sign", lambda x: cp.sign(x), [(2,)], "unrestricted",
                   skip_reason="zero gradient everywhere"),
    AtomTestConfig("cumprod", lambda x: cp.cumprod(x), [(4,)], "positive",
                   skip_reason="_grad returns empty list (TODO in source)"),
    AtomTestConfig("length", lambda x: cp.length(x), [(4,)], "unrestricted",
                   skip_reason="_grad returns None (discrete)"),
    AtomTestConfig("one_minus_pos", lambda x: cp.one_minus_pos(x), [(3,)],
                   "unrestricted", skip_reason="constraint atom, trivial affine"),
    AtomTestConfig("eye_minus_inv", lambda x: cp.eye_minus_inv(x), [(3, 3)],
                   "unrestricted", skip_reason="specialized spectral atom"),
    AtomTestConfig("von_neumann_entr", lambda x: cp.von_neumann_entr(x),
                   [(3, 3)], "psd", skip_reason="_grad has TODO (scipy CSC)"),
    AtomTestConfig("quantum_rel_entr", lambda x: cp.quantum_rel_entr(x, x),
                   [(3, 3)], "psd", skip_reason="two-arg quantum atom"),
]


# =============================================================================
# Multi-Variable Atom Configs
# =============================================================================

MULTI_VAR_ATOM_CONFIGS = [
    # === Binary elementwise atoms ===
    MultiVarAtomConfig("kl_div", lambda x, y: cp.kl_div(x, y),
                       [("positive", (3,)), ("positive", (3,))]),
    MultiVarAtomConfig("rel_entr", lambda x, y: cp.rel_entr(x, y),
                       [("positive", (3,)), ("positive", (3,))]),
    MultiVarAtomConfig("maximum", lambda x, y: cp.maximum(x, y),
                       [("unrestricted", (3,)), ("unrestricted", (3,))]),
    MultiVarAtomConfig("minimum", lambda x, y: cp.minimum(x, y),
                       [("unrestricted", (3,)), ("unrestricted", (3,))]),
    MultiVarAtomConfig("multiply", lambda x, y: cp.multiply(x, y),
                       [("unrestricted", (3,)), ("unrestricted", (3,))]),
    MultiVarAtomConfig("multiply_broadcast", lambda x, y: cp.multiply(x, y),
                       [("unrestricted", (3, 1)), ("unrestricted", (1, 3))]),

    # === Matrix operations ===
    MultiVarAtomConfig("matmul", lambda x, y: x @ y,
                       [("unrestricted", (2, 3)), ("unrestricted", (3, 2))]),
    MultiVarAtomConfig("quad_over_lin", lambda x, y: cp.quad_over_lin(x, y),
                       [("unrestricted", (3,)), ("positive", (1,))]),
    MultiVarAtomConfig("matrix_frac", lambda x, P: cp.matrix_frac(x, P),
                       [("unrestricted", (3,)), ("psd", (3, 3))]),
]


# =============================================================================
# Pytest Fixtures
# =============================================================================

@pytest.fixture(params=[42, 123, 456])
def random_seed(request):
    """Multiple random seeds for robust testing."""
    return request.param


# =============================================================================
# Systematic Gradient Tests
# =============================================================================


class TestSingleVarAtomGradients:
    """Systematic gradient tests for single-variable atoms."""

    @pytest.mark.parametrize(
        "config",
        SINGLE_VAR_ATOM_CONFIGS,
        ids=[c.name for c in SINGLE_VAR_ATOM_CONFIGS]
    )
    def test_single_var_atom(self, config: AtomTestConfig, random_seed: int):
        """Test gradient correctness for single-variable atoms."""
        if config.skip_reason:
            pytest.skip(config.skip_reason)

        for var_shape in config.var_shapes:
            var_value = AtomInputGenerator.generate(
                config.input_generator, var_shape, seed=random_seed
            )

            if config.symmetric:
                # Use symmetric gradcheck for PSD matrix atoms
                n = var_shape[0]
                passed, message = expression_gradcheck_symmetric(
                    config.atom_factory, n, var_value,
                    rtol=config.rtol, atol=config.atol
                )
            else:
                passed, message = expression_gradcheck(
                    config.atom_factory, var_shape, var_value,
                    rtol=config.rtol, atol=config.atol
                )

            assert passed, f"{config.name} shape={var_shape}: {message}"


class TestMultiVarAtomGradients:
    """Systematic gradient tests for multi-variable atoms."""

    @pytest.mark.parametrize(
        "config",
        MULTI_VAR_ATOM_CONFIGS,
        ids=[c.name for c in MULTI_VAR_ATOM_CONFIGS]
    )
    def test_multi_var_atom(self, config: MultiVarAtomConfig, random_seed: int):
        """Test gradient correctness for multi-variable atoms."""
        if config.skip_reason:
            pytest.skip(config.skip_reason)

        var_shapes = [spec[1] for spec in config.var_specs]
        var_values = [
            AtomInputGenerator.generate(spec[0], spec[1], seed=random_seed + i)
            for i, spec in enumerate(config.var_specs)
        ]

        passed, message = expression_gradcheck_multi(
            config.atom_factory, var_shapes, var_values,
            rtol=config.rtol, atol=config.atol
        )
        assert passed, f"{config.name}: {message}"


# Filter configs that have restricted domains (auto-derive from input_generator)
DOMAIN_VIOLATION_CONFIGS = [
    c for c in SINGLE_VAR_ATOM_CONFIGS
    if c.input_generator in DOMAIN_VIOLATION_MAP
    and c.test_domain
    and not c.skip_reason
]


class TestDomainViolations:
    """Automated tests that gradient returns None outside domain."""

    @pytest.mark.parametrize(
        "config",
        DOMAIN_VIOLATION_CONFIGS,
        ids=[c.name for c in DOMAIN_VIOLATION_CONFIGS]
    )
    def test_domain_violation(self, config: AtomTestConfig):
        """Test that gradient is None when input violates domain."""
        bad_gen = DOMAIN_VIOLATION_MAP[config.input_generator]
        var_shape = config.var_shapes[0]

        if config.symmetric:
            var = cp.Variable(var_shape, symmetric=True)
        else:
            var = cp.Variable(var_shape)

        var.value = AtomInputGenerator.generate(bad_gen, var_shape, seed=42)
        expr = config.atom_factory(var)

        assert expr.grad[var] is None, \
            f"{config.name}: expected None grad for {bad_gen} input"


class TestCompositeExpressions:
    """Tests for composite expressions (multiple atoms combined)."""

    @pytest.mark.parametrize("seed", [42, 123])
    def test_sum_of_squares(self, seed: int):
        """Test sum(square(x)) gradient."""
        var_value = AtomInputGenerator.unrestricted((3,), seed)
        passed, msg = expression_gradcheck(
            lambda x: cp.sum(cp.square(x)),
            (3,),
            var_value
        )
        assert passed, f"sum(square(x)): {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_log_sum_exp_composition(self, seed: int):
        """Test log_sum_exp of scaled input."""
        var_value = AtomInputGenerator.unrestricted((3,), seed)
        passed, msg = expression_gradcheck(
            lambda x: cp.log_sum_exp(2 * x),
            (3,),
            var_value
        )
        assert passed, f"log_sum_exp(2*x): {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_norm_of_affine(self, seed: int):
        """Test norm(Ax + b) gradient."""
        var_value = AtomInputGenerator.unrestricted((3,), seed)
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
        b = np.array([1, 2, 3])
        passed, msg = expression_gradcheck(
            lambda x: cp.pnorm(A @ x + b, 2),
            (3,),
            var_value
        )
        assert passed, f"norm(Ax+b): {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_entropy_sum(self, seed: int):
        """Test sum(entr(x)) gradient."""
        var_value = AtomInputGenerator.positive((3,), seed)
        passed, msg = expression_gradcheck(
            lambda x: cp.sum(cp.entr(x)),
            (3,),
            var_value
        )
        assert passed, f"sum(entr(x)): {msg}"


# =============================================================================
# Special Case Tests (require specific expected values or special handling)
# =============================================================================


class TestSpecialCases:
    """Tests that require specific expected values or special handling."""

    def test_linearize(self):
        """Test linearize method."""
        x = cp.Variable(2)

        # Affine.
        expr = (2*x - 5)[0]
        x.value = [1, 2]
        lin_expr = linearize(expr)
        x.value = [55, 22]
        np.testing.assert_almost_equal(lin_expr.value, expr.value)
        x.value = [-1, -5]
        np.testing.assert_almost_equal(lin_expr.value, expr.value)

        # Convex.
        A = cp.Variable((2, 2))
        expr = A**2 + 5

        with pytest.raises(Exception, match="Cannot linearize non-affine expression"):
            linearize(expr)

        A.value = [[1, 2], [3, 4]]
        lin_expr = linearize(expr)
        manual = expr.value + 2*cp.reshape(
            cp.diag(cp.vec(A, order='F')).value @ cp.vec(A - A.value, order='F'),
            (2, 2),
            order='F'
        )
        np.testing.assert_array_almost_equal(lin_expr.value, expr.value)
        A.value = [[-5, -5], [8.2, 4.4]]
        assert (lin_expr.value <= expr.value).all()
        np.testing.assert_array_almost_equal(lin_expr.value, manual.value)

        # Concave.
        expr = cp.log(x)/2
        x.value = [1, 2]
        lin_expr = linearize(expr)
        manual = expr.value + cp.diag(0.5*x**-1).value @ (x - x.value)
        np.testing.assert_array_almost_equal(lin_expr.value, expr.value)
        x.value = [3, 4.4]
        assert (lin_expr.value >= expr.value).all()
        np.testing.assert_array_almost_equal(lin_expr.value, manual.value)

    def test_partial_problem(self):
        """Test grad for partial minimization/maximization problems."""
        a = cp.Variable(name='a')
        x = cp.Variable(2, name='x')

        for obj in [Minimize((a)**-1), Maximize(cp.entr(a))]:
            prob = Problem(obj, [x + a >= [5, 8]])
            # Optimize over nothing.
            expr = partial_optimize(prob, dont_opt_vars=[x, a], solver=cp.CLARABEL)
            a.value = None
            x.value = None
            grad = expr.grad
            assert grad[a] is None
            assert grad[x] is None
            # Outside domain.
            a.value = 1.0
            x.value = [5, 5]
            grad = expr.grad
            assert grad[a] is None
            assert grad[x] is None

            a.value = 1
            x.value = [10, 10]
            grad = expr.grad
            np.testing.assert_almost_equal(grad[a], obj.args[0].grad[a])
            # Gradient w.r.t. x should be zero (scalar output, no dependence on x)
            np.testing.assert_array_almost_equal(
                grad[x].toarray().flatten(), [0, 0])

            # Optimize over x.
            expr = partial_optimize(prob, opt_vars=[x], solver=cp.CLARABEL)
            a.value = 1
            grad = expr.grad
            np.testing.assert_almost_equal(grad[a], obj.args[0].grad[a] + 0)

            # Optimize over a.
            fix_prob = Problem(obj, [x + a >= [5, 8], x == 0])
            fix_prob.solve(solver=cp.CLARABEL)
            dual_val = fix_prob.constraints[0].dual_variables[0].value
            expr = partial_optimize(prob, opt_vars=[a], solver=cp.CLARABEL)
            x.value = [0, 0]
            grad = expr.grad
            np.testing.assert_array_almost_equal(grad[x].toarray().flatten(), dual_val)

            # Optimize over x and a.
            expr = partial_optimize(prob, opt_vars=[x, a], solver=cp.CLARABEL)
            grad = expr.grad
            assert grad == {}

    def test_quad_form_issue_1260(self):
        """Test quad_form gradient access after solving (issue 1260)."""
        n = 10
        np.random.seed(1)
        P = np.random.randn(n, n)
        P = P.T @ P
        q = np.random.randn(n)

        # define the optimization problem with the 2nd constraint as quad_form
        x = cp.Variable(n)
        prob = cp.Problem(
            cp.Maximize(q.T @ x - (1/2)*cp.quad_form(x, P)),
            [cp.norm(x, 1) <= 1.0,
             cp.quad_form(x, P) <= 10,
             cp.abs(x) <= 0.01]
        )
        prob.solve(solver=cp.SCS)

        # access quad_form.expr.grad without error
        prob.constraints[1].expr.grad

        # define the optimization problem with a two-dimensional decision variable
        x = cp.Variable((n, 1))
        prob = cp.Problem(
            cp.Maximize(q.T @ x - (1 / 2) * cp.quad_form(x, P)),
            [cp.norm(x, 1) <= 1.0,
             cp.quad_form(x, P) <= 10,
             cp.abs(x) <= 0.01],
        )
        prob.solve(solver=cp.SCS)

        # access quad_form.expr.grad without error
        prob.constraints[1].expr.grad


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
