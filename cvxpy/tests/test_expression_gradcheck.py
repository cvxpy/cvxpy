"""
Copyright 2025, the CVXPY authors.

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
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pytest

import cvxpy as cp

"""
Systematic gradient validation for CVXPY expressions.

This module provides a PyTorch-style gradcheck utility that validates
expression-level gradients (expr.grad[var]) against numerical finite
differences for all CVXPY atoms.
"""

# =============================================================================
# Core Gradcheck Utility
# =============================================================================


def expression_gradcheck(
    expr_factory: Callable[[cp.Variable], cp.Expression],
    var_shape: Tuple[int, ...],
    var_value: np.ndarray,
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
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

    # Compute numerical gradient via central differences
    # CVXPY gradient shape is (input_size, output_size)
    # Convention: grad[i, j] = d(output[j]) / d(input[i])
    # This is the TRANSPOSE of the standard Jacobian
    input_size = var_value.size
    output_size = expr.size

    # Build standard Jacobian first: jacobian[out_idx, in_idx] = d(out)/d(in)
    jacobian = np.zeros((output_size, input_size))

    flat_value = var_value.flatten(order='F')  # CVXPY uses Fortran order

    for idx in range(input_size):
        # Perturb in positive direction
        perturbed_plus = flat_value.copy()
        perturbed_plus[idx] += eps
        var.value = perturbed_plus.reshape(var_shape, order='F')
        result_plus = np.asarray(expr.value).flatten(order='F')

        # Perturb in negative direction
        perturbed_minus = flat_value.copy()
        perturbed_minus[idx] -= eps
        var.value = perturbed_minus.reshape(var_shape, order='F')
        result_minus = np.asarray(expr.value).flatten(order='F')

        # Central difference gives d(output)/d(input[idx]) as column idx
        jacobian[:, idx] = (result_plus - result_minus) / (2 * eps)

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
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
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
        Size of the n×n symmetric matrix
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
                # But CVXPY stores full n×n gradient even for symmetric
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
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-4,
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

        # CVXPY gradient shape is (input_size, output_size)
        # Convention: grad[i, j] = d(output[j]) / d(input[i])
        input_size = var_value.size
        output_size = expr.size
        jacobian = np.zeros((output_size, input_size))

        flat_value = var_value.flatten(order='F')

        for idx in range(input_size):
            # Perturb in positive direction
            perturbed_plus = flat_value.copy()
            perturbed_plus[idx] += eps
            var.value = perturbed_plus.reshape(var.shape, order='F')
            result_plus = np.asarray(expr.value).flatten(order='F')

            # Perturb in negative direction
            perturbed_minus = flat_value.copy()
            perturbed_minus[idx] -= eps
            var.value = perturbed_minus.reshape(var.shape, order='F')
            result_minus = np.asarray(expr.value).flatten(order='F')

            jacobian[:, idx] = (result_plus - result_minus) / (2 * eps)

        # Restore value
        var.value = var_value
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

    @staticmethod
    def generate(input_type: str, shape: Tuple[int, ...],
                 seed: int = 42) -> np.ndarray:
        """Generate input based on type string."""
        if input_type == "unrestricted":
            return AtomInputGenerator.unrestricted(shape, seed)
        elif input_type == "positive":
            return AtomInputGenerator.positive(shape, seed)
        elif input_type == "nonnegative":
            return AtomInputGenerator.nonnegative(shape, seed)
        elif input_type == "psd":
            assert len(shape) == 2 and shape[0] == shape[1]
            return AtomInputGenerator.psd_matrix(shape[0], seed)
        elif input_type == "symmetric":
            return AtomInputGenerator.symmetric(shape, seed)
        else:
            raise ValueError(f"Unknown input type: {input_type}")


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
    extra_kwargs: dict = field(default_factory=dict)


# =============================================================================
# Atom Registry - Single Variable Atoms
# =============================================================================

SINGLE_VAR_ATOM_CONFIGS = [
    # === Elementwise atoms (unrestricted domain) ===
    AtomTestConfig("exp", lambda x: cp.exp(x), [(2,), (2, 2)], "unrestricted"),
    AtomTestConfig("logistic", lambda x: cp.logistic(x), [(2,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("huber", lambda x: cp.huber(x), [(2,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("huber_M2", lambda x: cp.huber(x, M=2), [(2,)],
                   "unrestricted"),
    AtomTestConfig("abs", lambda x: cp.abs(x), [(3,), (2, 2)], "unrestricted"),
    AtomTestConfig("pos", lambda x: cp.pos(x), [(3,)], "unrestricted"),
    AtomTestConfig("neg", lambda x: cp.neg(x), [(3,)], "unrestricted"),
    AtomTestConfig("square", lambda x: cp.square(x), [(2,), (2, 2)],
                   "unrestricted"),

    # === Elementwise atoms (positive domain) ===
    AtomTestConfig("log", lambda x: cp.log(x), [(2,), (2, 2)], "positive"),
    AtomTestConfig("log1p", lambda x: cp.log1p(x), [(2,), (2, 2)], "positive"),
    AtomTestConfig("sqrt", lambda x: cp.sqrt(x), [(2,), (2, 2)], "positive"),
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
    AtomTestConfig("sum", lambda x: cp.sum(x), [(2, 3)], "unrestricted"),
    AtomTestConfig("sum_axis0", lambda x: cp.sum(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("sum_axis1", lambda x: cp.sum(x, axis=1), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("max", lambda x: cp.max(x), [(5,)], "unrestricted"),
    AtomTestConfig("max_axis0", lambda x: cp.max(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("min", lambda x: cp.min(x), [(5,)], "unrestricted"),
    AtomTestConfig("min_axis0", lambda x: cp.min(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("geo_mean", lambda x: cp.geo_mean(x), [(3,)], "positive"),
    AtomTestConfig("harmonic_mean", lambda x: cp.harmonic_mean(x), [(3,)],
                   "positive"),
    AtomTestConfig("log_sum_exp", lambda x: cp.log_sum_exp(x), [(3,), (2, 2)],
                   "unrestricted"),
    AtomTestConfig("log_sum_exp_axis0",
                   lambda x: cp.log_sum_exp(x, axis=0), [(2, 3)],
                   "unrestricted"),
    AtomTestConfig("prod", lambda x: cp.prod(x), [(3,)], "positive"),

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
    AtomTestConfig("cummax", lambda x: cp.cummax(x), [(4,)], "unrestricted",
                   skip_reason="cummax gradient has subdifferential issues"),
    AtomTestConfig("diff", lambda x: cp.diff(x), [(4,)], "unrestricted"),
    AtomTestConfig("upper_tri", lambda x: cp.upper_tri(x), [(3, 3)], "unrestricted"),
    AtomTestConfig("index_single", lambda x: x[0], [(5,)], "unrestricted"),
    AtomTestConfig("index_slice", lambda x: x[1:4], [(5,)], "unrestricted"),

    # === Matrix atoms requiring PSD input ===
    # Note: These are tested separately in TestSymmetricMatrixAtoms using
    # expression_gradcheck_symmetric which handles symmetry constraints.
    AtomTestConfig("lambda_max", lambda x: cp.lambda_max(x), [(3, 3)], "psd",
                   skip_reason="requires symmetric gradcheck - tested in TestSymmetricMatrixAtoms"),
    AtomTestConfig("lambda_min", lambda x: cp.lambda_min(x), [(3, 3)], "psd",
                   skip_reason="requires symmetric gradcheck - tested in TestSymmetricMatrixAtoms"),
    AtomTestConfig("lambda_sum_largest",
                   lambda x: cp.lambda_sum_largest(x, 2), [(3, 3)], "psd",
                   skip_reason="lambda_sum_largest _grad raises NotImplementedError"),
    AtomTestConfig("log_det", lambda x: cp.log_det(x), [(3, 3)], "psd",
                   skip_reason="requires symmetric gradcheck - tested in TestSymmetricMatrixAtoms"),
    AtomTestConfig("tr_inv", lambda x: cp.tr_inv(x), [(3, 3)], "psd",
                   skip_reason="requires symmetric gradcheck - tested in TestSymmetricMatrixAtoms"),

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

    # === Atoms to skip (specialized or incomplete grad) ===
    AtomTestConfig("loggamma", lambda x: cp.loggamma(x), [(2,)], "positive",
                   skip_reason="loggamma _grad not implemented"),
    AtomTestConfig("log_normcdf", lambda x: cp.log_normcdf(x), [(2,)],
                   "unrestricted",
                   skip_reason="log_normcdf _grad not implemented"),
    AtomTestConfig("ceil", lambda x: cp.ceil(x), [(2,)], "unrestricted",
                   skip_reason="ceil has zero gradient everywhere"),
    AtomTestConfig("floor", lambda x: cp.floor(x), [(2,)], "unrestricted",
                   skip_reason="floor has zero gradient everywhere"),
    AtomTestConfig("sign", lambda x: cp.sign(x), [(2,)], "unrestricted",
                   skip_reason="sign has zero gradient everywhere"),
]

# =============================================================================
# Pytest Fixtures and Tests
# =============================================================================


def get_atom_config_ids():
    """Generate test IDs for atom configs."""
    return [c.name for c in SINGLE_VAR_ATOM_CONFIGS]


@pytest.fixture(params=[42, 123, 456])
def random_seed(request):
    """Multiple random seeds for robust testing."""
    return request.param


class TestExpressionGradcheck:
    """Systematic gradient tests for single-variable atoms."""

    @pytest.mark.parametrize(
        "config",
        SINGLE_VAR_ATOM_CONFIGS,
        ids=get_atom_config_ids()
    )
    def test_single_var_atom(self, config: AtomTestConfig, random_seed: int):
        """Test gradient correctness for single-variable atoms."""
        if config.skip_reason:
            pytest.skip(config.skip_reason)

        for var_shape in config.var_shapes:
            var_value = AtomInputGenerator.generate(
                config.input_generator, var_shape, seed=random_seed
            )

            passed, message = expression_gradcheck(
                config.atom_factory,
                var_shape,
                var_value,
                rtol=config.rtol,
                atol=config.atol
            )

            assert passed, f"{config.name} shape={var_shape}: {message}"


class TestMultiVariableAtoms:
    """Tests for atoms with multiple variable arguments."""

    @pytest.mark.parametrize("seed", [42, 123])
    def test_kl_div(self, seed: int):
        """Test kl_div(x, y) gradient with respect to both variables."""
        x_val = AtomInputGenerator.positive((3,), seed)
        y_val = AtomInputGenerator.positive((3,), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.kl_div(x, y),
            [(3,), (3,)],
            [x_val, y_val]
        )
        assert passed, f"kl_div: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_rel_entr(self, seed: int):
        """Test rel_entr(x, y) gradient with respect to both variables."""
        x_val = AtomInputGenerator.positive((3,), seed)
        y_val = AtomInputGenerator.positive((3,), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.rel_entr(x, y),
            [(3,), (3,)],
            [x_val, y_val]
        )
        assert passed, f"rel_entr: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_quad_over_lin(self, seed: int):
        """Test quad_over_lin(x, y) gradient."""
        x_val = AtomInputGenerator.unrestricted((3,), seed)
        y_val = np.array([AtomInputGenerator.positive((), seed + 1) + 1.0])

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.quad_over_lin(x, y),
            [(3,), (1,)],
            [x_val, y_val]
        )
        assert passed, f"quad_over_lin: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_matrix_frac(self, seed: int):
        """Test matrix_frac(x, P) gradient."""
        n = 3
        x_val = AtomInputGenerator.unrestricted((n,), seed)
        P_val = AtomInputGenerator.psd_matrix(n, seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, P: cp.matrix_frac(x, P),
            [(n,), (n, n)],
            [x_val, P_val]
        )
        assert passed, f"matrix_frac: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_maximum_two_vars(self, seed: int):
        """Test maximum(x, y) gradient with two variables."""
        x_val = AtomInputGenerator.unrestricted((3,), seed)
        y_val = AtomInputGenerator.unrestricted((3,), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.maximum(x, y),
            [(3,), (3,)],
            [x_val, y_val]
        )
        assert passed, f"maximum(x,y): {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_minimum_two_vars(self, seed: int):
        """Test minimum(x, y) gradient with two variables."""
        x_val = AtomInputGenerator.unrestricted((3,), seed)
        y_val = AtomInputGenerator.unrestricted((3,), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.minimum(x, y),
            [(3,), (3,)],
            [x_val, y_val]
        )
        assert passed, f"minimum(x,y): {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_matmul(self, seed: int):
        """Test matrix multiplication gradient."""
        x_val = AtomInputGenerator.unrestricted((2, 3), seed)
        y_val = AtomInputGenerator.unrestricted((3, 2), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: x @ y,
            [(2, 3), (3, 2)],
            [x_val, y_val]
        )
        assert passed, f"matmul: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_multiply_elementwise(self, seed: int):
        """Test elementwise multiplication gradient."""
        x_val = AtomInputGenerator.unrestricted((3,), seed)
        y_val = AtomInputGenerator.unrestricted((3,), seed + 1)

        passed, msg = expression_gradcheck_multi(
            lambda x, y: cp.multiply(x, y),
            [(3,), (3,)],
            [x_val, y_val]
        )
        assert passed, f"multiply: {msg}"


class TestSymmetricMatrixAtoms:
    """Tests for atoms that require symmetric/PSD matrix inputs."""

    @pytest.mark.parametrize("seed", [42, 123])
    def test_lambda_max(self, seed: int):
        """Test lambda_max gradient with symmetric gradcheck."""
        n = 3
        psd_val = AtomInputGenerator.psd_matrix(n, seed)
        passed, msg = expression_gradcheck_symmetric(
            lambda x: cp.lambda_max(x), n, psd_val
        )
        assert passed, f"lambda_max: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_lambda_min(self, seed: int):
        """Test lambda_min gradient with symmetric gradcheck."""
        n = 3
        psd_val = AtomInputGenerator.psd_matrix(n, seed)
        passed, msg = expression_gradcheck_symmetric(
            lambda x: cp.lambda_min(x), n, psd_val
        )
        assert passed, f"lambda_min: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_log_det(self, seed: int):
        """Test log_det gradient with symmetric gradcheck."""
        n = 3
        psd_val = AtomInputGenerator.psd_matrix(n, seed)
        passed, msg = expression_gradcheck_symmetric(
            lambda x: cp.log_det(x), n, psd_val
        )
        assert passed, f"log_det: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_tr_inv(self, seed: int):
        """Test tr_inv gradient with symmetric gradcheck."""
        n = 3
        psd_val = AtomInputGenerator.psd_matrix(n, seed)
        passed, msg = expression_gradcheck_symmetric(
            lambda x: cp.tr_inv(x), n, psd_val
        )
        assert passed, f"tr_inv: {msg}"


class TestEdgeCases:
    """Tests for edge cases and domain violations."""

    def test_log_negative_returns_none(self):
        """Test that log returns None gradient for negative values."""
        var = cp.Variable(2)
        var.value = np.array([-1.0, 1.0])
        expr = cp.log(var)
        assert expr.grad[var] is None

    def test_sqrt_zero_boundary(self):
        """Test sqrt gradient at zero boundary."""
        var = cp.Variable(2)
        var.value = np.array([0.0, 1.0])
        expr = cp.sqrt(var)
        # At zero, sqrt gradient is undefined (infinity)
        assert expr.grad[var] is None

    def test_log_det_non_psd(self):
        """Test that log_det returns None for non-PSD matrix."""
        var = cp.Variable((2, 2), symmetric=True)
        var.value = np.array([[-1.0, 0], [0, 1.0]])
        expr = cp.log_det(var)
        assert expr.grad[var] is None

    def test_inv_pos_at_zero(self):
        """Test inv_pos gradient at zero."""
        var = cp.Variable(2)
        var.value = np.array([0.0, 1.0])
        expr = cp.inv_pos(var)
        assert expr.grad[var] is None

    def test_scalar_variable(self):
        """Test gradcheck with scalar variables."""
        var_value = np.array(2.0)
        passed, msg = expression_gradcheck(
            lambda x: cp.square(x),
            (),
            var_value
        )
        assert passed, f"scalar square: {msg}"

    def test_large_matrix(self):
        """Test gradcheck with larger matrices."""
        var_value = AtomInputGenerator.unrestricted((5, 5), seed=42)
        passed, msg = expression_gradcheck(
            lambda x: cp.sum(x),
            (5, 5),
            var_value
        )
        assert passed, f"large matrix sum: {msg}"

    def test_3d_array(self):
        """Test gradcheck with 3D arrays."""
        var_value = AtomInputGenerator.unrestricted((2, 3, 4), seed=42)
        passed, msg = expression_gradcheck(
            lambda x: cp.sum(x),
            (2, 3, 4),
            var_value
        )
        assert passed, f"3D array sum: {msg}"

    def test_cumsum_3d(self):
        """Test cumsum gradient with 3D arrays."""
        var_value = AtomInputGenerator.unrestricted((2, 3, 4), seed=42)
        for axis in [0, 1, 2]:
            passed, msg = expression_gradcheck(
                lambda x, ax=axis: cp.cumsum(x, axis=ax),
                (2, 3, 4),
                var_value
            )
            assert passed, f"3D cumsum axis={axis}: {msg}"


class TestQuadraticAtoms:
    """Tests for quadratic atoms with fixed parameters."""

    @pytest.mark.parametrize("seed", [42, 123])
    def test_quad_form_identity(self, seed: int):
        """Test quad_form(x, I) gradient."""
        n = 3
        x_val = AtomInputGenerator.unrestricted((n,), seed)
        P = np.eye(n)

        passed, msg = expression_gradcheck(
            lambda x: cp.quad_form(x, P),
            (n,),
            x_val
        )
        assert passed, f"quad_form identity: {msg}"

    @pytest.mark.parametrize("seed", [42, 123])
    def test_quad_form_psd(self, seed: int):
        """Test quad_form(x, P) gradient with PSD P."""
        n = 3
        x_val = AtomInputGenerator.unrestricted((n,), seed)
        rng = np.random.default_rng(seed + 100)
        A = rng.standard_normal((n, n))
        P = A @ A.T + np.eye(n)

        passed, msg = expression_gradcheck(
            lambda x: cp.quad_form(x, P),
            (n,),
            x_val
        )
        assert passed, f"quad_form PSD: {msg}"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
