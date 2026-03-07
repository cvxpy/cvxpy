"""
Copyright 2026 The CVXPY Developers

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
from typing import List, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp_sparse
from scipy.special import kl_div as kl_div_scipy
from scipy.special import logsumexp
from scipy.special import rel_entr as rel_entr_scipy

# Type alias for bounds: (lower_bound, upper_bound)
Bounds = Tuple[np.ndarray, np.ndarray]


def _ensure_dense(arr, shape: Optional[Tuple[int, ...]] = None):
    """Convert sparse matrix to dense numpy array and expand to shape if given.

    Parameters
    ----------
    arr : array-like
        The array to densify. May be sparse, a broadcast view, or a scalar.
    shape : tuple of int, optional
        Target shape to broadcast to. If None, returns the array as-is after
        densifying sparse matrices.

    Returns
    -------
    np.ndarray
        Dense array, optionally broadcast to the target shape.
    """
    if sp_sparse.issparse(arr):
        arr = np.asarray(arr.todense())
    else:
        arr = np.asarray(arr)

    if shape is not None and arr.shape != shape:
        # Use np.broadcast_to for memory-efficient expansion, then copy
        # to get a writable array (some solvers may need to modify bounds)
        arr = np.broadcast_to(arr, shape).copy()

    return arr


def _safe_maximum(a, b):
    """Element-wise maximum, handling sparse matrices.

    Uses scipy sparse's .maximum() when either operand is sparse,
    avoiding explicit densification of large sparse matrices.
    """
    if sp_sparse.issparse(a):
        return a.maximum(b)
    if sp_sparse.issparse(b):
        return b.maximum(a)
    return np.maximum(a, b)


def _safe_minimum(a, b):
    """Element-wise minimum, handling sparse matrices.

    Uses scipy sparse's .minimum() when either operand is sparse,
    avoiding explicit densification of large sparse matrices.
    """
    if sp_sparse.issparse(a):
        return a.minimum(b)
    if sp_sparse.issparse(b):
        return b.minimum(a)
    return np.minimum(a, b)


def _all_isinf(arr) -> bool:
    """Check if all values are inf, sparse-aware (O(nnz) for sparse)."""
    if sp_sparse.issparse(arr):
        # Structural zeros are 0, not inf. If there are any structural zeros
        # then not all values are inf.
        if arr.nnz < np.prod(arr.shape):
            return False
        return bool(np.all(np.isinf(arr.data)))
    return bool(np.all(np.isinf(arr)))


def _any_isnan(arr) -> bool:
    """Check if any values are NaN, sparse-aware (O(nnz) for sparse)."""
    if sp_sparse.issparse(arr):
        if arr.nnz == 0:
            return False
        return bool(np.any(np.isnan(arr.data)))
    return bool(np.any(np.isnan(arr)))


def _all_zero_or_inf(arr) -> bool:
    """Check if all values are 0 or inf, sparse-aware (O(nnz) for sparse)."""
    if sp_sparse.issparse(arr):
        # Structural zeros are 0, which satisfies the condition.
        # Only need to check explicitly stored values.
        if arr.nnz == 0:
            return True
        return bool(np.all((arr.data == 0) | np.isinf(arr.data)))
    return bool(np.all((arr == 0) | np.isinf(arr)))


def unbounded(shape: Tuple[int, ...]) -> Bounds:
    """Return unbounded interval (-inf, inf) for given shape.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the bounds arrays.

    Returns
    -------
    Bounds
        A tuple of (lower, upper) arrays filled with -inf and inf.
    """
    lower = np.full(shape, -np.inf)
    upper = np.full(shape, np.inf)
    return (lower, upper)


def uniform_bounds(shape: Tuple[int, ...], lb: float, ub: float) -> Bounds:
    """Return uniform bounds as memory-efficient broadcast views.

    This creates read-only views that broadcast a scalar to the given shape
    without allocating memory for each element. Use this when all elements
    have the same bounds.

    Parameters
    ----------
    shape : tuple of ints
        The shape of the bounds arrays.
    lb : float
        The lower bound value (same for all elements).
    ub : float
        The upper bound value (same for all elements).

    Returns
    -------
    Bounds
        A tuple of broadcast views (lower, upper).
    """
    lb_scalar = np.array(lb)
    ub_scalar = np.array(ub)
    return (np.broadcast_to(lb_scalar, shape), np.broadcast_to(ub_scalar, shape))


def scalar_bounds(lb: float, ub: float) -> Bounds:
    """Return bounds for a scalar.

    Parameters
    ----------
    lb : float
        Lower bound.
    ub : float
        Upper bound.

    Returns
    -------
    Bounds
        A tuple of scalar arrays.
    """
    return (np.array(lb), np.array(ub))


def add_bounds(lb1: np.ndarray, ub1: np.ndarray,
               lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise addition: x + y.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for the first operand.
    lb2, ub2 : np.ndarray
        Bounds for the second operand.

    Returns
    -------
    Bounds
        Bounds for the sum.
    """
    return (lb1 + lb2, ub1 + ub2)


def _sparse_sum(arr, axis, keepdims):
    """Sum a sparse array, handling keepdims which scipy doesn't support."""
    result = arr.sum(axis=axis)
    # scipy sparse .sum() returns np.matrix; convert to ndarray.
    result = np.asarray(result)
    if keepdims:
        result = np.expand_dims(result, axis=axis)
    else:
        result = result.squeeze()
    return result


def sum_bounds(lb: np.ndarray, ub: np.ndarray,
               axis: Optional[Union[int, Tuple[int, ...]]] = None,
               keepdims: bool = False) -> Bounds:
    """Bounds for sum reduction.

    Parameters
    ----------
    lb, ub : array-like
        Bounds for the expression being summed (may be sparse).
    axis : None or int or tuple of ints
        Axis or axes along which to sum.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the sum.
    """
    if sp_sparse.issparse(lb):
        new_lb = _sparse_sum(lb, axis, keepdims)
    else:
        new_lb = np.sum(lb, axis=axis, keepdims=keepdims)
    if sp_sparse.issparse(ub):
        new_ub = _sparse_sum(ub, axis, keepdims)
    else:
        new_ub = np.sum(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def neg_bounds(lb, ub) -> Bounds:
    """Bounds for negation: -x.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.

    Returns
    -------
    Bounds
        Bounds for the negation.
    """
    return (-ub, -lb)


def mul_bounds(lb1, ub1, lb2, ub2) -> Bounds:
    """Bounds for elementwise multiplication: x * y.

    Uses interval arithmetic: the result is [min(products), max(products)]
    where products are all combinations of endpoints.

    Handles sparse matrices without densifying them unnecessarily.

    Parameters
    ----------
    lb1, ub1 : array-like
        Bounds for the first operand (may be sparse).
    lb2, ub2 : array-like
        Bounds for the second operand (may be sparse).

    Returns
    -------
    Bounds
        Bounds for the product.
    """
    # All four products of interval endpoints
    p1 = lb1 * lb2
    p2 = lb1 * ub2
    p3 = ub1 * lb2
    p4 = ub1 * ub2

    # Use iterative min/max when sparse matrices are involved
    if any(sp_sparse.issparse(p) for p in [p1, p2, p3, p4]):
        new_lb = p1
        new_ub = p1
        for p in [p2, p3, p4]:
            new_lb = _safe_minimum(new_lb, p)
            new_ub = _safe_maximum(new_ub, p)
        return (new_lb, new_ub)

    # Dense path: stack and take min/max
    products = np.stack([p1, p2, p3, p4], axis=0)
    new_lb = np.min(products, axis=0)
    new_ub = np.max(products, axis=0)
    return (new_lb, new_ub)


def div_bounds(lb1, ub1, lb2, ub2) -> Bounds:
    """Bounds for elementwise division: x / y.

    Note: If the divisor interval contains zero, returns unbounded.

    Parameters
    ----------
    lb1, ub1 : array-like
        Bounds for the numerator (may be sparse).
    lb2, ub2 : array-like
        Bounds for the divisor (may be sparse).

    Returns
    -------
    Bounds
        Bounds for the quotient.
    """
    # Reciprocal of a sparse matrix is inherently dense (1/0 for structural zeros),
    # so convert sparse divisor bounds to dense.
    lb2 = _ensure_dense(lb2)
    ub2 = _ensure_dense(ub2)

    # Check for division by interval containing zero
    contains_zero = (lb2 <= 0) & (ub2 >= 0)

    # Compute 1/[lb2, ub2]
    # When lb2 and ub2 have the same sign, reciprocal reverses the interval
    inv_lb = np.where(contains_zero, -np.inf, 1.0 / ub2)
    inv_ub = np.where(contains_zero, np.inf, 1.0 / lb2)

    # Now multiply by numerator bounds
    return mul_bounds(lb1, ub1, inv_lb, inv_ub)


def scale_bounds(lb: np.ndarray, ub: np.ndarray, c: float) -> Bounds:
    """Bounds for scalar multiplication: c * x.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    c : float
        The scalar multiplier.

    Returns
    -------
    Bounds
        Bounds for the scaled expression.
    """
    if c >= 0:
        return (c * lb, c * ub)
    else:
        return (c * ub, c * lb)


def abs_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise absolute value: |x|.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.

    Returns
    -------
    Bounds
        Bounds for the absolute value.
    """
    # When interval spans zero, lower bound is 0
    # When interval is entirely positive, |x| = x
    # When interval is entirely negative, |x| = -x
    spans_zero = (lb <= 0) & (ub >= 0)
    entirely_positive = lb >= 0
    entirely_negative = ub <= 0

    new_lb = np.where(spans_zero, 0.0,
                      np.where(entirely_positive, lb, -ub))
    new_ub = np.where(entirely_positive, ub,
                      np.where(entirely_negative, -lb,
                               np.maximum(-lb, ub)))
    return (new_lb, new_ub)


def maximum_bounds(bounds_list: List[Bounds]) -> Bounds:
    """Bounds for elementwise maximum: max(x1, x2, ...).

    Parameters
    ----------
    bounds_list : list of Bounds
        List of (lb, ub) tuples for each argument.

    Returns
    -------
    Bounds
        Bounds for the maximum.
    """
    # Use reduce with np.maximum to handle broadcasting between different shapes
    lb_result = bounds_list[0][0]
    ub_result = bounds_list[0][1]
    for lb, ub in bounds_list[1:]:
        lb_result = np.maximum(lb_result, lb)
        ub_result = np.maximum(ub_result, ub)
    return (lb_result, ub_result)


def minimum_bounds(bounds_list: List[Bounds]) -> Bounds:
    """Bounds for elementwise minimum: min(x1, x2, ...).

    Parameters
    ----------
    bounds_list : list of Bounds
        List of (lb, ub) tuples for each argument.

    Returns
    -------
    Bounds
        Bounds for the minimum.
    """
    # Use reduce with np.minimum to handle broadcasting between different shapes
    lb_result = bounds_list[0][0]
    ub_result = bounds_list[0][1]
    for lb, ub in bounds_list[1:]:
        lb_result = np.minimum(lb_result, lb)
        ub_result = np.minimum(ub_result, ub)
    return (lb_result, ub_result)


def max_reduction_bounds(lb: np.ndarray, ub: np.ndarray,
                         axis: Optional[Union[int, Tuple[int, ...]]] = None,
                         keepdims: bool = False) -> Bounds:
    """Bounds for max reduction: max(x, axis=axis).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis or axes along which to take max.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the max.
    """
    new_lb = np.max(lb, axis=axis, keepdims=keepdims)
    new_ub = np.max(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def min_reduction_bounds(lb: np.ndarray, ub: np.ndarray,
                         axis: Optional[Union[int, Tuple[int, ...]]] = None,
                         keepdims: bool = False) -> Bounds:
    """Bounds for min reduction: min(x, axis=axis).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis or axes along which to take min.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the min.
    """
    new_lb = np.min(lb, axis=axis, keepdims=keepdims)
    new_ub = np.min(ub, axis=axis, keepdims=keepdims)
    return (new_lb, new_ub)


def power_bounds(lb: np.ndarray, ub: np.ndarray, p: float) -> Bounds:
    """Bounds for elementwise power: x^p.

    Handles different cases based on p:
    - p > 0: behavior depends on whether p is even/odd and sign of interval
    - p < 0: requires positive interval
    - p = 0: constant 1

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the base expression.
    p : float
        The power.

    Returns
    -------
    Bounds
        Bounds for the power.
    """
    if p == 0:
        return (np.ones_like(lb), np.ones_like(ub))

    if p > 0:
        if p == int(p) and int(p) % 2 == 0:
            # Even integer power: x^(2k)
            # If interval entirely positive: [lb^p, ub^p]
            # If interval entirely negative: [ub^p, lb^p]
            # If interval spans zero: [0, max(lb^p, ub^p)]
            spans_zero = (lb <= 0) & (ub >= 0)
            entirely_positive = lb >= 0

            lb_power = np.abs(lb) ** p
            ub_power = np.abs(ub) ** p

            new_lb = np.where(spans_zero, 0.0,
                              np.where(entirely_positive, lb_power, ub_power))
            new_ub = np.where(spans_zero, np.maximum(lb_power, ub_power),
                              np.where(entirely_positive, ub_power, lb_power))
            return (new_lb, new_ub)
        else:
            # Odd integer power or non-integer positive power
            # For odd powers, x^p is monotonic: [lb^p, ub^p]
            # For non-integer powers, typically only defined for x >= 0
            # We'll handle the general case conservatively
            if p == int(p):
                # Odd integer power: monotonic
                return (lb ** p, ub ** p)
            else:
                # Non-integer power: requires x >= 0 for real result
                # If lb < 0, we have undefined behavior, return unbounded for those
                valid = lb >= 0
                new_lb = np.where(valid, lb ** p, -np.inf)
                new_ub = np.where(valid, ub ** p, np.inf)
                return (new_lb, new_ub)
    else:
        # Negative power: x^(-|p|) = 1/x^|p|
        # Only defined for x != 0
        # For positive intervals: [ub^p, lb^p] (reverses order)
        # For negative intervals with odd |p|: [ub^p, lb^p]
        # Intervals spanning zero: unbounded
        spans_zero = (lb <= 0) & (ub >= 0)
        entirely_positive = lb > 0

        if p == int(p) and int(-p) % 2 == 1:
            # Negative odd power: monotonically decreasing
            new_lb = np.where(spans_zero, -np.inf, np.minimum(lb ** p, ub ** p))
            new_ub = np.where(spans_zero, np.inf, np.maximum(lb ** p, ub ** p))
        else:
            # Negative even power or non-integer: only for positive x
            new_lb = np.where(entirely_positive, ub ** p, -np.inf)
            new_ub = np.where(entirely_positive, lb ** p, np.inf)

        return (new_lb, new_ub)


def exp_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise exponential: exp(x).

    exp is monotonically increasing, so bounds are [exp(lb), exp(ub)].

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for the exponential.
    """
    return (np.exp(lb), np.exp(ub))


def log_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise natural log: log(x).

    log is monotonically increasing on (0, inf), undefined for x <= 0.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for the logarithm.
    """
    # log is only defined for positive arguments
    valid = lb > 0
    new_lb = np.where(valid, np.log(lb), -np.inf)
    new_ub = np.where(ub > 0, np.log(ub), np.inf)
    return (new_lb, new_ub)


def sqrt_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise square root: sqrt(x).

    sqrt is monotonically increasing on [0, inf).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument (must be >= 0).

    Returns
    -------
    Bounds
        Bounds for the square root.
    """
    # sqrt is only defined for non-negative arguments
    new_lb = np.where(lb >= 0, np.sqrt(lb), -np.inf)
    new_ub = np.where(ub >= 0, np.sqrt(ub), np.inf)
    return (new_lb, new_ub)


def norm1_bounds(lb: np.ndarray, ub: np.ndarray,
                 axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 keepdims: bool = False) -> Bounds:
    """Bounds for 1-norm: sum(|x|).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis along which to compute the norm.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the 1-norm.
    """
    abs_lb, abs_ub = abs_bounds(lb, ub)
    return sum_bounds(abs_lb, abs_ub, axis=axis, keepdims=keepdims)


def norm_inf_bounds(lb: np.ndarray, ub: np.ndarray,
                    axis: Optional[Union[int, Tuple[int, ...]]] = None,
                    keepdims: bool = False) -> Bounds:
    """Bounds for infinity-norm: max(|x|).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression.
    axis : None or int or tuple of ints
        Axis along which to compute the norm.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the infinity norm.
    """
    abs_lb, abs_ub = abs_bounds(lb, ub)
    return max_reduction_bounds(abs_lb, abs_ub, axis=axis, keepdims=keepdims)


def broadcast_bounds(lb, ub, target_shape: Tuple[int, ...]) -> Bounds:
    """Broadcast bounds to a target shape.

    Handles sparse matrices: if a sparse bound already has the target shape,
    it is kept as-is; otherwise it is converted to dense for broadcasting.

    Parameters
    ----------
    lb, ub : array-like
        Bounds to broadcast (may be sparse).
    target_shape : tuple of ints
        Target shape.

    Returns
    -------
    Bounds
        Broadcasted bounds.
    """
    def _broadcast_one(arr, shape):
        if sp_sparse.issparse(arr):
            if arr.shape == shape:
                return arr
            # Can't broadcast sparse to a different shape; convert to dense.
            return np.broadcast_to(arr.toarray(), shape)
        return np.broadcast_to(arr, shape)

    return (_broadcast_one(lb, target_shape), _broadcast_one(ub, target_shape))


def reshape_bounds(lb: np.ndarray, ub: np.ndarray,
                   new_shape: Tuple[int, ...],
                   order: str = 'F') -> Bounds:
    """Reshape bounds to a new shape.

    Parameters
    ----------
    lb, ub : array-like
        Bounds to reshape (may be sparse).
    new_shape : tuple of ints
        New shape.
    order : str
        'C' for row-major (C-style) or 'F' for column-major (Fortran-style).

    Returns
    -------
    Bounds
        Reshaped bounds.
    """
    return (lb.reshape(new_shape, order=order), ub.reshape(new_shape, order=order))


def transpose_bounds(lb: np.ndarray, ub: np.ndarray,
                     axes: Optional[Tuple[int, ...]] = None) -> Bounds:
    """Transpose bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to transpose.
    axes : tuple of ints or None
        Permutation of axes.

    Returns
    -------
    Bounds
        Transposed bounds.
    """
    return (np.transpose(lb, axes), np.transpose(ub, axes))


def index_bounds(lb: np.ndarray, ub: np.ndarray, key) -> Bounds:
    """Index into bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to index.
    key : index
        Index/slice to apply.

    Returns
    -------
    Bounds
        Indexed bounds.
    """
    return (lb[key], ub[key])


def matmul_bounds(lb1, ub1, lb2, ub2) -> Bounds:
    """Bounds for matrix multiplication: x @ y.

    When one operand is a point (lb == ub, i.e. a constant), uses the
    exact split formula: A_pos @ lb2 + A_neg @ ub2. When both operands
    are intervals, returns unbounded since there is no efficient exact
    formula.

    Handles sparse matrices efficiently using scipy sparse operations
    to avoid densifying large sparse matrices.

    Parameters
    ----------
    lb1, ub1 : array-like
        Bounds for the first matrix (may be sparse).
    lb2, ub2 : array-like
        Bounds for the second matrix (may be sparse).

    Returns
    -------
    Bounds
        Bounds for the matrix product.
    """
    # The split formula (A_pos @ B_l + A_neg @ B_u) is only valid when
    # one operand is a known point (lb == ub). When both are intervals
    # the per-element minimum of a*b requires element-wise operations
    # that can't be expressed as matrix products, so we return unbounded.
    def _is_point(lb, ub):
        if sp_sparse.issparse(lb) and sp_sparse.issparse(ub):
            diff = lb - ub
            return diff.nnz == 0
        return np.array_equal(lb, ub)

    lhs_point = _is_point(lb1, ub1)
    rhs_point = _is_point(lb2, ub2)

    if lhs_point:
        # A is a constant matrix. C = A @ B where B in [lb2, ub2].
        # C_ij = sum_k A_ik * B_kj; min over B gives:
        #   A_pos @ lb2 + A_neg @ ub2
        a_pos = _safe_maximum(lb1, 0)
        a_neg = _safe_minimum(lb1, 0)
        new_lb = a_pos @ lb2 + a_neg @ ub2
        new_ub = a_pos @ ub2 + a_neg @ lb2
        return (new_lb, new_ub)

    if rhs_point:
        # B is a constant matrix. C = A @ B where A in [lb1, ub1].
        # C_ij = sum_k A_ik * B_kj; min over A gives:
        #   lb1 @ B_pos + ub1 @ B_neg  (for lower bound)
        b_pos = _safe_maximum(lb2, 0)
        b_neg = _safe_minimum(lb2, 0)
        new_lb = lb1 @ b_pos + ub1 @ b_neg
        new_ub = ub1 @ b_pos + lb1 @ b_neg
        return (new_lb, new_ub)

    # Both operands are intervals — no efficient exact formula.
    shape = (np.shape(lb1)[0],) if np.ndim(lb2) == 1 else (
        np.shape(lb1)[0], np.shape(lb2)[1])
    return unbounded(shape)


def get_expr_bounds_if_supported(expr, solver_context) -> Optional[list]:
    """Get bounds from expression for use on auxiliary variables.

    Returns a [lb, ub] list suitable for passing to Variable(bounds=...),
    or None if bounds should not be set. Returns None when:
    - The solver does not support bounded variables.
    - The expression bounds are entirely unbounded (-inf, inf).
    - The bounds contain NaN values.
    - The bounds only encode sign information that is already captured
      by the expression's sign attributes (avoids redundant constraints).

    Parameters
    ----------
    expr : Expression
        The expression whose bounds to compute.
    solver_context : SolverInfo or None
        Solver context; if None or the solver does not support bounds,
        returns None immediately.

    Returns
    -------
    list or None
        [lb, ub] arrays, or None.
    """
    if solver_context is None or not solver_context.solver_supports_bounds:
        return None
    lb, ub = expr.get_bounds()
    # Check if bounds are finite and worth using.
    # Use sparse-aware checks to avoid densifying large sparse matrices.
    if _all_isinf(lb) and _all_isinf(ub):
        return None
    # Check for NaN values which are not valid bounds
    if _any_isnan(lb) or _any_isnan(ub):
        return None
    # Check if bounds only match sign info (avoid redundant constraints)
    # If lb is all 0 or -inf, and ub is all inf, and expr is nonneg, skip
    lb_trivial = _all_zero_or_inf(lb)
    ub_trivial = _all_isinf(ub)
    if lb_trivial and ub_trivial and expr.is_nonneg():
        return None
    # If ub is all 0 or inf, and lb is all -inf, and expr is nonpos, skip
    ub_trivial_nonpos = _all_zero_or_inf(ub)
    lb_trivial_nonpos = _all_isinf(lb)
    if lb_trivial_nonpos and ub_trivial_nonpos and expr.is_nonpos():
        return None
    # Ensure dense and expand to full shape for solver consumption.
    # This handles broadcast views and sparse matrices.
    return [_ensure_dense(lb, expr.shape), _ensure_dense(ub, expr.shape)]


def get_expr_bounds(expr) -> Optional[list]:
    """Get bounds from expression for use on auxiliary variables.

    Like get_expr_bounds_if_supported but without a solver_context check.
    Returns a [lb, ub] list suitable for passing to Variable(bounds=...),
    or None if bounds are trivial.

    Parameters
    ----------
    expr : Expression
        The expression whose bounds to compute.

    Returns
    -------
    list or None
        [lb, ub] arrays, or None.
    """
    lb, ub = expr.get_bounds()
    if _all_isinf(lb) and _all_isinf(ub):
        return None
    if _any_isnan(lb) or _any_isnan(ub):
        return None
    lb_trivial = _all_zero_or_inf(lb)
    ub_trivial = _all_isinf(ub)
    if lb_trivial and ub_trivial and expr.is_nonneg():
        return None
    ub_trivial_nonpos = _all_zero_or_inf(ub)
    lb_trivial_nonpos = _all_isinf(lb)
    if lb_trivial_nonpos and ub_trivial_nonpos and expr.is_nonpos():
        return None
    return [_ensure_dense(lb, expr.shape), _ensure_dense(ub, expr.shape)]


def logistic_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise logistic: log(1 + exp(x)).

    logistic is monotonically increasing with range (0, inf).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for the logistic function.
    """
    return (np.logaddexp(0, lb), np.logaddexp(0, ub))


def sinh_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise sinh.

    sinh is monotonically increasing with range (-inf, inf).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for sinh.
    """
    return (np.sinh(lb), np.sinh(ub))


def asinh_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise asinh (inverse hyperbolic sine).

    asinh is monotonically increasing with range (-inf, inf).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for asinh.
    """
    return (np.arcsinh(lb), np.arcsinh(ub))


def tanh_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise tanh.

    tanh is monotonically increasing with range (-1, 1).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for tanh.
    """
    return (np.tanh(lb), np.tanh(ub))


def atanh_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise atanh (inverse hyperbolic tangent).

    atanh is monotonically increasing on (-1, 1).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument (should be in (-1, 1)).

    Returns
    -------
    Bounds
        Bounds for atanh.
    """
    valid_lb = lb > -1
    valid_ub = ub < 1
    safe_lb = np.clip(lb, -1 + 1e-15, 1 - 1e-15)
    safe_ub = np.clip(ub, -1 + 1e-15, 1 - 1e-15)
    new_lb = np.where(valid_lb, np.arctanh(safe_lb), -np.inf)
    new_ub = np.where(valid_ub, np.arctanh(safe_ub), np.inf)
    return (new_lb, new_ub)


def sin_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise sin.

    sin has range [-1, 1]. Returns conservative bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for sin.
    """
    shape = np.broadcast_shapes(np.shape(lb), np.shape(ub))
    return uniform_bounds(shape, -1.0, 1.0)


def cos_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise cos.

    cos has range [-1, 1]. Returns conservative bounds.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for cos.
    """
    shape = np.broadcast_shapes(np.shape(lb), np.shape(ub))
    return uniform_bounds(shape, -1.0, 1.0)


def tan_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise tan.

    tan is monotonically increasing on (-pi/2, pi/2).

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for tan.
    """
    half_pi = np.pi / 2
    in_domain = (lb > -half_pi) & (ub < half_pi)
    with np.errstate(invalid='ignore'):
        new_lb = np.where(in_domain, np.tan(lb), -np.inf)
        new_ub = np.where(in_domain, np.tan(ub), np.inf)
    return (new_lb, new_ub)


def entr_bounds(lb: np.ndarray, ub: np.ndarray) -> Bounds:
    """Bounds for elementwise entropy: -x*log(x).

    entr is concave with domain x >= 0, maximum 1/e at x = 1/e.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.

    Returns
    -------
    Bounds
        Bounds for entropy.
    """
    INV_E = 1.0 / np.e

    def _entr_val(x):
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(x > 0, -x * np.log(x), np.where(x == 0, 0.0, -np.inf))

    entr_at_lb = _entr_val(np.maximum(lb, 0))
    entr_at_ub = _entr_val(np.maximum(ub, 0))

    # Lower bound: min of endpoints (concave function attains min at boundary)
    new_lb = np.minimum(entr_at_lb, entr_at_ub)

    # Upper bound: 1/e if maximum point is in [lb, ub], otherwise max of endpoints
    contains_max = (lb <= INV_E) & (ub >= INV_E)
    new_ub = np.where(contains_max, INV_E, np.maximum(entr_at_lb, entr_at_ub))

    return (new_lb, new_ub)


def rel_entr_bounds(lb1: np.ndarray, ub1: np.ndarray,
                    lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise relative entropy: x*log(x/y).

    rel_entr is jointly convex in (x, y) with domain x >= 0, y >= 0.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for x.
    lb2, ub2 : np.ndarray
        Bounds for y.

    Returns
    -------
    Bounds
        Bounds for relative entropy.
    """
    # For a convex function, maximum over a box is at a corner.
    # Compute all four corners and take the max for upper bound.
    x_lb = np.maximum(lb1, 0.0)
    x_ub = np.maximum(ub1, 0.0)
    y_lb = np.maximum(lb2, 1e-300)  # avoid log(0)
    y_ub = np.maximum(ub2, 1e-300)

    with np.errstate(divide='ignore', invalid='ignore'):
        corners = np.stack([
            rel_entr_scipy(x_lb, y_lb),
            rel_entr_scipy(x_lb, y_ub),
            rel_entr_scipy(x_ub, y_lb),
            rel_entr_scipy(x_ub, y_ub),
        ], axis=0)
    corners = np.nan_to_num(corners, nan=np.inf, posinf=np.inf, neginf=-np.inf)

    new_ub = np.max(corners, axis=0)

    # For the lower bound of a convex function over a box, use -inf conservatively
    # since the minimum can be in the interior.
    # But we know rel_entr(x, y) >= -x (since log(x/y) >= 1 - y/x is not useful)
    # Actually rel_entr(x,x) = 0 and it can be negative (when x < y).
    # Use min of corners as a (possibly loose) lower bound.
    new_lb = np.min(corners, axis=0)

    return (new_lb, new_ub)


def kl_div_bounds(lb1: np.ndarray, ub1: np.ndarray,
                  lb2: np.ndarray, ub2: np.ndarray) -> Bounds:
    """Bounds for elementwise KL divergence: x*log(x/y) - x + y.

    kl_div is jointly convex in (x, y) with domain x >= 0, y >= 0.
    Always nonneg by Gibbs' inequality.

    Parameters
    ----------
    lb1, ub1 : np.ndarray
        Bounds for x.
    lb2, ub2 : np.ndarray
        Bounds for y.

    Returns
    -------
    Bounds
        Bounds for KL divergence.
    """
    x_lb = np.maximum(lb1, 0.0)
    x_ub = np.maximum(ub1, 0.0)
    y_lb = np.maximum(lb2, 1e-300)
    y_ub = np.maximum(ub2, 1e-300)

    with np.errstate(divide='ignore', invalid='ignore'):
        corners = np.stack([
            kl_div_scipy(x_lb, y_lb),
            kl_div_scipy(x_lb, y_ub),
            kl_div_scipy(x_ub, y_lb),
            kl_div_scipy(x_ub, y_ub),
        ], axis=0)
    corners = np.nan_to_num(corners, nan=np.inf, posinf=np.inf, neginf=0.0)

    new_ub = np.max(corners, axis=0)
    # kl_div >= 0 always, so lower bound is 0
    new_lb = np.zeros_like(new_ub)

    return (new_lb, new_ub)


def huber_bounds(lb: np.ndarray, ub: np.ndarray, M: float) -> Bounds:
    """Bounds for elementwise Huber function.

    huber(x, M) = x^2 for |x| <= M, 2M|x| - M^2 for |x| >= M.
    Always nonneg, even function.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.
    M : float
        Huber threshold parameter.

    Returns
    -------
    Bounds
        Bounds for the Huber function.
    """
    abs_lb, abs_ub = abs_bounds(lb, ub)

    def _huber_val(x, m):
        return np.where(x <= m, x**2, 2 * m * x - m**2)

    new_lb = _huber_val(abs_lb, M)
    new_ub = _huber_val(abs_ub, M)
    return (new_lb, new_ub)


def log_sum_exp_bounds(lb: np.ndarray, ub: np.ndarray,
                       axis: Optional[Union[int, Tuple[int, ...]]] = None,
                       keepdims: bool = False) -> Bounds:
    """Bounds for log-sum-exp: log(sum(exp(x))).

    log_sum_exp is monotonically increasing in each argument.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.
    axis : None or int or tuple of ints
        Axis along which to reduce.
    keepdims : bool
        Whether to keep reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for log-sum-exp.
    """
    new_lb = logsumexp(lb, axis=axis, keepdims=keepdims)
    new_ub = logsumexp(ub, axis=axis, keepdims=keepdims)
    return (np.asarray(new_lb), np.asarray(new_ub))


def sum_largest_bounds(lb: np.ndarray, ub: np.ndarray, k: float) -> Bounds:
    """Bounds for sum_largest: sum of k largest values.

    sum_largest is monotonically increasing in each argument.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.
    k : float
        Number of largest values to sum (may be fractional).

    Returns
    -------
    Bounds
        Bounds for sum_largest.
    """
    def _sum_largest_val(arr, k):
        flat = arr.flatten()
        n = len(flat)
        k_floor = int(np.floor(k))
        k_frac = k - k_floor
        sorted_vals = np.sort(flat)[::-1]  # descending
        result = np.sum(sorted_vals[:k_floor]) if k_floor > 0 else 0.0
        if k_frac > 0 and k_floor < n:
            result += k_frac * sorted_vals[k_floor]
        return result

    new_lb = _sum_largest_val(np.asarray(lb), k)
    new_ub = _sum_largest_val(np.asarray(ub), k)
    return (np.asarray(new_lb), np.asarray(new_ub))


def quad_over_lin_bounds(lb_x: np.ndarray, ub_x: np.ndarray,
                         lb_y: np.ndarray, ub_y: np.ndarray,
                         axis: Optional[Union[int, Tuple[int, ...]]] = None,
                         keepdims: bool = False) -> Bounds:
    """Bounds for quad_over_lin: sum(x^2)/y.

    Always nonneg when y > 0.

    Parameters
    ----------
    lb_x, ub_x : np.ndarray
        Bounds for x.
    lb_y, ub_y : np.ndarray
        Bounds for y (scalar).
    axis : None or int or tuple of ints
        Axis along which to sum x^2.
    keepdims : bool
        Whether to keep reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for quad_over_lin.
    """
    # Compute bounds on x^2 elementwise
    x_sq_lb, x_sq_ub = power_bounds(lb_x, ub_x, 2.0)

    # Sum along axis
    sum_sq_lb = np.sum(x_sq_lb, axis=axis, keepdims=keepdims)
    sum_sq_ub = np.sum(x_sq_ub, axis=axis, keepdims=keepdims)

    # Divide by y (y must be positive for valid domain)
    y_lo = float(np.min(lb_y))
    y_hi = float(np.max(ub_y))

    if y_lo <= 0:
        # y might be zero or negative, bounds are unbounded
        shape = np.shape(sum_sq_lb)
        return unbounded(shape)

    # sum(x^2)/y: min when numerator min and denominator max
    new_lb = sum_sq_lb / y_hi
    new_ub = sum_sq_ub / y_lo
    return (new_lb, new_ub)


def geo_mean_bounds(lb: np.ndarray, ub: np.ndarray,
                    w: tuple) -> Bounds:
    """Bounds for weighted geometric mean: prod(x_i^w_i).

    geo_mean is monotonically increasing in each argument (for positive args).
    Always nonneg.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument vector (should be nonneg).
    w : tuple
        Weights summing to 1.

    Returns
    -------
    Bounds
        Scalar bounds for the geometric mean.
    """
    lb_flat = np.maximum(np.asarray(lb).flatten(), 0.0)
    ub_flat = np.maximum(np.asarray(ub).flatten(), 0.0)
    w_arr = np.array([float(wi) for wi in w])

    # geo_mean is increasing in each arg, so bounds are at corners
    new_lb = np.prod(lb_flat ** w_arr)
    new_ub = np.prod(ub_flat ** w_arr)
    return (np.asarray(new_lb), np.asarray(new_ub))


def pnorm_bounds(lb: np.ndarray, ub: np.ndarray, p: float,
                 axis: Optional[Union[int, Tuple[int, ...]]] = None,
                 keepdims: bool = False) -> Bounds:
    """Bounds for p-norm: (sum |x_i|^p)^(1/p).

    Always nonneg.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.
    p : float
        The norm parameter (p != 0, 1, inf).
    axis : None or int or tuple of ints
        Axis along which to compute the norm.
    keepdims : bool
        Whether to keep reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the p-norm.
    """
    p_float = float(p)

    if p_float > 1:
        # Convex norm: ||x||_p. Increasing in |x_i|.
        abs_lb, abs_ub = abs_bounds(lb, ub)
        if axis is None:
            abs_lb = abs_lb.flatten()
            abs_ub = abs_ub.flatten()
        # Lower bound: norm of minimum absolute values
        # Upper bound: norm of maximum absolute values
        new_lb = np.linalg.norm(abs_lb, p_float, axis=axis, keepdims=keepdims)
        new_ub = np.linalg.norm(abs_ub, p_float, axis=axis, keepdims=keepdims)
        return (np.asarray(new_lb), np.asarray(new_ub))
    elif 0 < p_float < 1:
        # Concave: requires x >= 0. Increasing in each x_i.
        x_lb = np.maximum(lb, 0.0)
        x_ub = np.maximum(ub, 0.0)
        if axis is None:
            x_lb = x_lb.flatten()
            x_ub = x_ub.flatten()
        new_lb = np.linalg.norm(x_lb, p_float, axis=axis, keepdims=keepdims)
        new_ub = np.linalg.norm(x_ub, p_float, axis=axis, keepdims=keepdims)
        return (np.asarray(new_lb), np.asarray(new_ub))
    else:
        # Negative p or unusual cases: return nonneg bounds
        if axis is not None:
            result = np.linalg.norm(lb, p_float, axis=axis, keepdims=keepdims)
            return uniform_bounds(result.shape, 0.0, np.inf)
        return uniform_bounds(tuple(), 0.0, np.inf)


def prod_bounds(lb: np.ndarray, ub: np.ndarray,
                axis: Optional[Union[int, Tuple[int, ...]]] = None,
                keepdims: bool = False) -> Bounds:
    """Bounds for product: prod(x).

    When all entries are nonneg, prod is monotonically increasing.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the argument.
    axis : None or int or tuple of ints
        Axis along which to take the product.
    keepdims : bool
        Whether to keep reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the product.
    """
    all_nonneg = np.all(lb >= 0)
    if all_nonneg:
        new_lb = np.prod(lb, axis=axis, keepdims=keepdims)
        new_ub = np.prod(ub, axis=axis, keepdims=keepdims)
        return (new_lb, new_ub)

    # General case: product can flip sign. Return unbounded.
    result_shape = np.prod(lb, axis=axis, keepdims=keepdims).shape
    return unbounded(result_shape)


def coords_equal(coords1, coords2) -> bool:
    """Check if two coordinate tuples represent the same sparsity pattern.

    Parameters
    ----------
    coords1 : tuple of array-like
        First coordinate tuple (e.g., (rows, cols) for 2D sparse).
    coords2 : tuple of array-like
        Second coordinate tuple.

    Returns
    -------
    bool
        True if the coordinate tuples are equal.
    """
    if len(coords1) != len(coords2):
        return False
    return all(np.array_equal(c1, c2) for c1, c2 in zip(coords1, coords2))


def refine_bounds_from_sign(lb, ub,
                            is_nonneg: bool, is_nonpos: bool) -> Bounds:
    """Refine bounds based on sign information.

    Parameters
    ----------
    lb, ub : array-like
        Current bounds (may be sparse).
    is_nonneg : bool
        Whether the expression is known to be non-negative.
    is_nonpos : bool
        Whether the expression is known to be non-positive.

    Returns
    -------
    Bounds
        Refined bounds.
    """
    if is_nonneg:
        lb = _safe_maximum(lb, 0)
    if is_nonpos:
        ub = _safe_minimum(ub, 0)
    return (lb, ub)
