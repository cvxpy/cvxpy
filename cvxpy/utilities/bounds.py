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

# Type alias for bounds: (lower_bound, upper_bound)
Bounds = Tuple[np.ndarray, np.ndarray]


def _ensure_dense(arr):
    """Convert sparse matrix to dense numpy array. Return dense arrays unchanged."""
    if sp_sparse.issparse(arr):
        return np.asarray(arr.todense())
    return np.asarray(arr)


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


def sum_bounds(lb: np.ndarray, ub: np.ndarray,
               axis: Optional[Union[int, Tuple[int, ...]]] = None,
               keepdims: bool = False) -> Bounds:
    """Bounds for sum reduction.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds for the expression being summed.
    axis : None or int or tuple of ints
        Axis or axes along which to sum.
    keepdims : bool
        Whether to keep the reduced dimensions.

    Returns
    -------
    Bounds
        Bounds for the sum.
    """
    new_lb = np.sum(lb, axis=axis, keepdims=keepdims)
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
                   new_shape: Tuple[int, ...]) -> Bounds:
    """Reshape bounds to a new shape.

    Parameters
    ----------
    lb, ub : np.ndarray
        Bounds to reshape.
    new_shape : tuple of ints
        New shape.

    Returns
    -------
    Bounds
        Reshaped bounds.
    """
    return (lb.reshape(new_shape), ub.reshape(new_shape))


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
    # Ensure dense for Variable(bounds=...) consumption.
    return [_ensure_dense(lb), _ensure_dense(ub)]


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
