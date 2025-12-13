"""
Backwards compatibility shim for lazy_tensor_view.

All implementation has been moved to cvxpy.lin_ops.backends.lazy_backend.
This module re-exports all public classes for backwards compatibility.
"""
from cvxpy.lin_ops.backends.lazy_backend import (
    CompactTensor,
    LazyCanonBackend,
    LazyTensorView,
    compact_matmul,
    compact_mul_elem,
    compact_reshape,
)

__all__ = [
    "CompactTensor",
    "LazyCanonBackend",
    "LazyTensorView",
    "compact_matmul",
    "compact_mul_elem",
    "compact_reshape",
]
