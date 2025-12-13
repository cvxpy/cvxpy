"""
Debugging utilities for canon backends.

Provides tools for:
- Tensor inspection: Pretty-print CompactTensor and TensorView contents
- Backend comparison: Run same linop on multiple backends, diff outputs
- Memory profiling: Compare memory footprint between backends
"""
from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import scipy.sparse as sp

if TYPE_CHECKING:
    from cvxpy.lin_ops.backends import TensorRepresentation, TensorView
    from cvxpy.lin_ops.backends.coo_backend import CompactTensor


# =============================================================================
# Tensor Inspection
# =============================================================================

def describe_compact_tensor(ct: CompactTensor, name: str = "tensor") -> str:
    """Pretty-print CompactTensor structure."""
    total_size = ct.m * ct.n * ct.param_size
    density = len(ct.data) / max(1, total_size) * 100

    lines = [
        f"{name}: CompactTensor",
        f"  shape: ({ct.m}, {ct.n}) x {ct.param_size} params",
        f"  nnz: {len(ct.data):,}",
        f"  density: {density:.2f}%",
    ]

    if ct.param_size > 1:
        counts = Counter(ct.param_idx)
        lines.append(f"  nnz per param: {dict(sorted(counts.items()))}")

    if len(ct.data) > 0:
        lines.append(f"  data range: [{ct.data.min():.4g}, {ct.data.max():.4g}]")
        lines.append(f"  row range: [{ct.row.min()}, {ct.row.max()}]")
        lines.append(f"  col range: [{ct.col.min()}, {ct.col.max()}]")

    return "\n".join(lines)


def describe_tensor_view(view: TensorView, indent: int = 0) -> str:
    """Describe TensorView structure (Dict[var_id, Dict[param_id, tensor]])."""
    prefix = "  " * indent
    lines = [f"{prefix}TensorView:"]
    lines.append(f"{prefix}  rows: {view.rows}")
    lines.append(f"{prefix}  var_length: {view.var_length}")
    lines.append(f"{prefix}  param_free: {view.is_parameter_free}")
    lines.append(f"{prefix}  variable_ids: {view.variable_ids}")

    if view.tensor is not None:
        lines.append(f"{prefix}  tensors:")
        for var_id, var_tensor in view.tensor.items():
            lines.append(f"{prefix}    var_id={var_id}:")
            for param_id, tensor in var_tensor.items():
                if hasattr(tensor, 'nnz'):
                    lines.append(f"{prefix}      param_id={param_id}: nnz={tensor.nnz}")
                elif hasattr(tensor, 'shape'):
                    lines.append(f"{prefix}      param_id={param_id}: shape={tensor.shape}")

    return "\n".join(lines)


def tensor_to_dense(tr: TensorRepresentation, param_offset: int = 0) -> np.ndarray:
    """Convert TensorRepresentation to dense array for a specific param offset."""
    mask = tr.parameter_offset == param_offset
    return sp.coo_matrix(
        (tr.data[mask], (tr.row[mask], tr.col[mask])),
        shape=tr.shape
    ).toarray()


# =============================================================================
# Backend Comparison
# =============================================================================

def compare_tensor_representations(
    tr1: TensorRepresentation,
    tr2: TensorRepresentation,
    rtol: float = 1e-12,
    atol: float = 1e-12,
    name1: str = "tr1",
    name2: str = "tr2"
) -> dict:
    """Compare two TensorRepresentations, return diff summary."""
    result = {
        "match": True,
        "shape_match": tr1.shape == tr2.shape,
        "details": [],
    }

    if not result["shape_match"]:
        result["match"] = False
        result["details"].append(f"Shape mismatch: {tr1.shape} vs {tr2.shape}")
        return result

    # Get unique param offsets
    offsets1 = set(tr1.parameter_offset)
    offsets2 = set(tr2.parameter_offset)

    if offsets1 != offsets2:
        result["match"] = False
        result["details"].append(f"Param offsets differ: {offsets1} vs {offsets2}")

    # Compare each param slice
    for offset in offsets1 & offsets2:
        dense1 = tensor_to_dense(tr1, offset)
        dense2 = tensor_to_dense(tr2, offset)

        if not np.allclose(dense1, dense2, rtol=rtol, atol=atol):
            result["match"] = False
            diff = np.abs(dense1 - dense2)
            max_diff = diff.max()
            max_loc = np.unravel_index(diff.argmax(), diff.shape)
            result["details"].append(
                f"Param offset {offset}: max diff {max_diff:.2e} at {max_loc}"
            )

    return result


def compare_backends_on_problem(prob, backends: list[str] = None):
    """
    Run same problem on multiple backends, compare results.

    Args:
        prob: CVXPY Problem
        backends: List of backend names (default: ["SCIPY", "COO"])

    Returns:
        dict with comparison results
    """
    import cvxpy.settings as s

    if backends is None:
        backends = [s.SCIPY_CANON_BACKEND, s.COO_CANON_BACKEND]

    results = {}
    for backend in backends:
        try:
            data = prob.get_problem_data(solver=None, canon_backend=backend)
            results[backend] = data
        except Exception as e:
            results[backend] = {"error": str(e)}

    return results


# =============================================================================
# Memory Profiling
# =============================================================================

def memory_footprint_compact_tensor(ct: CompactTensor) -> dict:
    """Estimate memory usage of a CompactTensor in bytes."""
    return {
        "data": ct.data.nbytes,
        "row": ct.row.nbytes,
        "col": ct.col.nbytes,
        "param_idx": ct.param_idx.nbytes,
        "total": ct.data.nbytes + ct.row.nbytes + ct.col.nbytes + ct.param_idx.nbytes,
    }


def memory_footprint_sparse(matrix: sp.spmatrix) -> dict:
    """Estimate memory usage of a scipy sparse matrix in bytes."""
    csr = matrix.tocsr()
    return {
        "data": csr.data.nbytes,
        "indices": csr.indices.nbytes,
        "indptr": csr.indptr.nbytes,
        "total": csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes,
    }


def compare_memory(compact: CompactTensor) -> dict:
    """Compare memory usage of CompactTensor vs equivalent stacked sparse."""
    compact_mem = memory_footprint_compact_tensor(compact)
    stacked = compact.to_stacked_sparse()
    stacked_mem = memory_footprint_sparse(stacked)

    return {
        "compact_bytes": compact_mem["total"],
        "stacked_bytes": stacked_mem["total"],
        "ratio": stacked_mem["total"] / max(1, compact_mem["total"]),
        "savings_pct": (1 - compact_mem["total"] / max(1, stacked_mem["total"])) * 100,
    }


# =============================================================================
# Validation
# =============================================================================

def validate_compact_tensor(ct: CompactTensor) -> list[str]:
    """Check CompactTensor for common issues."""
    issues = []

    # Check array lengths match
    lengths = [len(ct.data), len(ct.row), len(ct.col), len(ct.param_idx)]
    if len(set(lengths)) > 1:
        issues.append(f"Array length mismatch: {lengths}")

    # Check row bounds
    if len(ct.row) > 0:
        if ct.row.min() < 0:
            issues.append(f"Negative row index: {ct.row.min()}")
        if ct.row.max() >= ct.m:
            issues.append(f"Row index out of bounds: {ct.row.max()} >= {ct.m}")

    # Check col bounds
    if len(ct.col) > 0:
        if ct.col.min() < 0:
            issues.append(f"Negative col index: {ct.col.min()}")
        if ct.col.max() >= ct.n:
            issues.append(f"Col index out of bounds: {ct.col.max()} >= {ct.n}")

    # Check param_idx bounds
    if len(ct.param_idx) > 0:
        if ct.param_idx.min() < 0:
            issues.append(f"Negative param_idx: {ct.param_idx.min()}")
        if ct.param_idx.max() >= ct.param_size:
            issues.append(f"param_idx out of bounds: {ct.param_idx.max()} >= {ct.param_size}")

    # Check for NaN/Inf
    if np.any(np.isnan(ct.data)):
        issues.append("Data contains NaN")
    if np.any(np.isinf(ct.data)):
        issues.append("Data contains Inf")

    return issues
