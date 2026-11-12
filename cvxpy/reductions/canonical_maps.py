"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Building blocks of the canonical parameter API.

``Problem.get_canonical_maps()`` exports three things from a compiled
reduction chain:

1. **Parameter preparation** -- the reductions' parameter transformations
   (:class:`ParameterTransform`) and the packing of compiled parameters
   into ``param_vec``.
2. **Solver inputs** -- for every solver input, a sparse matrix ``M`` and a
   constant vector ``v`` such that ``M @ param_vec + v`` is that input (the
   stored values, for a sparse matrix input; see :class:`MatrixStructure`).
3. **Solution recovery** -- index arrays (or sparse matrices) that map the
   solver's primal and dual vectors to the requested variables and duals.

This module holds the data types plus the small amount of linear-algebra
plumbing that the reductions and solver interfaces use to build them. It
deliberately depends on NumPy/SciPy only, so every reduction can import it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import scipy.sparse as sp

# ---------------------------------------------------------------------------
# Parameter transformations
# ---------------------------------------------------------------------------


class ParameterTransformKind(Enum):
    GATHER = "gather"
    LOG = "log"
    REAL = "real"
    IMAG = "imag"


@dataclass
class ParameterTransform:
    """One parameter transformation performed by a reduction.

    ``target = op(source)``. ``LOG``, ``REAL`` and ``IMAG`` act elementwise and
    preserve shape. ``GATHER`` returns the vector ``source_flat[indices]``,
    where ``source_flat`` is the source value flattened in Fortran order, or,
    for a source with a ``sparsity`` attribute, the vector of its stored
    entries in CVXPY's sparse-index order.
    """
    source_id: int
    target_id: int
    kind: ParameterTransformKind
    indices: np.ndarray | None = None  # Required only for GATHER.


def transform_source_value(param):
    """The value of ``param`` as the parameter transformations consume it.

    Sparse parameters contribute their stored entries; every other parameter
    contributes its (dense or SciPy-sparse) value. ``None`` if unset.
    """
    if param.sparse_idx is not None:
        return None if param._value is None else np.asarray(param._value)
    return param.value


def apply_param_transform(transform: ParameterTransform, value):
    """Evaluate a single transformation with NumPy/SciPy."""
    kind = transform.kind
    if kind == ParameterTransformKind.GATHER:
        if sp.issparse(value):
            rows, cols = np.unravel_index(transform.indices, value.shape, order='F')
            return np.asarray(sp.csr_array(value)[rows, cols]).ravel()
        return np.asarray(value).ravel(order='F')[transform.indices]
    if sp.issparse(value):
        value = value.toarray()
    if kind == ParameterTransformKind.LOG:
        return np.log(value)
    if kind == ParameterTransformKind.REAL:
        return np.real(value)
    if kind == ParameterTransformKind.IMAG:
        return np.imag(value)
    raise ValueError(f"Unknown parameter transform kind {kind}.")


def evaluate_param_transforms(transforms, values: dict[int, Any]) -> dict[int, Any]:
    """Evaluate ``transforms`` in order, starting from ``values``.

    ``values`` maps parameter IDs to values (see :func:`transform_source_value`).
    Returns a new dict that additionally holds every target, including
    intermediate IDs. A ``None`` source yields a ``None`` target.
    """
    values = dict(values)
    for t in transforms:
        src = values.get(t.source_id)
        values[t.target_id] = None if src is None else apply_param_transform(t, src)
    return values


def update_transformed_parameters(transforms, params: dict[int, Any]) -> None:
    """Assign the targets of ``transforms`` from their sources, in place.

    ``params`` maps IDs to the ``Parameter`` objects a reduction owns, sources
    and targets alike. IDs without a ``Parameter`` are intermediate values.
    Targets whose source is unset are left untouched.
    """
    targets = {t.target_id for t in transforms}
    values = {pid: transform_source_value(p) for pid, p in params.items()
              if pid not in targets}
    values = evaluate_param_transforms(transforms, values)
    for pid in targets:
        param = params.get(pid)
        value = values.get(pid)
        if param is None or value is None:
            continue
        if param.sparse_idx is not None:
            # The value holds the stored entries.
            param.value_sparse = sp.coo_array((value, param.sparse_idx), shape=param.shape)
        else:
            param.value = np.reshape(value, param.shape, order='F')


def pack_param_vec(values: dict[int, Any], param_id_to_col: dict[int, int],
                   param_id_to_size: dict[int, int], total_param_size: int) -> np.ndarray:
    """Pack compiled parameter values into ``param_vec`` (no constant sentinel)."""
    param_vec = np.zeros(total_param_size)
    for pid, col in param_id_to_col.items():
        value = values[pid]
        if value is None:
            raise ValueError(f"Parameter {pid} has no value.")
        if sp.issparse(value):
            value = value.toarray()
        size = param_id_to_size[pid]
        param_vec[col:col + size] = np.asarray(value, dtype=float).ravel(order='F')
    return param_vec


# ---------------------------------------------------------------------------
# Solver inputs
# ---------------------------------------------------------------------------


@dataclass
class MatrixStructure:
    """Sparsity structure of a sparse solver input.

    The input's affine map produces its stored values in this order.
    """
    format: Literal["csc", "csr"]
    shape: tuple[int, int]
    indices: np.ndarray
    indptr: np.ndarray

    def assemble(self, data: np.ndarray) -> sp.csc_array | sp.csr_array:
        cls = sp.csc_array if self.format == "csc" else sp.csr_array
        return cls((data, self.indices, self.indptr), shape=self.shape)


def _scale_csr_rows(M: sp.csr_array, s: np.ndarray) -> sp.csr_array:
    """``diag(s) @ M`` for a CSR matrix, without a sparse product."""
    M = sp.csr_array(M, copy=True)
    M.data *= np.repeat(s, np.diff(M.indptr))
    return M


class AffineVector:
    """A dense vector ``M @ param_vec + v``."""

    def __init__(self, M: sp.csr_array, v: np.ndarray) -> None:
        self.M = sp.csr_array(M)
        self.v = np.asarray(v, dtype=float)

    @property
    def size(self) -> int:
        return self.v.size

    def take(self, idx) -> AffineVector:
        idx = np.asarray(idx, dtype=np.int64)
        return AffineVector(self.M[idx], self.v[idx])

    def scale(self, s) -> AffineVector:
        s = np.broadcast_to(np.asarray(s, dtype=float), self.v.shape)
        return AffineVector(_scale_csr_rows(self.M, s), s * self.v)

    @staticmethod
    def constant(v, n_params: int) -> AffineVector:
        v = np.asarray(v, dtype=float).ravel()
        return AffineVector(sp.csr_array((v.size, n_params)), v)

    @staticmethod
    def concat(vecs, n_params: int) -> AffineVector:
        if not vecs:
            return AffineVector.constant(np.zeros(0), n_params)
        return AffineVector(sp.vstack([x.M for x in vecs], format='csr'),
                            np.concatenate([x.v for x in vecs]))


class AffineSparseMatrix:
    """A sparse matrix whose stored values are ``M @ param_vec + v``.

    Entries are kept in COO form (``rows``, ``cols``); ``M`` has one row per
    entry. The structure never depends on parameter values, so entries that
    evaluate to zero for particular values stay present. Each ``(row, col)``
    position holds at most one entry: :func:`tensor_matrix_block` produces
    unique entries and the other methods only drop or rescale them.
    """

    def __init__(self, shape, rows, cols, M, v) -> None:
        self.shape = (int(shape[0]), int(shape[1]))
        self.rows = np.asarray(rows, dtype=np.int64)
        self.cols = np.asarray(cols, dtype=np.int64)
        self.M = sp.csr_array(M)
        self.v = np.asarray(v, dtype=float)

    def row_block(self, start: int, stop: int) -> AffineSparseMatrix:
        """Rows ``start:stop``, renumbered from zero (entry order is kept)."""
        keep = np.flatnonzero((self.rows >= start) & (self.rows < stop))
        return AffineSparseMatrix((stop - start, self.shape[1]), self.rows[keep] - start,
                                  self.cols[keep], self.M[keep], self.v[keep])

    def scale_rows(self, s) -> AffineSparseMatrix:
        """Scale matrix row ``i`` by ``s[i]``."""
        s = np.asarray(s, dtype=float)[self.rows]
        return AffineSparseMatrix(self.shape, self.rows, self.cols,
                                  _scale_csr_rows(self.M, s), s * self.v)

    def triu(self) -> AffineSparseMatrix:
        keep = np.flatnonzero(self.rows <= self.cols)
        return AffineSparseMatrix(self.shape, self.rows[keep], self.cols[keep],
                                  self.M[keep], self.v[keep])

    def to_structure(self, fmt: Literal["csc", "csr"] = "csc"):
        """Order the entries for ``fmt``; return ``(structure, M, v)``."""
        major, minor = (self.cols, self.rows) if fmt == "csc" else (self.rows, self.cols)
        n_major, n_minor = (self.shape[1], self.shape[0]) if fmt == "csc" else self.shape
        key = major * n_minor + minor
        if np.all(key[1:] > key[:-1]):
            # Already ordered, as entries extracted from a stuffed tensor
            # are for CSC.
            order = slice(None)
        else:
            order = np.argsort(key, kind='stable')
            sorted_key = key[order]
            if np.any(sorted_key[1:] == sorted_key[:-1]):
                raise ValueError("AffineSparseMatrix has duplicate (row, col) entries.")
        indptr = np.zeros(n_major + 1, dtype=np.int64)
        np.cumsum(np.bincount(major, minlength=n_major), out=indptr[1:])
        structure = MatrixStructure(fmt, self.shape, minor[order].astype(np.int64), indptr)
        return structure, self.M[order], self.v[order]


class StuffedTensor:
    """A stuffed parameter tensor, reduced to its nonzero rows.

    The tensor has one row per entry of a (flattened) problem-data matrix,
    which can be far more rows than nonzeros, so everything here is linear
    in the number of nonzeros. It has ``n_params + 1`` columns, the last
    holding the constant offset.

    ``nonzero_rows`` lists the tensor rows with a nonzero coefficient, in
    increasing order, and row ``k`` of the CSR matrix ``reduced`` is tensor
    row ``nonzero_rows[k]``. This is the representation ``ReducedMat``
    caches.
    """

    def __init__(self, tensor) -> None:
        coo = sp.coo_array(tensor)
        keep = coo.data != 0
        row, col, data = coo.row[keep].astype(np.int64), coo.col[keep], coo.data[keep]
        if np.any(row[1:] < row[:-1]):
            order = np.argsort(row, kind='stable')
            row, col, data = row[order], col[order], data[order]
        new_row = np.empty(row.size, dtype=bool)
        new_row[:1] = True
        np.not_equal(row[1:], row[:-1], out=new_row[1:])
        starts = np.flatnonzero(new_row)
        self.nonzero_rows = row[starts]
        indptr = np.append(starts, row.size).astype(np.int64)
        self.reduced = sp.csr_array((data, col, indptr),
                                    shape=(self.nonzero_rows.size, coo.shape[1]))
        self.reduced.sum_duplicates()
        n_params = coo.shape[1] - 1
        # The parameter columns and the constant column, split once.
        self.linear = self.reduced[:, :n_params]
        self.constant = self.reduced[:, [n_params]].toarray().ravel()

    def row_range(self, start: int, stop: int):
        """``(rows, M, v)`` for the nonzero rows in ``start:stop``."""
        lo, hi = np.searchsorted(self.nonzero_rows, [start, stop])
        return self.nonzero_rows[lo:hi], self.linear[lo:hi], self.constant[lo:hi]


def prepare_tensor(tensor) -> StuffedTensor | None:
    """Wrap a stuffed parameter tensor for block extraction."""
    return None if tensor is None else StuffedTensor(tensor)


def tensor_matrix_block(tensor: StuffedTensor | None, n_rows: int, n_cols: int,
                        n_params: int) -> AffineSparseMatrix:
    """The ``n_rows x n_cols`` matrix in the leading rows of a stuffed tensor.

    The tensor has one row per matrix entry, in Fortran order. Only rows
    with a nonzero coefficient become structural entries; they come out in
    CSC order.
    """
    if tensor is None or n_rows * n_cols == 0:
        return AffineSparseMatrix((n_rows, n_cols), [], [], sp.csr_array((0, n_params)), [])
    flat, M, v = tensor.row_range(0, n_rows * n_cols)
    return AffineSparseMatrix((n_rows, n_cols), flat % n_rows, flat // n_rows, M, v)


def tensor_vector_block(tensor: StuffedTensor | None, start: int, stop: int,
                        n_params: int) -> AffineVector:
    """Rows ``start:stop`` of a stuffed tensor, as a dense affine vector."""
    if tensor is None or stop == start:
        return AffineVector.constant(np.zeros(stop - start), n_params)
    rows, M, v = tensor.row_range(start, stop)
    # Scatter the nonzero rows into the dense block.
    counts = np.zeros(stop - start + 1, dtype=np.int64)
    counts[rows - start + 1] = np.diff(M.indptr)
    M_full = sp.csr_array((M.data, M.indices, np.cumsum(counts)),
                          shape=(stop - start, n_params))
    v_full = np.zeros(stop - start)
    v_full[rows - start] = v
    return AffineVector(M_full, v_full)


@dataclass
class SolverInputMaps:
    """The affine maps from ``param_vec`` to one solver interface's inputs."""
    maps: dict[str, tuple[sp.csr_array, np.ndarray]] = field(default_factory=dict)
    structures: dict[str, MatrixStructure] = field(default_factory=dict)

    def add_vector(self, name: str, vec: AffineVector) -> None:
        self.maps[name] = (vec.M, vec.v)

    def add_matrix(self, name: str, mat: AffineSparseMatrix,
                   fmt: Literal["csc", "csr"] = "csc") -> None:
        structure, M, v = mat.to_structure(fmt)
        self.maps[name] = (M, v)
        self.structures[name] = structure

    def evaluate(self, param_vec: np.ndarray) -> dict[str, np.ndarray | sp.sparray]:
        """Evaluate every input at ``param_vec``; sparse inputs are assembled."""
        values = {}
        for name, (M, v) in self.maps.items():
            data = M @ param_vec + v if M.shape[1] else v.copy()
            structure = self.structures.get(name)
            values[name] = data if structure is None else structure.assemble(data)
        return values


# ---------------------------------------------------------------------------
# Solution recovery
# ---------------------------------------------------------------------------


@dataclass
class Exp:
    """Recovery ``exp(solver_x[indices])``, used for DGP primal variables."""
    indices: np.ndarray


class UnsupportedRecovery:
    """Marks an output that cannot be expressed by canonical recovery maps.

    Composition propagates it; it only raises when the output is requested.
    """

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def take(self, idx) -> UnsupportedRecovery:
        return self

    def linear(self, mat) -> UnsupportedRecovery:
        return self

    def with_exp(self) -> UnsupportedRecovery:
        return self

    def finalize(self) -> UnsupportedRecovery:
        return self


class Recovery:
    """A linear map from a solver output vector to a block of values.

    ``source`` names the solver vector (``'x'`` for the primal, ``'y'`` for
    the dual) and ``n_source`` its length. The map is either a gather
    (``indices``) or a sparse matrix (``matrix``). ``exp`` marks a gather
    followed by elementwise exponentiation.
    """

    def __init__(self, source: str, n_source: int, indices=None, matrix=None,
                 exp: bool = False) -> None:
        assert (indices is None) != (matrix is None)
        self.source = source
        self.n_source = int(n_source)
        self.indices = None if indices is None else np.asarray(indices, dtype=np.int64)
        self.matrix = None if matrix is None else sp.csr_array(matrix)
        self.exp = exp

    @staticmethod
    def identity(source: str, n_source: int, start: int = 0, size: int | None = None):
        size = n_source - start if size is None else size
        return Recovery(source, n_source, indices=np.arange(start, start + size))

    @staticmethod
    def zeros(source: str, n_source: int, size: int) -> Recovery:
        return Recovery(source, n_source, matrix=sp.csr_array((size, n_source)))

    @property
    def size(self) -> int:
        return self.indices.size if self.indices is not None else self.matrix.shape[0]

    def as_matrix(self) -> sp.csr_array:
        if self.matrix is not None:
            return self.matrix
        n = self.indices.size
        return sp.csr_array((np.ones(n), (np.arange(n), self.indices)),
                            shape=(n, self.n_source))

    def take(self, idx) -> Recovery:
        idx = np.asarray(idx, dtype=np.int64)
        if self.indices is not None:
            return Recovery(self.source, self.n_source, indices=self.indices[idx],
                            exp=self.exp)
        return Recovery(self.source, self.n_source, matrix=self.matrix[idx], exp=self.exp)

    def linear(self, mat) -> Recovery | UnsupportedRecovery:
        """The map ``mat @ self``."""
        if self.exp:
            return UnsupportedRecovery("a linear map cannot follow exp()")
        return Recovery(self.source, self.n_source,
                        matrix=sp.csr_array(mat) @ self.as_matrix())

    def with_exp(self) -> Recovery:
        return Recovery(self.source, self.n_source, indices=self.indices,
                        matrix=self.matrix, exp=True)

    @staticmethod
    def concat(recs) -> Recovery | UnsupportedRecovery:
        """Stack ``recs`` (at least one) into one recovery."""
        if not recs:
            raise ValueError("Recovery.concat needs at least one recovery to "
                             "determine the source vector.")
        for rec in recs:
            if isinstance(rec, UnsupportedRecovery):
                return rec
        first = recs[0]
        if any(r.source != first.source or r.exp != first.exp for r in recs):
            raise ValueError("Cannot concatenate recoveries from different sources.")
        if all(r.indices is not None for r in recs):
            return Recovery(first.source, first.n_source,
                            indices=np.concatenate([r.indices for r in recs]),
                            exp=first.exp)
        return Recovery(first.source, first.n_source,
                        matrix=sp.vstack([r.as_matrix() for r in recs], format='csr'),
                        exp=first.exp)

    def finalize(self) -> np.ndarray | Exp | sp.csr_array | UnsupportedRecovery:
        """The exported form: indices, ``Exp(indices)``, or a sparse matrix.

        An exponentiated recovery that is not a gather has no exported form.
        """
        indices = self.indices
        if indices is None:
            mat = sp.csr_array(self.matrix, copy=True)
            mat.sum_duplicates()
            mat.eliminate_zeros()
            if np.all(np.diff(mat.indptr) == 1) and np.all(mat.data == 1):
                indices = mat.indices.astype(np.int64)
            elif self.exp:
                return UnsupportedRecovery("an exponentiated recovery must be a gather")
            else:
                return mat
        return Exp(indices) if self.exp else indices


@dataclass
class RecoveryMap:
    """Recoveries of all primal variables and constraint duals at one level.

    ``primal`` maps variable IDs to the Fortran-flattened variable value;
    ``dual`` maps constraint IDs to the constraint's dual value as CVXPY
    stores it at that level, flattened in Fortran order (multi-part duals,
    such as RSOC's, concatenate their parts).
    """
    n_x: int
    n_y: int
    primal: dict[int, Recovery | UnsupportedRecovery] = field(default_factory=dict)
    dual: dict[int, Recovery | UnsupportedRecovery] = field(default_factory=dict)

    def get_primal(self, var_id: int) -> Recovery | UnsupportedRecovery:
        return self.primal.get(var_id, UnsupportedRecovery(
            f"variable {var_id} is not recoverable from the compiled problem"))

    def get_dual(self, constr_id: int) -> Recovery | UnsupportedRecovery:
        return self.dual.get(constr_id, UnsupportedRecovery(
            f"the dual of constraint {constr_id} is not recoverable "
            "from the compiled problem"))

    def replace(self, primal=None, dual=None) -> RecoveryMap:
        return RecoveryMap(self.n_x, self.n_y,
                           self.primal if primal is None else primal,
                           self.dual if dual is None else dual)

    def pass_through(self, var_ids, cons_id_map) -> RecoveryMap:
        """Keep ``var_ids``; rename duals via ``{outer_id: inner_id}``."""
        return self.replace(
            primal={vid: self.get_primal(vid) for vid in var_ids},
            dual={orig: self.get_dual(new) for orig, new in cons_id_map.items()})


def trace_scaled_selection(fn, n_in: int) -> sp.csr_array:
    """Matrix ``S`` with ``fn(v) == S @ v`` for a scaled-selection map ``fn``.

    ``fn`` must be linear (as dual recovery code is: it indexes, reshapes
    and scales), and each of its output entries must be a multiple of at
    most one input entry. This reuses the NumPy recovery code of a reduction
    as the source of truth, rather than duplicating its index arithmetic.

    Applying ``fn`` to ones gives each output's scale, and to ``1, ..., n_in``
    its input index. The inferred ``S`` is then checked on two random
    inputs: for a linear ``fn`` that differs from ``S``, a random input
    agrees with ``S`` only with probability zero. Raises
    ``NotImplementedError`` if ``fn`` is not such a map; reductions turn this
    into an :class:`UnsupportedRecovery`.
    """
    def call(v):
        return np.asarray(fn(v), dtype=float).ravel(order='F')

    ones = call(np.ones(n_in))
    ramp = call(np.arange(1.0, n_in + 1))
    nz = np.flatnonzero(ones)
    ratio = ramp[nz] / ones[nz]
    cols = np.rint(ratio).astype(np.int64) - 1
    if not (np.allclose(ratio, cols + 1) and np.all((cols >= 0) & (cols < n_in))):
        raise NotImplementedError("Recovery is not a scaled selection.")
    S = sp.csr_array((ones[nz], (nz, cols)), shape=(ones.size, n_in))
    rng = np.random.default_rng(0)
    for _ in range(2):
        probe = rng.standard_normal(n_in)
        if not np.allclose(call(probe), S @ probe):
            raise NotImplementedError("Recovery is not a scaled selection.")
    return S


# ---------------------------------------------------------------------------
# The exported object
# ---------------------------------------------------------------------------


@dataclass
class VarInfo:
    var_id: int
    shape: tuple[int, ...]  # Logical shape.
    size: int  # Number of recovered entries.
    sparse_idx: tuple[np.ndarray, ...] | None  # None for dense outputs.
    is_symmetric: bool


@dataclass
class DualInfo:
    dual_var_id: int
    constraint_id: int
    shape: tuple[int, ...]
    size: int
    cone_type: str
    is_psd: bool


@dataclass
class CanonicalMaps:
    """Maps between user parameters, solver inputs, and solution values.

    See :meth:`cvxpy.Problem.get_canonical_maps`. The ``prepare_param_vec``,
    ``evaluate`` and ``recover`` methods are NumPy reference implementations
    of the three stages.
    """
    maps: dict[str, tuple[sp.csr_array, np.ndarray]]
    structures: dict[str, MatrixStructure]
    primal_recovery: dict[int, np.ndarray | Exp | sp.csr_array]
    dual_recovery: dict[int, np.ndarray | sp.csr_array]
    primal_vars: list[VarInfo]
    dual_vars: list[DualInfo]
    cone_dims: Any
    constraints: list
    parameters: list  # Compiled parameters.
    param_transforms: list[ParameterTransform]  # In execution order.
    param_id_to_col: dict[int, int]  # Excludes the constant sentinel.
    param_id_to_size: dict[int, int]
    total_param_size: int

    def prepare_param_vec(self, values: dict[int, Any]) -> np.ndarray:
        """``param_vec`` from original parameter values, keyed by parameter ID.

        Values follow :func:`transform_source_value`: sparse parameters
        contribute their stored entries.
        """
        values = evaluate_param_transforms(self.param_transforms, values)
        return pack_param_vec(values, self.param_id_to_col,
                              self.param_id_to_size, self.total_param_size)

    def evaluate(self, param_vec: np.ndarray) -> dict[str, np.ndarray | sp.sparray]:
        """Solver inputs at ``param_vec``; sparse inputs are assembled."""
        return SolverInputMaps(self.maps, self.structures).evaluate(param_vec)

    def recover(self, solver_x: np.ndarray | None = None,
                solver_y: np.ndarray | None = None) -> dict[int, np.ndarray]:
        """Requested values from solver outputs, keyed by variable ID.

        Dense outputs are reshaped to their logical shape in Fortran order;
        sparse and diagonal outputs are returned as their stored values.
        """
        out = {}
        for info in self.primal_vars:
            value = _apply_recovery(self.primal_recovery[info.var_id], solver_x)
            out[info.var_id] = (value if info.sparse_idx is not None
                                else np.reshape(value, info.shape, order='F'))
        for info in self.dual_vars:
            value = _apply_recovery(self.dual_recovery[info.dual_var_id], solver_y)
            out[info.dual_var_id] = np.reshape(value, info.shape, order='F')
        return out


def _apply_recovery(rec, vec) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    if isinstance(rec, Exp):
        return np.exp(vec[rec.indices])
    if sp.issparse(rec):
        return rec @ vec
    return vec[rec]
