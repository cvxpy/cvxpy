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

Structural-key helpers for the per-apply subexpression cache used by
canonicalization reductions.

When the same Expression subtree appears in two places in a problem (e.g.
``cp.norm1(x)`` in both the objective and a constraint), a recursive
canonicalizer that runs blindly will emit a fresh set of auxiliary
variables and epigraph constraints for each occurrence. The reduction
caches the canonicalization result for each subtree it sees, keyed by
structure, so the second occurrence reuses the first one's canonical
expression and auxiliary constraints.

These helpers build an opaque hashable key that is equal exactly when two
subtrees are structurally identical for canonicalization purposes: same atom
types, same shapes, same ``get_data()`` payloads, same Variable/Parameter ids
at the leaves. The caller is responsible for adding any reduction-specific
bits (e.g. ``affine_above`` for Dcp2Cone's quad branch) on top of
``expr_key``.
"""

from collections.abc import Hashable
from typing import TypeAlias

import numpy as np
import scipy.sparse as sp

from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable

# A structural signature: a heterogeneous, hashable tuple describing one
# Expression node locally, with its children already reduced to interned int
# keys. Two nodes are structurally identical for caching purposes exactly when
# their signatures compare equal. Signatures are used as dict keys, so every
# component must be hashable.
Signature: TypeAlias = tuple[Hashable, ...]

# The opaque key returned for a subtree: a compact interned integer (see
# StructuralKeyCache.intern_signature). Equal keys mean structurally identical
# subtrees.
ExprKey: TypeAlias = int


class UncacheableError(Exception):
    """Raised when a structural cache key cannot be safely built.

    Callers should treat this as "skip the cache for this subtree" rather than
    risking incorrect reuse.
    """


class StructuralKeyCache:
    """Per-apply state for building compact structural expression keys."""

    def __init__(self) -> None:
        self.expression_keys: dict[int, ExprKey] = {}
        self.signature_keys: dict[Signature, ExprKey] = {}

    def intern_signature(self, signature: Signature) -> ExprKey:
        """Return a compact key for a local structural signature."""
        if signature not in self.signature_keys:
            self.signature_keys[signature] = len(self.signature_keys)
        return self.signature_keys[signature]


def expr_key(expr: object, key_cache: StructuralKeyCache) -> ExprKey:
    """Build a hashable structural key for an Expression subtree.

    Variables/Parameters key by their ``.id`` (same source leaf -> same key).
    Constants key by value for small arrays and by object identity for large
    ones; see ``_constant_key`` for why.
    Composite Expressions key by ``(type, shape, get_data(), child keys)``.
    Raises ``UncacheableError`` for anything we can't hash safely.

    ``key_cache`` is required and scoped to one reduction apply. It maps
    ``id(expr)`` to a previously built key, so a caller that computes keys for
    every node in a tree only walks each node once. It also interns local
    structural signatures, so the returned key is a compact integer instead of
    a recursively nested tuple.
    """
    expr_id = id(expr)
    if expr_id in key_cache.expression_keys:
        return key_cache.expression_keys[expr_id]

    if isinstance(expr, Variable):
        signature = ("var", expr.id)
    elif isinstance(expr, Parameter):
        signature = ("param", expr.id)
    elif isinstance(expr, Constant):
        signature = _constant_key(expr)
    elif isinstance(expr, Expression):
        # A composite atom: key on its type, shape, get_data() payload, and the
        # already-interned child keys. An unhashable get_data() entry
        # propagates UncacheableError out of _hashable_value.
        child_keys = tuple(expr_key(arg, key_cache) for arg in expr.args)

        data = expr.get_data()
        if data is None:
            data_key: Signature = ()
        else:
            data_key = tuple(_hashable_value(d, key_cache) for d in data)

        signature = ("atom", type(expr), tuple(expr.shape), data_key, child_keys)
    else:
        # Not an Expression at all (e.g. an objective reached via a
        # partial_problem's args): refuse to cache rather than risk reuse.
        raise UncacheableError()

    key = key_cache.intern_signature(signature)
    key_cache.expression_keys[expr_id] = key
    return key


# Constants below this many elements are hashed by value so that
# default-parameter scalars (e.g. the implicit ``M=0.5`` in ``cp.huber(x)``)
# from two separate ``cp.huber(x)`` calls still produce equal cache keys.
# Above the threshold we fall back to id() to avoid copying large matrices
# into cache keys.
_CONSTANT_VALUE_HASH_MAX_SIZE = 64


def _constant_key(expr: Constant) -> Signature:
    """Key a Constant in one of three ways, in order of preference:

    1. Small array (<= 64 elements): value hash. Catches the case where two
       structurally identical user expressions embed distinct ``Constant``
       objects with equal data -- e.g. each ``cp.huber(x)`` call mints a fresh
       ``Constant(0.5)`` for the default ``M`` argument, which would otherwise
       defeat the CSE merge.

    2. Large float64 ndarray: id of the underlying ndarray. ``Constant``'s
       ndarray-interface stores a reference to a float64 ndarray without
       copying (see ``ndarray_interface.const_to_matrix``), so two
       ``Constant(arr)`` calls on the same source array share ``_value``.
       Keying on ``id(expr.value)`` catches that without copying problem-data
       bytes into the cache key. Other dtypes (int, bool, ...) get copied via
       ``astype(float64)`` so this branch only fires when it's safe.

    3. Sparse values: value hash for small sparse arrays, using sparse storage
       directly rather than ``np.asarray``. Large sparse values fall back to the
       Constant wrapper id.

    4. Fallback: id of the Constant wrapper. The common large-data pattern of
       binding ``c = cp.Constant(arr)`` and reusing ``c`` deduplicates here.
    """
    value = expr.value
    if sp.issparse(value):
        return _sparse_constant_key(expr, value)
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return ("const", id(expr))
    if arr.dtype != object and arr.size <= _CONSTANT_VALUE_HASH_MAX_SIZE:
        return ("const-val", arr.shape, str(arr.dtype), arr.tobytes())
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        return ("const-ref", value.shape, id(value))
    return ("const", id(expr))


def _sparse_constant_key(expr: Constant, value: sp.sparray) -> Signature:
    """Key sparse Constant values without converting them to object arrays."""
    if value.ndim == 2 and value.nnz <= _CONSTANT_VALUE_HASH_MAX_SIZE:
        sparse = value.tocsc(copy=False)
        if not sparse.has_canonical_format:
            sparse = sparse.copy()
            sparse.sum_duplicates()
            sparse.sort_indices()
        return (
            "sparse-const-val",
            sparse.shape,
            str(sparse.dtype),
            sparse.data.tobytes(),
            sparse.indices.tobytes(),
            sparse.indptr.tobytes(),
        )
    if value.ndim > 2 and value.nnz <= _CONSTANT_VALUE_HASH_MAX_SIZE:
        sparse = value.tocoo(copy=False)
        sparse.sum_duplicates()
        coords = sparse.coords
        return (
            "sparse-const-val",
            sparse.shape,
            str(sparse.dtype),
            tuple(coord.tobytes() for coord in coords),
            sparse.data.tobytes(),
        )
    return ("sparse-const", id(expr))


def _hashable_value(v: object, key_cache: StructuralKeyCache) -> Hashable:
    """Best-effort conversion of a ``get_data()`` entry to a hashable form."""
    if v is None or isinstance(v, (int, float, bool, str, bytes)):
        return v
    if isinstance(v, tuple):
        return tuple(_hashable_value(e, key_cache) for e in v)
    if isinstance(v, list):
        return ("list", tuple(_hashable_value(e, key_cache) for e in v))
    if isinstance(v, slice):
        return ("slice", v.start, v.stop, v.step)
    if isinstance(v, range):
        return ("range", v.start, v.stop, v.step)
    if isinstance(v, np.ndarray):
        return ("ndarray", v.shape, str(v.dtype), v.tobytes())
    if isinstance(v, Expression):
        return ("expr", expr_key(v, key_cache))
    # Unknown / unsafe data — refuse to cache this subtree.
    raise UncacheableError()
