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
canonicalization reductions (Dcp2Cone, Dnlp2Smooth, ...).

When the same Expression subtree appears in two places in a problem (e.g.
``cp.norm1(x)`` in both the objective and a constraint), a recursive
canonicalizer that runs blindly will emit a fresh set of auxiliary
variables and epigraph constraints for each occurrence. The reduction
caches the canonicalization result for each subtree it sees, keyed by
structure, so the second occurrence reuses the first one's canonical
expression and auxiliary constraints.

These helpers build a hashable key that is equal exactly when two subtrees
are structurally identical for canonicalization purposes: same atom types,
same shapes, same ``get_data()`` payloads, same Variable/Parameter ids at
the leaves. The caller is responsible for adding any reduction-specific
bits (e.g. ``affine_above`` for Dcp2Cone's quad branch) on top of
``expr_key``.
"""

import numpy as np

from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.constants.parameter import Parameter
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.variable import Variable


class UncacheableError(Exception):
    """Raised when a structural cache key cannot be safely built.

    Callers should treat this as "skip the cache for this subtree" rather than
    risking incorrect reuse.
    """


def expr_key(expr):
    """Build a hashable structural key for an Expression subtree.

    Variables/Parameters key by their ``.id`` (same source leaf -> same key).
    Constants key by value for small arrays and by object identity for large
    ones; see ``_constant_key`` for why.
    Composite Expressions key by ``(type, shape, get_data(), child keys)``.
    Raises ``UncacheableError`` for anything we can't hash safely.
    """
    if isinstance(expr, Variable):
        return ("var", expr.id)
    if isinstance(expr, Parameter):
        return ("param", expr.id)
    if isinstance(expr, Constant):
        return _constant_key(expr)
    if not isinstance(expr, Expression):
        raise UncacheableError()

    child_keys = tuple(expr_key(arg) for arg in expr.args)

    data = expr.get_data()
    if data is None:
        data_key: tuple = ()
    else:
        data_key = tuple(_hashable_value(d) for d in data)

    return ("atom", type(expr), tuple(expr.shape), data_key, child_keys)


# Constants below this many elements are hashed by value so that
# default-parameter scalars (e.g. the implicit ``M=0.5`` in ``cp.huber(x)``)
# from two separate ``cp.huber(x)`` calls still produce equal cache keys.
# Above the threshold we fall back to id() to avoid copying large matrices
# into cache keys.
_CONSTANT_VALUE_HASH_MAX_SIZE = 64


def _constant_key(expr: Constant):
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

    3. Fallback: id of the Constant wrapper. The common large-data pattern of
       binding ``c = cp.Constant(arr)`` and reusing ``c`` deduplicates here.
    """
    value = expr.value
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return ("const", id(expr))
    if arr.size <= _CONSTANT_VALUE_HASH_MAX_SIZE:
        return ("const-val", arr.shape, str(arr.dtype), arr.tobytes())
    if isinstance(value, np.ndarray) and value.dtype == np.float64:
        return ("const-ref", value.shape, id(value))
    return ("const", id(expr))


def _hashable_value(v):
    """Best-effort conversion of a ``get_data()`` entry to a hashable form."""
    if v is None or isinstance(v, (int, float, bool, str, bytes)):
        return v
    if isinstance(v, tuple):
        return tuple(_hashable_value(e) for e in v)
    if isinstance(v, list):
        return ("list", tuple(_hashable_value(e) for e in v))
    if isinstance(v, slice):
        return ("slice", v.start, v.stop, v.step)
    if isinstance(v, range):
        return ("range", v.start, v.stop, v.step)
    if isinstance(v, np.ndarray):
        return ("ndarray", v.shape, str(v.dtype), v.tobytes())
    if isinstance(v, Expression):
        return ("expr", expr_key(v))
    # Unknown / unsafe data — refuse to cache this subtree.
    raise UncacheableError()
