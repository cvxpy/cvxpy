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
    Constants key by object identity; see ``_constant_key`` for why.
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
    """Key a small Constant by value, larger Constants by object identity.

    Many atoms construct fresh ``Constant`` objects for default scalar
    parameters (e.g. ``huber`` defaults ``M=0.5`` by minting ``Constant(0.5)``
    inside its constructor). Two structurally identical user expressions like
    ``cp.huber(x) + cp.huber(x)`` therefore embed two distinct ``Constant``
    objects with equal data; keying purely by ``id()`` would mark those as
    distinct and skip the CSE merge. For small constants the value hash is
    cheap, so we use it; for large arrays we keep ``id()`` to avoid copying
    O(problem-data) bytes into cache keys (sharing the matrix reference --
    the common large-data pattern -- still deduplicates).
    """
    value = expr.value
    try:
        arr = np.asarray(value)
    except (TypeError, ValueError):
        return ("const", id(expr))
    if arr.size <= _CONSTANT_VALUE_HASH_MAX_SIZE:
        return ("const-val", arr.shape, str(arr.dtype), arr.tobytes())
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
