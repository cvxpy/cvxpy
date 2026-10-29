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

Cone matrix stuffing via the C diff engine. ``ConeMatrixStuffing.apply``
dispatches to ``stuff_cone_program`` for canon_backend="DIFFENGINE", which
produces a ``DiffengineConeProg``: a cone program that owns a live engine
program and re-evaluates the expression trees at the current parameter values,
rather than multiplying parameter tensors.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeProg, order_cone_constraints
from cvxpy.reductions.matrix_stuffing import (
    _has_parametric_bounds,
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.extractor import DiffEngineExtractor


class _Matrices(NamedTuple):
    """The concrete cone matrices at one set of parameter values."""

    q: np.ndarray
    d: float
    A: sp.csc_array
    b: np.ndarray
    P: sp.csc_array | None


class _RowLayout:
    """The solver's cone row layout, with a plan for applying it to A.

    The layout is a signed permutation, as returned by
    ``cone_format.restruct_permutation``: ``new_row`` and ``sign`` are indexed
    by the *old* row, so old row i becomes row ``new_row[i]``, scaled by
    ``sign[i]``.

    Permuting rows keeps every stored value in its column, so a CSC matrix
    keeps its ``indptr``; only the order of the values within each column
    changes. That order depends on the sparsity pattern alone, which the
    re-extractions of one program share, so it is computed once and each
    later extraction is a gather of ``A.data`` and a sign flip, with no
    format conversion.
    """

    def __init__(self, new_row: np.ndarray, sign: np.ndarray) -> None:
        self.new_row = new_row
        self.sign = sign
        # The (indptr, indices) the plan was computed for, and the plan:
        # (order of the values, their signs, their new row indices).
        self._pattern = None
        self._plan = None

    def _plan_for(self, A: sp.csc_array):
        pattern = self._pattern
        if (pattern is not None and np.array_equal(pattern[0], A.indptr)
                and np.array_equal(pattern[1], A.indices)):
            return self._plan
        # A different pattern (or the first extraction): plan again. The
        # stable sort keeps the columns in place and orders each by new row,
        # whether or not A's indices were sorted.
        new_idx = self.new_row[A.indices].astype(A.indices.dtype, copy=False)
        col = np.repeat(np.arange(A.shape[1]), np.diff(A.indptr))
        order = np.lexsort((new_idx, col))
        self._pattern = (A.indptr.copy(), A.indices.copy())
        self._plan = (order, self.sign[A.indices][order], new_idx[order])
        return self._plan

    def apply(self, matrices: _Matrices) -> _Matrices:
        A = sp.csc_array(matrices.A)
        order, signs, new_indices = self._plan_for(A)
        A = sp.csc_array((A.data[order] * signs, new_indices, A.indptr),
                         shape=A.shape)
        A.has_sorted_indices = True
        b = np.empty_like(matrices.b)
        b[self.new_row] = matrices.b * self.sign
        return matrices._replace(A=A, b=b)


def _restructure(matrices: _Matrices, perm: _RowLayout | None) -> _Matrices:
    """Apply the solver's cone row layout to one extraction."""
    if perm is None:
        return matrices
    return perm.apply(matrices)


class DiffengineConeProg(ConeProg):
    """A cone program whose matrices are re-extracted by the C diff engine.

    Parameters stay symbolic in the compiled engine program; each
    ``apply_parameters()`` pushes the current values and re-evaluates
    ``(q, d, A, b, P)`` directly, instead of multiplying parameter tensors.
    A parameter-free problem takes the same path with an empty parameter
    vector, so the first extraction is also the last.

    There are no parameter tensors here, and nothing that reaches this program
    asks for them: the solver interfaces consume ``apply_parameters``' return
    values and ask ``has_quad_obj``. ``ExtractDirectCones``, the one reduction
    that does read tensors, is kept away by ``_diffengine_eligible``'s
    ``dir_cone_kinds`` check.
    """

    def __init__(self, extractor, x, variables, var_id_to_col, constraints,
                 parameters, param_id_to_col, matrices: _Matrices,
                 formatted: bool = False, restruct_perm=None,
                 lower_bounds=None, upper_bounds=None) -> None:
        super().__init__(x, variables, var_id_to_col, constraints,
                         parameters, param_id_to_col, formatted=formatted,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        self.extractor = extractor
        # The row layout ``with_row_layout`` applied, kept so it can be
        # re-applied to every later extraction.
        self._restruct_perm = restruct_perm
        # The concrete matrices at the currently-pushed parameter values,
        # post-restructuring.
        self._matrices = matrices
        # Memo key for ``self._matrices``: the parameter vector they were
        # extracted at, so an unchanged re-solve skips the extraction (and so
        # the extraction done to construct this instance is not repeated).
        # Per instance, not on the extractor: ``with_row_layout`` gives a
        # restructured copy the same extractor.
        self._extracted_param_vec = self._param_vec()

    @property
    def has_quad_obj(self) -> bool:
        return self._matrices.P is not None

    def _param_vec(self, id_to_param_value=None) -> np.ndarray:
        """Flatten and concatenate the parameter values, in the extractor's order.

        Reading ``p.value`` re-runs a CallbackParam's fold closure, so
        composite parametric coefficients are refreshed here as well.
        """
        if not self.parameters:
            return np.zeros(0)
        return np.concatenate([
            np.asarray(p.value if id_to_param_value is None
                       else id_to_param_value[p.id],
                       dtype=np.float64).flatten(order='F')
            for p in self.parameters])

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Re-evaluate the engine program at the current parameter values."""
        if zero_offset or keep_zeros:
            # These flags belong to the DPP-tensor differentiation contract
            # (diffcp / problem.derivative), which the diff engine does not
            # implement: it re-evaluates a nonlinear map rather than applying
            # a stored linear one. Solver interfaces that need them declare
            # REQUIRES_PARAM_TENSORS, which keeps them off this backend; this
            # is the backstop for problem.derivative, which reaches the
            # program directly.
            raise NotImplementedError(
                "The DIFFENGINE backend does not support the parameter "
                "differentiation contract (zero_offset/keep_zeros); solve "
                "without requires_grad, or use a tensor canon backend.")
        theta = self._param_vec(id_to_param_value)
        stored = self._matrices
        if (np.array_equal(theta, self._extracted_param_vec)
                and (stored.P is not None or not quad_obj)):
            # The stored matrices already correspond to these values.
            if quad_obj:
                return stored.P, stored.q, stored.d, stored.A, stored.b
            return stored.q, stored.d, stored.A, stored.b
        self.extractor.update_parameters(theta)
        matrices = _restructure(
            _Matrices(*self.extractor.extract(quad_obj)), self._restruct_perm)
        self._matrices = matrices
        self._extracted_param_vec = theta
        if quad_obj:
            return (matrices.P, matrices.q, matrices.d, matrices.A, matrices.b)
        return matrices.q, matrices.d, matrices.A, matrices.b

    def with_row_layout(self, perm):
        """Apply the cone row layout without losing re-extraction.

        ``ParamConeProg`` rebuilds itself out of the permuted parameter
        tensors, which this program does not have and which would drop the
        engine program besides. The layout is structural, so it is kept here
        and re-applied to every later extraction instead.
        """
        # ConeFormat only formats an unformatted program, so there is never an
        # earlier layout to compose with.
        assert self._restruct_perm is None, "already formatted"
        if perm is not None:
            perm = _RowLayout(*perm)
        formatted = DiffengineConeProg(
            self.extractor, self.x, self.variables, self.var_id_to_col,
            self.constraints, self.parameters, self.param_id_to_col,
            _restructure(self._matrices, perm), formatted=True,
            restruct_perm=perm,
            lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)
        # The restructured matrices correspond to THIS instance's values.
        formatted._extracted_param_vec = self._extracted_param_vec
        return formatted

    def split_adjoint(self, del_vars=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")

    def apply_param_jac(self, delc, delA, delb, active_params=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")


def stuff_cone_program(problem, cons, inverse_data, quad_obj):
    """Stuff a problem by evaluating the expression trees with the C diff
    engine instead of building parameter tensors.

    Produces a DiffengineConeProg, which keeps the engine program alive and
    re-extracts the cone matrices on every apply_parameters(). Parameters stay
    symbolic; a parameter-free problem is the degenerate case, where the
    parameter vector is empty and the extraction done here is the only one.

    ``cons`` are the lowered (but not yet ordered) constraints from
    ``ConeMatrixStuffing.apply``. Returns ``(new_prob, inverse_data)``.
    """
    variables = problem.variables()
    if _has_parametric_bounds(variables):
        raise NotImplementedError(
            f"The {s.DIFFENGINE_CANON_BACKEND} canonicalization backend "
            "does not support parametric variable bounds.")

    ordered_cons = order_cone_constraints(cons)
    inverse_data.cons_id_map = {con.id: con.id for con in ordered_cons}
    inverse_data.constraints = ordered_cons
    inverse_data.minimize = type(problem.objective) == Minimize

    # One-shot extraction of the concrete cone matrices at x = 0.
    expr_list = [arg for c in ordered_cons for arg in c.args]
    params = problem.parameters()
    extractor = DiffEngineExtractor(inverse_data).build(
        problem.objective.expr, expr_list, params, quad_obj)
    matrices = _Matrices(*extractor.extract(quad_obj))

    n = inverse_data.x_length
    boolean, integer = extract_mip_idx(variables)
    x = Variable(n, boolean=boolean, integer=integer)
    lower_bounds = extract_lower_bounds(variables, n)
    upper_bounds = extract_upper_bounds(variables, n)

    new_prob = DiffengineConeProg(
        extractor, x, variables, inverse_data.var_offsets, ordered_cons,
        params, inverse_data.param_id_map, matrices,
        lower_bounds=lower_bounds, upper_bounds=upper_bounds)
    return new_prob, inverse_data
