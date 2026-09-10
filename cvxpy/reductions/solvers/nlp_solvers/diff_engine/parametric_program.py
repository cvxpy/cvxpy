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

The cone program of the DIFFENGINE backend: a ParamConeProg subclass that
owns a live diff-engine extractor and re-evaluates the expression trees at
the current parameter values on every apply_parameters() call.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import scipy.sparse as sp

from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solvers.conic_solvers.conic_solver import build_restruct_mat_sparse
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.cone_stuffing import encode_cone_tensors
from cvxpy.reductions.utilities import ReducedMat


class _Matrices(NamedTuple):
    """The concrete cone matrices at one set of parameter values."""

    q: np.ndarray
    d: float
    A: sp.csc_array
    b: np.ndarray
    P: sp.csc_array | None


class _lazy_tensor:
    """One of ``ParamConeProg``'s tensors, encoded on first read.

    The tensors are an encoding of ``self._matrices`` that costs more to
    build than the extraction that produced them, and nothing on the solve
    path reads them: the solver interfaces consume ``apply_parameters``'
    return values and ask ``has_quad_obj`` rather than inspecting ``P``.
    Encoding them eagerly on every solve would erase most of what caching
    the compiled program saves.
    """

    def __set_name__(self, owner, name: str) -> None:
        self.name = name

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return obj._encode_tensors()[self.name]


class DiffengineParamConeProg(ParamConeProg):
    """A ParamConeProg whose matrices are re-extracted by the C diff engine.

    Parameters stay symbolic in the compiled engine program; each
    ``apply_parameters()`` pushes the current values and re-evaluates
    ``(q, d, A, b, P)`` directly, instead of multiplying parameter tensors.
    A parameter-free problem takes the same path with an empty parameter
    vector, so the first extraction is also the last.
    """

    def __init__(self, extractor, x, variables, var_id_to_col, constraints,
                 parameters, param_id_to_col, q, d, A, b, P,
                 formatted: bool = False, restruct_mat=None,
                 lower_bounds=None, upper_bounds=None) -> None:
        self.extractor = extractor
        # The restructuring matrix R that ``format_for`` applied, kept so it
        # can be re-applied to every later extraction.
        self._restruct_mat = restruct_mat
        # The concrete matrices at the currently-pushed parameter values,
        # post-restructuring. Everything else here is derived from them.
        self._matrices = _Matrices(q, d, A, b, P)
        self._tensor_cache: dict | None = None
        # No tensors are passed: _store_tensors defers them until read. The
        # base's lb_tensor/ub_tensor and dir_cones are likewise unused --
        # bounds arrive concrete, and ExtractDirectCones keeps problems with
        # direct cones off this backend.
        super().__init__(None, x, None, variables, var_id_to_col, constraints,
                         parameters, param_id_to_col, P=None,
                         formatted=formatted,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        # Memo key for ``self._matrices``: the parameter vector they were
        # extracted at, so an unchanged re-solve skips the extraction (and so
        # the extraction done to construct this instance is not repeated).
        # Per instance, not on the extractor: ``format_for`` gives a
        # restructured copy the same extractor.
        self._extracted_param_vec = self._param_vec()

    def _store_tensors(self, q, A, P) -> None:
        """No-op: the tensors are encoded on demand, see ``_lazy_tensor``."""

    q = _lazy_tensor()
    A = _lazy_tensor()
    P = _lazy_tensor()
    reduced_A = _lazy_tensor()
    reduced_P = _lazy_tensor()

    @property
    def has_quad_obj(self) -> bool:
        return self._matrices.P is not None

    def _encode_tensors(self) -> dict:
        """Encode the current matrices into ParamConeProg's tensor layout."""
        if self._tensor_cache is None:
            q_t, A_t, P_t = encode_cone_tensors(*self._matrices, self.x.size)
            self._tensor_cache = {
                'q': q_t, 'A': A_t, 'P': P_t,
                'reduced_A': ReducedMat(A_t, self.x.size),
                'reduced_P': ReducedMat(P_t, self.x.size, quad_form=True),
            }
        return self._tensor_cache

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
            # a stored linear one.
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
        q, d, A, b, P = self.extractor.extract(quad_obj)
        if self._restruct_mat is not None:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()
        self._matrices = _Matrices(q, d, A, b, P)
        self._extracted_param_vec = theta
        self._tensor_cache = None
        if quad_obj:
            return P, q, d, A, b
        return q, d, A, b

    def format_for(self, solver):
        """Apply the solver's cone row layout without losing re-extraction.

        The base implementation rebuilds a plain ParamConeProg, which would
        drop the engine program. R is structural (constraint types and shapes
        only), so it is built once here and re-applied to every later
        extraction.
        """
        R = build_restruct_mat_sparse(self.constraints, solver.EXP_CONE_ORDER)
        q, d, A, b, P = self._matrices
        if R is not None:
            A = R @ A
            b = np.asarray(R @ b).flatten()
        formatted = DiffengineParamConeProg(
            self.extractor, self.x, self.variables, self.var_id_to_col,
            self.constraints, self.parameters, self.param_id_to_col,
            q, d, A, b, P, formatted=True, restruct_mat=R,
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
