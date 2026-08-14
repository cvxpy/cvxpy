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

The symbolic-parametric cone program of the DIFFENGINE backend: a
ParamConeProg subclass that owns a live diff-engine extractor and
re-evaluates the expression trees at the current parameter values on
every apply_parameters() call.
"""
from __future__ import annotations

import numpy as np

from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.utilities import ReducedMat


class DiffengineParamConeProg(ParamConeProg):
    """A ParamConeProg whose matrices are re-extracted by the C diff engine.

    Parameters stay symbolic in the compiled engine program; each
    ``apply_parameters()`` pushes the current values and re-evaluates
    ``(q, d, A, b, P)`` directly, instead of multiplying parameter tensors.
    The instance always reaches the solver already ``formatted=True`` (the
    cone-restructuring matrix ``R`` pre-applied by ``DiffengineConeFormat``),
    so ``ConicSolver.format_constraints`` never replaces it with a stock
    program that could not re-extract.
    """

    def __init__(self, extractor, x, variables, var_id_to_col, constraints,
                 parameters, param_id_to_col, q, d, A, b, P,
                 formatted: bool = False, restruct_mat=None,
                 lower_bounds=None, upper_bounds=None) -> None:
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.cone_stuffing import (
            encode_cone_tensors,
        )
        self.extractor = extractor
        self._restruct_mat = restruct_mat
        # The concrete matrices at the currently-pushed parameter values,
        # post-restructuring. with_restruct() and the tensor refresh both
        # read from here.
        self._raw = (q, d, A, b, P)
        q_t, A_t, P_t = encode_cone_tensors(q, d, A, b, P, x.size)
        super().__init__(q_t, x, A_t, variables, var_id_to_col, constraints,
                         parameters, param_id_to_col, P=P_t,
                         formatted=formatted,
                         lower_bounds=lower_bounds, upper_bounds=upper_bounds)

    def _param_vec(self, id_to_param_value=None):
        """Flatten and concatenate parameter values (the extractor's encoding).

        CallbackParam values re-run their fold closure on access, so composite
        parametric coefficients are refreshed here as well.
        """
        values = ((lambda p: id_to_param_value[p.id])
                  if id_to_param_value is not None else (lambda p: p.value))
        return np.concatenate([
            np.asarray(values(p), dtype=np.float64).flatten(order='F')
            for p in self.parameters])

    def _refresh_tensors(self) -> None:
        """Re-encode the stored raw matrices into the ParamConeProg tensor
        attributes, so direct readers of .q/.A/.P stay consistent."""
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.cone_stuffing import (
            encode_cone_tensors,
        )
        q, d, A, b, P = self._raw
        self.q, self.A, self.P = encode_cone_tensors(q, d, A, b, P, self.x.size)
        self.reduced_A = ReducedMat(self.A, self.x.size)
        self.reduced_P = ReducedMat(self.P, self.x.size, quad_form=True)

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
        self.extractor.update_parameters(theta)
        q, d, A, b, P = self.extractor.extract(quad_obj)
        if self._restruct_mat is not None:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()
        self._raw = (q, d, A, b, P)
        self._refresh_tensors()
        if quad_obj:
            return P, q, d, A, b
        return q, d, A, b

    def with_restruct(self, R):
        """Return the formatted sibling with the cone-restructuring matrix
        ``R`` pre-applied. Shares the extractor (and its engine program)."""
        q, d, A, b, P = self._raw
        if R is not None:
            A = R @ A
            b = np.asarray(R @ b).flatten()
        return DiffengineParamConeProg(
            self.extractor, self.x, self.variables, self.var_id_to_col,
            self.constraints, self.parameters, self.param_id_to_col,
            q, d, A, b, P, formatted=True, restruct_mat=R,
            lower_bounds=self.lower_bounds, upper_bounds=self.upper_bounds)

    def split_adjoint(self, del_vars=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")

    def apply_param_jac(self, delc, deld, delA, delb, active_params=None):
        raise NotImplementedError(
            "The DIFFENGINE backend does not support the parameter "
            "differentiation contract (problem.backward/derivative).")
