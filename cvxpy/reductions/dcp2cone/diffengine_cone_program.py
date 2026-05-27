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
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sparsediffpy import _sparsediffengine as _diffengine

from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims, ParamConeProg
from cvxpy.reductions.matrix_stuffing import (
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import convert_expr
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import (
    build_var_dict,
    normalize_shape,
    to_dense_float,
)
from cvxpy.reductions.utilities import group_constraints


def build_capsule(objective_expr, constraint_exprs, inverse_data, params=None, verbose=False):
    """Build a C diff engine problem capsule from CVXPY expressions.

    Returns
    -------
    capsule : PyCapsule
        The C diff engine problem capsule.
    n_vars : int
        Total number of scalar decision variables.
    param_dict : dict
        Mapping {param_id: C parameter capsule}.
    """
    var_dict, n_vars = build_var_dict(inverse_data)

    param_dict = {}
    if params:
        for param_id, offset in inverse_data.param_id_map.items():
            if param_id not in inverse_data.param_shapes:
                continue
            d1, d2 = normalize_shape(inverse_data.param_shapes[param_id])
            param = next(p for p in params if p.id == param_id)
            p = to_dense_float(param.value)
            param_dict[param_id] = _diffengine.make_parameter(
                d1, d2, offset, n_vars, p.flatten(order='F'))

    c_obj = convert_expr(objective_expr, var_dict, n_vars, param_dict)
    c_constraints = [convert_expr(e, var_dict, n_vars, param_dict) for e in constraint_exprs]

    capsule = _diffengine.make_problem(c_obj, c_constraints, verbose)

    if param_dict and params:
        _diffengine.problem_register_params(capsule, list(param_dict.values()))
        theta = np.concatenate([
            np.asarray(p.value, dtype=np.float64).flatten(order='F')
            for p in params
        ])
        _diffengine.problem_update_params(capsule, theta)

    return capsule, n_vars, param_dict


class DiffengineConeProgram(ParamConeProg):
    """A cone program with matrices extracted via the diffengine.

    Duck-type compatible with ParamConeProg. On first solve, stores the sparsity
    pattern of A and P. When parameters are present, re-evaluates the converted
    C expression trees on subsequent solves via apply_parameters().

    minimize   q'x + d + [(1/2)x'Px]
    subject to cone_constr(A*x + b) in cones
    """

    def __init__(
        self,
        x: Variable,
        A: sp.spmatrix,
        b: np.ndarray,
        q: np.ndarray,
        d: float,
        P,
        constraints: list,
        inverse_data,
        formatted: bool = False,
        lower_bounds=None,
        upper_bounds=None,
        capsule=None,
        parameters=None,
        quad_obj: bool = False,
    ) -> None:
        self.A = A
        self.b = b
        self.q = q
        self.d = d
        self.x = x
        self.P = P

        self.constraints = constraints
        self.constr_map = group_constraints(constraints)
        self.cone_dims = ConeDims(self.constr_map)

        self.inverse_data = inverse_data
        self.formatted = formatted
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

        self._capsule = capsule
        self.quad_obj = quad_obj
        self._restruct_mat = None
        self.parameters = list(parameters) if parameters else []
        self.param_id_to_size = {p.id: p.size for p in self.parameters}

        # Parametric bounds not yet supported on the diffengine path.
        self.lb_tensor = None
        self.ub_tensor = None

    @property
    def variables(self):
        return [self.inverse_data.id2var[vid]
                for vid in self.inverse_data.var_offsets]

    @property
    def var_id_to_col(self):
        return self.inverse_data.var_offsets

    @property
    def id_to_var(self):
        return self.inverse_data.id2var

    @property
    def param_id_to_col(self):
        return self.inverse_data.param_id_map

    def is_mixed_integer(self) -> bool:
        return self.x.attributes['boolean'] or self.x.attributes['integer']

    def apply_parameters(self, id_to_param_value=None, zero_offset: bool = False,
                         keep_zeros: bool = False, quad_obj: bool = False):
        """Return problem matrices, re-evaluating if parameters are present."""
        if not self.parameters or self._capsule is None:
            A = self.A
            if quad_obj and self.P is not None:
                return self.P, self.q, self.d, A, self.b
            return self.q, self.d, A, self.b

        if id_to_param_value is not None:
            parts = [np.asarray(id_to_param_value[p.id],
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        else:
            parts = [np.asarray(p.value,
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        theta = np.concatenate(parts)

        _diffengine.problem_update_params(self._capsule, theta)

        n_vars = self.inverse_data.x_length
        x0 = np.zeros(n_vars, dtype=np.float64)

        d = float(_diffengine.problem_objective_forward(self._capsule, x0))
        q = _diffengine.problem_gradient(self._capsule).copy()

        if self.constraints:
            b_vec = _diffengine.problem_constraint_forward(self._capsule, x0)
            jac_data, jac_indices, jac_indptr, jac_shape = \
                _diffengine.problem_jacobian(self._capsule)
            A = sp.csr_matrix(
                (jac_data, jac_indices, jac_indptr),
                shape=(jac_shape[0], n_vars)).tocsc()
        else:
            b_vec = np.array([], dtype=np.float64)
            A = sp.csc_matrix((0, n_vars))

        b = np.atleast_1d(b_vec)

        if self._restruct_mat is not None and self._restruct_mat is not False:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()

        self.A, self.b, self.q, self.d = A, b, q, d

        if quad_obj:
            duals = np.zeros(b.shape[0], dtype=np.float64)
            h_data, h_indices, h_indptr, h_shape = \
                _diffengine.problem_hessian(self._capsule, 1.0, duals)
            P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
            self.P = P_csr.tocsc()
            return self.P, q, d, A, b
        return q, d, A, b

    def apply_restruct_mat(self, restruct_mat, restruct_mat_op=None):
        """Materialize the cone-restructuring block-diagonal, cache it, apply to A, b.

        `restruct_mat` is the per-constraint list from `ConicSolver.format_constraints`;
        `restruct_mat_op` is the linear-operator form used by `ParamConeProg` and is
        ignored here. The materialized CSC matrix is cached so `apply_parameters` can
        re-apply it on subsequent solves.
        """
        R = None
        if restruct_mat:
            sparse_mats = []
            for mat in restruct_mat:
                if sp.issparse(mat):
                    sparse_mats.append(sp.csc_matrix(mat))
                elif callable(mat):
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat(eye)))
                else:
                    eye = sp.eye_array(mat.shape[1], format='csc')
                    sparse_mats.append(sp.csc_matrix(mat @ eye))

            R = sp.block_diag(sparse_mats, format='csc')
            new_A = R @ self.A
            new_b = np.asarray(R @ self.b).flatten()
        else:
            new_A, new_b = self.A, self.b

        new_prog = DiffengineConeProgram(
            self.x, new_A, new_b, self.q, self.d, self.P,
            self.constraints, self.inverse_data,
            formatted=True,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            capsule=self._capsule,
            parameters=self.parameters,
            quad_obj=self.quad_obj,
        )
        if R is not None:
            new_prog._restruct_mat = R
        return new_prog

    def split_solution(self, sltn, active_vars=None):
        from cvxpy.reductions import cvx_attr2constr
        if active_vars is None:
            active_vars = [v.id for v in self.variables]
        sltn_dict = {}
        for var_id, col in self.var_id_to_col.items():
            if var_id in active_vars:
                var = self.id_to_var[var_id]
                value = sltn[col:var.size + col]
                if var.attributes_were_lowered():
                    orig_var = var.leaf_of_provenance()
                    value = cvx_attr2constr.recover_value_for_leaf(
                        orig_var, value, project=False)
                    sltn_dict[orig_var.id] = np.reshape(
                        value, orig_var.shape, order='F')
                else:
                    sltn_dict[var_id] = np.reshape(
                        value, var.shape, order='F')
        return sltn_dict

    @classmethod
    def from_problem(cls, problem, ordered_cons, inverse_data, quad_obj):
        """Build a DiffengineConeProgram by evaluating expressions at x=0."""
        expr_list = [arg for c in ordered_cons for arg in c.args]
        params = problem.parameters()
        capsule, n_vars, _ = build_capsule(
            problem.objective.expr, expr_list, inverse_data, params=params, verbose=True)

        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(n_vars, boolean=boolean, integer=integer)

        x0 = np.zeros(n_vars, dtype=np.float64)

        if quad_obj:
            _diffengine.problem_init_derivatives(capsule)
        else:
            _diffengine.problem_init_jacobian(capsule)

        d = float(_diffengine.problem_objective_forward(capsule, x0))
        q = _diffengine.problem_gradient(capsule).copy()

        if expr_list:
            b_vec = _diffengine.problem_constraint_forward(capsule, x0)
            jac_data, jac_indices, jac_indptr, jac_shape = \
                _diffengine.problem_jacobian(capsule)
            A = sp.csr_matrix(
                (jac_data, jac_indices, jac_indptr),
                shape=(jac_shape[0], n_vars)).tocsc()
        else:
            b_vec = np.array([], dtype=np.float64)
            A = sp.csc_matrix((0, n_vars))

        P = None
        if quad_obj:
            duals = np.zeros(b_vec.shape[0], dtype=np.float64)
            h_data, h_indices, h_indptr, h_shape = \
                _diffengine.problem_hessian(capsule, 1.0, duals)
            P_csr = sp.csr_matrix((h_data, h_indices, h_indptr), shape=h_shape)
            P = P_csr.tocsc()

        lower_bounds = extract_lower_bounds(problem.variables(), n_vars)
        upper_bounds = extract_upper_bounds(problem.variables(), n_vars)

        return cls(x, A, np.atleast_1d(b_vec), q, d, P,
                   ordered_cons, inverse_data,
                   lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                   capsule=capsule, parameters=params, quad_obj=quad_obj)
