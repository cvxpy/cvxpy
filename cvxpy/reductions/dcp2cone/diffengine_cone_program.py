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

from cvxpy.expressions.variable import Variable
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeDims
from cvxpy.reductions.matrix_stuffing import (
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem
from cvxpy.reductions.utilities import group_constraints


def _build_jacobian_csc(c_problem, jac_structure, m, n_vars):
    """Evaluate the constraint Jacobian and assemble it as a CSC matrix."""
    if m == 0:
        return sp.csc_matrix((0, n_vars))
    rows, cols = jac_structure
    vals = c_problem.eval_jacobian_vals()
    return sp.coo_matrix((vals, (rows, cols)), shape=(m, n_vars)).tocsc()


def _build_hessian_csc(c_problem, duals):
    """Evaluate the full (symmetric) Lagrangian Hessian and assemble it as CSC.

    The engine already holds the full symmetric Hessian as CSR, so we fetch it
    directly -- no lower-triangular mirror / diagonal-doubling fixup needed.
    """
    data, indices, indptr, shape = c_problem.eval_hessian_csr(1.0, duals)
    return sp.csr_matrix((data, indices, indptr), shape=shape).tocsc()


def _extract_matrices(c_problem, jac_structure, n_vars, has_constraints, quad_obj):
    """Evaluate the (affine) objective and constraints at x=0 to recover the cone
    program matrices: gradient ``q``, constant ``d``, constraint Jacobian ``A``,
    offset ``b``, and (for a quadratic objective) the Hessian ``P``.

    Shared by ``from_problem`` (initial build) and ``apply_parameters`` (re-evaluation
    after parameter values change). The returned matrices are pre-restructuring.
    """
    x0 = np.zeros(n_vars, dtype=np.float64)
    d = float(c_problem.objective_forward(x0))
    q = c_problem.gradient().copy()
    if has_constraints:
        b_vec = c_problem.constraint_forward(x0)
        A = _build_jacobian_csc(c_problem, jac_structure, b_vec.shape[0], n_vars)
    else:
        b_vec = np.array([], dtype=np.float64)
        A = sp.csc_matrix((0, n_vars))
    b = np.atleast_1d(b_vec)
    P = None
    if quad_obj:
        duals = np.zeros(b.shape[0], dtype=np.float64)
        P = _build_hessian_csc(c_problem, duals)
    return q, d, A, b, P


class DiffengineConeProgram(ParamProb):
    """A cone program whose matrices are extracted via the C diff engine.

    Sibling of ParamConeProg under the ParamProb interface. Wraps a `C_problem`
    instance (the same class used by the NLP solver path) and re-evaluates the
    converted expression trees on each solve via apply_parameters().

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
        c_problem: C_problem,
        jac_structure,
        formatted: bool = False,
        lower_bounds=None,
        upper_bounds=None,
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

        self.c_problem = c_problem
        self._jac_structure = jac_structure
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
        if not self.parameters:
            if quad_obj and self.P is not None:
                return self.P, self.q, self.d, self.A, self.b
            return self.q, self.d, self.A, self.b

        if id_to_param_value is not None:
            parts = [np.asarray(id_to_param_value[p.id],
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        else:
            parts = [np.asarray(p.value,
                                dtype=np.float64).flatten(order='F')
                     for p in self.parameters]
        self.c_problem.update_params(np.concatenate(parts))

        q, d, A, b, P = _extract_matrices(
            self.c_problem, self._jac_structure, self.inverse_data.x_length,
            bool(self.constraints), quad_obj)

        if self._restruct_mat is not None:
            A = self._restruct_mat @ A
            b = np.asarray(self._restruct_mat @ b).flatten()

        self.A, self.b, self.q, self.d, self.P = A, b, q, d, P

        if quad_obj:
            return P, q, d, A, b
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
            c_problem=self.c_problem,
            jac_structure=self._jac_structure,
            formatted=True,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
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
        c_problem = C_problem.from_exprs(
            problem.objective.expr, expr_list, params, inverse_data, verbose=False)

        c_problem.init_jacobian_coo()
        if quad_obj:
            c_problem.init_hessian()
        jac_structure = c_problem.get_jacobian_sparsity_coo() if expr_list else None

        n_vars = inverse_data.x_length
        q, d, A, b, P = _extract_matrices(
            c_problem, jac_structure, n_vars, bool(expr_list), quad_obj)

        boolean, integer = extract_mip_idx(problem.variables())
        x = Variable(n_vars, boolean=boolean, integer=integer)
        lower_bounds = extract_lower_bounds(problem.variables(), n_vars)
        upper_bounds = extract_upper_bounds(problem.variables(), n_vars)

        return cls(x, A, b, q, d, P,
                   ordered_cons, inverse_data,
                   c_problem=c_problem,
                   jac_structure=jac_structure,
                   lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                   parameters=params, quad_obj=quad_obj)
