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

from cvxpy.reductions.solvers.nlp_solvers.diff_engine.c_problem import C_problem


def _build_jacobian_csc(c_problem, jac_structure, m, n_vars):
    """Evaluate the constraint Jacobian and assemble it as a CSC matrix."""
    if m == 0:
        return sp.csc_matrix((0, n_vars))
    rows, cols = jac_structure
    vals = c_problem.eval_jacobian_vals()
    return sp.coo_matrix((vals, (rows, cols)), shape=(m, n_vars)).tocsc()


def _full_symmetric_hess_structure(rows, cols, n_vars):
    """Precompute the *full* symmetric Hessian sparsity from the engine's lower-triangular
    COO sparsity (rows >= cols). Off-diagonal entries are mirrored across the diagonal.

    Returns ``(full_rows, full_cols, offdiag_mask, shape)`` so that, given fresh
    lower-triangular values ``v``, the full symmetric values are
    ``concatenate([v, v[offdiag_mask]])`` -- see ``_build_hessian_csc``.
    """
    offdiag = rows != cols
    full_rows = np.concatenate([rows, cols[offdiag]])
    full_cols = np.concatenate([cols, rows[offdiag]])
    return full_rows, full_cols, offdiag, (n_vars, n_vars)


def _build_hessian_csc(c_problem, hess_structure, duals):
    """Assemble the full symmetric Lagrangian Hessian P as CSC.

    Re-evaluates only the lower-triangular Hessian *values* (the structure is fixed
    across parameter updates and was captured once in ``hess_structure``), then mirrors
    the off-diagonal entries to form the full symmetric matrix the cone solvers expect.
    """
    full_rows, full_cols, offdiag, shape = hess_structure
    vals = c_problem.eval_hessian_vals_coo_lower_tri(1.0, duals)
    full_vals = np.concatenate([vals, vals[offdiag]])
    return sp.csc_matrix((full_vals, (full_rows, full_cols)), shape=shape)


class DiffEngineExtractor:
    """Non-DPP analog of ``CoeffExtractor``: builds a C autodiff problem from
    the expressions and recovers the concrete cone matrices ``(q, d, A, b, P)``
    by evaluating it at ``x = 0``, re-evaluating when parameter values change.
    Owns the ``C_problem``; ``DiffengineConeProgram`` delegates extraction to it.
    """

    def __init__(self, inverse_data) -> None:
        self.inverse_data = inverse_data
        self.n_vars = inverse_data.x_length
        self.c_problem = None
        self.jac_structure = None
        self.hess_structure = None
        self.has_constraints = False

    def build(self, objective_expr, constraint_exprs, parameters, quad_obj: bool):
        """Compile the objective and constraint expressions into a ``C_problem``.

        ``constraint_exprs`` is the flat list of cone-constraint argument expressions.
        Returns ``self`` for chaining.
        """
        self.has_constraints = bool(constraint_exprs)
        self.c_problem = C_problem.from_exprs(
            objective_expr, constraint_exprs, parameters, self.inverse_data,
            verbose=False)
        self.c_problem.init_jacobian_coo()
        self.jac_structure = (self.c_problem.get_jacobian_sparsity_coo()
                              if self.has_constraints else None)
        if quad_obj:
            # Capture the (fixed) lower-triangular Hessian sparsity once; re-solves then
            # fetch values only and mirror to full symmetric (see _build_hessian_csc).
            self.c_problem.init_hessian_coo_lower_tri()
            rows, cols = self.c_problem.get_problem_hessian_sparsity_coo()
            self.hess_structure = _full_symmetric_hess_structure(rows, cols, self.n_vars)
        return self

    def update_parameters(self, theta: np.ndarray) -> None:
        """Push new (flattened, concatenated) parameter values into the C problem."""
        self.c_problem.update_params(theta)

    def extract(self, quad_obj: bool):
        """Evaluate the (affine) objective and constraints at ``x = 0`` to recover the
        cone matrices: gradient ``q``, constant ``d``, constraint Jacobian ``A``, offset
        ``b``, and (for a quadratic objective) the Hessian ``P``. Pre-restructuring.
        """
        x0 = np.zeros(self.n_vars, dtype=np.float64)
        d = float(self.c_problem.objective_forward(x0))
        q = self.c_problem.gradient().copy()
        if self.has_constraints:
            b_vec = self.c_problem.constraint_forward(x0)
            A = _build_jacobian_csc(
                self.c_problem, self.jac_structure, b_vec.shape[0], self.n_vars)
        else:
            b_vec = np.array([], dtype=np.float64)
            A = sp.csc_matrix((0, self.n_vars))
        b = np.atleast_1d(b_vec)
        P = None
        if quad_obj:
            duals = np.zeros(b.shape[0], dtype=np.float64)
            P = _build_hessian_csc(self.c_problem, self.hess_structure, duals)
        return q, d, A, b, P
