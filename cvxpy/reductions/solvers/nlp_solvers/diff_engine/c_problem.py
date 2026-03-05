""" Copyright 2025, the CVXPY developers

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
import numpy as np
from sparsediffpy import _sparsediffengine as _diffengine

import cvxpy as cp
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    build_variable_dict,
    convert_expr,
)


class C_problem:
    """Wrapper around C problem struct for CVXPY problems."""

    def __init__(self, cvxpy_problem: cp.Problem, verbose: bool = True):
        var_dict, n_vars = build_variable_dict(cvxpy_problem.variables())
        c_obj = convert_expr(cvxpy_problem.objective.expr, var_dict, n_vars)
        c_constraints = [convert_expr(c.expr, var_dict, n_vars) for c in cvxpy_problem.constraints]
        self._capsule = _diffengine.make_problem(c_obj, c_constraints, verbose)

    def init_jacobian_coo(self):
        """Fill sparsity for the constraint Jacobian in COO format.

        Must be called once before get_jacobian_sparsity_coo() or eval_jacobian_vals().
        """
        _diffengine.problem_init_jacobian_coo(self._capsule)

    def init_hessian_coo_lower_tri(self):
        """Fill sparsity for the Lagrangian Hessian (lower triangle, COO).

        Must be called once before get_problem_hessian_sparsity_coo() or
        eval_hessian_vals_coo_lower_tri().
        """
        _diffengine.problem_init_hessian_coo_lower_triangular(self._capsule)

    def objective_forward(self, u: np.ndarray) -> float:
        """Evaluate objective. Returns obj_value float."""
        return _diffengine.problem_objective_forward(self._capsule, u)

    def constraint_forward(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraints only. Returns constraint_values array."""
        return _diffengine.problem_constraint_forward(self._capsule, u)

    def gradient(self) -> np.ndarray:
        """Compute gradient of objective. Call objective_forward first. Returns gradient array."""
        return _diffengine.problem_gradient(self._capsule)
    
    def get_jacobian_sparsity_coo(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the sparsity pattern (row, col) of the constraint Jacobian.

        Does not evaluate the Jacobian; only returns structural nonzero indices.
        Call init_jacobian_coo() first.
        """
        rows, cols, unused_shape = _diffengine.get_jacobian_sparsity_coo(self._capsule)
        return rows, cols

    def eval_jacobian_vals(self) -> np.ndarray:
        """Evaluate the constraint Jacobian and return its nonzero values.

        The values correspond to the sparsity pattern from get_jacobian_sparsity_coo().
        Call constraint_forward() first to set the evaluation point.
        """
        return _diffengine.problem_eval_jacobian_vals(self._capsule)
    
    def get_problem_hessian_sparsity_coo(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the sparsity pattern (row, col) of the lower-triangular Lagrangian Hessian.

        Does not evaluate the Hessian; only returns structural nonzero indices.
        Call init_hessian_coo_lower_tri() first.
        """
        rows, cols, unused_shape = _diffengine.get_problem_hessian_sparsity_coo(self._capsule)
        return rows, cols

    def eval_hessian_vals_coo_lower_tri(
        self, obj_factor: float, lagrange: np.ndarray
    ) -> np.ndarray:
        """Evaluate the lower-triangular Lagrangian Hessian and return its nonzero values.

        Computes obj_factor * hess_f + sum(lagrange[i] * hess_gi), where f is the objective
        and gi are the constraints. The values correspond to the sparsity pattern from
        get_problem_hessian_sparsity_coo(). Only the lower triangle is returned.

        Call objective_forward() and constraint_forward() first to set the evaluation point.
        """
        return _diffengine.problem_eval_hessian_vals_coo(self._capsule, obj_factor, lagrange)
    
