"""Wrapper around C problem struct for CVXPY problems.

Copyright 2025, the CVXPY developers

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
from scipy import sparse

import cvxpy as cp

# Import the low-level C bindings
try:
    import _diffengine
except ImportError as e:
    raise ImportError(
        "NLP support requires diff-engine. Rebuild with: pip install -e ."
    ) from e

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
        self._jacobian_allocated = False
        self._hessian_allocated = False

    def init_jacobian(self):
        """Initialize Jacobian structures only. Must be called before jacobian()."""
        _diffengine.problem_init_jacobian(self._capsule)
        self._jacobian_allocated = True

    def init_hessian(self):
        """Initialize Hessian structures only. Must be called before hessian()."""
        _diffengine.problem_init_hessian(self._capsule)
        self._hessian_allocated = True

    def objective_forward(self, u: np.ndarray) -> float:
        """Evaluate objective. Returns obj_value float."""
        return _diffengine.problem_objective_forward(self._capsule, u)

    def constraint_forward(self, u: np.ndarray) -> np.ndarray:
        """Evaluate constraints only. Returns constraint_values array."""
        return _diffengine.problem_constraint_forward(self._capsule, u)

    def gradient(self) -> np.ndarray:
        """Compute gradient of objective. Call objective_forward first. Returns gradient array."""
        return _diffengine.problem_gradient(self._capsule)

    def jacobian(self) -> sparse.csr_matrix:
        """Compute constraint Jacobian. Call constraint_forward first."""
        data, indices, indptr, shape = _diffengine.problem_jacobian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    
    def get_jacobian(self) -> sparse.csr_matrix:
        """Get constraint Jacobian. This function does not evaluate the jacobian. """
        data, indices, indptr, shape = _diffengine.get_jacobian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)

    def hessian(self, obj_factor: float, lagrange: np.ndarray) -> sparse.csr_matrix:
        """Compute Lagrangian Hessian.

        Computes: obj_factor * H_obj + sum(lagrange_i * H_constraint_i)

        Call objective_forward and constraint_forward before this.

        Args:
            obj_factor: Weight for objective Hessian
            lagrange: Array of Lagrange multipliers (length = total_constraint_size)

        Returns:
            scipy CSR matrix of shape (n_vars, n_vars)
        """
        data, indices, indptr, shape = _diffengine.problem_hessian(
            self._capsule, obj_factor, lagrange
        )
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
    
    def get_hessian(self) -> sparse.csr_matrix:
        """Get Lagrangian Hessian. This function does not evaluate the hessian."""
        data, indices, indptr, shape = _diffengine.get_hessian(self._capsule)
        return sparse.csr_matrix((data, indices, indptr), shape=shape)
