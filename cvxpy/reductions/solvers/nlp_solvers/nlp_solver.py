"""
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

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)

if TYPE_CHECKING:
    from cvxpy.problems.problem import Problem


class NLPsolver(Solver):
    """
    A non-linear programming (NLP) solver.
    """
    REQUIRES_CONSTR = False
    MIP_CAPABLE = False
    BOUNDED_VARIABLES = True

    def accepts(self, problem: Problem) -> bool:
        """
        Only accepts disciplined nonlinear programs.
        """
        return problem.is_dnlp()

    def apply(self, problem: Problem) -> tuple[dict, InverseData]:
        """
        Construct NLP problem data stored in a dictionary.
        The NLP has the following form

            minimize      f(x)
            subject to    g^l <= g(x) <= g^u
                          x^l <= x <= x^u
        where f and g are non-linear (and possibly non-convex) functions
        """
        problem, data, inv_data = self._prepare_data_and_inv_data(problem)

        return data, inv_data

    def _prepare_data_and_inv_data(
        self, problem: Problem
    ) -> tuple[Problem, dict, InverseData]:
        data = dict()
        bounds = Bounds(problem)
        inverse_data = InverseData(bounds.new_problem)
        inverse_data.offset = 0.0
        data["problem"] = bounds.new_problem
        data["cl"], data["cu"] = bounds.cl, bounds.cu
        data["lb"], data["ub"] = bounds.lb, bounds.ub
        data["x0"] = bounds.x0
        data["_bounds"] = bounds  # Store for deferred Oracles creation in solve_via_data
        return problem, data, inverse_data

class Bounds:
    """Extracts variable and constraint bounds from a CVXPY problem.

    Converts the problem into the standard NLP form::

        g^l <= g(x) <= g^u,   x^l <= x <= x^u

    Inequalities are lowered to nonneg form and equalities to zero
    constraints.  The resulting ``new_problem`` attribute holds the
    canonicalized problem used by the solver oracles.
    """

    def __init__(self, problem: Problem) -> None:
        self.problem = problem
        self.main_var = problem.variables()
        self.get_constraint_bounds()
        self.get_variable_bounds()
        self.construct_initial_point()

    def get_constraint_bounds(self) -> None:
        """
        Get constraint bounds for all constraints.
        Also converts inequalities to nonneg form,
        as well as equalities to zero constraints and forms
        a new problem from the canonicalized problem.
        """
        lower, upper = [], []
        new_constr = []
        for constraint in self.problem.constraints:
            if isinstance(constraint, Equality):
                lower.extend([0.0] * constraint.size)
                upper.extend([0.0] * constraint.size)
                new_constr.append(lower_equality(constraint))
            elif isinstance(constraint, Inequality):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(lower_ineq_to_nonneg(constraint))
            elif isinstance(constraint, NonPos):
                lower.extend([0.0] * constraint.size)
                upper.extend([np.inf] * constraint.size)
                new_constr.append(nonpos2nonneg(constraint))
        canonicalized_prob = self.problem.copy([self.problem.objective, new_constr])
        self.new_problem = canonicalized_prob
        self.cl = np.array(lower)
        self.cu = np.array(upper)

    def get_variable_bounds(self) -> None:
        """
        Get variable bounds for all variables.
        Uses the variable's get_bounds() method which handles bounds attributes,
        nonneg/nonpos attributes, and properly broadcasts scalar bounds.
        """
        var_lower, var_upper = [], []
        for var in self.main_var:
            # get_bounds() returns arrays broadcastable to var.shape
            # and handles all edge cases (scalar bounds, sign attributes, etc.)
            lb, ub = var.get_bounds()
            # Flatten in column-major (Fortran) order and convert to contiguous array
            # (broadcast_to creates read-only views that need to be copied)
            lb_flat = np.asarray(lb).flatten(order='F')
            ub_flat = np.asarray(ub).flatten(order='F')
            var_lower.extend(lb_flat)
            var_upper.extend(ub_flat)
        self.lb = np.array(var_lower)
        self.ub = np.array(var_upper)

    def construct_initial_point(self) -> None:
        """ Loop through all variables and collect the intial point."""
        x0 = []
        for var in self.main_var:
            if var.value is None:
                raise ValueError("Variable %s has no value. This is a bug and should be reported."
                                  % var.name())

            x0.append(np.atleast_1d(var.value).flatten(order='F'))
        self.x0 = np.concatenate(x0, axis=0)

class Oracles:
    """Oracle interface for NLP solvers using the C-based diff engine.

    Provides function and derivative oracles (objective, gradient, constraints,
    Jacobian, Hessian) by wrapping the ``C_problem`` class from the diff engine.

    Forward passes are cached per solver iteration: calling ``objective`` or
    ``constraints`` sets a flag so that ``gradient``/``jacobian``/``hessian``
    can reuse the cached forward values.  The ``intermediate`` callback resets
    these flags at the start of each new solver iteration.

    Sparsity structures (Jacobian and Hessian) are computed once on first
    access and cached for the lifetime of the object.
    """

    def __init__(
        self,
        problem: Problem,
        initial_point: np.ndarray,
        num_constraints: int,
        verbose: bool = True,
        use_hessian: bool = True,
    ) -> None:
        # Import from cvxpy's diff_engine integration layer
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine import C_problem

        self.c_problem = C_problem(problem, verbose=verbose)
        self.use_hessian = use_hessian

        # Always initialize Jacobian
        self.c_problem.init_jacobian()

        # Only initialize Hessian if needed (not for quasi-Newton methods)
        if use_hessian:
            self.c_problem.init_hessian()

        self.initial_point = initial_point
        self.num_constraints = num_constraints
        self.iterations = 0

        # Cached sparsity structures
        self._jac_structure: tuple[np.ndarray, np.ndarray] | None = None
        self._hess_structure: tuple[np.ndarray, np.ndarray] | None = None
        self.constraints_forward_passed = False
        self.objective_forward_passed = False

    def objective(self, x: np.ndarray) -> float:
        """Returns the scalar value of the objective given x."""
        self.objective_forward_passed = True
        return self.c_problem.objective_forward(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        """Returns the gradient of the objective with respect to x."""

        if not self.objective_forward_passed:
            self.objective(x)

        return self.c_problem.gradient()

    def constraints(self, x: np.ndarray) -> np.ndarray:
        """Returns the constraint values."""
        self.constraints_forward_passed = True
        return self.c_problem.constraint_forward(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        """Returns the Jacobian values in COO format at the sparsity structure. """

        if not self.constraints_forward_passed:
            self.constraints(x)

        jac_csr = self.c_problem.jacobian()
        jac_coo = jac_csr.tocoo()
        return jac_coo.data.copy()

    def jacobianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the sparsity structure of the Jacobian."""
        if self._jac_structure is not None:
            return self._jac_structure

        jac_csr = self.c_problem.get_jacobian()
        jac_coo = jac_csr.tocoo()

        self._jac_structure = (jac_coo.row.astype(np.int32),
                               jac_coo.col.astype(np.int32))
        return self._jac_structure

    def hessian(self, x: np.ndarray, duals: np.ndarray, obj_factor: float) -> np.ndarray:
        """Returns the lower triangular Hessian values in COO format. """
        if not self.use_hessian:
            # Shouldn't be called when using quasi-Newton, but return empty array
            return np.array([])

        if not self.objective_forward_passed:
            self.objective(x)
        if not self.constraints_forward_passed:
            self.constraints(x)

        hess_csr = self.c_problem.hessian(obj_factor, duals)
        hess_coo = hess_csr.tocoo()

        # Extract lower triangular values
        mask = hess_coo.row >= hess_coo.col

        return hess_coo.data[mask]

    def hessianstructure(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns the sparsity structure of the lower triangular Hessian."""
        if not self.use_hessian:
            # Return empty structure when using quasi-Newton approximation
            return (np.array([], dtype=np.int32), np.array([], dtype=np.int32))

        if self._hess_structure is not None:
            return self._hess_structure

        hess_csr = self.c_problem.get_hessian()
        hess_coo = hess_csr.tocoo()

        # Keep only lower triangular
        mask = hess_coo.row >= hess_coo.col
        self._hess_structure = (
            hess_coo.row[mask].astype(np.int32),
            hess_coo.col[mask].astype(np.int32)
        )
        return self._hess_structure

    def intermediate(
        self,
        alg_mod: int,
        iter_count: int,
        obj_value: float,
        inf_pr: float,
        inf_du: float,
        mu: float,
        d_norm: float,
        regularization_size: float,
        alpha_du: float,
        alpha_pr: float,
        ls_trials: int,
    ) -> None:
        """Prints information at every Ipopt iteration."""
        self.iterations = iter_count
        self.objective_forward_passed = False
        self.constraints_forward_passed = False
