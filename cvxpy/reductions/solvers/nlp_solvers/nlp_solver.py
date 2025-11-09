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


class NLPsolver(Solver):
    """
    A non-linear programming (NLP) solver.
    """
    REQUIRES_CONSTR = False
    MIP_CAPABLE = False

    def accepts(self, problem):
        """
        Only accepts disciplined nonlinear programs.
        """
        return problem.is_dnlp()

    def apply(self, problem):
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

    def _prepare_data_and_inv_data(self, problem):
        data = dict()
        bounds = Bounds(problem)
        inverse_data = InverseData(bounds.new_problem)
        data["problem"] = bounds.new_problem
        data["cl"], data["cu"] = bounds.cl, bounds.cu
        data["lb"], data["ub"] = bounds.lb, bounds.ub
        data["x0"] = bounds.x0
        inverse_data.offset = 0.0
        return problem, data, inverse_data

class Bounds():
    def __init__(self, problem):
        self.problem = problem
        self.main_var = problem.variables()
        self.get_constraint_bounds()
        self.get_variable_bounds()
        self.construct_initial_point()

    def get_constraint_bounds(self):
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

    def get_variable_bounds(self):
        """
        Get variable bounds for all variables.
        Also takes into account nonneg/nonpos attributes.
        """
        var_lower, var_upper = [], []
        for var in self.main_var:
            size = var.size
            if var.bounds:
                lb = var.bounds[0].flatten(order='F')
                ub = var.bounds[1].flatten(order='F')
                if var.is_nonneg():
                    lb = np.maximum(lb, 0)
                if var.is_nonpos():
                    ub = np.minimum(ub, 0)
                var_lower.extend(lb)
                var_upper.extend(ub)
            else:
                # No bounds specified, use infinite bounds or bounds
                # set by the nonnegative or nonpositive attribute
                if var.is_nonneg():
                    var_lower.extend([0.0] * size)
                else:
                    var_lower.extend([-np.inf] * size)
                if var.is_nonpos():
                    var_upper.extend([0.0] * size)
                else:
                    var_upper.extend([np.inf] * size)
        self.lb = np.array(var_lower)
        self.ub = np.array(var_upper)
    
    def construct_initial_point(self):
        """
        Constructs an initial point for the optimization problem.
        If no initial value is specified, look at the bounds.
        If both lb and ub are specified, we initialize the
        variables to be their midpoints. If only one of them
        is specified, we initialize the variable one unit
        from the bound. If none of them is specified, we
        initialize it to zero.
        """
        initial_values = []
        offset = 0
        lbs = self.lb
        ubs = self.ub
        for var in self.problem.variables():
            if var.value is not None:
                initial_values.append(np.atleast_1d(var.value).flatten(order='F'))
            else:
                lb = lbs[offset:offset + var.size]
                ub = ubs[offset:offset + var.size]
                lb_finite = np.isfinite(lb)
                ub_finite = np.isfinite(ub)
                # Replace infs with zero for arithmetic
                lb0 = np.where(lb_finite, lb, 0.0)
                ub0 = np.where(ub_finite, ub, 0.0)
                # Midpoint if both finite, one from bound if only one finite, zero if none
                init = (lb_finite * ub_finite * 0.5 * (lb0 + ub0) +
                        lb_finite * (~ub_finite) * (lb0 + 1.0) +
                        (~lb_finite) * ub_finite * (ub0 - 1.0))
                initial_values.append(init)
            offset += var.size
        self.x0 = np.concatenate(initial_values, axis=0)
