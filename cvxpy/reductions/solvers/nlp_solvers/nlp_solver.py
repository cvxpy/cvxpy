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
import scipy.sparse as sp

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
        inverse_data.offset = 0.0
        data["problem"] = bounds.new_problem
        data["cl"], data["cu"] = bounds.cl, bounds.cu
        data["lb"], data["ub"] = bounds.lb, bounds.ub
        data["x0"] = bounds.x0
        oracles = Oracles(bounds.new_problem, bounds.x0, len(bounds.cl))
        data["objective"] = oracles.objective
        data["gradient"] = oracles.gradient
        data["constraints"] = oracles.constraints
        data["jacobian"] = oracles.jacobian
        data["jacobianstructure"] = oracles.jacobianstructure
        data["hessian"] = oracles.hessian
        data["hessianstructure"] = oracles.hessianstructure
        data["oracles"] = oracles
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


class Oracles():
    def __init__(self, problem, initial_point, num_constraints):
        self.problem = problem
        self.grad_obj = np.zeros(initial_point.size, dtype=np.float64)

        # for evaluating hessian
        self.hess_lagrangian_coo = ([], [], [])
        self.hess_lagrangian_coo_rows_cols = ([], [])
        self.has_computed_hess_sparsity = False
        self.num_constraints = num_constraints

        # for evaluating jacobian
        self.jacobian_coo = ([], [], [])
        self.jacobian_coo_rows_cols = ([], [])
        self.jacobian_affine_coo = ([], [], [])
        self.has_computed_jac_sparsity = False
        self.has_stored_affine_jacobian = False

        self.main_var = []
        self.initial_point = initial_point
        self.iterations = 0
        for var in self.problem.variables():
            self.main_var.append(var)

    def set_variable_value(self, x):
        offset = 0
        for var in self.main_var:
            size = var.size
            var.value = x[offset:offset+size].reshape(var.shape, order='F')
            offset += size

    def objective(self, x):
        """Returns the scalar value of the objective given x."""
        self.set_variable_value(x)
        obj_value = self.problem.objective.args[0].value
        return obj_value
    
    def gradient(self, x):
        """Returns the gradient of the objective with respect to x."""
        self.set_variable_value(x)

        # fill with zeros to reset from previous call
        self.grad_obj.fill(0)

        grad_offset = 0
        grad_dict = self.problem.objective.expr.jacobian()

        for var in self.main_var:
            size = var.size
            if var in grad_dict:
                _, cols, vals = grad_dict[var]
                self.grad_obj[grad_offset + cols] = vals
            grad_offset += size

        return self.grad_obj

    def constraints(self, x):
        """Returns the constraint values."""
        self.set_variable_value(x)
        # Evaluate all constraints
        constraint_values = []
        for constraint in self.problem.constraints:
            constraint_values.append(np.asarray(constraint.args[0].value).flatten(order='F'))
        return np.concatenate(constraint_values)

    def parse_jacobian_dict(self, grad_dict, constr_offset, is_affine):
        col_offset = 0
        for var in self.main_var:
            if var in grad_dict:
                rows, cols, vals = grad_dict[var]
                if not isinstance(rows, np.ndarray):
                    rows = np.array(rows)
                if not isinstance(cols, np.ndarray):
                    cols = np.array(cols)

                self.jacobian_coo[0].extend(rows + constr_offset)
                self.jacobian_coo[1].extend(cols + col_offset)
                self.jacobian_coo[2].extend(vals)

                if is_affine:
                    self.jacobian_affine_coo[0].extend(rows + constr_offset)
                    self.jacobian_affine_coo[1].extend(cols + col_offset)
                    self.jacobian_affine_coo[2].extend(vals)

            col_offset += var.size
    
    def insert_missing_zeros_jacobian(self):
        rows, cols, vals = self.jacobian_coo
        rows_true, cols_true = self.jacobian_coo_rows_cols
        if not self.permutation_needed:
            return vals
        dim = self.initial_point.size
        m = self.num_constraints
        J = sp.csr_matrix((vals, (rows, cols)), shape=(m, dim))
        vals_true = J[rows_true, cols_true].data
        return vals_true

    def jacobian(self, x):
        self.set_variable_value(x)
    
        # reset previous call
        if not self.has_stored_affine_jacobian:
            self.jacobian_coo = ([], [], [])
        else:
            self.jacobian_coo = (self.jacobian_affine_coo[0].copy(), 
                                    self.jacobian_affine_coo[1].copy(),
                                    self.jacobian_affine_coo[2].copy())

        # compute jacobian of each constraint
        constr_offset = 0
        for constraint in self.problem.constraints:
            is_affine = constraint.expr.is_affine()
            if is_affine and self.has_stored_affine_jacobian:
                constr_offset += constraint.size
                continue
            
            grad_dict = constraint.expr.jacobian()
            self.parse_jacobian_dict(grad_dict, constr_offset, is_affine)
            constr_offset += constraint.size
        
        # insert missing zeros (ie., entries that turned out to be zero but
        # are not structurally zero)
        if self.has_computed_jac_sparsity:
            vals = self.insert_missing_zeros_jacobian()
        else:
            vals = self.jacobian_coo[2]
        return vals
        
    def jacobianstructure(self):
        # if we have already computed the sparsity structure, return it
        # (Ipopt only calls this function once, so this if is not strictly
        #  necessary)
        if self.has_computed_jac_sparsity:
            return self.jacobian_coo_rows_cols
        
        # set values to nans for full jacobian structure
        x = np.nan * np.ones(self.initial_point.size)
        self.jacobian(x)
        self.has_computed_jac_sparsity = True
        self.has_stored_affine_jacobian = True

        # permutation inside "insert_missing_zeros_jacobian" is needed if the 
        # problem has both affine and none-affine constraints.
        self.permutation_needed = False
        for constraint in self.problem.constraints:
            if not constraint.expr.is_affine():
                self.permutation_needed = True
                break

        # store sparsity pattern
        rows, cols = self.jacobian_coo[0], self.jacobian_coo[1]
        self.jacobian_coo_rows_cols = (rows, cols)
        return self.jacobian_coo_rows_cols

    def parse_hess_dict(self, hess_dict):
        """ Adds the contribution of blocks defined in hess_dict to the full
            hessian matrix 
        """
        row_offset = 0
        for var1 in self.main_var:
            col_offset = 0
            for var2 in self.main_var:
                if (var1, var2) in hess_dict:
                    rows, cols, vals = hess_dict[(var1, var2)]
                    if not isinstance(rows, np.ndarray):
                        rows = np.array(rows)
                    if not isinstance(cols, np.ndarray):
                        cols = np.array(cols)

                    self.hess_lagrangian_coo[0].extend(rows + row_offset)
                    self.hess_lagrangian_coo[1].extend(cols + col_offset)
                    self.hess_lagrangian_coo[2].extend(vals)

                col_offset += var2.size
            row_offset += var1.size

    def sum_coo(self):
        shape = (self.initial_point.size, self.initial_point.size)
        rows, cols, vals = self.hess_lagrangian_coo
        coo = sp.coo_matrix((vals, (rows, cols)), shape=shape)
        coo.sum_duplicates()
        self.hess_lagrangian_coo = (coo.row, coo.col, coo.data)
    
    def insert_missing_zeros_hessian(self):
        rows, cols, vals = self.hess_lagrangian_coo
        rows_true, cols_true = self.hess_lagrangian_coo_rows_cols
        dim = self.initial_point.size
        H = sp.csr_matrix((vals, (rows, cols)), shape=(dim, dim))
        vals_true = H[rows_true, cols_true].data
        return vals_true

    def hessianstructure(self):            
        # if we have already computed the sparsity structure, return it
        # (Ipopt only calls this function once, so this if is not strictly
        #  necessary)
        if self.has_computed_hess_sparsity:
            return self.hess_lagrangian_coo_rows_cols
        
        # set values to nans for full hessian structure
        x = np.nan * np.ones(self.initial_point.size)
        self.hessian(x, np.ones(self.num_constraints), 1.0)
        self.has_computed_hess_sparsity = True

        # extract lower triangular part
        rows, cols = self.hess_lagrangian_coo[0], self.hess_lagrangian_coo[1]
        mask = rows >= cols 
        rows = rows[mask]
        cols = cols[mask]
        self.hess_lagrangian_coo_rows_cols = (rows, cols)
        return self.hess_lagrangian_coo_rows_cols
        
    def hessian(self, x, duals, obj_factor):
        self.set_variable_value(x)
        
        # reset previous call
        self.hess_lagrangian_coo = ([], [], [])
        
        # compute hessian of objective times obj_factor
        obj_hess_dict = self.problem.objective.expr.hess_vec(np.array([obj_factor]))
        self.parse_hess_dict(obj_hess_dict)

        # compute hessian of constraints times duals
        constr_offset = 0
        for constraint in self.problem.constraints:    
            lmbda = duals[constr_offset:constr_offset + constraint.size]
            constraint_hess_dict = constraint.expr.hess_vec(lmbda)
            self.parse_hess_dict(constraint_hess_dict)
            constr_offset += constraint.size

        # merge duplicate entries together
        self.sum_coo()

        # insert missing zeros (ie., entries that turned out to be zero but are not 
        # structurally zero)
        if self.has_computed_hess_sparsity:
            vals = self.insert_missing_zeros_hessian()
        else:
            vals = self.hess_lagrangian_coo[2]
        return vals

    def intermediate(self, alg_mod, iter_count, obj_value, inf_pr, inf_du, mu,
                    d_norm, regularization_size, alpha_du, alpha_pr,
                    ls_trials):
        """Prints information at every Ipopt iteration."""
        self.iterations = iter_count
