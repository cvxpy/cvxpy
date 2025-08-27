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

import cvxpy.settings as s
from cvxpy.constraints import (
    Equality,
    Inequality,
    NonPos,
)
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
from cvxpy.reductions.utilities import (
    lower_equality,
    lower_ineq_to_nonneg,
    nonpos2nonneg,
)
from cvxpy.utilities.citations import CITATION_DICT


class IPOPT(NLPsolver):
    """
    NLP interface for the IPOPT solver
    """
    # Map between IPOPT status and CVXPY status
    # taken from https://github.com/jump-dev/Ipopt.jl/blob/master/src/C_wrapper.jl#L485-L511
    STATUS_MAP = {
        # Success cases
        0: s.OPTIMAL,                    # Solve_Succeeded
        1: s.OPTIMAL_INACCURATE,         # Solved_To_Acceptable_Level
        6: s.OPTIMAL,                    # Feasible_Point_Found
        
        # Infeasibility/Unboundedness
        2: s.INFEASIBLE,                 # Infeasible_Problem_Detected
        4: s.UNBOUNDED,                  # Diverging_Iterates
        
        # Numerical/Algorithm issues
        3: s.SOLVER_ERROR,               # Search_Direction_Becomes_Too_Small
        -2: s.SOLVER_ERROR,              # Restoration_Failed
        -3: s.SOLVER_ERROR,              # Error_In_Step_Computation
        -13: s.SOLVER_ERROR,             # Invalid_Number_Detected
        -100: s.SOLVER_ERROR,            # Unrecoverable_Exception
        -101: s.SOLVER_ERROR,            # NonIpopt_Exception_Thrown
        -199: s.SOLVER_ERROR,            # Internal_Error
        
        # User/Resource limits
        5: s.USER_LIMIT,                 # User_Requested_Stop
        -1: s.USER_LIMIT,                # Maximum_Iterations_Exceeded
        -4: s.USER_LIMIT,                # Maximum_CpuTime_Exceeded
        -5: s.USER_LIMIT,                # Maximum_WallTime_Exceeded
        -102: s.USER_LIMIT,              # Insufficient_Memory
        
        # Problem definition issues
        -10: s.SOLVER_ERROR,             # Not_Enough_Degrees_Of_Freedom
        -11: s.SOLVER_ERROR,             # Invalid_Problem_Definition
        -12: s.SOLVER_ERROR,             # Invalid_Option
    }

    def name(self):
        """
        The name of solver.
        """
        return 'IPOPT'

    def import_solver(self):
        """
        Imports the solver.
        """
        import cyipopt  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        attr = {}
        status = self.STATUS_MAP[solution['status']]
        # the info object does not contain all the attributes we want
        # see https://github.com/mechmotum/cyipopt/issues/17
        # attr[s.SOLVE_TIME] = solution.solve_time
        # attr[s.NUM_ITERS] = solution.iterations
        # more detailed statistics here when available
        # attr[s.EXTRA_STATS] = solution.extra.FOO
    
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['obj_val']
            opt_val = primal_val + inverse_data.offset
            primal_vars = {}
            x_opt = solution['x']
            for id, offset in inverse_data.var_offsets.items():
                shape = inverse_data.var_shapes[id]
                size = np.prod(shape, dtype=int)
                primal_vars[id] = np.reshape(x_opt[offset:offset+size], shape, order='F')
            return Solution(status, opt_val, primal_vars, {}, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cyipopt
        bounds = self.Bounds(data["problem"])
        x0 = self.construct_initial_point(bounds)
        nlp = cyipopt.Problem(
        n=len(x0),
        m=len(bounds.cl),
        problem_obj=self.Oracles(bounds.new_problem, x0),
        lb=bounds.lb,
        ub=bounds.ub,
        cl=bounds.cl,
        cu=bounds.cu,
        )
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-7)
        #nlp.add_option('honor_original_bounds', 'yes')
        nlp.add_option('bound_relax_factor', 0.0)
        nlp.add_option('hessian_approximation', "limited-memory")
        nlp.add_option('derivative_test', 'first-order')
        nlp.add_option('least_square_init_duals', 'yes')
        #nlp.add_option('constr_mult_init_max', 1e10) 
        #nlp.add_option('derivative_test_perturbation', 1e-5)
        #nlp.add_option('point_perturbation_radius', 0.1)
        _, info = nlp.solve(x0)
        return info

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["IPOPT"]
    
    # TODO (DCED): Ask WZ where to put this.
    def construct_initial_point(self, bounds):
            initial_values = []
            offset = 0
            lbs = bounds.lb 
            ubs = bounds.ub
            for var in bounds.main_var:
                if var.value is not None:
                    initial_values.append(var.value.flatten(order='F'))
                else:
                    # If no initial value is specified, look at the bounds.
                    # If both lb and ub are specified, we initialize the
                    # variables to be their midpoints. If only one of them 
                    # is specified, we initialize the variable one unit 
                    # from the bound. If none of them is specified, we 
                    # initialize it to zero.
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
            x0 = np.concatenate(initial_values, axis=0)
            return x0

    class Oracles():
        def __init__(self, problem, inital_point):
            self.problem = problem
            self.main_var = []
            self.initial_point = inital_point
            for var in self.problem.variables():
                self.main_var.append(var)

        def objective(self, x):
            """Returns the scalar value of the objective given x."""
            # Set the variable value
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
            # Evaluate the objective
            obj_value = self.problem.objective.args[0].value
            return obj_value
        
        def gradient(self, x):
            #import pdb 
            #pdb.set_trace()
            """Returns the gradient of the objective with respect to x."""
            # compute the gradient using _grad
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
            grad_offset = 0
            grad = np.zeros(x.size, dtype=np.float64)
            grad_dict = self.problem.objective.expr.grad
            for var in self.main_var:
                size = var.size
                if var in grad_dict:
                    array = grad_dict[var]
                    if sp.issparse(array):
                        array = array.toarray().flatten(order='F')
                    grad[grad_offset:grad_offset+size] = array
                grad_offset += size
            return grad

        def constraints(self, x):
            """Returns the constraint values."""
            # Set the variable value
            offset = 0
            #import pdb 
            #pdb.set_trace()
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
            
            # Evaluate all constraints
            constraint_values = []
            for constraint in self.problem.constraints:
                constraint_values.append(np.asarray(constraint.args[0].value).flatten(order='F'))
            return np.concatenate(constraint_values)

        def jacobian(self, x):
            """Returns only the non-zero values of the Jacobian."""
            #import pdb 
            #pdb.set_trace()
            # Set variable values
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
            
            values = []
            for constraint in self.problem.constraints:
                # get the jacobian of the constraint
                grad_dict = constraint.expr.grad
                for var in self.main_var:
                    if var in grad_dict:
                        rows, cols = self.jacobian_idxs[constraint][var] 
                        jacobian = grad_dict[var].T
                        if sp.issparse(jacobian):
                            jacobian = sp.dok_matrix(jacobian)
                            data = np.array([
                                jacobian.get((r, c), 0)
                                for r, c in zip(rows, cols)
                            ])
                            values.append(np.atleast_1d(data))
                        else:
                            values.append(np.atleast_1d(jacobian))
            return np.concatenate(values)
        
        def jacobianstructure(self):
            """Returns the sparsity structure of the Jacobian."""
            # this dict stores the jacobian for each constraint for each variable
            self.jacobian_idxs = {}
            # Set dummy values to get gradient structure
            #offset = 0
            for var in self.main_var:
                if var.shape == ():
                    #var.value = self.initial_point[offset]
                    var.value = np.nan
                else:
                    var.value = np.nan * np.ones(var.size).reshape(var.shape, order='F')
                    #var.value = np.atleast_1d(self.initial_point[offset:offset + var.size])
                #offset += var.size
            rows, cols = [], []
            row_offset = 0
            for constraint in self.problem.constraints:
                grad_dict = constraint.expr.grad
                col_offset = 0
                constraint_jac = {}
                for var in self.main_var:
                    if var in grad_dict:
                        jacobian = grad_dict[var].T
                        if sp.issparse(jacobian):
                            jacobian = jacobian.tocoo()
                            rows.extend(jacobian.row + row_offset)
                            cols.extend(jacobian.col + col_offset)
                            constraint_jac[var] = (jacobian.row, jacobian.col)
                        else:
                            rows.extend(np.ones(jacobian.size)*row_offset)
                            cols.extend(np.arange(col_offset, col_offset + var.size))
                            assert(jacobian.shape == ())
                            constraint_jac[var] = (0, 0)
                    col_offset += var.size
                row_offset += constraint.size
                self.jacobian_idxs[constraint] = constraint_jac

            return (np.array(rows), np.array(cols))

    class Bounds():
        def __init__(self, problem):
            self.problem = problem
            self.main_var = problem.variables()
            self.get_constraint_bounds()
            self.get_variable_bounds()

        def get_constraint_bounds(self):
            """Also normalizes the constraints and creates a new problem"""
            lower = []
            upper = []
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
            
            lowered_con_problem = self.problem.copy([self.problem.objective, new_constr])
            self.new_problem = lowered_con_problem
            self.cl = np.array(lower)
            self.cu = np.array(upper)

        def get_variable_bounds(self):
            var_lower = []
            var_upper = []
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
