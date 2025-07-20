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
import torch

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
from cvxtorch import TorchExpression


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
        initial_values = []
        for var in bounds.main_var:
            if var.value is not None:
                initial_values.append(var.value.flatten(order='F'))
            else:
                # If no initial value, use zero
                initial_values.append(np.zeros(var.size))
        x0 = np.concatenate(initial_values, axis=0)
        nlp = cyipopt.Problem(
        n=len(x0),
        m=len(bounds.cl),
        problem_obj=self.Oracles(bounds.new_problem),
        lb=bounds.lb,
        ub=bounds.ub,
        cl=bounds.cl,
        cu=bounds.cu,
        )
        nlp.add_option('mu_strategy', 'adaptive')
        nlp.add_option('tol', 1e-7)
        nlp.add_option('hessian_approximation', "limited-memory")
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

    class Oracles():
        def __init__(self, problem):
            self.problem = problem
            self.main_var = []
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
            """Returns the gradient of the objective with respect to x."""
            # Convert to torch tensor with gradient tracking
            offset = 0
            torch_exprs = []
            for var in self.main_var:
                size = var.size
                slice = x[offset:offset+size].reshape(var.shape, order='F')
                torch_exprs.append(torch.from_numpy(slice.astype(np.float64)).requires_grad_(True))
                offset += size
            
            torch_expr = TorchExpression(self.problem.objective.args[0])
            torch_obj = torch_expr.torch_expression(*torch_exprs)
            
            # Compute gradient
            torch_obj.backward()
            gradients = []
            for tensor in torch_exprs:
                if tensor.grad is not None:
                    gradients.append(tensor.grad.detach().numpy().flatten())
                else:
                    gradients.append(np.array([0] * tensor.numel()))
            return np.concatenate(gradients)
        
        def constraints(self, x):
            """Returns the constraint values."""
            # Set the variable value
            offset = 0
            for var in self.main_var:
                size = var.size
                var.value = x[offset:offset+size].reshape(var.shape, order='F')
                offset += size
            
            # Evaluate all constraints
            constraint_values = []
            for constraint in self.problem.constraints:
                constraint_values.append(np.asarray(constraint.args[0].value).flatten())
            return np.concat(constraint_values)
        
        def jacobian(self, x):
            """Returns the Jacobian of the constraints with respect to x."""
            # Convert to torch tensor with gradient tracking
            x = torch.from_numpy(x.astype(np.float64)).requires_grad_(True)

            # Define a function that computes all constraint values
            def constraint_function(x):
                from cvxtorch.utils.torch_utils import tensor_reshape_fortran
                offset = 0
                torch_vars_dict = {}
                torch_exprs = []
                for var in self.main_var:
                    size = var.size
                    slice = x[offset:offset+size].reshape(var.shape)
                    #slice = x[offset:offset+size].reshape(var.shape, order='F')
                    torch_vars_dict[var.id] = slice # Map CVXPY variable ID to torch tensor
                    torch_exprs.append(slice)
                    offset += size
                
                # Create mapping from torch tensors back to CVXPY variables
                torch_to_var = {}
                for i, var in enumerate(self.main_var):
                    torch_to_var[var.id] = torch_exprs[i]
                
                constraint_values = []
                for constraint in self.problem.constraints:
                    # all constraints have a single argument
                    # because they are "normalized" in the reduction
                    constraint_expr = constraint.args[0]
                    constraint_vars = constraint_expr.variables()
                    
                    # Create ordered list of torch tensors for this specific constraint
                    # in the order that the constraint expression expects them
                    constr_torch_args = []
                    for var in constraint_vars:
                        if var.id in torch_to_var:
                            constr_torch_args.append(torch_to_var[var.id])
                        else:
                            raise ValueError(f"Variable {var} not found in torch mapping")
                    
                    torch_expr = TorchExpression(constraint_expr).torch_expression(
                        *constr_torch_args
                    )
                    constraint_values.append(torch_expr)
                return torch.cat([cv.flatten() for cv in constraint_values])

            # Compute Jacobian using torch.autograd.functional.jacobian
            if len(self.problem.constraints) > 0:
                jacobian_tuple = torch.autograd.functional.jacobian(constraint_function, x)
                # Handle the case where jacobian_tuple is a tuple (multiple variables)
                if isinstance(jacobian_tuple, tuple):
                    # Concatenate along the last dimension (variable dimension)
                    jacobian_matrix = torch.cat(
                        [jac.reshape(jac.size(0), -1) for jac in jacobian_tuple],
                        dim=1
                    )
                else:
                    # Single variable case
                    jacobian_matrix = jacobian_tuple
                return jacobian_matrix.detach().numpy()

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
                    var_lower.extend(var.bounds[0].flatten(order='F'))
                    var_upper.extend(var.bounds[1].flatten(order='F'))
                else:
                    # No bounds specified, use infinite bounds
                    var_lower.extend([-np.inf] * size)
                    var_upper.extend([np.inf] * size)

            self.lb = np.array(var_lower)
            self.ub = np.array(var_upper)
