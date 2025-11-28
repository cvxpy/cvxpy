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

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.nlp_solvers.nlp_solver import NLPsolver
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
        attr[s.NUM_ITERS] = solution['iterations']
        # more detailed statistics here when available
        # attr[s.EXTRA_STATS] = solution.extra.FOO
        if 'all_objs_from_best_of' in solution:
            attr[s.EXTRA_STATS] = {'all_objs_from_best_of':
                                    solution['all_objs_from_best_of']}
    
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
        This consists of:
        - "oracles": An Oracles object that computes the objective and constraints
        - "x0": Initial guess for the primal variables
        - "lb": Lower bounds on the primal variables
        - "ub": Upper bounds on the primal variables
        - "cl": Lower bounds on the constraints
        - "cu": Upper bounds on the constraints
        - "objective": Function to compute the objective value
        - "gradient": Function to compute the objective gradient
        - "constraints": Function to compute the constraint values
        - "jacobian": Function to compute the constraint Jacobian
        - "jacobianstructure": Function to compute the structure of the Jacobian
        - "hessian": Function to compute the Hessian of the Lagrangian
        - "hessianstructure": Function to compute the structure of the Hessian
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
        # Create oracles object
        oracles = data["oracles"]
        nlp = cyipopt.Problem(
        n=len(data["x0"]),
        m=len(data["cl"]),
        problem_obj=oracles,
        lb=data["lb"],
        ub=data["ub"],
        cl=data["cl"],
        cu=data["cu"],
        )
        # Set default IPOPT options, but use solver_opts if provided
        default_options = {
            'mu_strategy': 'adaptive',
            'tol': 1e-7,
            'bound_relax_factor': 0.0,
            'hessian_approximation': 'exact',
            'derivative_test': 'first-order',
            'least_square_init_duals': 'yes'
        }
        # Update defaults with user-provided options
        if solver_opts:
            default_options.update(solver_opts)
        if not verbose and 'print_level' not in default_options:
            default_options['print_level'] = 3
        # Apply all options to the nlp object
        for option_name, option_value in default_options.items():
            nlp.add_option(option_name, option_value)

        _, info = nlp.solve(data["x0"])

        # add number of iterations to info dict from oracles
        info['iterations'] = oracles.iterations
        return info

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["IPOPT"]
