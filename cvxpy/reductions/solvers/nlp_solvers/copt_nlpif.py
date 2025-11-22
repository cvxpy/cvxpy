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


class COPT(NLPsolver):
    """
    NLP interface for the COPT solver.
    """
    STATUS_MAP = {
        # Success cases
        0: s.OPTIMAL,                    # Solve_Succeeded
        1: s.OPTIMAL_INACCURATE,         # Solved_To_Acceptable_Level
        6: s.OPTIMAL,                    # Feasible_Point_Found
        
        # Infeasibility/Unboundedness
        2: s.INFEASIBLE,                 # Infeasible_Problem_Detected
        4: s.UNBOUNDED,                  # Diverging_Iterates
    }

    def name(self):
        """
        The name of solver.
        """
        return 'COPT'

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
        attr[s.NUM_ITERS] = solution['iterations']
    
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
        raise NotImplementedError("COPT NLP interface is not yet implemented.")

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COPT"]
