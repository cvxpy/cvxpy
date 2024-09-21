"""
Copyright, the CVXPY developers

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


import cvxpy.settings as s
from cvxpy.constraints.nonpos import NonNeg
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


class HIGHS(ConicSolver):
    """An interface for HiGHS solver."""
    
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS
    MIP_CAPABLE = True
    MI_SUPPORTED_CONSTRAINTS = SUPPORTED_CONSTRAINTS

    STATUS_MAP = {0: s.OPTIMAL,  # Optimal
                  1: s.OPTIMAL_INACCURATE,  # Iteration/time limit reached
                  2: s.INFEASIBLE,  # Infeasible
                  3: s.UNBOUNDED,  # Unbounded
                  4: s.SOLVER_ERROR  # Numerical difficulties encountered
                  }
    def name(self):
        """The name of the solver."""
        return 'HIGHS'

    def import_solver(self) -> None:
        """Imports the solver."""
        import highspy  # noqa F401

    def accepts(self, problem):
        """Can HiGHS solve the problem?"""
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True
    
    def apply(self, problem):
        """
        Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg]
        len_eq = problem.cone_dims.zero

        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A[:len_eq]
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b[:len_eq].flatten(order='F')
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        data[s.G] = -A[len_eq:]
        if 0 in data[s.G].shape:
            data[s.G] = None
        data[s.H] = b[len_eq:].flatten(order='F')
        if 0 in data[s.H].shape:
            data[s.H] = None
        return data, inv_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        pass

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data."""
        status = self.STATUS_MAP[solution["status"]]

        # Sometimes when the solver's time limit is reached, the solver doesn't return a solution.
        # In these situations we correct the status from s.OPTIMAL_INACCURATE to s.SOLVER_ERROR
        if (status == s.OPTIMAL_INACCURATE) and (solution.x is None):
            status = s.SOLVER_ERROR

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            primal_val = solution['fun']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[self.VAR_ID]: solution['x']}

            # SciPy linprog only returns duals for version >= 1.7.0
            # and method is one of 'highs', 'highs-ds' or 'highs-ipm'
            # MIP problems don't have duals and thus are not updated.
            if 'ineqlin' in solution and not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    -solution['eqlin']['marginals'],
                    utilities.extract_dual_value,
                    inverse_data[self.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    -solution['ineqlin']['marginals'],
                    utilities.extract_dual_value,
                    inverse_data[self.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
            
            attr = {}
            if "nit" in solution: # Number of interior-point or simplex iterations
                attr[s.NUM_ITERS] = solution['nit']
            if "mip_gap" in solution: # Branch and bound statistics
                attr[s.EXTRA_STATS] = {"mip_gap": solution['mip_gap'], 
                                       "mip_node_count": solution['mip_node_count'], 
                                       "mip_dual_bound": solution['mip_dual_bound']}

            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status)
