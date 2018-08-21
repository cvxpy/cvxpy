"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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
from cvxpy.constraints import SOC
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities

from .conic_solver import ConicSolver


class XPRESS(ConicSolver):
    """An interface for the Gurobi solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.SOLVER_ERROR,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  # TODO could be anything.
                  # means time expired.
                  9: s.OPTIMAL_INACCURATE,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.XPRESS

    def import_solver(self):
        """Imports the solver.
        """
        import xpress
        self.version = xpress.getversion()

    def accepts(self, problem):
        """Can Gurobi solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in XPRESS.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        objective, _ = problem.objective.canonical_form
        constraints = [con for c in problem.constraints for con in c.canonical_form[1]]
        data["objective"] = objective
        data["constraints"] = constraints
        variables = problem.variables()[0]
        data[s.BOOL_IDX] = [t[0] for t in variables.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in variables.integer_idx]

        # Order and group constraints.
        inv_data = {self.VAR_ID: problem.variables()[0].id}
        inv_data[s.EQ_CONSTR] = problem.constraints
        inv_data['is_mip'] = problem.is_mixed_integer()
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE]
            primal_vars = {inverse_data[XPRESS.VAR_ID]: solution['primal']}
            if not inverse_data['is_mip']:
                dual_vars = utilities.get_dual_values(
                    solution[s.EQ_DUAL],
                    utilities.extract_dual_value,
                    inverse_data[s.EQ_CONSTR])
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None

        other = {}
        other[s.XPRESS_IIS] = solution[s.XPRESS_IIS]
        other[s.XPRESS_TROW] = solution[s.XPRESS_TROW]
        return Solution(status, opt_val, primal_vars, dual_vars, other)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.xpress_intf import XPRESS as XPRESS_OLD
        solver = XPRESS_OLD()
        solver_opts[s.BOOL_IDX] = data[s.BOOL_IDX]
        solver_opts[s.INT_IDX] = data[s.INT_IDX]
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
