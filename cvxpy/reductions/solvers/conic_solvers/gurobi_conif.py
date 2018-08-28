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
from cvxpy.constraints import SOC, NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities

from .conic_solver import ConicSolver


class GUROBI(ConicSolver):
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
        return s.GUROBI

    def import_solver(self):
        """Imports the solver.
        """
        import gurobipy
        gurobipy  # For flake8

    def accepts(self, problem):
        """Can Gurobi solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in GUROBI.SUPPORTED_CONSTRAINTS:
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
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[GUROBI.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        inv_data[GUROBI.NEQ_CONSTR] = leq_constr + soc_constr
        inv_data['is_mip'] = len(data[s.BOOL_IDX]) > 0 or len(data[s.INT_IDX]) > 0
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
            primal_vars = {inverse_data[GUROBI.VAR_ID]: solution['primal']}
            if not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[GUROBI.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None

        return Solution(status, opt_val, primal_vars, dual_vars, {})

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.gurobi_intf import GUROBI as GUROBI_OLD
        solver = GUROBI_OLD()
        solver_opts[s.BOOL_IDX] = data[s.BOOL_IDX]
        solver_opts[s.INT_IDX] = data[s.INT_IDX]
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
