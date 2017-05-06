"""
Copyright 2015 Enzo Busseti, 2017 Robin Verschueren

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import SDP, SOC, NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solution import Solution

from .conic_solver import ConicSolver


class MOSEK(ConicSolver):
    """An interface for the Mosek solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Map of Mosek status to CVXPY status.
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

    def import_solver(self):
        """Imports the solver.
        """
        import mosek
        mosek  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

    def accepts(self, problem):
        """Can Mosek solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos, SOC, SDP]:
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
        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Order and group constraints.
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[MOSEK.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        sd_constr = [c for c in problem.constraints if type(c) == SDP]
        inv_data[MOSEK.NEQ_CONSTR] = leq_constr + soc_constr + sd_constr
        return data, inv_data
    
    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution['status']

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['value']
            primal_vars = {inverse_data[MOSEK.VAR_ID]: solution['primal']}
            eq_dual = ConicSolver.get_dual_values(solution['eq_dual'], inverse_data[MOSEK.EQ_CONSTR])
            leq_dual = ConicSolver.get_dual_values(solution['ineq_dual'], inverse_data[MOSEK.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None
            primal_vars = None
            dual_vars = None

        return Solution(status, opt_val, primal_vars, dual_vars, None)

    def solve(self, problem, warm_start, verbose, solver_opts):
        from cvxpy.problems.solvers.mosek_intf import MOSEK as MOSEK_OLD
        solver = MOSEK_OLD()
        _, inv_data = self.apply(problem)
        objective, _ = problem.objective.canonical_form
        constraints = [constraint for c in problem.constraints for constraint in c.canonical_form[1]]
        sol = solver.solve(objective, constraints, {self.name():ProblemData()}, \
            warm_start, verbose, solver_opts)
        
        return self.invert(sol, inv_data)
