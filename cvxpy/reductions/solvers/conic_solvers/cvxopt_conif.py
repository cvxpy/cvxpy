"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

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

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solvers.solver import group_constraints
from .conic_solver import ConeDims, ConicSolver


class CVXOPT(ConicSolver):
    """An interface for the CVXOPT solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC,
                                                                 ExpCone,
                                                                 PSD]

    # Map of CVXOPT status to CVXPY status.
    STATUS_MAP = {'optimal': s.OPTIMAL,
                  'infeasible': s.INFEASIBLE,
                  'unbounded': s.UNBOUNDED,
                  'solver_error': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CVXOPT

    def import_solver(self):
        """Imports the solver.
        """
        import cvxopt
        cvxopt  # For flake8

    def accepts(self, problem):
        """Can CVXOPT solve the problem?
        """
        # TODO check if is matrix stuffed.
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
        data[ConicSolver.DIMS] = ConeDims(
            group_constraints(problem.constraints))
        variables = problem.variables()[0]
        data[s.BOOL_IDX] = [t[0] for t in variables.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in variables.integer_idx]

        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Order and group constraints.
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[CVXOPT.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        sdp_constr = [c for c in problem.constraints if type(c) == PSD]
        exp_constr = [c for c in problem.constraints if type(c) == ExpCone]
        inv_data[CVXOPT.NEQ_CONSTR] = leq_constr + soc_constr + sdp_constr + exp_constr
        return data, inv_data

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        from cvxpy.problems.solvers.cvxopt_intf import CVXOPT as CVXOPT_OLD
        solver = CVXOPT_OLD()
        return solver.solve(
            data["objective"],
            data["constraints"],
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)
