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

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, NonPos, Zero
from cvxpy.reductions.solution import Solution

from .conic_solver import ConicSolver


class SCS(ConicSolver):
    """An interface for the SCS solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC,
                                                                 ExpCone,
                                                                 PSD]
    MIN_CONSTRAINTS = 1

    # Map of SCS status to CVXPY status.
    STATUS_MAP = {"Solved": s.OPTIMAL,
                  "Solved/Inaccurate": s.OPTIMAL_INACCURATE,
                  "Unbounded": s.UNBOUNDED,
                  "Unbounded/Inaccurate": s.UNBOUNDED_INACCURATE,
                  "Infeasible": s.INFEASIBLE,
                  "Infeasible/Inaccurate": s.INFEASIBLE_INACCURATE,
                  "Failure": s.SOLVER_ERROR,
                  "Indeterminate": s.SOLVER_ERROR}

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 1, 2]

    def name(self):
        """The name of the solver.
        """
        return s.SCS

    def import_solver(self):
        """Imports the solver.
        """
        import scs
        scs  # For flake8

    def accepts(self, problem):
        """Can SCS solve the problem?
        """
        return (type(problem.objective) == Minimize
                and is_stuffed_cone_objective(problem.objective)
                and len(problem.constraints) >= SCS.MIN_CONSTRAINTS
                and all(type(c) in SCS.SUPPORTED_CONSTRAINTS for c in
                        problem.constraints))

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.variables()[0].id}
        data[s.C], data[s.OFFSET] = ConicSolver.get_coeff_offset(
            problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        # SCS only has inequalities
        inv_data[SCS.EQ_CONSTR] = []

        # Order and group nonlinear constraints.
        data[s.DIMS] = {}
        zero_constr = [c for c in problem.constraints if type(c) == Zero]
        data[s.DIMS]['f'] = sum([np.prod(c.size) for c in zero_constr])
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        data[s.DIMS]['l'] = sum([np.prod(c.size) for c in leq_constr])
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        data[s.DIMS]['q'] = [size for cons in soc_constr
                                  for size in cons.cone_sizes()]
        exp_constr = [c for c in problem.constraints if type(c) == ExpCone]
        data[s.DIMS]['ep'] = sum([cons.num_cones() for cons in exp_constr])
        constr = zero_constr + leq_constr + soc_constr + exp_constr
        inv_data[SCS.NEQ_CONSTR] = constr
        data[s.A], data[s.B] = self.group_coeff_offset(constr,
                                                       self.EXP_CONE_ORDER)
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution['info']['status']]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = solution["info"]["solveTime"]
        attr[s.SETUP_TIME] = solution["info"]["setupTime"]
        attr[s.NUM_ITERS] = solution["info"]["iter"]

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pobj']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[SCS.VAR_ID]: solution['x']}
            ineq_dual = ConicSolver.get_dual_values(
                solution['y'], inverse_data[SCS.NEQ_CONSTR])
            dual_vars = ineq_dual
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None
            primal_vars = None
            dual_vars = None

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        ...

        Returns
        -------
        tuple
        ...
        """
        import scs
        return scs.solve(
            data,
            data[s.DIMS],
            verbose=verbose,
            **solver_opts)
