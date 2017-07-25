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
from cvxpy.reductions.solution import failure_solution, Solution
from cvxpy.reductions.solvers.solver import group_constraints

from .conic_solver import ConicSolver


class SCS(ConicSolver):
    """An interface for the SCS solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    # TODO(akshayka): Support for PSD needs to be implemented.
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC,
                                                                 ExpCone,
                                                                 PSD]
    REQUIRES_CONSTR = True

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

    def format_constr(self, constr, exp_cone_order):
        if isinstance(constr, PSD):
            raise NotImplementedError("SCS formatting of PSD constraints "
                                      "not yet implemented.")
        else:
            return super(SCS, self).format_constr(constr, exp_cone_order)

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.variables()[0].id}

        # Parse the coefficient vector from the objective.
        data[s.C], data[s.OFFSET] = self.get_coeff_offset(
            problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        # Order and group nonlinear constraints.
        constr_map = group_constraints(problem.constraints)
        data[s.DIMS] = {}
        data[s.DIMS]['f'] = sum([np.prod(c.size) for c in constr_map[Zero]])
        data[s.DIMS]['l'] = sum([np.prod(c.size) for c in constr_map[NonPos]])
        data[s.DIMS]['ep'] = sum([c.num_cones() for c in constr_map[ExpCone]])
        data[s.DIMS]['q'] = [sz for c in constr_map[SOC]
                                for sz in c.cone_sizes()]
        # TODO(akshayka): Assemble a list of PSD cone sizes.

        zero_constr = constr_map[Zero]
        neq_constr = (constr_map[NonPos] + constr_map[SOC]
                      + constr_map[PSD] + constr_map[ExpCone])
        inv_data[SCS.NEQ_CONSTR] = neq_constr
        inv_data[SCS.EQ_CONSTR] = zero_constr
        inv_data[s.DIMS] = data[s.DIMS]

        # Obtain A, b such that Ax + s = b, s lies in cone.
        #
        # Note that scs mandates that the cones MUST be ordered with
        # zero cones first, then non-nonnegative orthant, then SOC,
        # then PSD, then exponential.
        data[s.A], data[s.B] = self.group_coeff_offset(
            zero_constr + neq_constr, self.EXP_CONE_ORDER)
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
            eq_dual_vars = ConicSolver.get_dual_values(
                solution['y'][:inverse_data[s.DIMS]['f']],
                inverse_data[SCS.EQ_CONSTR])
            # TODO(akshayka): This is not entirely correct; logic
            # is needed for PSD constraints. See scs_intf.format_results.
            ineq_dual_vars = ConicSolver.get_dual_values(
                solution['y'][inverse_data[s.DIMS]['f']:],
                inverse_data[SCS.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status)

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
