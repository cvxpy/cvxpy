"""
Copyright 2013 Steven Diamond, Copyright 2017 Robin Verschueren

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
from cvxpy.constraints import SOC, ExpCone, NonPos, Zero
from cvxpy.reductions.solution import Solution

from .conic_solver import ConicSolver
from .reduction_solver import ReductionSolver


class ECOS(ReductionSolver):
    """An interface for the ECOS solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = True
    MIP_CAPABLE = False

    # EXITCODES from ECOS
    # ECOS_OPTIMAL  (0)   Problem solved to optimality
    # ECOS_PINF     (1)   Found certificate of primal infeasibility
    # ECOS_DINF     (2)   Found certificate of dual infeasibility
    # ECOS_INACC_OFFSET (10)  Offset exitflag at inaccurate results
    # ECOS_MAXIT    (-1)  Maximum number of iterations reached
    # ECOS_NUMERICS (-2)  Search direction unreliable
    # ECOS_OUTCONE  (-3)  s or z got outside the cone, numerics?
    # ECOS_SIGINT   (-4)  solver interrupted by a signal/ctrl-c
    # ECOS_FATAL    (-7)  Unknown problem in solver

    # Map of ECOS status to CVXPY status.
    STATUS_MAP = {0: s.OPTIMAL,
                  1: s.INFEASIBLE,
                  2: s.UNBOUNDED,
                  10: s.OPTIMAL_INACCURATE,
                  11: s.INFEASIBLE_INACCURATE,
                  12: s.UNBOUNDED_INACCURATE,
                  -1: s.SOLVER_ERROR,
                  -2: s.SOLVER_ERROR,
                  -3: s.SOLVER_ERROR,
                  -4: s.SOLVER_ERROR,
                  -7: s.SOLVER_ERROR}

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 2, 1]

    def import_solver(self):
        """Imports the solver.
        """
        import ecos
        ecos  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.ECOS

    def is_mat_stuffed(self, expr):
        """Returns whether the expression is reshape(A*x + b).
        """
        # TODO

    def accepts(self, problem):
        """Can ECOS solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos, SOC, ExpCone]:
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
        data[s.C], data[s.OFFSET] = ConicSolver.get_coeff_offset(problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[ECOS.EQ_CONSTR] = constr
        data[s.A], data[s.B] = ConicSolver.group_coeff_offset(constr, ECOS.EXP_CONE_ORDER)

        # Order and group nonlinear constraints.
        data[s.DIMS] = {}
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        data[s.DIMS]['l'] = sum([np.prod(c.size) for c in leq_constr])
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        data[s.DIMS]['q'] = [size for cons in soc_constr for size in cons.cone_sizes()]
        exp_constr = [c for c in problem.constraints if type(c) == ExpCone]
        data[s.DIMS]['e'] = sum([cons.num_cones() for cons in exp_constr])
        other_constr = leq_constr + soc_constr + exp_constr
        inv_data[ECOS.NEQ_CONSTR] = other_constr
        data[s.G], data[s.H] = ConicSolver.group_coeff_offset(other_constr, ECOS.EXP_CONE_ORDER)
        return data, inv_data

    def solve(self, problem, warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        ...

        Returns
        -------
        tuple
        ...
        """
        import ecos
        data, inv_data = self.apply(problem)
        solution = ecos.solve(data[s.C], data[s.G], data[s.H],
                              data[s.DIMS], data[s.A], data[s.B],
                              verbose=verbose,
                              **solver_opts)
        return self.invert(solution, inv_data)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution['info']['exitFlag']]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = solution["info"]["timing"]["tsolve"]
        attr[s.SETUP_TIME] = solution["info"]["timing"]["tsetup"]
        attr[s.NUM_ITERS] = solution["info"]["iter"]

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pcost']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[ECOS.VAR_ID]: solution['x']}
            eq_dual = ConicSolver.get_dual_values(solution['y'], inverse_data[ECOS.EQ_CONSTR])
            leq_dual = ConicSolver.get_dual_values(solution['z'], inverse_data[ECOS.NEQ_CONSTR])
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

        return Solution(status, opt_val, primal_vars, dual_vars, attr)
