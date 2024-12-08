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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import SOC, ExpCone, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


# Utility method for formatting a ConeDims instance into a dictionary
# that can be supplied to ecos.
def dims_to_solver_dict(cone_dims):
    cones = {
        'l': cone_dims.nonneg,
        "q": cone_dims.soc,
        'e': cone_dims.exp,
    }
    return cones


class ECOS(ConicSolver):
    """An interface for the ECOS solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC, ExpCone]

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

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import ecos  # noqa F401

    def name(self):
        """The name of the solver.
        """
        return s.ECOS

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        # Format constraints
        #
        # ECOS requires constraints to be specified in the following order:
        # 1. zero cone
        # 2. non-negative orthant
        # 3. soc
        # 4. exponential
        if not problem.formatted:
            problem = self.format_constraints(problem, self.EXP_CONE_ORDER)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + constr_map[ExpCone]
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
        import ecos
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])
        if data[s.A] is not None and data[s.A].nnz == 0 and np.prod(data[s.A].shape) > 0:
            raise ValueError(
                "ECOS cannot handle sparse data with nnz == 0; "
                "this is a bug in ECOS, and it indicates that your problem "
                "might have redundant constraints.")
        solution = ecos.solve(data[s.C], data[s.G], data[s.H],
                              cones, data[s.A], data[s.B],
                              verbose=verbose,
                              **solver_opts)
        return solution

    def invert(self, solution, inverse_data):
        """Returns solution to original problem, given inverse_data.
        """
        status = self.STATUS_MAP[solution['info']['exitFlag']]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = solution["info"]["timing"]["tsolve"]
        attr[s.SETUP_TIME] = solution["info"]["timing"]["tsetup"]
        attr[s.NUM_ITERS] = solution["info"]["iter"]
        attr[s.EXTRA_STATS] = solution

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pcost']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]: intf.DEFAULT_INTF.const_to_matrix(solution['x'])
            }
            dual_vars = utilities.get_dual_values(solution['z'],
                                                  utilities.extract_dual_value,
                                                  inverse_data[self.NEQ_CONSTR])
            for con in inverse_data[self.NEQ_CONSTR]:
                if isinstance(con, ExpCone):
                    cid = con.id
                    n_cones = con.num_cones()
                    perm = utilities.expcone_permutor(n_cones, ECOS.EXP_CONE_ORDER)
                    dual_vars[cid] = dual_vars[cid][perm]
            eq_duals = utilities.get_dual_values(solution['y'],
                                                 utilities.extract_dual_value,
                                                 inverse_data[self.EQ_CONSTR])
            dual_vars.update(eq_duals)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)
