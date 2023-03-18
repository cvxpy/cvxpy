"""
Copyright 2013 Steven Diamond

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


import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import ExpCone
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import (
    ECOS,
    dims_to_solver_dict,
)


class ECOS_BB(ECOS):
    """An interface for the ECOS BB solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = True
    MI_SUPPORTED_CONSTRAINTS = ECOS.SUPPORTED_CONSTRAINTS

    # Exit flags from ECOS_BB
    # ECOS_BB found optimal solution.
    # MI_OPTIMAL_SOLN (ECOS_OPTIMAL)
    # ECOS_BB proved problem is infeasible.
    # MI_INFEASIBLE (ECOS_PINF)
    # ECOS_BB proved problem is unbounded.
    # MI_UNBOUNDED (ECOS_DINF)
    # ECOS_BB hit maximum iterations but a feasible solution was found and
    # the best seen feasible solution was returned.
    # MI_MAXITER_FEASIBLE_SOLN (ECOS_OPTIMAL + ECOS_INACC_OFFSET)
    # ECOS_BB hit maximum iterations without finding a feasible solution.
    # MI_MAXITER_NO_SOLN (ECOS_PINF + ECOS_INACC_OFFSET)
    # ECOS_BB hit maximum iterations without finding a feasible solution
    #   that was unbounded.
    # MI_MAXITER_UNBOUNDED (ECOS_DINF + ECOS_INACC_OFFSET)

    def name(self):
        """The name of the solver.
        """
        return s.ECOS_BB

    def apply(self, problem):
        data, inv_data = super(ECOS_BB, self).apply(problem)
        # Because the problem variable is single dimensional, every
        # boolean/integer index has length one.
        var = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in var.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in var.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]
        return data, inv_data

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
            dual_vars = None
            if not inverse_data['is_mip']:
                dual_vars = utilities.get_dual_values(
                    solution['z'],
                    utilities.extract_dual_value,
                    inverse_data[self.NEQ_CONSTR]
                )
                for con in inverse_data[self.NEQ_CONSTR]:
                    if isinstance(con, ExpCone):
                        cid = con.id
                        n_cones = con.num_cones()
                        perm = utilities.expcone_permutor(n_cones,
                                                          ECOS.EXP_CONE_ORDER)
                        dual_vars[cid] = dual_vars[cid][perm]
                eq_duals = utilities.get_dual_values(
                    solution['y'],
                    utilities.extract_dual_value,
                    inverse_data[self.EQ_CONSTR]
                )
                dual_vars.update(eq_duals)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import ecos
        cones = dims_to_solver_dict(data[ConicSolver.DIMS])
        # Default verbose to false for BB wrapper.
        if 'mi_verbose' in solver_opts:
            mi_verbose = solver_opts['mi_verbose']
            del solver_opts['mi_verbose']
        else:
            mi_verbose = verbose
        solution = ecos.solve(data[s.C], data[s.G], data[s.H],
                              cones, data[s.A], data[s.B],
                              verbose=verbose,
                              mi_verbose=mi_verbose,
                              bool_vars_idx=data[s.BOOL_IDX],
                              int_vars_idx=data[s.INT_IDX],
                              **solver_opts)
        return solution
