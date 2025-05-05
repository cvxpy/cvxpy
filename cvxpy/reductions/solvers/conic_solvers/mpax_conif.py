"""
Copyright 2025, the CVXPY Authors

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
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class MPAX(ConicSolver):
    """An interface for the MPAX solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    BOUNDED_VARIABLES = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS

    STATUS_MAP = {
                    # 1: UNSPECIFIED
                    2: s.OPTIMAL,      # 2: OPTIMAL
                    3: s.INFEASIBLE_OR_UNBOUNDED,   # 3: PRIMAL_INFEASIBLE
                    4: s.INFEASIBLE_OR_UNBOUNDED,    # 4: DUAL_INFEASIBLE
                    5: s.USER_LIMIT,   # 5: TIME_LIMIT
                    6: s.USER_LIMIT,   # 6: ITERATION_LIMIT
                    7: s.SOLVER_ERROR, # 7: NUMERICAL_ERROR
                    8: s.SOLVER_ERROR, # 8# INVALID_PROBLEM
                    # 9# OTHER
                }

    def name(self):
        """The name of the solver.
        """
        return 'MPAX'

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import jax  # noqa F401
        import mpax  # noqa F401

    def supports_quad_obj(self) -> bool:
        """MPAX supports quadratic objective with any combination
        of conic constraints.
        """
        return True

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """

        attr = {}
        status = self.STATUS_MAP[int(solution.termination_status)]
        attr[s.NUM_ITERS] = solution.iteration_count

        if status in s.SOLUTION_PRESENT:
            opt_val = float(solution.primal_objective)
            primal_vars = {
                inverse_data[MPAX.VAR_ID]: solution.primal_solution.tolist()
            }
            eq_dual_vars = utilities.get_dual_values(
                (solution.dual_solution).tolist()[:inverse_data[ConicSolver.DIMS].zero],
                utilities.extract_dual_value,
                inverse_data[MPAX.EQ_CONSTR]
            )
            ineq_dual_vars = utilities.get_dual_values(
                solution.dual_solution.tolist()[inverse_data[ConicSolver.DIMS].zero:],
                utilities.extract_dual_value,
                inverse_data[MPAX.NEQ_CONSTR]
            )
            dual_vars = {}
            dual_vars.update(eq_dual_vars)
            dual_vars.update(ineq_dual_vars)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)


    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        warm_start : Bool
            Whether to warm_start mpax.
        verbose : Bool
            Control the verbosity.
        solver_opts : dict
            mpax-specific solver options.

        Returns
        -------
        The result returned by a call to clarabel.solve().
        """
        import jax
        import mpax

        A = data[s.A]
        b = data[s.B]
        c = data[s.C]

        lb = data[s.LOWER_BOUNDS]
        if lb is None:
            lb = np.full_like(c, -np.inf)
        ub = data[s.UPPER_BOUNDS]
        if ub is None:
            ub = np.full_like(c, np.inf)

        num_eq = data[ConicSolver.DIMS].zero
        G = -A[num_eq:]
        h = -b[num_eq:]
        A = -A[:num_eq]
        b = -b[:num_eq]

        if s.P in data:
            P = data[s.P] # TODO: check if P is symmetric.
            model = mpax.create_qp(P, c, A, b, G, h, lb, ub)
        else:
            model = mpax.create_lp(c, A, b, G, h, lb, ub)

        # solver = mpax.raPDHG(warm_start=warm_start, verbose=verbose, **solver_opts)
        solver = mpax.raPDHG(warm_start=False, verbose=verbose, **solver_opts)

        jit_optimize = jax.jit(solver.optimize)

        if warm_start and solver_cache is not None and \
            self.name() in solver_cache:
            initial_primal_solution = solver_cache[self.name()].primal_solution
            initial_dual_solution = solver_cache[self.name()].dual_solution
            results = jit_optimize(model,
                                   initial_primal_solution=initial_primal_solution,
                                   initial_dual_solution=initial_dual_solution)
        else:
            initial_primal_solution = None
            initial_dual_solution = None
            results = jit_optimize(model)

        if solver_cache is not None:
            solver_cache[self.name()] = results

        return results


    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["MPAX"]
