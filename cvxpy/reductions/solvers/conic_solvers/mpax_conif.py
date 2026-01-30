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
from cvxpy.reductions.solvers.conic_solvers.affine_qp_mixin import AffineQpMixin
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class MPAX(AffineQpMixin, ConicSolver):
    """Conic interface for the MPAX solver.

    MPAX is a JAX-based QP solver that handles quadratic objectives with affine
    constraints. This conic interface allows MPAX to be used through the standard
    conic pathway.
    """

    MIP_CAPABLE = False
    REQUIRES_CONSTR = False

    STATUS_MAP = {
        1: s.SOLVER_ERROR,  # UNSPECIFIED
        2: s.OPTIMAL,       # OPTIMAL
        3: s.INFEASIBLE_OR_UNBOUNDED,  # PRIMAL_INFEASIBLE
        4: s.INFEASIBLE_OR_UNBOUNDED,  # DUAL_INFEASIBLE
        5: s.USER_LIMIT,    # TIME_LIMIT
        6: s.USER_LIMIT,    # ITERATION_LIMIT
        7: s.SOLVER_ERROR,  # NUMERICAL_ERROR
        8: s.SOLVER_ERROR,  # INVALID_PROBLEM
        9: s.SOLVER_ERROR,  # OTHER
    }

    def name(self):
        return s.MPAX

    def import_solver(self) -> None:
        import jax  # noqa F401
        import mpax  # noqa F401

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import jax
        import mpax

        # Convert conic format to QP format
        cone_dims = data[self.DIMS]
        qp_data = self.conic_to_qp_format(data, cone_dims)

        P = qp_data[s.P] if s.P in qp_data else None
        c = qp_data[s.Q]
        A = qp_data[s.A]
        b = qp_data[s.B]
        G = qp_data[s.F]
        h = qp_data[s.G]

        # No variable bounds in conic interface
        lb = np.full_like(c, -np.inf)
        ub = np.full_like(c, np.inf)

        solver_opts = solver_opts.copy()
        if P is not None and P.nnz != 0:
            model = mpax.create_qp(P, c, A, b, G, h, lb, ub)
        else:
            model = mpax.create_lp(c, A, b, G, h, lb, ub)

        algorithm = solver_opts.pop('algorithm', None)
        if algorithm is None or algorithm == 'raPDHG':
            alg = mpax.raPDHG
        elif algorithm == 'r2HPDHG':
            alg = mpax.r2HPDHG
        else:
            raise ValueError('Invalid MPAX algorithm')

        if warm_start and solver_cache is not None and \
                self.name() in solver_cache:
            solver = alg(warm_start=True, verbose=verbose, **solver_opts)
            jit_optimize = jax.jit(solver.optimize)
            initial_primal_solution = solver_cache[self.name()].primal_solution
            initial_dual_solution = solver_cache[self.name()].dual_solution
            results = jit_optimize(model,
                                   initial_primal_solution=initial_primal_solution,
                                   initial_dual_solution=initial_dual_solution)
        else:
            solver = alg(warm_start=False, verbose=verbose, **solver_opts)
            jit_optimize = jax.jit(solver.optimize)
            results = jit_optimize(model)

        if solver_cache is not None:
            solver_cache[self.name()] = results

        return results

    def invert(self, solution, inverse_data):
        attr = {}
        status = self.STATUS_MAP[int(solution.termination_status)]
        attr[s.NUM_ITERS] = solution.iteration_count

        if status in s.SOLUTION_PRESENT:
            opt_val = float(solution.primal_objective) + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]: np.array(solution.primal_solution, dtype=float)
            }
            # Build dual vars dict keyed by constraint IDs
            # MPAX returns dual_solution as [eq_duals; ineq_duals]
            y = np.array(solution.dual_solution, dtype=float)
            n_eq = inverse_data[self.DIMS].zero
            eq_dual = utilities.get_dual_values(
                y[:n_eq],
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR])
            ineq_dual = utilities.get_dual_values(
                y[n_eq:],
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual)
            dual_vars.update(ineq_dual)
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def cite(self, data):
        return CITATION_DICT["MPAX"]
