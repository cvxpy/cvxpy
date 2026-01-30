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

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.affine_qp_mixin import AffineQpMixin
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class PROXQP(AffineQpMixin, ConicSolver):
    """Conic interface for the PROXQP solver.

    PROXQP is a QP solver that handles quadratic objectives with affine constraints.
    This conic interface allows PROXQP to be used through the standard conic pathway.
    """

    MIP_CAPABLE = False
    REQUIRES_CONSTR = False

    # Map of PROXQP status to CVXPY status.
    STATUS_MAP = {
        "PROXQP_SOLVED": s.OPTIMAL,
        "PROXQP_MAX_ITER_REACHED": s.USER_LIMIT,
        "PROXQP_PRIMAL_INFEASIBLE": s.INFEASIBLE,
        "PROXQP_DUAL_INFEASIBLE": s.UNBOUNDED,
    }

    # Variable name mapping for PROXQP API
    VAR_MAP = {
        "P": "H",
        "q": "g",
        "A": "A",
        "b": "b",
        "F": "C",
        "lb": "lb",
        "G": "ub",
    }

    def name(self):
        return s.PROXQP

    def import_solver(self) -> None:
        import proxsuite
        proxsuite

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import proxsuite

        # Convert conic format to QP format
        cone_dims = data[self.DIMS]
        qp_data = self.conic_to_qp_format(data, cone_dims)

        solver_opts = solver_opts.copy()
        solver_opts['backend'] = solver_opts.get('backend', 'dense')
        backend = solver_opts['backend']

        if backend == "dense":
            P = qp_data[s.P].toarray() if s.P in qp_data else None
            A = qp_data[s.A].toarray()
            F = qp_data[s.F].toarray()
        elif backend == "sparse":
            P = qp_data[s.P] if s.P in qp_data else None
            A = qp_data[s.A]
            F = qp_data[s.F]
        else:
            raise ValueError("Wrong input, backend must be either dense or sparse")

        q = qp_data[s.Q]
        b = qp_data[s.B]
        g = qp_data[s.G]
        lb = -np.inf * np.ones(g.shape)

        n_var = A.shape[1] if A.shape[0] > 0 else (F.shape[1] if F.shape[0] > 0 else 0)
        n_eq = A.shape[0]
        n_ineq = F.shape[0]

        # Store for caching
        cache_data = {
            s.P: P, s.A: A, s.F: F,
            s.Q: q, s.B: b, s.G: g, 'lb': lb,
        }

        # Overwrite default values
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-8)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 0.)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)
        solver_opts['rho'] = solver_opts.get('rho', 1e-6)
        solver_opts['mu_eq'] = solver_opts.get('mu_eq', 1e-3)
        solver_opts['mu_in'] = solver_opts.get('mu_in', 1e-1)

        if warm_start and solver_cache is not None and self.name() in solver_cache:
            solver, old_data, results = solver_cache[self.name()]
            new_args = {}

            for key in ['q', 'b', 'G', 'lb']:
                if key == 'q':
                    cache_key = s.Q
                elif key == 'b':
                    cache_key = s.B
                elif key == 'G':
                    cache_key = s.G
                else:
                    cache_key = 'lb'
                if any(cache_data[cache_key] != old_data[cache_key]):
                    new_args[self.VAR_MAP[key]] = cache_data[cache_key]

            if P is not None and old_data[s.P] is not None:
                if P.data.shape != old_data[s.P].data.shape or any(
                        P.data != old_data[s.P].data):
                    new_args['H'] = P
            if A.data.shape != old_data[s.A].data.shape or any(
                    A.data != old_data[s.A].data):
                new_args['A'] = A
            if F.data.shape != old_data[s.F].data.shape or any(
                    F.data != old_data[s.F].data):
                new_args['C'] = F

            if new_args:
                solver.update(**new_args)

            status = self.STATUS_MAP.get(results.info.status.name, s.SOLVER_ERROR)

            if status == s.OPTIMAL:
                solver.solve(results.x, results.y, results.z)
            else:
                solver.solve()
        else:
            if backend == "dense":
                solver = proxsuite.proxqp.dense.QP(n_var, n_eq, n_ineq)
            elif backend == "sparse":
                solver = proxsuite.proxqp.sparse.QP(n_var, n_eq, n_ineq)

            solver.init(
                H=P,
                g=q,
                A=A,
                b=b,
                C=F,
                l=lb,
                u=g,
                rho=solver_opts['rho'],
                mu_eq=solver_opts['mu_eq'],
                mu_in=solver_opts['mu_in'],
            )

            solver.settings.eps_abs = solver_opts['eps_abs']
            solver.settings.eps_rel = solver_opts['eps_rel']
            solver.settings.max_iter = solver_opts['max_iter']
            solver.settings.verbose = verbose

            solver.solve()

        results = solver.results

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, cache_data, results)

        return results

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"solution": solution}

        # Map PROXQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(solution.info.status.name, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.objValue + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }

            # Build dual vars dict keyed by constraint IDs
            # PROXQP returns solution.y (eq_duals) and solution.z (ineq_duals)
            eq_dual = utilities.get_dual_values(
                solution.y,
                utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR])
            ineq_dual = utilities.get_dual_values(
                solution.z,
                utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR])
            dual_vars = {}
            dual_vars.update(eq_dual)
            dual_vars.update(ineq_dual)
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def cite(self, data):
        return CITATION_DICT["PROXQP"]
