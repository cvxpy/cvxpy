"""
Copyright 2022, the CVXPY Authors

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
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


class PROXQP(QpSolver):
    """QP interface for the PROXQP solver"""

    MIP_CAPABLE = False

    # Map of Proxqp status to CVXPY status.
    STATUS_MAP = {"PROXQP_SOLVED": s.OPTIMAL,
                  "PROXQP_MAX_ITER_REACHED": s.USER_LIMIT,
                  "PROXQP_PRIMAL_INFEASIBLE": s.INFEASIBLE,
                  "PROXQP_DUAL_INFEASIBLE": s.UNBOUNDED}

    VAR_MAP = {"P": "H",
               "q": "g",
               "A": "A",
               "b": "b",
               "F": "C",
               "l": "lb",
               "G": "ub"}

    def name(self):
        return s.PROXQP

    def import_solver(self) -> None:
        import proxsuite
        proxsuite

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"solution": solution}

        # Map PROXQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(solution.info.status.name, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.objValue + inverse_data[s.OFFSET]
            primal_vars = {
                PROXQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }

            dual_vars = {PROXQP.DUAL_VAR_ID: np.concatenate(
                (solution.y, solution.z))}
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import proxsuite

        solver_opts['backend'] = solver_opts.get('backend', 'dense')
        backend = solver_opts['backend']

        if backend == "dense":
            # Convert sparse to dense matrices
            P = data[s.P].toarray()
            A = data[s.A].toarray()
            F = data[s.F].toarray()
        elif backend == "sparse":
            P = data[s.P]
            A = data[s.A]
            F = data[s.F]
        else:
            raise ValueError("Wrong input, backend most be either dense or sparse")

        q = data[s.Q]
        b = data[s.B]
        g = data[s.G]

        lb = -np.inf*np.ones(data[s.G].shape)
        data['lb'] = lb

        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        # Overwrite default values
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-8)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 0.)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)
        solver_opts['rho'] = solver_opts.get('rho', 1e-6)
        solver_opts['mu_eq'] = solver_opts.get('mu_eq', 1e-3)
        solver_opts['mu_in'] = solver_opts.get('mu_in', 1e-1)
        # Use cached data
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            solver, old_data, results = solver_cache[self.name()]
            new_args = {}
            for key in ['q', 'b', 'G', 'lb']:
                if any(data[key] != old_data[key]):
                    new_args[self.VAR_MAP[key]] = data[key]
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

            solver.init(H=P,
                        g=q,
                        A=A,
                        b=b,
                        C=F,
                        l=lb,
                        u=g,
                        rho=solver_opts['rho'],
                        mu_eq=solver_opts['mu_eq'],
                        mu_in=solver_opts['mu_in'])

            solver.settings.eps_abs = solver_opts['eps_abs']
            solver.settings.eps_rel = solver_opts['eps_rel']
            solver.settings.max_iter = solver_opts['max_iter']
            solver.settings.verbose = verbose

            solver.solve()

        results = solver.results

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, results)

        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["PROXQP"]