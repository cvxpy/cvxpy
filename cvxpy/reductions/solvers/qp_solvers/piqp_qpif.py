"""
Copyright 2023, the CVXPY Authors

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


class PIQP(QpSolver):
    """QP interface for the PIQP solver"""

    MIP_CAPABLE = False

    # Map of PIQP status to CVXPY status.
    STATUS_MAP = {"PIQP_SOLVED": s.OPTIMAL,
                  "PIQP_MAX_ITER_REACHED": s.USER_LIMIT,
                  "PIQP_PRIMAL_INFEASIBLE": s.INFEASIBLE,
                  "PIQP_DUAL_INFEASIBLE": s.UNBOUNDED}

    def name(self):
        return s.PIQP

    def import_solver(self) -> None:
        import piqp
        piqp

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"solution": solution}

        # Map PIQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(solution.info.status.name, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.primal_obj + inverse_data[s.OFFSET]
            primal_vars = {
                PIQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }

            dual_vars = {PIQP.DUAL_VAR_ID: np.concatenate(
                (solution.y, solution.z))}
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import piqp

        solver_opts = solver_opts.copy()

        solver_opts['backend'] = solver_opts.get('backend', 'sparse')
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

        if backend == "dense":
            solver = piqp.DenseSolver()
        elif backend == "sparse":
            solver = piqp.SparseSolver()

        del solver_opts['backend']
        for opt in solver_opts.keys():
            try:
                solver.settings.__setattr__(opt, solver_opts[opt])
            except TypeError as e:
                raise TypeError(f"PIQP: Incorrect type for setting '{opt}'.") from e
            except AttributeError as e:
                raise TypeError(f"PIQP: unrecognized solver setting '{opt}'.") from e
        solver.settings.verbose = verbose

        solver.setup(P=P,
                     c=q,
                     A=A,
                     b=b,
                     G=F,
                     h=g)

        solver.solve()

        result = solver.result

        if solver_cache is not None:
            solver_cache[self.name()] = result

        return result
