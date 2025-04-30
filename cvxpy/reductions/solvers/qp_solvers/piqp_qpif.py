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
from cvxpy.utilities.citations import CITATION_DICT


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
                (solution.y, solution.z if hasattr(solution, 'z') else solution.z_u))}
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import piqp
        old_interface = float(piqp.__version__.split('.')[0]) == 0 \
                        and float(piqp.__version__.split('.')[1]) <= 5

        solver_opts = solver_opts.copy()

        solver_opts['backend'] = solver_opts.get('backend', 'sparse')
        backend = solver_opts['backend']
        del solver_opts['backend']

        if backend not in ['dense', 'sparse']:
            raise ValueError("Wrong input, backend must be either dense or sparse")

        def update_solver_settings(solver):
            for opt in solver_opts.keys():
                try:
                    solver.settings.__setattr__(opt, solver_opts[opt])
                except TypeError as e:
                    raise TypeError(f"PIQP: Incorrect type for setting '{opt}'.") from e
                except AttributeError as e:
                    raise TypeError(f"PIQP: Unrecognized solver setting '{opt}'.") from e
            solver.settings.verbose = verbose

        structure_changed = True
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            structure_changed = False

            solver, old_data, _ = solver_cache[self.name()]
            new_args = {}

            for key, param in [(s.Q, 'c'), (s.B, 'b'), (s.G, 'h' if old_interface else 'h_u')]:
                if any(data[key] != old_data[key]):
                    new_args[param] = data[key]

            if backend == 'sparse' and data[s.P].data.shape != old_data[s.P].data.shape:
                    structure_changed = True
            elif data[s.P].data.shape != old_data[s.P].data.shape or any(
                    data[s.P].data != old_data[s.P].data):
                new_args['P'] = data[s.P] if backend == 'sparse' else data[s.P].toarray()

            if backend == 'sparse' and data[s.A].data.shape != old_data[s.A].data.shape:
                    structure_changed = True
            elif data[s.A].data.shape != old_data[s.A].data.shape or any(
                    data[s.A].data != old_data[s.A].data):
                new_args['A'] = data[s.A] if backend == 'sparse' else data[s.A].toarray()

            if backend == 'sparse' and data[s.F].data.shape != old_data[s.F].data.shape:
                    structure_changed = True
            elif data[s.F].data.shape != old_data[s.F].data.shape or any(
                    data[s.F].data != old_data[s.F].data):
                new_args['G'] = data[s.F] if backend == 'sparse' else data[s.F].toarray()

            if backend == 'dense' and not isinstance(solver, piqp.DenseSolver):
                structure_changed = True
            if backend == 'sparse' and not isinstance(solver, piqp.SparseSolver):
                structure_changed = True

            update_solver_settings(solver)

            if not structure_changed and new_args:
                solver.update(**new_args)

        if structure_changed:
            if backend == 'dense':
                solver = piqp.DenseSolver()
            else:
                solver = piqp.SparseSolver()

            update_solver_settings(solver)

            if backend == 'dense':
                # Convert sparse to dense matrices
                P = data[s.P].toarray()
                A = data[s.A].toarray()
                F = data[s.F].toarray()
            else:
                P = data[s.P]
                A = data[s.A]
                F = data[s.F]

            q = data[s.Q]
            b = data[s.B]
            g = data[s.G]

            if old_interface:
                solver.setup(P=P, c=q, A=A, b=b, G=F, h=g)
            else:
                solver.setup(P=P, c=q, A=A, b=b, G=F, h_u=g)

        solver.solve()

        result = solver.result

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, result)

        return result

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["PIQP"]