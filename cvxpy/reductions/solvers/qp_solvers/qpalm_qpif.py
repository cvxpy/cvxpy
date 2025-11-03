"""
Copyright, the CVXPY authors

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
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


class QPALM(QpSolver):
    """QP interface for the QPALM solver"""

    MIP_CAPABLE = False

    def name(self):
        return s.QPALM

    def import_solver(self) -> None:
        import qpalm
        qpalm

    def invert(self, solution, inverse_data, options):
        import qpalm

        # Map of QPALM status to CVXPY status.
        STATUS_MAP = {
            qpalm.Info.SOLVED: s.OPTIMAL,
            qpalm.Info.PRIMAL_INFEASIBLE: s.INFEASIBLE,
            qpalm.Info.DUAL_INFEASIBLE: s.UNBOUNDED,
            qpalm.Info.MAX_ITER_REACHED: s.USER_LIMIT,
            qpalm.Info.TIME_LIMIT_REACHED: s.USER_LIMIT,
            qpalm.Info.UNSOLVED: s.SOLVER_ERROR,
            qpalm.Info.ERROR: s.SOLVER_ERROR,
        }

        # Map QPALM statuses back to CVXPY statuses
        status = STATUS_MAP.get(solution.info.status_val, s.SOLVER_ERROR)

        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = {"info": solution.info, "solver": solution}

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.objective + inverse_data[s.OFFSET]
            primal_vars = {
                QPALM.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(solution.solution.x.copy())
            }
            dual_vars = {
                QPALM.DUAL_VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(solution.solution.y.copy())
            }
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import qpalm

        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        b_max = np.concatenate((data[s.B], data[s.G]))
        b_min = np.concatenate([data[s.B], -np.inf * np.ones_like(data[s.G])])
        n_con, n_var = A.shape

        qp_data = qpalm.Data(n_var, n_con)
        qp_data.Q = sp.triu(P).tocsc()
        qp_data.q = q
        qp_data.A = A
        qp_data.bmin = b_min
        qp_data.bmax = b_max

        settings = qpalm.Settings()
        # Chosen to match PIQP's default tolerances:
        # https://github.com/PREDICT-EPFL/piqp/blob/5115f0c08b86de40aff90f7f717956f0a573c627/include/piqp/settings.hpp#L48-L49
        settings.eps_abs = 1e-8
        settings.eps_rel = 1e-9
        # By default, QPALM is a bit too eager in declaring infeasibility when
        # decreasing eps_{abs,rel}, so also decrease the feasibility tolerances
        settings.eps_dual_inf = 1e-8
        settings.eps_prim_inf = 1e-8
        settings.verbose = verbose
        for k, v in solver_opts.items():
            try:
                setattr(settings, k, v)
            except TypeError as e:
                raise TypeError(f"QPALM: Incorrect type for setting '{k}'.") from e
            except AttributeError as e:
                raise TypeError(f"QPALM: Unrecognized solver setting '{k}'.") from e

        if warm_start and self.name() in solver_cache:
            solver, old_data = solver_cache[self.name()]
            def sp_neq(a, b):
                return a.data.shape != b.data.shape or any(a.data != b.data)

            if sp_neq(old_data.Q, qp_data.Q) or sp_neq(old_data.A, qp_data.A):
                solver.update_Q_A(qp_data.Q.data, qp_data.A.data)
            if (old_data.q != qp_data.q).any():
                solver.update_q(qp_data.q)
            if (old_data.bmin != qp_data.bmin).any() or (old_data.bmax != qp_data.bmax).any():
                solver.update_bounds(bmin=qp_data.bmin, bmax=qp_data.bmax)
            solver.update_settings(settings)
            solver.warm_start(solver.solution.x, solver.solution.y)
        else:
            solver = qpalm.Solver(qp_data, settings)
        solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = solver, qp_data

        return solver

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["QPALM"]
