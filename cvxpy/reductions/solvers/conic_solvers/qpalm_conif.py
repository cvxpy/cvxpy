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
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.affine_qp_mixin import AffineQpMixin
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class QPALM(AffineQpMixin, ConicSolver):
    """Conic interface for the QPALM solver.

    QPALM is a QP solver that handles quadratic objectives with affine constraints.
    This conic interface allows QPALM to be used through the standard conic pathway.
    """

    MIP_CAPABLE = False
    REQUIRES_CONSTR = False

    def name(self):
        return s.QPALM

    def import_solver(self) -> None:
        import qpalm
        qpalm

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import qpalm

        # Convert conic format to QP format
        cone_dims = data[self.DIMS]
        qp_data = self.conic_to_qp_format(data, cone_dims)

        P = qp_data[s.P] if s.P in qp_data else sp.csc_array((0, 0))
        q = qp_data[s.Q]
        A = sp.vstack([qp_data[s.A], qp_data[s.F]]).tocsc()
        b_max = np.concatenate((qp_data[s.B], qp_data[s.G]))
        b_min = np.concatenate([qp_data[s.B], -np.inf * np.ones_like(qp_data[s.G])])
        n_con, n_var = A.shape

        qpalm_data = qpalm.Data(n_var, n_con)
        qpalm_data.Q = sp.triu(P).tocsc()
        qpalm_data.q = q
        qpalm_data.A = A
        qpalm_data.bmin = b_min
        qpalm_data.bmax = b_max

        settings = qpalm.Settings()
        # Chosen to match PIQP's default tolerances
        settings.eps_abs = 1e-8
        settings.eps_rel = 1e-9
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

        if warm_start and solver_cache is not None and self.name() in solver_cache:
            solver, old_qpalm_data = solver_cache[self.name()]

            def sp_neq(a, b):
                return a.data.shape != b.data.shape or any(a.data != b.data)

            if sp_neq(old_qpalm_data.Q, qpalm_data.Q) or sp_neq(old_qpalm_data.A, qpalm_data.A):
                solver.update_Q_A(qpalm_data.Q.data, qpalm_data.A.data)
            if (old_qpalm_data.q != qpalm_data.q).any():
                solver.update_q(qpalm_data.q)
            if ((old_qpalm_data.bmin != qpalm_data.bmin).any() or
                    (old_qpalm_data.bmax != qpalm_data.bmax).any()):
                solver.update_bounds(bmin=qpalm_data.bmin, bmax=qpalm_data.bmax)
            solver.update_settings(settings)
            solver.warm_start(solver.solution.x, solver.solution.y)
        else:
            solver = qpalm.Solver(qpalm_data, settings)

        solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = solver, qpalm_data

        return solver

    def invert(self, solution, inverse_data):
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
                inverse_data[self.VAR_ID]:
                intf.DEFAULT_INTF.const_to_matrix(solution.solution.x.copy())
            }
            # Build dual vars dict keyed by constraint IDs
            # QPALM returns duals for [eq_constrs; ineq_constrs]
            y = solution.solution.y.copy()
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
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def cite(self, data):
        return CITATION_DICT["QPALM"]
