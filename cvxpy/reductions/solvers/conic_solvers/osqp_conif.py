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
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.affine_qp_mixin import AffineQpMixin
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.utilities.citations import CITATION_DICT


class OSQP(AffineQpMixin, ConicSolver):
    """Conic interface for the OSQP solver.

    OSQP is a QP solver that handles quadratic objectives with affine constraints.
    This conic interface allows OSQP to be used through the standard conic pathway.
    """

    # Map of OSQP status to CVXPY status.
    # Note: Status map has changed in versions >= 1
    STATUS_MAP_PRE_V1 = {
        1: s.OPTIMAL,
        2: s.OPTIMAL_INACCURATE,
        -2: s.USER_LIMIT,           # Maxiter reached
        -3: s.INFEASIBLE,
        3: s.INFEASIBLE_INACCURATE,
        -4: s.UNBOUNDED,
        4: s.UNBOUNDED_INACCURATE,
        -6: s.USER_LIMIT,
        -5: s.SOLVER_ERROR,         # Interrupted by user
        -10: s.SOLVER_ERROR,        # Unsolved
    }
    STATUS_MAP = {
        1: s.OPTIMAL,
        2: s.OPTIMAL_INACCURATE,
        3: s.INFEASIBLE,
        4: s.INFEASIBLE_INACCURATE,
        5: s.UNBOUNDED,
        6: s.UNBOUNDED_INACCURATE,
        7: s.USER_LIMIT,            # Maxiter reached
        8: s.USER_LIMIT,
        10: s.SOLVER_ERROR,         # Interrupted by user
        11: s.SOLVER_ERROR,         # Unsolved
    }

    # Solver capabilities
    MIP_CAPABLE = False
    REQUIRES_CONSTR = False

    def name(self):
        return s.OSQP

    def import_solver(self) -> None:
        import osqp
        osqp

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        """Returns the result of the call to OSQP solver.

        Parameters
        ----------
        data : dict
            Data generated via ConicSolver.apply().
        warm_start : bool
            Whether to warm start OSQP.
        verbose : bool
            Control verbosity.
        solver_opts : dict
            OSQP-specific solver options.
        solver_cache : dict, optional
            Cache for warm starting.

        Returns
        -------
        OSQP solution object.
        """
        import osqp
        is_pre_v1 = float(osqp.__version__.split('.')[0]) < 1

        # Convert conic format to QP format
        cone_dims = data[self.DIMS]
        qp_data = self.conic_to_qp_format(data, cone_dims)

        P = qp_data[s.P] if s.P in qp_data else None
        q = qp_data[s.Q]

        # Stack equality and inequality into OSQP format:
        # l <= Ax <= u
        # For equality: l = u = b_eq
        # For inequality: l = -inf, u = g
        A = sp.vstack([qp_data[s.A], qp_data[s.F]]).tocsc()
        uA = np.concatenate((qp_data[s.B], qp_data[s.G]))
        lA = np.concatenate([qp_data[s.B], -np.inf * np.ones(qp_data[s.G].shape)])

        # Store for caching
        cache_data = {
            'q': q, 'l': lA, 'u': uA,
            s.P: P, 'Ax': A,
        }

        if P is not None:
            P = sp.csc_matrix((P.data, P.indices, P.indptr), shape=P.shape)
        if A is not None:
            A = sp.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)

        # Overwrite defaults eps_abs=eps_rel=1e-3, max_iter=4000
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-5)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-5)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)

        # Use cached data for warm start
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            solver, old_data, results = solver_cache[self.name()]
            new_args = {}
            for key in ['q', 'l', 'u']:
                if any(cache_data[key] != old_data[key]):
                    new_args[key] = cache_data[key]
            factorizing = False
            if P is not None and old_data[s.P] is not None:
                if P.data.shape != old_data[s.P].data.shape or any(
                        P.data != old_data[s.P].data):
                    P_triu = sp.csc_array(sp.triu(P, format='csc'))
                    new_args['Px'] = P_triu.data
                    factorizing = True
            if A.data.shape != old_data['Ax'].data.shape or any(
                    A.data != old_data['Ax'].data):
                new_args['Ax'] = A.data
                factorizing = True

            if new_args:
                solver.update(**new_args)
            # Map OSQP statuses back to CVXPY statuses
            status_map = self.STATUS_MAP_PRE_V1 if is_pre_v1 else self.STATUS_MAP
            status = status_map.get(results.info.status_val, s.SOLVER_ERROR)
            if status == s.OPTIMAL:
                solver.warm_start(results.x, results.y)
            # Polish if factorizing.
            polish_param = 'polish' if is_pre_v1 else 'polishing'
            solver_opts[polish_param] = solver_opts.get(polish_param, factorizing)
            solver.update_settings(verbose=verbose, **solver_opts)
        else:
            # Initialize and solve problem
            polish_param = 'polish' if is_pre_v1 else 'polishing'
            solver_opts[polish_param] = solver_opts.get(polish_param, True)
            solver = osqp.OSQP()
            try:
                solver.setup(P, q, A, lA, uA, verbose=verbose, **solver_opts)
            except Exception as e:
                raise SolverError(e)

        results = solver.solve(raise_error=False)

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, cache_data, results)
        return results

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem.

        Parameters
        ----------
        solution : OSQP solution object
        inverse_data : dict
            Data for inverting the solution.

        Returns
        -------
        Solution object.
        """
        import osqp
        is_pre_v1 = float(osqp.__version__.split('.')[0]) < 1

        attr = {s.SOLVE_TIME: solution.info.run_time}
        attr[s.EXTRA_STATS] = solution

        # Map OSQP statuses back to CVXPY statuses
        status_map = self.STATUS_MAP_PRE_V1 if is_pre_v1 else self.STATUS_MAP
        status = status_map.get(solution.info.status_val, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.obj_val + inverse_data[s.OFFSET]
            primal_vars = {
                inverse_data[self.VAR_ID]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }
            # Build dual vars dict keyed by constraint IDs
            # OSQP returns y as [eq_duals; ineq_duals]
            y = solution.y
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
        """Returns bibtex citation for the solver."""
        return CITATION_DICT["OSQP"]
