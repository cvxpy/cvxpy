import numpy as np
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


class OSQP(QpSolver):
    """QP interface for the OSQP solver"""

    # Map of OSQP status to CVXPY status.
    # Note: Status map has changed in versions >= 1
    STATUS_MAP_PRE_V1 = {1: s.OPTIMAL,
                  2: s.OPTIMAL_INACCURATE,
                  -2: s.USER_LIMIT,           # Maxiter reached
                  -3: s.INFEASIBLE,
                  3: s.INFEASIBLE_INACCURATE,
                  -4: s.UNBOUNDED,
                  4: s.UNBOUNDED_INACCURATE,
                  -6: s.USER_LIMIT,
                  -5: s.SOLVER_ERROR,           # Interrupted by user
                  -10: s.SOLVER_ERROR}          # Unsolved
    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.OPTIMAL_INACCURATE,
                  3: s.INFEASIBLE,
                  4: s.INFEASIBLE_INACCURATE,
                  5: s.UNBOUNDED,
                  6: s.UNBOUNDED_INACCURATE,
                  7: s.USER_LIMIT,           # Maxiter reached
                  8: s.USER_LIMIT,
                  10: s.SOLVER_ERROR,          # Interrupted by user
                  11: s.SOLVER_ERROR}          # Unsolved

    def name(self):
        return s.OSQP

    def import_solver(self) -> None:
        import osqp
        osqp

    def invert(self, solution, inverse_data):
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
                OSQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }
            dual_vars = {OSQP.DUAL_VAR_ID: solution.y}
            attr[s.NUM_ITERS] = solution.info.iter
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        import osqp
        is_pre_v1 = float(osqp.__version__.split('.')[0]) < 1
        
        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        data['Ax'] = A
        uA = np.concatenate((data[s.B], data[s.G]))
        data['u'] = uA
        lA = np.concatenate([data[s.B], -np.inf*np.ones(data[s.G].shape)])
        data['l'] = lA

        if P is not None:
            P = sp.csc_matrix((P.data, P.indices, P.indptr), shape=P.shape)
        if A is not None:
            A = sp.csc_matrix((A.data, A.indices, A.indptr), shape=A.shape)

        # Overwrite defaults eps_abs=eps_rel=1e-3, max_iter=4000
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-5)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-5)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)

        # Use cached data
        if warm_start and solver_cache is not None and self.name() in solver_cache:
            solver, old_data, results = solver_cache[self.name()]
            new_args = {}
            for key in ['q', 'l', 'u']:
                if any(data[key] != old_data[key]):
                    new_args[key] = data[key]
            factorizing = False
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
            solver_cache[self.name()] = (solver, data, results)
        return results

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["OSQP"]