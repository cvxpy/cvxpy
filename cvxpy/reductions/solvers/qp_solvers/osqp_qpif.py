import cvxpy.settings as s
from cvxpy.reductions.solvers import utilities
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np
import scipy.sparse as sp


class OSQP(QpSolver):
    """QP interface for the OSQP solver"""

    # Map of OSQP status to CVXPY status.
    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.OPTIMAL_INACCURATE,
                  -2: s.SOLVER_ERROR,           # Maxiter reached
                  -3: s.INFEASIBLE,
                  3: s.INFEASIBLE_INACCURATE,
                  -4: s.UNBOUNDED,
                  4: s.UNBOUNDED_INACCURATE,
                  -5: s.SOLVER_ERROR,           # Interrupted by user
                  -10: s.SOLVER_ERROR}          # Unsolved

    def name(self):
        return s.OSQP

    def import_solver(self):
        import osqp
        osqp

    def invert(self, solution, inverse_data):
        attr = {s.SOLVE_TIME: solution.info.run_time}

        # Map OSQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(solution.info.status_val, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution.info.obj_val
            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(solution.x))
            }
            dual_vars = utilities.get_dual_values(
                intf.DEFAULT_INTF.const_to_matrix(solution.y),
                utilities.extract_dual_value,
                inverse_data.sorted_constraints)
            attr[s.NUM_ITERS] = solution.info.iter
        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf
        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        import osqp
        P = data[s.P]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        data['full_A'] = A
        uA = np.concatenate((data[s.B], data[s.G]))
        data['u'] = uA
        lA = np.concatenate([data[s.B], -np.inf*np.ones(data[s.G].shape)])
        data['l'] = lA

        # Overwrite defaults eps_abs=eps_rel=1e-4, max_iter=4000
        solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-4)
        solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-4)
        solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)

        if solver_cache is not None and self.name() in solver_cache:
            # Use cached data.
            solver, old_data, results = solver_cache[self.name()]
            same_pattern = (P.shape == old_data[s.P].shape and
                            all(P.indptr == old_data[s.P].indptr) and
                            all(P.indices == old_data[s.P].indices)) and \
                           (A.shape == old_data['full_A'].shape and
                            all(A.indptr == old_data['full_A'].indptr) and
                            all(A.indices == old_data['full_A'].indices))
        else:
            same_pattern = False

        # If sparsity pattern differs need to do setup.
        if warm_start and same_pattern:
            new_args = {}
            for key in ['q', 'l', 'u']:
                if any(data[key] != old_data[key]):
                    new_args[key] = data[key]
            factorizing = False
            if any(P.data != old_data[s.P].data):
                P_triu = sp.triu(P).tocsc()
                new_args['Px'] = P_triu.data
                factorizing = True
            if any(A.data != old_data['full_A'].data):
                new_args['Ax'] = A.data
                factorizing = True

            if new_args:
                solver.update(**new_args)
            # Map OSQP statuses back to CVXPY statuses
            status = self.STATUS_MAP.get(results.info.status_val, s.SOLVER_ERROR)
            if status == s.OPTIMAL:
                solver.warm_start(results.x, results.y)
            # Polish if factorizing.
            solver_opts['polish'] = solver_opts.get('polish', factorizing)
            solver.update_settings(verbose=verbose, **solver_opts)
        else:
            # Initialize and solve problem
            solver_opts['polish'] = solver_opts.get('polish', True)
            solver = osqp.OSQP()
            solver.setup(P, q, A, lA, uA, verbose=verbose, **solver_opts)

        results = solver.solve()

        if solver_cache is not None:
            solver_cache[self.name()] = (solver, data, results)
        return results
