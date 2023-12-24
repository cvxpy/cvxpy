from ctypes import c_double, c_int

import numpy as np
import scipy.sparse as sp

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.error import SolverError
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver


class DAQP(QpSolver):
    """QP interface for the DAQP solver.
    
    This is a simple implementation based on the official documentation.

    .. note::

        DAQP allows for marking individual constraints as "soft", meaning that
        a small user-defined violation is allowed. We currently don't support
        that option. DAQP also supports MIP-like problems, which we also don't
        compile here.

    """

    # TODO to be tested
    # REQUIRES_CONSTR = False

    # Map of DAQP exit flags to CVXPY status.
    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.OPTIMAL, # soft optimal, returned if soft constraints
                                # enabled through solver opts (which we don't
                                # currently support)
                  -2: s.SOLVER_ERROR, # Cycling detected (shouldn't happen?)
                  -1: s.INFEASIBLE,
                  -3: s.UNBOUNDED,
                  # TODO to be tested
                  -4: s.USER_LIMIT, # iter limit reached, should it be error?
                  -5: s.SOLVER_ERROR, # non-convex problem, shouldn't happen
                  -6: s.SOLVER_ERROR} # active set infeasible, shouldn't happen

    # these are solver options that can be passed; we drop any other - good idea?
    ALLOWED_SOLVER_OPTS = (
        'primal_tol',
        'dual_tol',
        'zero_tol',
        'pivot_tol',
        'progress_tol',
        'cycle_tol',
        'iter_limit',
        'fval_bound',
        'eps_prox',
        'eta_prox'
        # ignore the ones about soft constraints and branch-and-bound for now
    )

    def name(self):
        return s.DAQP

    def import_solver(self) -> None:
        import daqp
        daqp

    def invert(self, solution, inverse_data):

        (xstar,fval,exitflag,info) = solution

        print((xstar,fval,exitflag,info))

        attr = {s.SOLVE_TIME: info['solve_time'] + info['setup_time']}
        attr[s.EXTRA_STATS] = info

        # Map DAQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(exitflag, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = fval + inverse_data[s.OFFSET]
            primal_vars = {
                DAQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(xstar))
            }
            dual_vars = {DAQP.DUAL_VAR_ID: np.array(info['lam'])}
            attr[s.NUM_ITERS] = info['iterations']
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts,
                       solver_cache=None):
        """Solve using DAQP.

        .. note::

            We transform from scipy sparse to numpy dense arrays here;
            if the canonicalization code can handle dense canonicalization
            natively it may be better to do so beforehand.

        """
        import daqp

        # using naming conventions in
        # https://darnstrom.github.io/daqp/start/python 

        H = np.array(data[s.P].todense(), dtype=c_double)
        f = np.array(data[s.Q], dtype=c_double)

        print('H.shape', H.shape)
        print('f.shape', f.shape)

        # this can probably be made more efficient
        A = np.array(np.concatenate(
                [data[s.A].todense(), data[s.F].todense()]), dtype=c_double)
        
        print('A.shape', A.shape)

        bupper = np.array(np.concatenate((
                np.ones(len(f), dtype=c_double) * np.inf,
                data[s.B], 
                data[s.G])),
            dtype=c_double)

        blower = np.array(
            np.concatenate((
                -np.inf * np.ones(len(f), dtype=c_double),
                data[s.B], 
                -np.inf*np.ones(data[s.G].shape))),
            dtype=c_double)

        print('bupper.shape', bupper.shape)
        print('blower.shape', blower.shape)

        sense = np.array(
            np.concatenate((
                # variable bounds, unused
                np.zeros(len(f), dtype=c_int),
                # equality constraints, always active and immutable
                np.ones(len(data[s.B]), dtype=c_int)*5,
                # inequality constraints
                np.zeros(len(data[s.G]), dtype=c_int))),
            dtype=c_int
        )

        print('sense', sense)

        # Overwrite defaults eps_abs=eps_rel=1e-3, max_iter=4000
        # solver_opts['eps_abs'] = solver_opts.get('eps_abs', 1e-5)
        # solver_opts['eps_rel'] = solver_opts.get('eps_rel', 1e-5)
        # solver_opts['max_iter'] = solver_opts.get('max_iter', 10000)

        used_solver_opts = {
            k:solver_opts[k] for k in solver_opts
            if k in self.ALLOWED_SOLVER_OPTS}

        (xstar,fval,exitflag,info) = daqp.solve(
            H,f,A,bupper,blower,sense,**used_solver_opts)

        return (xstar,fval,exitflag,info)
