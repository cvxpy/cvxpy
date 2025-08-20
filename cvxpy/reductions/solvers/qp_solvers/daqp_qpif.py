from ctypes import c_double, c_int

import numpy as np

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


class DAQP(QpSolver):
    """QP interface for the DAQP solver.
    
    This is a simple implementation based on the `official DAQP documentation
    <https://darnstrom.github.io/daqp/>`_.

    .. note::

        DAQP allows for marking individual constraints as "soft", meaning that
        a small user-defined violation is allowed. We currently don't support
        that option. DAQP also supports MIP QPs, which can be compiled by
        CVXPY and could be supported here as a future addition.

    """

    REQUIRES_CONSTR = False
    BOUNDED_VARIABLES = True

    # Map of DAQP exit flags to CVXPY status.
    STATUS_MAP = {1: s.OPTIMAL,
                  2: s.OPTIMAL, # soft optimal, returned if soft constraints
                                # enabled through solver opts (which we don't
                                # currently support)
                  -2: s.SOLVER_ERROR, # Cycling detected (shouldn't happen?)
                  -1: s.INFEASIBLE,
                  -3: s.UNBOUNDED,
                  # TODO to be tested
                  -4: s.USER_LIMIT, # iter limit reached
                  -5: s.SOLVER_ERROR, # non-convex problem, shouldn't happen
                  -6: s.INFEASIBLE} # provided active set (equality
                                    # constraints) infeasible

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

        attr = {
            s.SOLVE_TIME: info['solve_time'], 
            s.SETUP_TIME: info['setup_time'],
        }
        attr[s.EXTRA_STATS] = info

        # Map DAQP statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(exitflag, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = fval + inverse_data[s.OFFSET]
            primal_vars = {
                DAQP.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(xstar))
            }
            # dual variables associated with var bounds: TODO
            len_primal = len(xstar)
            dual_vars = {DAQP.DUAL_VAR_ID: np.array(info['lam'][len_primal:])}
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

        # this can probably be made more efficient
        A = np.array(np.concatenate(
                [data[s.A].todense(), data[s.F].todense()]), dtype=c_double)

        # Upper bounds on problem variable.
        if data[s.UPPER_BOUNDS] is None:
            var_upper_bounds = np.ones(len(f), dtype=c_double) * np.inf
        else:
            var_upper_bounds = data[s.UPPER_BOUNDS]
                    
        bupper = np.array(np.concatenate((
                var_upper_bounds,
                data[s.B], 
                data[s.G])),
            dtype=c_double)

        # Lower bounds on problem variable.
        if data[s.LOWER_BOUNDS] is None:
            var_lower_bounds = np.ones(len(f), dtype=c_double) * -np.inf
        else:
            var_lower_bounds = data[s.LOWER_BOUNDS]

        blower = np.array(
            np.concatenate((
                var_lower_bounds,
                data[s.B], 
                -np.inf*np.ones(data[s.G].shape))),
            dtype=c_double)

        sense = np.array(
            np.concatenate((
                # variable bounds, maybe unused
                np.zeros(len(f), dtype=c_int),
                # equality constraints, always active and immutable
                np.ones(len(data[s.B]), dtype=c_int)*5,
                # inequality constraints
                np.zeros(len(data[s.G]), dtype=c_int))),
            dtype=c_int
        )

        # according to https://stackoverflow.com/questions/16266720/find-out-if-a-matrix-is-positive-definite-with-numpy
        # best way to check positive definiteness (since we already know it is symmetric)

        if 'eps_prox' not in solver_opts:
            try:
                np.linalg.cholesky(H)
                is_positive_definite = True
            except np.linalg.LinAlgError:
                is_positive_definite = False
        else:
            is_positive_definite = False # shouldn't be used

        # Overwrite defaults eps_prox
        solver_opts['eps_prox'] = solver_opts.get('eps_prox', 0. if is_positive_definite else 1e-5)
        # This is chosen by benchmarking to a typical problem class
        # https://gist.github.com/enzbus/3d0236c7ed93cff5f0ec3f8587bcd67e
        # Zero (which is default) can't be used in most cases because Cvxpy's
        # canonicalization may add variables that have no quadratic cost.

        used_solver_opts = {
            k:solver_opts[k] for k in solver_opts
            if k in self.ALLOWED_SOLVER_OPTS}

        (xstar,fval,exitflag,info) = daqp.solve(
            H,f,A,bupper,blower,sense,**used_solver_opts)

        return (xstar,fval,exitflag,info)

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["DAQP"]