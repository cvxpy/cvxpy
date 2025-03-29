"""
This file is the CVXPY QP extension of the Cardinal Optimizer
"""
import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.utilities.citations import CITATION_DICT


class COPT(QpSolver):
    """
    QP interface for the COPT solver
    """
    # Solve capabilities
    MIP_CAPABLE = True

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map between COPT status and CVXPY status
    STATUS_MAP = {
                  1: s.OPTIMAL,             # optimal
                  2: s.INFEASIBLE,          # infeasible
                  3: s.UNBOUNDED,           # unbounded
                  4: s.INF_OR_UNB,          # infeasible or unbounded
                  5: s.SOLVER_ERROR,        # numerical
                  6: s.USER_LIMIT,          # node limit
                  7: s.OPTIMAL_INACCURATE,  # imprecise
                  8: s.USER_LIMIT,          # time out
                  9: s.SOLVER_ERROR,        # unfinished
                  10: s.USER_LIMIT          # interrupted
                 }

    def name(self):
        """
        The name of solver.
        """
        return 'COPT'

    def import_solver(self):
        """
        Imports the solver.
        """
        import coptpy  # noqa F401

    def invert(self, solution, inverse_data):
        """
        Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]
        attr = {s.SOLVE_TIME: solution[s.SOLVE_TIME],
                s.NUM_ITERS: solution[s.NUM_ITERS],
                s.EXTRA_STATS: solution['model']}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[COPT.VAR_ID]: solution[s.PRIMAL]}
            if not inverse_data[COPT.IS_MIP]:
                dual_vars = {COPT.DUAL_VAR_ID: solution['y']}
            return Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            return failure_solution(status, attr)

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        """
        Returns the result of the call to the solver.

        Parameters
        ----------
        data : dict
            Data used by the solver.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.
        solver_cache: None
            None

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import coptpy as copt

        # Create COPT environment and model
        envconfig = copt.EnvrConfig()
        if not verbose:
            envconfig.set('nobanner', '1')

        env = copt.Envr(envconfig)
        model = env.createModel()

        # Pass through verbosity
        model.setParam(copt.COPT.Param.Logging, verbose)

        # Get the problem data from cvxpy
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        F = data[s.F]
        g = data[s.G]

        # Build COPT problem data
        n = data['n_var']

        if A.shape[0] > 0 and F.shape[0] == 0:
            Amat = A
            lhs = b
            rhs = b
        elif A.shape[0] == 0 and F.shape[0] > 0:
            Amat = F
            lhs = np.full(F.shape[0], -copt.COPT.INFINITY)
            rhs = g
        elif A.shape[0] > 0 and F.shape[0] > 0:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            lhs = np.hstack((b, np.full(F.shape[0], -copt.COPT.INFINITY)))
            rhs = np.hstack((b, g))
        else:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            lhs = None
            rhs = None

        lb = np.full(n, -copt.COPT.INFINITY)
        ub = np.full(n, +copt.COPT.INFINITY)

        vtype = None
        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            vtype = np.array([copt.COPT.CONTINUOUS] * n)
            if data[s.BOOL_IDX]:
                vtype[data[s.BOOL_IDX]] = copt.COPT.BINARY
                lb[data[s.BOOL_IDX]] = 0
                ub[data[s.BOOL_IDX]] = 1
            if data[s.INT_IDX]:
                vtype[data[s.INT_IDX]] = copt.COPT.INTEGER

        # Load matrix data
        # TODO remove `sp.csc_matrix` when COPT starts supporting sparray
        model.loadMatrix(q, sp.csc_matrix(Amat), lhs, rhs, lb, ub, vtype)

        # Load Q data
        if P.count_nonzero():
            # TODO switch to `P = P.tocoo()` when COPT supports sparray
            P = sp.coo_matrix(P)
            model.loadQ(0.5*P)

        # Set parameters
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        # Solve problem
        solution = {}
        try:
            model.solve()
            # Reoptimize if INF_OR_UNBD, to get definitive answer.
            if model.status == copt.COPT.INF_OR_UNB and solver_opts.get('reoptimize', True):
                model.setParam(copt.COPT.Param.Presolve, 0)
                model.solve()
            if model.hasmipsol:
                solution[s.VALUE] = model.objval
                solution[s.PRIMAL] = np.array(model.getValues())
            elif model.haslpsol:
                solution[s.VALUE] = model.objval
                solution[s.PRIMAL] = np.array(model.getValues())
                solution['y'] = -np.array(model.getDuals())
        except Exception:
            pass

        solution[s.SOLVE_TIME] = model.solvingtime
        solution[s.NUM_ITERS] = model.barrieriter + model.simplexiter

        solution[s.STATUS] = self.STATUS_MAP.get(model.status, s.SOLVER_ERROR)
        if solution[s.STATUS] == s.USER_LIMIT and model.hasmipsol:
            solution[s.STATUS] = s.OPTIMAL_INACCURATE
        if solution[s.STATUS] == s.USER_LIMIT and not model.hasmipsol:
            solution[s.STATUS] = s.INFEASIBLE_INACCURATE

        solution['model'] = model

        return solution

    def cite(self, data):
        """Returns bibtex citation for the solver.

        Parameters
        ----------
        data : dict
            Data generated via an apply call.
        """
        return CITATION_DICT["COPT"]