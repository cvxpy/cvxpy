"""
Copyright 2018 Sascha-Dominic Schnug
"""

import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np
import scipy.sparse as sp


class BONMIN_QP(QpSolver):
    """QP interface for the BONMIN_QP solver"""

    MIP_CAPABLE = True

    # Map of BONMIN_QP status to CVXPY status.
    STATUS_MAP = {0: s.OPTIMAL,
                  1: s.INFEASIBLE,
                  2: s.UNBOUNDED}

    def name(self):
        return s.BONMIN_QP

    def import_solver(self):
        import pyMIQP
        pyMIQP

    def invert(self, solution, inverse_data):
        # Map BONMIN / pyMIQP statuses back to CVXPY statuses
        attr = {s.SOLVE_TIME: solution['time']}
        status = self.STATUS_MAP.get(solution['status'], s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = solution['obj']
            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(solution['x'])
            }
            # TODO consider adding support for duals
            dual_vars = None
            # dual_vars = utilities.get_dual_values(
            #     intf.DEFAULT_INTF.const_to_matrix(solution.y),
            #     utilities.extract_dual_value,
            #     inverse_data.sorted_constraints)
            # attr[s.NUM_ITERS] = solution.info.iter
        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf
        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts,
                       solver_cache=None):
        import pyMIQP
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form
            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g
        """
        P = data[s.P]
        n = P.shape[0]
        q = data[s.Q]
        A = sp.vstack([data[s.A], data[s.F]]).tocsc()
        uA = np.concatenate((data[s.B], data[s.G]))
        lA = np.concatenate([data[s.B], -np.inf*np.ones(data[s.G].shape)])
        var_np = np.zeros(n, dtype=int)
        var_np[data[s.BOOL_IDX]] = 1
        var_np[data[s.INT_IDX]] = 2

        # no solver_cache impl
        # no warmstart

        # Only interaction with pyMIQP
        hess_approx = False
        if 'hessian_approximation' in solver_opts:
            hess_approx = solver_opts['hessian_approximation']

        deriv_test = 'none'
        if 'derivative_test' in solver_opts:
            deriv_test = solver_opts['derivative_test']

        algorithm = 'B-BB'
        if 'algorithm' in solver_opts:
            algorithm = solver_opts['algorithm']

        solver = pyMIQP.MIQP(verbose=verbose, hessian_approximation=hess_approx,
                             derivative_test=deriv_test)
        solver.set_c(q)
        solver.set_Q(P)
        if A.size > 0:
            solver.set_A(A)
            solver.set_glb(lA)
            solver.set_gub(uA)
        solver.set_xlb(np.full(n, -np.inf))
        solver.set_xub(np.full(n, np.inf))
        solver.set_var_types(var_np)
        solver.solve(algorithm=algorithm)

        res_status = solver.get_sol_status()
        res_time = solver.get_sol_time()
        res_obj = None
        res_x = None
        if res_status == 0:
            res_obj = solver.get_sol_obj()
            res_x = solver.get_sol_x()

        results = {'x': res_x, 'obj': res_obj, 'status': res_status,
                   'time': res_time}

        return results
