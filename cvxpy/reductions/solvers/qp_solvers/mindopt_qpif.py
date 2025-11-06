import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from scipy.sparse import csr_matrix

import numpy as np


def constrain_mindopt_infty(v) -> None:
    '''
    Limit values of vector v between +/- infinity as
    defined in the MindOpt package
    '''
    import mindoptpy as mp
    n = len(v)

    for i in range(n):
        if v[i] >= 1e20:
            v[i] = mp.MDO.INFINITY
        if v[i] <= -1e20:
            v[i] = -mp.MDO.INFINITY


class MINDOPT(QpSolver):
    """QP interface for the MindOpt solver"""

    MIP_CAPABLE = True

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map of MindOpt status to CVXPY status.
    STATUS_MAP = {
        1: s.OPTIMAL,  # OPTIMAL
        2: s.INFEASIBLE,  # INFEASIBLE
        3: s.UNBOUNDED,  # UNBOUNDED
        4: s.INFEASIBLE_OR_UNBOUNDED,  # INF_OR_UBD
        5: s.USER_LIMIT,  # SUB_OPTIMAL
        6: s.USER_LIMIT,  # ITERATION_LIMIT
        7: s.USER_LIMIT,  # TIME_LIMIT
        8: s.USER_LIMIT,  # NODE_LIMIT
        9: s.USER_LIMIT,  # SOLUTION_LIMIT
        10: s.USER_LIMIT,  # STALLING_NODE_LIMIT
        11: s.USER_LIMIT,  # INTERRUPTED
    }
    def name(self):
        return s.MINDOPT

    def import_solver(self) -> None:
        import mindoptpy

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        import mindoptpy as mp
        data, inv_data = super(MINDOPT, self).apply(problem)
        # Add initial guess.
        data['init_value'] = utilities.stack_vals(problem.variables, mp.MDO.UNDEFINED)
        return data, inv_data

    def invert(self, results, inverse_data):
        model = results["model"]
        x_mdo = model.getVars()
        n = len(x_mdo)
        constraints_mdo = model.getConstrs()
        m = len(constraints_mdo)

        try:
            bar_iter_count = model.model.getAttr("IPM/NumIters")
        except AttributeError:
            bar_iter_count = 0
        try:
            simplex_iter_count = model.model.getAttr("SPX/NumIters")
        except AttributeError:
            simplex_iter_count = 0
        # Take the sum
        iter_count = bar_iter_count + simplex_iter_count

        # Start populating attribute dictionary
        attr = {s.SOLVE_TIME: model.SolverTime,
                s.NUM_ITERS: iter_count,
                s.EXTRA_STATS: model}

        # Map MindOpt statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)
        if status == s.USER_LIMIT and not model.SolCount:
            status = s.INFEASIBLE_INACCURATE

        if (status in s.SOLUTION_PRESENT) or (model.SolCount > 0):
            opt_val = model.objVal + inverse_data[s.OFFSET]
            x = np.array([x_mdo[i].X for i in range(n)])

            primal_vars = {
                MINDOPT.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data[MINDOPT.IS_MIP]:
                y = -np.array([constraints_mdo[i].DualSoln for i in range(m)])
                dual_vars = {MINDOPT.DUAL_VAR_ID: y}

            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import mindoptpy as mp

        # N.B. Here we assume that the matrices in data are in csc format
        P = data[s.P]
        q = data[s.Q]
        A = csr_matrix(data[s.A])       # Convert A matrix to csr format
        A.indptr = A.indptr.astype(np.int32)
        A.indices = A.indices.astype(np.int32)
        b = data[s.B]
        F = csr_matrix(data[s.F])       # Convert F matrix to csr format
        F.indptr = F.indptr.astype(np.int32)
        F.indices = F.indices.astype(np.int32)
        g = data[s.G]
        n = data['n_var']

        # Constrain values between bounds
        constrain_mindopt_infty(b)
        constrain_mindopt_infty(g)

        # Create a new model
        if 'env' in solver_opts:
            default_env = solver_opts['env']
            del solver_opts['env']
            model = mp.Model(env=default_env)
        else:
            # Create MindOpt model using default (unspecified) environment
            model = mp.Model()

        # Pass through verbosity
        model.setParam("OutputFlag", verbose)

        # Add variables
        vtypes = {}
        for ind in data[s.BOOL_IDX]:
            vtypes[ind] = mp.MDO.BINARY
        for ind in data[s.INT_IDX]:
            vtypes[ind] = mp.MDO.INTEGER
        for i in range(n):
            if i not in vtypes:
                vtypes[i] = mp.MDO.CONTINUOUS
        x_mdo = mp.tupledict()
        for i in range(n):
            x_mdo[i] = model.addVar(lb=-mp.MDO.INFINITY, ub=mp.MDO.INFINITY, vtype=vtypes[i])

        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.SolCount > 0):
                old_x_mdo = old_model.getVars()
                for idx in range(len(x_mdo)):
                    x_mdo[idx].start = old_x_mdo[idx].X
        elif warm_start:
            # Set the start value of MindOpt vars to user provided values.
            for idx in range(len(x_mdo)):
                x_mdo[idx].start = data['init_value'][idx]

        if A.shape[0] > 0:
            model.addMConstr(A, None, mp.MDO.EQUAL, b)

        if F.shape[0] > 0:
            model.addMConstr(F, None, mp.MDO.LESS_EQUAL, g)

        # Define objective
        P = P.tocoo()
        obj = mp.QuadExpr()
        obj.addTerms([0.5 * data1 for data1 in P.data], [x_mdo[i] for i in P.row], [x_mdo[j] for j in P.col])
        obj.addTerms(q, x_mdo)
        model.setObjective(obj, mp.MDO.MINIMIZE)
        # print(model.getObjective())

        # Set parameters
        for key, value in solver_opts.items():
            # Ignore arguments unique to the CVXPY interface.
            if key not in self.INTERFACE_ARGS:
                model.setParam(key, value)

        if 'save_file' in solver_opts:
            model.write(solver_opts['save_file'])

        # Solve problem
        results_dict = {}
        try:
            # Solve
            model.optimize()
            if model.Status == mp.MDO.INF_OR_UBD and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
                model.setParam("Presolve", 0)
                model.optimize()
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model

        if solver_cache is not None:
            solver_cache[self.name()] = model

        return results_dict
