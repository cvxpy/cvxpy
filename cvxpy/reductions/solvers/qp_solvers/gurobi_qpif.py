import numpy as np

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver


def constrain_gurobi_infty(v) -> None:
    '''
    Limit values of vector v between +/- infinity as
    defined in the Gurobi package
    '''
    import gurobipy as grb
    n = len(v)

    for i in range(n):
        if v[i] >= 1e20:
            v[i] = grb.GRB.INFINITY
        if v[i] <= -1e20:
            v[i] = -grb.GRB.INFINITY


class GUROBI(QpSolver):
    """QP interface for the Gurobi solver"""

    MIP_CAPABLE = True

    # Keyword arguments for the CVXPY interface.
    INTERFACE_ARGS = ["save_file", "reoptimize"]

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  4: s.INFEASIBLE_OR_UNBOUNDED,  # Triggers reoptimize.
                  5: s.UNBOUNDED,
                  6: s.SOLVER_ERROR,
                  7: s.USER_LIMIT, # ITERATION_LIMIT
                  8: s.USER_LIMIT, # NODE_LIMIT
                  9: s.USER_LIMIT,  # TIME_LIMIT
                  10: s.USER_LIMIT, # SOLUTION_LIMIT
                  11: s.USER_LIMIT, # INTERRUPTED
                  12: s.SOLVER_ERROR, # NUMERIC
                  13: s.USER_LIMIT, # SUBOPTIMAL
                  14: s.USER_LIMIT, # INPROGRESS
                  15: s.USER_LIMIT, # USER_OBJ_LIMIT
                  16: s.USER_LIMIT, # WORK_LIMIT
                  17: s.USER_LIMIT} # MEM_LIMIT

    def name(self):
        return s.GUROBI

    def import_solver(self) -> None:
        import gurobipy
        gurobipy

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        """
        import gurobipy as grb
        data, inv_data = super(GUROBI, self).apply(problem)
        # Add initial guess.
        data['init_value'] = utilities.stack_vals(problem.variables, grb.GRB.UNDEFINED)
        return data, inv_data

    def invert(self, results, inverse_data):
        model = results["model"]
        x_grb = model.getVars()
        n = len(x_grb)
        constraints_grb = model.getConstrs()
        m = len(constraints_grb)

        # Note: Gurobi does not always fill BarIterCount
        # and IterCount so better using try/except
        try:
            bar_iter_count = model.BarIterCount
        except AttributeError:
            bar_iter_count = 0
        try:
            simplex_iter_count = model.IterCount
        except AttributeError:
            simplex_iter_count = 0
        # Take the sum in case they both appear. One of them
        # will be 0 anyway
        iter_count = bar_iter_count + simplex_iter_count

        # Start populating attribute dictionary
        attr = {s.SOLVE_TIME: model.Runtime,
                s.NUM_ITERS: iter_count,
                s.EXTRA_STATS: model}

        # Map GUROBI statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)
        if status == s.USER_LIMIT and not model.SolCount:
            status = s.INFEASIBLE_INACCURATE

        if (status in s.SOLUTION_PRESENT) or (model.solCount > 0):
            opt_val = model.objVal + inverse_data[s.OFFSET]
            x = np.array([x_grb[i].X for i in range(n)])

            primal_vars = {
                GUROBI.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data[GUROBI.IS_MIP]:
                y = -np.array([constraints_grb[i].Pi for i in range(m)])
                dual_vars = {GUROBI.DUAL_VAR_ID: y}

            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status, attr)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import gurobipy as grb

        # N.B. Here we assume that the matrices in data are in csc format
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A].tocsr()       # Convert A matrix to csr format
        b = data[s.B]
        F = data[s.F].tocsr()       # Convert F matrix to csr format
        g = data[s.G]
        n = data['n_var']

        # Constrain values between bounds
        constrain_gurobi_infty(b)
        constrain_gurobi_infty(g)

        # Create a new model
        if 'env' in solver_opts:
            # Specifies environment to create Gurobi model for control over licensing and parameters
            # https://www.gurobi.com/documentation/9.1/refman/environments.html
            default_env = solver_opts['env']
            del solver_opts['env']
            model = grb.Model(env=default_env)
        else:
            # Create Gurobi model using default (unspecified) environment
            model = grb.Model()

        # Pass through verbosity
        model.setParam("OutputFlag", verbose)

        # Add variables
        vtypes = {}
        for ind in data[s.BOOL_IDX]:
            vtypes[ind] = grb.GRB.BINARY
        for ind in data[s.INT_IDX]:
            vtypes[ind] = grb.GRB.INTEGER
        for i in range(n):
            if i not in vtypes:
                vtypes[i] = grb.GRB.CONTINUOUS
        x_grb = model.addVars(int(n),
                              ub={i: grb.GRB.INFINITY for i in range(n)},
                              lb={i: -grb.GRB.INFINITY for i in range(n)},
                              vtype=vtypes)

        if warm_start and solver_cache is not None \
                and self.name() in solver_cache:
            old_model = solver_cache[self.name()]
            old_status = self.STATUS_MAP.get(old_model.Status,
                                             s.SOLVER_ERROR)
            if (old_status in s.SOLUTION_PRESENT) or (old_model.solCount > 0):
                old_x_grb = old_model.getVars()
                for idx in range(len(x_grb)):
                    x_grb[idx].start = old_x_grb[idx].X
        elif warm_start:
            # Set the start value of Gurobi vars to user provided values.
            for idx in range(len(x_grb)):
                x_grb[idx].start = data['init_value'][idx]

        if A.shape[0] > 0:
            # We can pass all of A @ x == b at once, use stable API
            # introduced with Gurobi v9.5
            model.addMConstr(A, None, grb.GRB.EQUAL, b)

        if F.shape[0] > 0:
            # We can pass all of A @ x == b at once, use stable API
            # introduced with Gurobi v9.5
            model.addMConstr(F, None, grb.GRB.LESS_EQUAL, g)

        # Define objective
        # Use stable API starting in Gurobi v9
        P = P.tocoo()
        model.setMObjective(0.5 * P, q, 0.0)

        # Set parameters
        model.setParam("QCPDual", True)
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
            if model.Status == 4 and solver_opts.get('reoptimize', False):
                # INF_OR_UNBD. Solve again to get a definitive answer.
                model.setParam("DualReductions", 0)
                model.optimize()
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model

        if solver_cache is not None:
            solver_cache[self.name()] = model

        return results_dict
