import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np


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

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.INFEASIBLE_INACCURATE,
                  6: s.INFEASIBLE,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  9: s.USER_LIMIT,  # Maximum time expired
                  # TODO could be anything.
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.OPTIMAL_INACCURATE}

    def name(self):
        return s.GUROBI

    def import_solver(self) -> None:
        import gurobipy
        gurobipy

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

        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
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
        model.update()

        x = np.array(model.getVars(), copy=False)

        if A.shape[0] > 0:
            if hasattr(model, 'addMConstrs'):
                # We can pass all of A @ x == b at once, use stable API
                # introduced with Gurobi v9
                model.addMConstrs(A, None, grb.GRB.EQUAL, b)
            elif hasattr(model, '_v811_addMConstrs'):
                # We can pass all of A @ x == b at once, API only for Gurobi
                # v811
                A.eliminate_zeros()  # Work around bug in gurobipy v811
                sense = np.repeat(grb.GRB.EQUAL, A.shape[0])
                model._v811_addMConstrs(A, sense, b)
            else:
                # Add equality constraints: iterate over the rows of A
                # adding each row into the model
                for i in range(A.shape[0]):
                    start = A.indptr[i]
                    end = A.indptr[i+1]
                    variables = x[A.indices[start:end]]
                    coeff = A.data[start:end]
                    expr = grb.LinExpr(coeff, variables)
                    model.addConstr(expr, grb.GRB.EQUAL, b[i])
        model.update()

        if F.shape[0] > 0:
            if hasattr(model, 'addMConstrs'):
                # We can pass all of F @ x <= g at once, use stable API
                # introduced with Gurobi v9
                model.addMConstrs(F, None, grb.GRB.LESS_EQUAL, g)
            elif hasattr(model, '_v811_addMConstrs'):
                # We can pass all of F @ x <= g at once, API only for Gurobi
                # v811.
                F.eliminate_zeros()  # Work around bug in gurobipy v811
                sense = np.repeat(grb.GRB.LESS_EQUAL, F.shape[0])
                model._v811_addMConstrs(F, sense, g)
            else:
                # Add inequality constraints: iterate over the rows of F
                # adding each row into the model
                for i in range(F.shape[0]):
                    start = F.indptr[i]
                    end = F.indptr[i+1]
                    variables = x[F.indices[start:end]]
                    coeff = F.data[start:end]
                    expr = grb.LinExpr(coeff, variables)
                    model.addConstr(expr, grb.GRB.LESS_EQUAL, g[i])
        model.update()

        # Define objective
        if hasattr(model, 'setMObjective'):
            # Use stable API starting in Gurobi v9
            P = P.tocoo()
            model.setMObjective(0.5 * P, q, 0.0)
        elif hasattr(model, '_v811_setMObjective'):
            # Use temporary API for Gurobi v811 only
            P = P.tocoo()
            model._v811_setMObjective(0.5 * P, q)
        else:
            obj = grb.QuadExpr()
            if P.count_nonzero():  # If there are any nonzero elms in P
                P = P.tocoo()
                obj.addTerms(0.5*P.data, vars=list(x[P.row]),
                             vars2=list(x[P.col]))
            obj.add(grb.LinExpr(q, x))  # Add linear part
            model.setObjective(obj)  # Set objective
        model.update()

        # Set parameters
        model.setParam("QCPDual", True)
        for key, value in solver_opts.items():
            model.setParam(key, value)

        # Update model
        model.update()

        # Solve problem
        results_dict = {}
        try:
            # Solve
            model.optimize()
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model

        if solver_cache is not None:
            solver_cache[self.name()] = model

        return results_dict
