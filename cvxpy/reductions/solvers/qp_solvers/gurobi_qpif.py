import cvxpy.settings as s
from cvxpy.reductions.solvers import utilities
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np


def constrain_gurobi_infty(v):
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
                  # TODO could be anything.
                  # means time expired.
                  9: s.SOLVER_ERROR,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.OPTIMAL_INACCURATE}

    def name(self):
        return s.GUROBI

    def import_solver(self):
        import gurobipy
        gurobipy

    def invert(self, results, inverse_data):
        model = results["model"]
        x_grb = model.getVars()
        n = len(x_grb)
        constraints_grb = model.getConstrs()
        m = len(constraints_grb)

        # Start populating attribute dictionary
        attr = {s.SOLVE_TIME: model.Runtime,
                s.NUM_ITERS: model.BarIterCount}

        # Map GUROBI statuses back to CVXPY statuses
        status = self.STATUS_MAP.get(model.Status, s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            opt_val = model.objVal
            x = np.array([x_grb[i].X for i in range(n)])

            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data.is_mip:
                y = -np.array([constraints_grb[i].Pi for i in range(m)])

                dual_vars = utilities.get_dual_values(
                    intf.DEFAULT_INTF.const_to_matrix(y),
                    utilities.extract_dual_value,
                    inverse_data.sorted_constraints)
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
        P = data[s.P].tocoo()       # Convert matrix to coo format
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

        # Add variables
        for i in range(n):
            # Set variable type.
            if i in data[s.BOOL_IDX]:
                vtype = grb.GRB.BINARY
            elif i in data[s.INT_IDX]:
                vtype = grb.GRB.INTEGER
            else:
                vtype = grb.GRB.CONTINUOUS
            model.addVar(ub=grb.GRB.INFINITY,
                         lb=-grb.GRB.INFINITY,
                         vtype=vtype)
        model.update()
        x = model.getVars()

        # Add equality constraints: iterate over the rows of A
        # adding each row into the model
        if A.shape[0] > 0:
            for i in range(A.shape[0]):
                start = A.indptr[i]
                end = A.indptr[i+1]
                variables = [x[j] for j in A.indices[start:end]]  # Get nnz
                coeff = A.data[start:end]
                expr = grb.LinExpr(coeff, variables)
                model.addConstr(expr, grb.GRB.EQUAL, b[i])
        model.update()

        # Add inequality constraints: iterate over the rows of F
        # adding each row into the model
        if F.shape[0] > 0:
            for i in range(F.shape[0]):
                start = F.indptr[i]
                end = F.indptr[i+1]
                variables = [x[j] for j in F.indices[start:end]]  # Get nnz
                coeff = F.data[start:end]
                expr = grb.LinExpr(coeff, variables)
                model.addConstr(expr, grb.GRB.LESS_EQUAL, g[i])
        model.update()

        # Define objective
        obj = grb.QuadExpr()
        if P.count_nonzero():  # If there are any nonzero elms in P
            for i in range(P.nnz):
                obj.add(.5*P.data[i]*x[P.row[i]]*x[P.col[i]])
        obj.add(grb.LinExpr(q, x))  # Add linear part
        model.setObjective(obj)  # Set objective
        model.update()

        # Set verbosity and other parameters
        model.setParam("OutputFlag", verbose)
        # TODO user option to not compute duals.
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

        return results_dict
