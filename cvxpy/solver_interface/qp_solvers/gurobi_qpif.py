# GUROBI interface to solve QP problems
import numpy as np
from quadprog.results import quadprogResults
import gurobipy as grb
import quadprog.problem as qp
import ipdb


class GUROBI(object):
    """
    An interface for the Gurobi QP solver.
    """

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: qp.OPTIMAL,
                  3: qp.INFEASIBLE,
                  5: qp.UNBOUNDED,
                  4: qp.SOLVER_ERROR,
                  6: qp.SOLVER_ERROR,
                  7: qp.SOLVER_ERROR,
                  8: qp.SOLVER_ERROR,
                  # TODO could be anything.
                  # means time expired.
                  9: qp.OPTIMAL_INACCURATE,
                  10: qp.SOLVER_ERROR,
                  11: qp.SOLVER_ERROR,
                  12: qp.SOLVER_ERROR,
                  13: qp.SOLVER_ERROR}

    def __init__(self, **kwargs):
        self.options = kwargs

    def solve(self, p):

        # Convert Matrices in CSR format
        p.A = p.A.tocsr()

        # Convert P matrix to COO format
        p.P = p.P.tocoo()

        # Get problem dimensions
        n = p.P.shape[0]
        m = p.A.shape[0]

        # Create a new model
        model = grb.Model("qp")

        # Add variables
        for i in range(n):
            model.addVar(ub=grb.GRB.INFINITY, lb=-grb.GRB.INFINITY)
        model.update()
        x = model.getVars()

        # Add inequality constraints: iterate over the rows of Aeq
        # adding each row into the model
        for i in range(m):
            start = p.A.indptr[i]
            end = p.A.indptr[i+1]
            variables = [x[j] for j in p.A.indices[start:end]]  # Get nnz
            coeff = p.A.data[start:end]
            expr = grb.LinExpr(coeff, variables)
            model.addRange(expr, lower=p.lA[i], upper=p.uA[i])
        model.update()

        # Define objective
        obj = grb.QuadExpr()  # Set quadratic part
        if p.P.count_nonzero():  # If there are any nonzero elms in P
            for i in range(p.P.nnz):
                obj.add(.5*p.P.data[i]*x[p.P.row[i]]*x[p.P.col[i]])
        obj.add(grb.LinExpr(p.q, x))  # Add linear part
        model.setObjective(obj)  # Set objective

        # Update model
        model.update()

        # Set parameters
        for param, value in self.options.iteritems():
            if param == "verbose":
                if value == 0:
                    model.setParam("OutputFlag", 0)
            else:
                model.setParam(param, value)

        # Update model
        model.update()

        # Solve problem
        try:
            # Solve
            model.optimize()
        except:  # Error in the solution
            print "Error in Gurobi solution\n"
            return quadprogResults(qp.SOLVER_ERROR, None, None, None,
                                   np.inf, None)

        # Return results
        # Get status
        status = self.STATUS_MAP.get(model.Status, qp.SOLVER_ERROR)

        if (status != qp.SOLVER_ERROR) & (status != qp.INFEASIBLE):
            # Get objective value
            objval = model.objVal

            # Get solution
            sol = np.array([x[i].X for i in range(n)])

            # Get dual variables  (Gurobi uses swapped signs (-1))
            constrs = model.getConstrs()
            dual = -np.array([constrs[i].Pi for i in range(m)])
            # sol_dual_ineq = -np.array([constrs[i+neq].Pi for i in range(m)])

            # Bounds
            # sol_dual_ub = np.zeros(n)
            # sol_dual_lb = np.zeros(n)

            # RCx = [x[i].RC for i in range(n)]  # Get reduced costs
            # for i in range(n):
            #     if RCx[i] >= 1e-07:
            #         sol_dual_lb[i] = RCx[i]
            #     else:
            #         sol_dual_ub[i] = -RCx[i]

            # Get computation time
            cputime = model.Runtime

            # Total Number of iterations
            total_iter = model.BarIterCount

            return quadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:  # Error
            # Get computation time
            cputime = model.Runtime

            return quadprogResults(status, None, None, None,
                                   cputime, None)
