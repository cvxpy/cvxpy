# GUROBI interface to solve QP problems
import numpy as np
import gurobipy as grb
import cvxpy.settings as s


class GUROBI(object):
    """
    An interface for the Gurobi QP solver.
    """

    # Map of Gurobi status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.SOLVER_ERROR,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  # TODO could be anything.
                  # means time expired.
                  9: s.OPTIMAL_INACCURATE,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

# Functions in old conic format
# def split_constr(self, constr_map)
# def matrix_intf(self):
# def vec_intf(self):

# TODO (Bart): Is this function needed?
    # def __init__(self, **kwargs):
    # self.options = kwargs

    # TODO: Fix how matrices used. At the moment it is using the old problem structure "p"
    # def solve(self, p):
    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):

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

        # TODO: Define results dictionary
        results_dict = model
        return self.format_results(results_dict, data, cached_data)

    def format_results(self, results_dict, data, cached_data):
        # Return results
        # Get status
        status = self.STATUS_MAP.get(results_dict.Status, s.SOLVER_ERROR)

        if (status != s.SOLVER_ERROR) & (status != s.INFEASIBLE):
            # Get objective value
            objval = results_dict.objVal

            # Get solution
            sol = np.array([x[i].X for i in range(len(x))])

            # Get dual variables  (Gurobi uses swapped signs (-1))
            constrs = results_dict.getConstrs()
            dual = -np.array([constrs[i].Pi for i in range(len(constrs))])

            # Get computation time
            cputime = results_dict.Runtime

            # Total Number of iterations
            total_iter = results_dict.BarIterCount

            # TODO: Add results structure
            return quadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:  # Error
            # Get computation time
            cputime = results_dict.Runtime

            # TODO: Add results structure
            return quadprogResults(status, None, None, None,
                                   cputime, None)
