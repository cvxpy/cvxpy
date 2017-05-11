# GUROBI interface to solve QP problems
import numpy as np
import gurobipy as grb
from cvxpy.problems.problem_data.problem_data import ProblemData
import cvxpy.settings as s
from cvxpy.reductions.solution import Solution
from cvxpy.solver_interface.conic_solvers.conic_solver import ConicSolver
from cvxpy.solver_interface.reduction_solver import ReductionSolver
from collections import namedtuple

class GUROBI(ReductionSolver):
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

    def name(self):
        return "GUROBI"
    
    def import_solver(self):
        import gurobipy as grb
        grb

    def accepts(self, problem):
        return problem.is_qp()

    def apply(self, problem):
        data = namedtuple("qp_struct", ['P', 'q', 'A', 'lb', 'ub'])
        obj = problem.objective
        data.P = obj.args[0].args[0].args[1].value
        data.q = obj.args[0].args[1].args[0].value.flatten()
        data.A, b = ConicSolver.get_coeff_offset(problem.constraints[0].args[0])
        data.uA = -b
        data.lA = -grb.GRB.INFINITY*np.ones(b.shape)
        inverse_data = {self.VAR_ID: problem.variables()[0].id}
        inverse_data[self.EQ_CONSTR] = None # Gurobi does not accept equality constraints
        inverse_data[self.NEQ_CONSTR] = problem.constraints[0].id
        
        return data, inverse_data

    def invert(self, solution, inverse_data):
        status = self.STATUS_MAP.get(solution.Status, s.SOLVER_ERROR)
        cputime = solution.Runtime
        attr = {s.SOLVE_TIME:cputime}
        if status in s.SOLUTION_PRESENT:
            opt_val = solution.objVal
            primal_vars = {inverse_data[self.VAR_ID]: np.array(solution.x)}
            constrs = solution.getConstrs()
            dual_vars = {} 
            # Gurobi uses swapped signs (-1) for dual variables
            dual_vars[inverse_data[self.NEQ_CONSTR]] = -np.array([constrs[i].Pi for i in range(len(constrs))])
            total_iter = solution.BarIterCount
            attr[s.NUM_ITERS] = total_iter
        else: # no solution
            primal_vars = None
            dual_vars = None 
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve(self, problem, warm_start, verbose, solver_opts):
        data, inverse_data = self.apply(problem)
        solution = self._solve(data, solver_opts)
        return self.invert(solution, inverse_data)

    def _solve(self, p, options):

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
        for param, value in options.iteritems():
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

        return model
