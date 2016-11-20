# CPLEX interface to solve QP problems
import numpy as np
import cvxpy.settings as s
import cplex


class CPLEX(object):
    """
    An interface for the CPLEX QP solver.
    """

    # Map of CPLEX status to CVXPY status. #TODO: add more!
    STATUS_MAP = {1: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  2: s.UNBOUNDED,
                  6: s.OPTIMAL_INACCURATE}

    def import_solver(self):
        """Imports the solver.
        """
        import cplex
        cplex  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.CPLEX

# Functions in old conic format
# def split_constr(self, constr_map)
# def matrix_intf(self):
# def vec_intf(self):

# TODO (Bart): Is this function needed?
    # def __init__(self, **kwargs):
    # self.options = kwargs

    # TODO: Fix how matrices used. At the moment it is using the old problem structure "p"
    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):

        # Convert Matrices in CSR format
        p.A = p.A.tocsr()
        p.P = p.P.tocsr()

        # Get problem dimensions
        n = p.P.shape[0]
        m = p.A.shape[0]

        # Adjust infinity values in bounds
        uA = np.copy(p.uA)
        lA = np.copy(p.lA)

        for i in range(m):
            if uA[i] == np.inf:
                uA[i] = cplex.infinity
            if lA[i] == -np.inf:
                lA[i] = -cplex.infinity

        # Define CPLEX problem
        model = cplex.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables
        model.variables.add(obj=p.q,
                            lb=-cplex.infinity*np.ones(n),
                            ub=cplex.infinity*np.ones(n))  # Linear obj part

        # Add constraints
        for i in range(m):  # Add inequalities
            start = p.A.indptr[i]
            end = p.A.indptr[i+1]
            row = [[p.A.indices[start:end].tolist(),
                   p.A.data[start:end].tolist()]]
            if (lA[i] != -cplex.infinity) & (uA[i] == cplex.infinity):
                model.linear_constraints.add(lin_expr=row,
                                             senses=["G"],
                                             rhs=[lA[i]])
            elif (lA[i] == -cplex.infinity) & (uA[i] != cplex.infinity):
                model.linear_constraints.add(lin_expr=row,
                                             senses=["L"],
                                             rhs=[uA[i]])
            else:
                model.linear_constraints.add(lin_expr=row,
                                             senses=["R"],
                                             range_values=[lA[i] - uA[i]],
                                             rhs=[uA[i]])

        # Set quadratic Cost
        if p.P.count_nonzero():  # Only if quadratic form is not null
            qmat = []
            for i in range(n):
                start = p.P.indptr[i]
                end = p.P.indptr[i+1]
                qmat.append([p.P.indices[start:end].tolist(),
                            p.P.data[start:end].tolist()])
            model.objective.set_quadratic(qmat)

        # TODO: Set solver options
        # for param, value in self.options.iteritems():
        #     if param == "verbose":
        #         if value == 0:
        #             model.set_results_stream(None)
        #             model.set_log_stream(None)
        #             model.set_error_stream(None)
        #             model.set_warning_stream(None)
        #     else:
        #         exec("model.parameters.%s.set(%d)" % (param, value))

        # Solve problem
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
        except:  # Error in the solution
            print "Error in CPLEX solution\n"
            # TODO: Fix it
            return None

        # Get computation time and store in solution structure
        model.solution.cputime = end-start

        # TODO: Define results dictionary
        results_dict = model
        return self.format_results(results_dict, data, cached_data)

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard QP form
        """
        # TODO: Complete funciton depending on required data

        # Get status
        status = self.STATUS_MAP.get(results_dict.get_status(),
                                     s.SOLVER_ERROR)

        if (status != s.SOLVER_ERROR) & (status != s.INFEASIBLE):
            # Get objective value
            objval = results_dict.get_objective_value()

            # Get solution
            sol = np.array(results_dict.get_values())

            # Get dual values
            dual = -np.array(results_dict.get_dual_values())

            # Get total number of iterations
            total_iter = int(results_dict.progress.get_num_barrier_iterations())

            # TODO: Add results structure
            return quadprogResults(status, objval, sol, dual,
                                   cputime, total_iter)
        else:
            return quadprogResults(status, None, None, None,
                                   cputime, None)
