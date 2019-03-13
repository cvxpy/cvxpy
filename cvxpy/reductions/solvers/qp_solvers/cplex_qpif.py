import cvxpy.settings as s
from cvxpy.reductions.solvers import utilities
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np


def constrain_cplex_infty(v):
    '''
    Limit values of vector v between +/- infinity as
    defined in the CPLEX package
    '''
    import cplex as cpx
    n = len(v)

    for i in range(n):
        if v[i] >= cpx.infinity:
            v[i] = cpx.infinity
        if v[i] <= -cpx.infinity:
            v[i] = -cpx.infinity


class CPLEX(QpSolver):
    """QP interface for the CPLEX solver"""

    MIP_CAPABLE = True

    # Map of CPLEX status to CVXPY status. #TODO: add more!
    STATUS_MAP = {1: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  2: s.UNBOUNDED,
                  21: s.UNBOUNDED,
                  22: s.INFEASIBLE,
                  4: s.INFEASIBLE,
                  10: s.USER_LIMIT,
                  101: s.OPTIMAL,
                  102: s.OPTIMAL,
                  103: s.INFEASIBLE,
                  107: s.USER_LIMIT,
                  118: s.UNBOUNDED}

    def name(self):
        return s.CPLEX

    def import_solver(self):
        import cplex
        cplex

    def invert(self, results, inverse_data):
        model = results["model"]
        attr = {}
        if "cputime" in results:
            attr[s.SOLVE_TIME] = results["cputime"]
        attr[s.NUM_ITERS] = \
            int(model.solution.progress.get_num_barrier_iterations())

        status = self.STATUS_MAP.get(model.solution.get_status(),
                                     s.SOLVER_ERROR)

        if status in s.SOLUTION_PRESENT:
            # Get objective value
            opt_val = model.solution.get_objective_value()

            # Get solution
            x = np.array(model.solution.get_values())
            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data.is_mip:
                y = -np.array(model.solution.get_dual_values())

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
        import cplex as cpx
        P = data[s.P].tocsr()       # Convert matrix to csr format
        q = data[s.Q]
        A = data[s.A].tocsr()       # Convert A matrix to csr format
        b = data[s.B]
        F = data[s.F].tocsr()       # Convert F matrix to csr format
        g = data[s.G]
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        # Constrain values between bounds
        constrain_cplex_infty(b)
        constrain_cplex_infty(g)

        # Define CPLEX problem
        model = cpx.Cplex()

        # Minimize problem
        model.objective.set_sense(model.objective.sense.minimize)

        # Add variables and linear objective
        var_idx = list(model.variables.add(obj=q,
                                           lb=-cpx.infinity*np.ones(n_var),
                                           ub=cpx.infinity*np.ones(n_var)))

        # Constrain binary/integer variables if present
        for i in data[s.BOOL_IDX]:
            model.variables.set_types(var_idx[i],
                                      model.variables.type.binary)
        for i in data[s.INT_IDX]:
            model.variables.set_types(var_idx[i],
                                      model.variables.type.integer)

        # Add constraints
        lin_expr, rhs = [], []
        for i in range(n_eq):  # Add equalities
            start = A.indptr[i]
            end = A.indptr[i+1]
            lin_expr.append([A.indices[start:end].tolist(),
                             A.data[start:end].tolist()])
            rhs.append(b[i])
        if lin_expr:
            model.linear_constraints.add(lin_expr=lin_expr,
                                         senses=["E"] * len(lin_expr),
                                         rhs=rhs)

        lin_expr, rhs = [], []
        for i in range(n_ineq):  # Add inequalities
            start = F.indptr[i]
            end = F.indptr[i+1]
            lin_expr.append([F.indices[start:end].tolist(),
                             F.data[start:end].tolist()])
            rhs.append(g[i])
        if lin_expr:
            model.linear_constraints.add(lin_expr=lin_expr,
                                         senses=["L"] * len(lin_expr),
                                         rhs=rhs)

        # Set quadratic Cost
        if P.count_nonzero():  # Only if quadratic form is not null
            qmat = []
            for i in range(n_var):
                start = P.indptr[i]
                end = P.indptr[i+1]
                qmat.append([P.indices[start:end].tolist(),
                            P.data[start:end].tolist()])
            model.objective.set_quadratic(qmat)

        # Set parameters
        if not verbose:
            model.set_results_stream(None)
            model.set_log_stream(None)
            model.set_error_stream(None)
            model.set_warning_stream(None)

        # TODO: The code in cvxpy/problems/solvers/cplex_intf.py sets
        #       CPLEX parameters in the same way and the code is
        #       duplicated here. This should be refactored.
        kwargs = sorted(solver_opts.keys())
        if "cplex_params" in kwargs:
            for param, value in solver_opts["cplex_params"].items():
                try:
                    eval("model.parameters.{0}.set({1})".format(param, value))
                except AttributeError:
                    raise ValueError(
                        "invalid CPLEX parameter, value pair ({0}, {1})".format(
                            param, value))
            kwargs.remove("cplex_params")
        if "cplex_filename" in kwargs:
            filename = solver_opts["cplex_filename"]
            if filename:
                model.write(filename)
            kwargs.remove("cplex_filename")
        if kwargs:
            raise ValueError("invalid keyword-argument '{0}'".format(kwargs[0]))

        # Solve problem
        results_dict = {}
        try:
            start = model.get_time()
            model.solve()
            end = model.get_time()
            results_dict["cputime"] = end - start
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = model

        return results_dict
