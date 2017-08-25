import cvxpy.settings as s
from cvxpy.reductions.solvers import utilities
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
import numpy as np
import scipy.sparse as spa


class MOSEK(QpSolver):
    """QP interface for the Mosek solver"""

    MIP_CAPABLE = True

    def name(self):
        return s.MOSEK

    def import_solver(self):
        import mosek
        mosek

    def invert(self, results, inverse_data):
        import mosek
        task = results["task"]
        status = results[s.STATUS]
        soltype = results["soltype"]

        # Start populating attribut dictionary
        cputime = task.getdouinf(mosek.dinfitem.optimizer_time) + \
            task.getdouinf(mosek.dinfitem.presolve_time)
        total_iter = task.getintinf(mosek.iinfitem.intpnt_iter)
        attr = {s.SOLVE_TIME: cputime,
                s.NUM_ITERS: total_iter}

        if status in s.SOLUTION_PRESENT:
            # get primal solution
            x = np.zeros(task.getnumvar())
            task.getxx(soltype, x)
            # get obj value
            opt_val = task.getprimalobj(soltype)

            primal_vars = {
                inverse_data.id_map.keys()[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data.is_mip:
                y = np.zeros(task.getnumcon())
                task.gety(soltype, y)
                y = -y    # MOSEK dual signs are inverted

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
        import mosek
        # N.B. Here we assume that the matrices in data are in csc format
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        F = data[s.F]
        g = data[s.G]
        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        # Create environment
        env = mosek.Env()

        # Create optimization task
        task = env.Task()

        if verbose:
            # Define a stream printer to grab output from MOSEK
            def streamprinter(text):
                import sys
                sys.stdout.write(text)
                sys.stdout.flush()
            env.set_Stream(mosek.streamtype.log, streamprinter)
            task.set_Stream(mosek.streamtype.log, streamprinter)

        # Append 'n' variables.
        # The variables will initially be fixed at zero (x=0).
        task.appendvars(n_var)

        # Add linear cost and variable type by iterating over all variables
        for j in range(n_var):
            task.putcj(j, q[j])

            if j in data[s.INT_IDX] or j in data[s.BOOL_IDX]:
                task.putvartype(j, mosek.variabletype.type_int)

            if j in data[s.BOOL_IDX]:
                task.putvarbound(j, mosek.boundkey.ra, 0, 1)
            else:
                task.putvarbound(j, mosek.boundkey.fr, -np.inf, np.inf)

        # Add constraints
        task.appendcons(n_eq + n_ineq)
        if A.shape[0] and F.shape[0]:
            constraints_matrix = spa.bmat([[A], [F]])
        else:
            constraints_matrix = A if A.shape[0] else F
        coefficients = np.concatenate([b, g])

        row, col, el = spa.find(constraints_matrix)
        task.putaijlist(row, col, el)

        type_constraint = [mosek.boundkey.fx] * n_eq
        type_constraint += [mosek.boundkey.up] * n_ineq
        task.putconboundlist(np.arange(n_eq + n_ineq, dtype=int),
                             type_constraint,
                             coefficients,
                             coefficients)

        # Add quadratic cost
        if P.count_nonzero():  # If there are any nonzero elms in P
            P = spa.tril(P, format='coo')
            task.putqobj(P.row, P.col, P.data)

        # Set problem minimization
        task.putobjsense(mosek.objsense.minimize)

        # Set solver parameters
        if not verbose:
            self._handle_str_param(task, 'MSK_IPAR_LOG'.strip(), 0)

        for param, value in solver_opts.items():
                if isinstance(param, str):
                    self._handle_str_param(task, param.strip(), value)
                else:
                    self._handle_enum_param(task, param, value)

        # Optimization and check termination code
        task.optimize()

        if verbose:   # Print solution summary
            task.solutionsummary(mosek.streamtype.msg)

        # Get solution type and status
        soltype, solsta = self.choose_solution(task)

        # Map status using statusmap
        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.integer_optimal: s.OPTIMAL,
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED,
                      mosek.solsta.near_optimal: s.OPTIMAL_INACCURATE,
                      mosek.solsta.near_prim_infeas_cer: s.INFEASIBLE_INACCURATE,
                      mosek.solsta.near_dual_infeas_cer: s.UNBOUNDED_INACCURATE,
                      mosek.solsta.unknown: s.SOLVER_ERROR}
        status = STATUS_MAP.get(solsta, s.SOLVER_ERROR)

        # Results dictionary
        results_dict = {}
        results_dict[s.STATUS] = status
        results_dict['task'] = task
        results_dict['soltype'] = soltype

        return results_dict

    def choose_solution(self, task):
        """Chooses between the basic, interior point solution or integer solution
        Parameters
        ----------
        task : mosek.Task
            The solver status interface.
        Returns
        -------
        soltype
            The preferred solution (mosek.soltype.*)
        solsta
            The status of the preferred solution (mosek.solsta.*)
        """
        import mosek

        def rank(status):
            # Rank solutions
            # optimal > near_optimal > anything else > None
            if status == mosek.solsta.optimal:
                return 3
            elif status == mosek.solsta.near_optimal:
                return 2
            elif status is not None:
                return 1
            else:
                return 0

        solsta_bas, solsta_itr = None, None

        # Integer solution
        if task.solutiondef(mosek.soltype.itg):
            solsta_itg = task.getsolsta(mosek.soltype.itg)
            return mosek.soltype.itg, solsta_itg

        # Continuous solution
        if task.solutiondef(mosek.soltype.bas):
            solsta_bas = task.getsolsta(mosek.soltype.bas)

        if task.solutiondef(mosek.soltype.itr):
            solsta_itr = task.getsolsta(mosek.soltype.itr)

        # As long as interior solution is not worse, take it
        # (for backward compatibility)
        if rank(solsta_itr) >= rank(solsta_bas):
            return mosek.soltype.itr, solsta_itr
        else:
            return mosek.soltype.bas, solsta_bas

    @staticmethod
    def _handle_str_param(task, param, value):
        if param.startswith("MSK_DPAR_"):
            task.putnadouparam(param, value)
        elif param.startswith("MSK_IPAR_"):
            task.putnaintparam(param, value)
        elif param.startswith("MSK_SPAR_"):
            task.putnastrparam(param, value)
        else:
            raise ValueError("Invalid MOSEK parameter '%s'." % param)

    @staticmethod
    def _handle_enum_param(task, param, value):
        import mosek
        if isinstance(param, mosek.dparam):
            task.putdouparam(param, value)
        elif isinstance(param, mosek.iparam):
            task.putintparam(param, value)
        elif isinstance(param, mosek.sparam):
            task.putstrparam(param, value)
        else:
            raise ValueError("Invalid MOSEK parameter '%s'." % param)
