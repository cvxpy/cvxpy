"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.problem_data.compr_matrix import compress_matrix
from cvxpy.problems.solvers.solver import Solver
from cvxpy.problems.kktsolver import get_kktsolver
import scipy.sparse as sp
import scipy
import numpy as np
import copy

class CVXOPT(Solver):
    """An interface for the CVXOPT solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = True
    EXP_CAPABLE = True
    MIP_CAPABLE = False

    # Map of CVXOPT status to CVXPY status.
    STATUS_MAP = {'optimal': s.OPTIMAL,
                  'primal infeasible': s.INFEASIBLE,
                  'dual infeasible': s.UNBOUNDED,
                  'unknown': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CVXOPT

    def import_solver(self):
        """Imports the solver.
        """
        import cvxopt

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.CVXOPT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.CVXOPT_DENSE_INTF

    def split_constr(self, constr_map):
        """Extracts the equality, inequality, and nonlinear constraints.

        Parameters
        ----------
        constr_map : dict
            A dict of the canonicalized constraints.

        Returns
        -------
        tuple
            (eq_constr, ineq_constr, nonlin_constr)
        """
        return (constr_map[s.EQ], constr_map[s.LEQ], constr_map[s.EXP])

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.
        warm_start : bool
            Not used.
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        import cvxopt, cvxopt.solvers
        data = self.get_problem_data(objective, constraints, cached_data)
        # Save old data in case need to use robust solver.
        old_data = {
                s.DIMS: data[s.DIMS],
                s.A: data[s.A],
                s.B: data[s.B],
                s.G: data[s.G],
                s.H: data[s.H],
                s.F: data[s.F],
            }
        data[s.DIMS] = copy.deepcopy(data[s.DIMS])
        # User chosen KKT solver option.
        kktsolver = self.get_kktsolver_opt(solver_opts)
        # Cannot have redundant rows unless using robust LDL kktsolver.
        if kktsolver != s.ROBUST_KKTSOLVER:
            # Will detect infeasibility.
            if self.remove_redundant_rows(data) == s.INFEASIBLE:
                return {s.STATUS: s.INFEASIBLE}
        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options.copy()
        # Silence cvxopt if verbose is False.
        cvxopt.solvers.options["show_progress"] = verbose

        # Apply any user-specific options.
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.solvers.options[key] = value

        # Always do 1 step of iterative refinement after solving KKT system.
        if not "refinement" in cvxopt.solvers.options:
            cvxopt.solvers.options["refinement"] = 1

        try:
            # Target cvxopt clp if nonlinear constraints exist.
            if data[s.DIMS][s.EXP_DIM]:
                results_dict = self.cpl_solve(data, kktsolver)
            else:
                results_dict = self.conelp_solve(data, kktsolver)
        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {"status": "unknown"}

        # Restore original cvxopt solver options.
        self._restore_solver_options(old_options)
        return self.format_results(results_dict, data, cached_data)

    def cpl_solve(self, data, kktsolver):
        """Solve using the cpl solver.

        Parameters
        ----------
        data : dict
            All the problem data.
        kktsolver : The kktsolver to use.
        robust : Use the robust kktsolver?

        Returns
        -------
        dict
            The solver output.

        Raises
        ------
        ValueError
            If CVXOPT fails.
        """
        import cvxopt, cvxopt.solvers
        if kktsolver == s.ROBUST_KKTSOLVER:
            # Get custom kktsolver.
            kktsolver = get_kktsolver(data[s.G],
                                      data[s.DIMS],
                                      data[s.A],
                                      data[s.F])
        return cvxopt.solvers.cpl(data[s.C],
                                  data[s.F],
                                  data[s.G],
                                  data[s.H],
                                  data[s.DIMS],
                                  data[s.A],
                                  data[s.B],
                                  kktsolver=kktsolver)

    def conelp_solve(self, data, kktsolver):
        """Solve using the conelp solver.

        Parameters
        ----------
        data : dict
            All the problem data.
        kktsolver : The kktsolver to use.
        robust : Use the robust kktsolver?

        Returns
        -------
        dict
            The solver output.

        Raises
        ------
        ValueError
            If CVXOPT fails.
        """
        import cvxopt, cvxopt.solvers
        if kktsolver == s.ROBUST_KKTSOLVER:
            # Get custom kktsolver.
            kktsolver = get_kktsolver(data[s.G],
                                      data[s.DIMS],
                                      data[s.A])
        return cvxopt.solvers.conelp(data[s.C],
                                     data[s.G],
                                     data[s.H],
                                     data[s.DIMS],
                                     data[s.A],
                                     data[s.B],
                                     kktsolver=kktsolver)

    @staticmethod
    def remove_redundant_rows(data):
        """Remove redundant constraints from A and G.

        Parameters
        ----------
        data : dict
            All the problem data.

        Returns
        -------
        str
            A status indicating if infeasibility was detected.
        """
        dims = data[s.DIMS]
        # Convert A, b, G, h to scipy sparse matrices and numpy 1D arrays.
        A = intf.DEFAULT_SPARSE_INTF.const_to_matrix(data[s.A],
            convert_scalars=True)
        G = intf.DEFAULT_SPARSE_INTF.const_to_matrix(data[s.G],
            convert_scalars=True)
        b = intf.DEFAULT_NP_INTF.const_to_matrix(data[s.B],
            convert_scalars=True)
        h = intf.DEFAULT_NP_INTF.const_to_matrix(data[s.H],
            convert_scalars=True)
        # Remove redundant rows in A.
        if A.shape[0] > 0:
            # The pivoting improves robustness.
            Q, R, P = scipy.linalg.qr(A.todense(), pivoting=True)
            rows_to_keep = []
            for i in range(R.shape[0]):
                if np.linalg.norm(R[i,:]) > 1e-10:
                    rows_to_keep.append(i)
            R = R[rows_to_keep,:]
            Q = Q[:, rows_to_keep]
            # Invert P from col -> var to var -> col.
            Pinv = np.zeros(P.size, dtype='int')
            for i in range(P.size):
                Pinv[P[i]] = i
            # Rearrage R.
            R = R[:,Pinv]
            A = R
            b_old = b
            b = Q.T.dot(b)
            # If b is not in the range of Q,
            # the problem is infeasible.
            if not np.allclose(b_old, Q.dot(b)):
                return s.INFEASIBLE
            dims[s.EQ_DIM] = int(b.shape[0])
            data["Q"] = intf.CVXOPT_DENSE_INTF.const_to_matrix(Q,
                convert_scalars=True)

        # Remove obviously redundant rows in G's <= constraints.
        if dims[s.LEQ_DIM] > 0:
            G = G.tocsr()
            G_leq = G[:dims[s.LEQ_DIM],:]
            h_leq = h[:dims[s.LEQ_DIM]]
            G_other = G[dims[s.LEQ_DIM]:,:]
            h_other = h[dims[s.LEQ_DIM]:]
            G_leq, h_leq, P_leq = compress_matrix(G_leq, h_leq)
            dims[s.LEQ_DIM] = int(h_leq.shape[0])
            data["P_leq"] = intf.CVXOPT_SPARSE_INTF.const_to_matrix(P_leq,
                convert_scalars=True)
            # Scipy 0.13 can't stack empty arrays.
            if G_leq.shape[0] > 0 and G_other.shape[0] > 0:
                G = sp.vstack([G_leq, G_other])
            elif G_leq.shape[0] > 0:
                G = G_leq
            else:
                G = G_other
            h = np.vstack([h_leq, h_other])
        # Convert A, b, G, h to CVXOPT matrices.
        data[s.A] = intf.CVXOPT_SPARSE_INTF.const_to_matrix(A,
            convert_scalars=True)
        data[s.G] = intf.CVXOPT_SPARSE_INTF.const_to_matrix(G,
            convert_scalars=True)
        data[s.B] = intf.CVXOPT_DENSE_INTF.const_to_matrix(b,
            convert_scalars=True)
        data[s.H] = intf.CVXOPT_DENSE_INTF.const_to_matrix(h,
            convert_scalars=True)
        return s.OPTIMAL

    @staticmethod
    def _restore_solver_options(old_options):
        import cvxopt.solvers
        for key, value in list(cvxopt.solvers.options.items()):
            if key in old_options:
                cvxopt.solvers.options[key] = old_options[key]
            else:
                del cvxopt.solvers.options[key]

    @staticmethod
    def get_kktsolver_opt(solver_opts):
        """Returns the KKT solver selected by the user.

        Removes the KKT solver from solver_opts.

        Parameters
        ----------
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        str or None
            The KKT solver chosen by the user.
        """
        if "kktsolver" in solver_opts:
            kktsolver = solver_opts["kktsolver"]
            del solver_opts["kktsolver"]
        else:
            kktsolver = 'chol'
        return kktsolver

    def format_results(self, results_dict, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        results_dict : dict
            The solver output.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """
        new_results = {}
        status = self.STATUS_MAP[results_dict['status']]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            new_results[s.VALUE] = primal_val + data[s.OFFSET]
            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.EQ_DUAL] = results_dict['y']
            if data[s.DIMS][s.EXP_DIM]:
                new_results[s.INEQ_DUAL] = results_dict['zl']
            else:
                new_results[s.INEQ_DUAL] = results_dict['z']
            # Need to multiply duals by Q and P_leq.
            if "Q" in data:
                y = new_results[s.EQ_DUAL]
                # Test if all constraints eliminated.
                if y.size[0] == 0:
                    dual_len = data["Q"].size[0]
                    new_results[s.EQ_DUAL] = self.vec_intf().zeros(dual_len, 1)
                else:
                    new_results[s.EQ_DUAL] = data["Q"]*y
            if "P_leq" in data:
                leq_len = data[s.DIMS][s.LEQ_DIM]
                P_rows = data["P_leq"].size[1]
                new_len = P_rows + new_results[s.INEQ_DUAL].size[0] - leq_len
                new_dual = self.vec_intf().zeros(new_len, 1)
                z = new_results[s.INEQ_DUAL][:leq_len]
                # Test if all constraints eliminated.
                if z.size[0] == 0:
                    new_dual[:P_rows] = 0
                else:
                    new_dual[:P_rows] = data["P_leq"].T*z
                new_dual[P_rows:] = new_results[s.INEQ_DUAL][leq_len:]
                new_results[s.INEQ_DUAL] = new_dual

        return new_results
