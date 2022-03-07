"""
Copyright 2013 Steven Diamond, 2017 Robin Verschueren

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Dict, List, Union

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, NonNeg, Zero
from cvxpy.reductions.solvers.compr_matrix import compress_matrix
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.kktsolver import setup_ldl_factor


# Utility method for formatting a ConeDims instance into a dictionary
# that can be supplied to cvxopt.
def dims_to_solver_dict(cone_dims) -> Dict[str, Union[List[int], int]]:
    cones = {
        "l": int(cone_dims.nonneg),
        "q": [int(v) for v in cone_dims.soc],
        "s": [int(v) for v in cone_dims.psd],
    }
    return cones


class CVXOPT(ConicSolver):
    """An interface for the CVXOPT solver.
    """

    # Solver capabilities.
    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC,
                                                                 PSD]

    # Map of CVXOPT status to CVXPY status.
    STATUS_MAP = {"optimal": s.OPTIMAL,
                  "feasible": s.OPTIMAL_INACCURATE,
                  "infeasible problem": s.INFEASIBLE,
                  "primal infeasible": s.INFEASIBLE,
                  "LP relaxation is primal infeasible": s.INFEASIBLE,
                  "LP relaxation is dual infeasible": s.UNBOUNDED,
                  "unbounded": s.UNBOUNDED,
                  "dual infeasible": s.UNBOUNDED,
                  "unknown": s.SOLVER_ERROR,
                  "undefined": s.SOLVER_ERROR,
                  "solver_error": s.SOLVER_ERROR}

    # When constraints aren't provided, use zero matrix and vector of this length
    MIN_CONSTRAINT_LENGTH = 0

    def name(self):
        """The name of the solver.
        """
        return s.CVXOPT

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import cvxopt
        cvxopt  # For flake8

    def accepts(self, problem) -> bool:
        """Can CVXOPT solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in self.SUPPORTED_CONSTRAINTS:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims

        constr_map = problem.constr_map
        inv_data[self.EQ_CONSTR] = constr_map[Zero]
        inv_data[self.NEQ_CONSTR] = constr_map[NonNeg] + constr_map[SOC] + constr_map[PSD]
        len_eq = problem.cone_dims.zero

        c, d, A, b = problem.apply_parameters()
        data[s.C] = c
        inv_data[s.OFFSET] = d
        data[s.A] = -A[:len_eq]
        if data[s.A].shape[0] == 0:
            data[s.A] = None
        data[s.B] = b[:len_eq].flatten()
        if data[s.B].shape[0] == 0:
            data[s.B] = None
        if len_eq >= A.shape[0]:
            # Then the given optimization problem has no conic constraints.
            # This is certainly a degenerate case, but we'll handle it downstream.
            data[s.G] = None
            data[s.H] = None
        else:
            data[s.G] = -A[len_eq:]
            data[s.H] = b[len_eq:].flatten()
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        return super(CVXOPT, self).invert(solution, inverse_data)

    @classmethod
    def _prepare_cvxopt_matrices(cls, data) -> dict:
        """Convert constraints and cost to solver-specific format
        """
        cvxopt_data = data.copy()

        # convert cost to cvxopt format
        cvxopt_data[s.C] = intf.dense2cvxopt(cvxopt_data[s.C])

        # handle missing constraints
        var_length = cvxopt_data[s.C].size[0]
        if cvxopt_data[s.A] is None:
            cvxopt_data[s.A] = np.zeros((cls.MIN_CONSTRAINT_LENGTH, var_length))
            cvxopt_data[s.B] = np.zeros((cls.MIN_CONSTRAINT_LENGTH, 1))
        if cvxopt_data[s.G] is None:
            cvxopt_data[s.G] = np.zeros((cls.MIN_CONSTRAINT_LENGTH, var_length))
            cvxopt_data[s.H] = np.zeros((cls.MIN_CONSTRAINT_LENGTH, 1))

        # convert constraints into cvxopt format
        cvxopt_data[s.A] = intf.sparse2cvxopt(cvxopt_data[s.A])
        cvxopt_data[s.B] = intf.dense2cvxopt(cvxopt_data[s.B])
        cvxopt_data[s.G] = intf.sparse2cvxopt(cvxopt_data[s.G])
        cvxopt_data[s.H] = intf.dense2cvxopt(cvxopt_data[s.H])
        return cvxopt_data

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        import cvxopt.solvers

        # Save original cvxopt solver options.
        old_options = cvxopt.solvers.options.copy()
        # Save old data in case need to use robust solver.
        data[s.DIMS] = dims_to_solver_dict(data[s.DIMS])
        # Do a preliminary check for a certain, problematic KKT solver.
        kktsolver = self.get_kktsolver_opt(solver_opts)
        if isinstance(kktsolver, str) and kktsolver == 'chol':
            if self.remove_redundant_rows(data) == s.INFEASIBLE:
                return {s.STATUS: s.INFEASIBLE}

        data = self._prepare_cvxopt_matrices(data)

        c, G, h, dims = data[s.C], data[s.G], data[s.H], data[s.DIMS]
        A, b = data[s.A], data[s.B]

        # Apply any user-specific options.
        # Silence solver.
        solver_opts["show_progress"] = verbose
        # Rename max_iters to maxiters.
        if "max_iters" in solver_opts:
            solver_opts["maxiters"] = solver_opts["max_iters"]
        for key, value in solver_opts.items():
            cvxopt.solvers.options[key] = value

        # Always do 1 step of iterative refinement after solving KKT system.
        if "refinement" not in cvxopt.solvers.options:
            cvxopt.solvers.options["refinement"] = 1

        # finalize the KKT solver.
        if isinstance(kktsolver, str) and kktsolver == s.ROBUST_KKTSOLVER:
            kktsolver = setup_ldl_factor(c, G, h, dims, A, b)
        elif not isinstance(kktsolver, str):
            kktsolver = kktsolver(c, G, h, dims, A, b)

        try:
            results_dict = cvxopt.solvers.conelp(c, G, h, dims, A, b,
                                                 kktsolver=kktsolver)
        # Catch exceptions in CVXOPT and convert them to solver errors.
        except ValueError:
            results_dict = {"status": "unknown"}

        # Restore original cvxopt solver options.
        self._restore_solver_options(old_options)

        # Construct solution.
        solution = {}
        status = self.STATUS_MAP[results_dict['status']]
        solution[s.STATUS] = status
        if solution[s.STATUS] in s.SOLUTION_PRESENT:
            primal_val = results_dict['primal objective']
            solution[s.VALUE] = primal_val
            solution[s.PRIMAL] = results_dict['x']
            solution[s.EQ_DUAL] = results_dict['y']
            solution[s.INEQ_DUAL] = results_dict['z']
            # Need to multiply duals by Q and P_leq.
            if "Q" in data:
                y = results_dict['y']
                # Test if all constraints eliminated.
                if y.size[0] == 0:
                    dual_len = data["Q"].size[0]
                    solution[s.EQ_DUAL] = cvxopt.matrix(0., (dual_len, 1))
                else:
                    solution[s.EQ_DUAL] = data["Q"]*y
            if "P_leq" in data:
                leq_len = data[s.DIMS][s.LEQ_DIM]
                P_rows = data["P_leq"].size[0]
                new_len = P_rows + solution[s.INEQ_DUAL].size[0] - leq_len
                new_dual = cvxopt.matrix(0., (new_len, 1))
                z = solution[s.INEQ_DUAL][:leq_len]
                # Test if all constraints eliminated.
                if z.size[0] == 0:
                    new_dual[:P_rows] = 0
                else:
                    new_dual[:P_rows] = data["P_leq"] * z
                new_dual[P_rows:] = solution[s.INEQ_DUAL][leq_len:]
                solution[s.INEQ_DUAL] = new_dual

            for key in [s.PRIMAL, s.EQ_DUAL, s.INEQ_DUAL]:
                solution[key] = intf.cvxopt2dense(solution[key])
        return solution

    @staticmethod
    def remove_redundant_rows(data):
        """Check if A has redundant rows. If it does, remove redundant constraints
        from A, and apply a presolve procedure for G.

        Parameters
        ----------
        data : dict
            All the problem data.

        Returns
        -------
        str
            A status indicating if infeasibility was detected.
        """
        # Extract data.
        dims = data[s.DIMS]
        A = data[s.A]
        G = data[s.G]
        b = data[s.B]
        h = data[s.H]
        if A is None:
            return s.OPTIMAL
        TOL = 1e-10
        #
        # Use a gram matrix approach to skip dense QR factorization, if possible.
        #
        gram = A @ A.T
        if gram.shape[0] == 1:
            gram = gram.toarray().item()  # we only have one equality constraint.
            if gram > 0:
                return s.OPTIMAL
            elif not b.item() == 0.0:
                return s.INFEASIBLE
            else:
                data[s.A] = None
                data[s.B] = None
                return s.OPTIMAL
        eig = eigsh(gram, k=1, which='SM', return_eigenvectors=False)
        if eig > TOL:
            return s.OPTIMAL
        #
        # Redundant constraints exist, up to numerical tolerance;
        # reformulate equality constraints to remove this redundancy.
        #
        Q, R, P = scipy.linalg.qr(A.todense(), pivoting=True)  # pivoting helps robustness
        rows_to_keep = []
        for i in range(R.shape[0]):
            if np.linalg.norm(R[i, :]) > TOL:
                rows_to_keep.append(i)
        R = R[rows_to_keep, :]
        Q = Q[:, rows_to_keep]
        # Invert P from col -> var to var -> col.
        Pinv = np.zeros(P.size, dtype='int')
        for i in range(P.size):
            Pinv[P[i]] = i
        # Rearrage R.
        R = R[:, Pinv]
        A = R
        b_old = b
        b = Q.T.dot(b)
        # If b is not in the range of Q, the problem is infeasible.
        if not np.allclose(b_old, Q.dot(b)):
            return s.INFEASIBLE
        dims[s.EQ_DIM] = int(b.shape[0])
        data["Q"] = intf.dense2cvxopt(Q)
        #
        # Since we're applying nontrivial presolve to A, apply to G as well.
        #
        if G is not None:
            G = G.tocsr()
            G_leq = G[:dims[s.LEQ_DIM], :]
            h_leq = h[:dims[s.LEQ_DIM]].ravel()
            G_other = G[dims[s.LEQ_DIM]:, :]
            h_other = h[dims[s.LEQ_DIM]:].ravel()
            G_leq, h_leq, P_leq = compress_matrix(G_leq, h_leq)
            dims[s.LEQ_DIM] = int(h_leq.shape[0])
            data["P_leq"] = intf.sparse2cvxopt(P_leq)
            G = sp.vstack([G_leq, G_other])
            h = np.hstack([h_leq, h_other])
        # Record changes, and return.
        data[s.A] = A
        data[s.G] = G
        data[s.B] = b
        data[s.H] = h
        return s.OPTIMAL

    @staticmethod
    def _restore_solver_options(old_options) -> None:
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
