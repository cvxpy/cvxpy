"""
Copyright 2016 Jaehyun Park

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
from cvxpy.problems.solvers.solver import Solver
import cvxpy.utilities as u

class LCLS(Solver):
    """An interface for the ECOS solver.
    """

    # Solver capabilities.
    LP_CAPABLE = False
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def import_solver(self):
        """Imports the solver.
        """
        import lcls

    def name(self):
        """The name of the solver.
        """
        return s.LCLS

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTF

    def vec_intf(self):
        """The interface for vectors passed to the solver.
        """
        return intf.DEFAULT_INTF

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
        return (constr_map[s.EQ], constr_map[s.LEQ], [])

    def solve(self, objective, constraints, id_map, N):
        """Returns the result of the call to the solver.

        Parameters
        ----------
        objective : CVXPY objective object
            Raw objective passed by CVXPY. Can be convex/concave.
        constraints : list
            The list of raw constraints.
        
        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """
        from cvxopt import matrix, spmatrix
        from cvxopt.lapack import posv, gesv
        import scipy.sparse as sp
        import numpy as np

        M = u.quad_coeffs(objective.args[0], id_map, N)[0]
        M = M.todense()
        P = M[:N, :N]
        q = (M[:N, N] + M[N, :N].transpose())/2
        r = M[N, N]
        As = np.empty([0, N])
        bs = np.empty([0, 1])
        for constr in constraints:
            (A, b) = u.affine_coeffs(constr._expr, id_map, N)
            A = A.todense()
            b = b.todense()
            As = np.vstack((As, A))
            bs = np.vstack((bs, b))

        m = As.shape[0]
        AA = matrix( np.bmat([[P, As.transpose()], [As, np.zeros((m, m))]]) )
        BB = matrix( np.vstack((-q, -bs)) )
        gesv(AA, BB)

        x = np.asmatrix(np.array(BB[:N, :]))
        nu = np.asmatrix(np.array(BB[N:, :]))
        p_star = np.dot(x.transpose(), np.dot(P, x)) + 2*np.dot(q.transpose(), x) + r
        p_star = p_star[0, 0]

        return self.format_results(p_star, x, nu)

    def format_results(self, p_star, x, nu):
    #def format_results(self, results_dict, data, cached_data):
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
        new_results[s.VALUE] = p_star
        new_results[s.STATUS] = s.OPTIMAL # just for now
        new_results[s.PRIMAL] = x
        new_results[s.EQ_DUAL] = nu
        return new_results
