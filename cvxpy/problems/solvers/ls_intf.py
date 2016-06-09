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

class LS(Solver):
    """An interface for the ECOS solver.
    """

    # Solver capabilities.
    # Incapable of solving any general cone program,
    # must be invoked through a special path.
    LP_CAPABLE = False
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def import_solver(self):
        """Imports the solver.
        """
        import ls

    def name(self):
        """The name of the solver.
        """
        return s.LS

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

    def suitable(self, prob, canon_constraints):
        """Temporary method to determine whether the given Problem object is suitable for LS solver.
        """
        import cvxpy.lin_ops as lo
        from cvxpy.constraints import SOC
        allowedConstrs = (lo.LinEqConstr, lo.LinLeqConstr, SOC)

        return (prob.is_dcp() and prob.objective.args[0].is_quadratic()
            and not prob.objective.args[0].is_affine() and
            all([constr.OP_NAME == '==' for constr in prob.constraints]) and
            all([isinstance(c, allowedConstrs) for c in canon_constraints]))

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
        from cvxopt import matrix, spmatrix, sparse
        from cvxopt.lapack import sysv
        from cvxopt.umfpack import linsolve
        import scipy.sparse as sp
        import numpy as np

        M = u.quad_coeffs(objective.args[0], id_map, N)[0]

        P = M[:N, :N]
        q = (M[:N, N] + M[N, :N].transpose())/2
        r = M[N, N]

        if len(constraints) > 0:
            Cs = [u.affine_coeffs(c._expr, id_map, N) for c in constraints]
            As = sp.vstack([C[0] for C in Cs])
            bs = sp.vstack([C[1] for C in Cs])
            AA = sp.bmat([[P, As.transpose()], [As, None]]).tocoo()
            BB = matrix(sp.vstack([-q, -bs]).todense())
        else:
            AA = P.tocoo()
            BB = matrix(-q.todense())

        AA = spmatrix(AA.data, AA.row, AA.col, AA.shape)
        try:
            linsolve(AA, BB)
            x = np.array(BB[:N, :])
            nu = np.array(BB[N:, :])
            s = np.dot(x.transpose(), P*x)
            t = q.transpose()*x
            p_star = (s+t)[0, 0] + r

        except ArithmeticError:
            x = None
            nu = None
            p_star = None

        return self.format_results(x, nu, p_star)

    def format_results(self, x, nu, p_star):
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
        if x is not None: # just for now
            new_results[s.VALUE] = p_star
            new_results[s.STATUS] = s.OPTIMAL
            new_results[s.PRIMAL] = x
            new_results[s.EQ_DUAL] = nu
        else:
            new_results[s.STATUS] = s.INFEASIBLE
        return new_results
