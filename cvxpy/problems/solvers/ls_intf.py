"""
Copyright 2016 Jaehyun Park

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

import cvxpy.interface as intf
from cvxpy.error import SolverError
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
from cvxpy.utilities import QuadCoeffExtractor
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as SLA
import warnings


class LS(Solver):
    """Linearly constrained least squares solver via SciPy.
    """

    # LS is incapable of solving any general cone program,
    # and must be invoked through a special path.
    LP_CAPABLE = False
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def import_solver(self):
        """Imports the solver.
        """
        pass

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

    def suitable(self, prob):
        """Temporary method to determine whether the given Problem object is suitable for LS solver.
        """
        import cvxpy.constraints.eq_constraint as eqc

        import cvxpy.expressions.variables as var
        allowedVariables = (var.variable.Variable, var.symmetric.SymmetricUpperTri)

        # TODO: handle affine objective
        return (
            prob.is_dcp() and prob.objective.args[0].is_quadratic() and
            not prob.objective.args[0].is_affine() and
            all([isinstance(c, eqc.EqConstraint) for c in prob.constraints]) and
            all([type(v) in allowedVariables for v in prob.variables()]) and
            all([not v.domain for v in prob.variables()])  # no implicit variable domains
            # (TODO: domains are not implemented yet)
        )

    def validate_solver(self, prob):
        if not self.suitable(prob):
            raise SolverError(
                "The solver %s cannot solve the problem." % self.name()
            )

    def get_sym_data(self, objective, constraints, cached_data=None):
        class FakeSymData(object):
            def __init__(self, objective, constraints):
                self.constr_map = {s.EQ: constraints}
                vars_ = objective.variables()
                for c in constraints:
                    vars_ += c.variables()
                vars_ = list(set(vars_))
                self.vars_ = vars_
                self.var_offsets, self.var_sizes, self.x_length = self.get_var_offsets(vars_)

            def get_var_offsets(self, variables):
                var_offsets = {}
                var_sizes = {}
                vert_offset = 0
                for x in variables:
                    var_sizes[x.id] = x.size
                    var_offsets[x.id] = vert_offset
                    vert_offset += x.size[0]*x.size[1]
                return (var_offsets, var_sizes, vert_offset)

        return FakeSymData(objective, constraints)

    def solve(self, objective, constraints, cached_data,
              warm_start, verbose, solver_opts):
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

        sym_data = self.get_sym_data(objective, constraints)

        id_map = sym_data.var_offsets
        N = sym_data.x_length

        extractor = QuadCoeffExtractor(id_map, N)

        # Extract the coefficients
        (Ps, Q, R) = extractor.get_coeffs(objective.args[0])

        P = Ps[0]
        q = np.asarray(Q.todense()).flatten()
        r = R[0]

        # Forming the KKT system
        if len(constraints) > 0:
            Cs = [extractor.get_coeffs(c._expr)[1:] for c in constraints]
            As = sp.vstack([C[0] for C in Cs])
            bs = np.array([C[1] for C in Cs]).flatten()
            lhs = sp.bmat([[2*P, As.transpose()], [As, None]], format='csr')
            rhs = np.concatenate([-q, -bs])
        else:  # avoiding calling vstack with empty list
            lhs = 2*P
            rhs = -q

        warnings.filterwarnings('error')

        # Actually solving the KKT system
        try:
            sol = SLA.spsolve(lhs.tocsr(), rhs)
            x = np.array(sol[:N])
            nu = np.array(sol[N:])
            p_star = np.dot(x.transpose(), P*x + q) + r
        except SLA.MatrixRankWarning:
            x = None
            nu = None
            p_star = None

        warnings.resetwarnings()

        result_dict = {
            s.PRIMAL: x,
            s.EQ_DUAL: nu,
            s.VALUE: objective.primal_to_result(p_star)
        }

        return self.format_results(result_dict, None, cached_data)

    def format_results(self, result_dict, data, cached_data):
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

        new_results = result_dict
        if result_dict[s.PRIMAL] is None:
            new_results[s.STATUS] = s.INFEASIBLE
        else:
            new_results[s.STATUS] = s.OPTIMAL
        return new_results
