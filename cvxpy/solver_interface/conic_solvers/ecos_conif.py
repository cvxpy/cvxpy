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

import cvxpy.settings as s
from cvxpy.atoms import reshape
from cvxpy.constraints import Zero, NonPos, SOC, ExpCone
from cvxpy.problems.solvers.solver import Solver
from cvxpy.constraints.utilities import format_axis
from cvxpy.reductions.solution import Solution
import numpy as np
import scipy.sparse as sp

class ECOS(object):
    """An interface for the ECOS solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = False
    EXP_CAPABLE = True
    MIP_CAPABLE = False

    # EXITCODES from ECOS
    # ECOS_OPTIMAL  (0)   Problem solved to optimality
    # ECOS_PINF     (1)   Found certificate of primal infeasibility
    # ECOS_DINF     (2)   Found certificate of dual infeasibility
    # ECOS_INACC_OFFSET (10)  Offset exitflag at inaccurate results
    # ECOS_MAXIT    (-1)  Maximum number of iterations reached
    # ECOS_NUMERICS (-2)  Search direction unreliable
    # ECOS_OUTCONE  (-3)  s or z got outside the cone, numerics?
    # ECOS_SIGINT   (-4)  solver interrupted by a signal/ctrl-c
    # ECOS_FATAL    (-7)  Unknown problem in solver

    # Map of ECOS status to CVXPY status.
    STATUS_MAP = {0: s.OPTIMAL,
                  1: s.INFEASIBLE,
                  2: s.UNBOUNDED,
                  10: s.OPTIMAL_INACCURATE,
                  11: s.INFEASIBLE_INACCURATE,
                  12: s.UNBOUNDED_INACCURATE,
                  -1: s.SOLVER_ERROR,
                  -2: s.SOLVER_ERROR,
                  -3: s.SOLVER_ERROR,
                  -4: s.SOLVER_ERROR,
                  -7: s.SOLVER_ERROR}

    def import_solver(self):
        """Imports the solver.
        """
        import ecos
        ecos  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.ECOS

    # def accepts(problem):
    #     """Can ECOS solve the problem?
    #     """
    #     if not problem.objective.is_affine():
    #         return False
    #     for constr in problem.constraints:
    #         if type(constr) not in [Eq, Ineq, SOC, ExpCone]:
    #             return False
    #         for arg in constr:
    #             if not arg.is_affine():
    #                 return False
    #     return True

    @staticmethod
    def get_coeff_offset(expr):
        """Return the coefficient and offset in A*x + b.

        Args:
          expr: A CVXPY expression.

        Returns:
          (SciPy COO sparse matrix, NumPy 1D array)
        """
        # May be a reshape as root.
        if type(expr) == reshape:
            expr = expr.args[0]
        # Convert data to float32.
        offset = expr.args[1].value.ravel().astype(np.float32)
        coeff = expr.args[0].args[0].value.astype(np.float32)
        # Convert scalars to sparse matrices.
        if np.isscalar(coeff):
            coeff = sp.coo_matrix(([coeff], ([0], [0])), shape=(1, 1))
        return (coeff, offset)

    @staticmethod
    def format_constr(constr):
        """Return the coefficient and offset for the constraint in ECOS format.

        Args:
          constr: A CVXPY constraint.

        Returns:
          (SciPy CSR sparse matrix, NumPy 1D array)
        """
        coeffs = []
        offsets = []
        for arg in constr.args:
            coeff, offset = ECOS.get_coeff_offset(arg)
            coeffs.append(coeff.tocsr())
            offsets.append(offset)
        height = sum([c.shape[0] for c in coeffs])
        # Specialize based on constraint type.
        if type(constr) in [NonPos, Zero]:
            return coeffs[0], -offsets[0]
        elif type(constr) == SOC:
            # Group each t row with appropriate X rows.
            mat_arr = []
            offset = np.zeros(height, dtype=np.float32)
            if constr.axis == 0:
                gap = constr.args[1].shape[0] + 1
            else:
                gap = constr.args[1].shape[1] + 1
            for i in range(constr.args[0].size):
                offset[i*gap] = offsets[0][i]
                mat_arr.append(coeffs[0][i, :])
                offset[i*gap+1:(i+1)*gap] = offsets[1][i*(gap-1):(i+1)*(gap-1)]
                mat_arr.append(coeffs[1][i*(gap-1):(i+1)*(gap-1), :])
            return -sp.vstack(mat_arr), offset
        elif type(constr) == ExpCone:
            # TODO
            return None
        else:
            raise ValueError("Unsupported constraint type.")

    @staticmethod
    def group_coeff_offset(constraints):
        """Combine the constraints into a single matrix, offset.

        Args:
          constraints: A list of CVXPY constraints.

        Returns:
          (SciPy CSC sparse matrix, NumPy 1D array)
        """
        matrices = []
        offsets = []
        for cons in constraints:
            coeff, offset = ECOS.format_constr(cons)
            matrices.append(coeff)
            offsets.append(offset)
        if len(constraints) > 0:
            coeff = sp.vstack(matrices).tocsc()
            offset = np.hstack(offsets)
        else:
            coeff = None
            offset = None
        return coeff, offset

    def get_problem_data(self, problem):
        """Returns the argument for the call to the solver.

        Parameters
        ----------
        objective : LinOp
            The canonicalized objective.
        constraints : list
            The list of canonicalized cosntraints.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The arguments needed for the solver.
        """
        data = {}
        data[s.C], data[s.OFFSET] = self.get_coeff_offset(problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        data[s.OFFSET] = data[s.OFFSET][0]
        constr = [c for c in problem.constraints if type(c) == Zero]
        data[s.A], data[s.B] = self.group_coeff_offset(constr)
        # Order and group nonlinear constraints.
        data[s.DIMS] = {}
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        data[s.DIMS]['l'] = sum([np.prod(c.size) for c in leq_constr])
        soc_constr = [c for c in problem.constraints if type(c) == SOC]
        data[s.DIMS]['q'] = []
        for cons in soc_constr:
            data[s.DIMS]['q'] += cons.cone_sizes()
        exp_constr = [c for c in problem.constraints if type(c) == ExpCone]
        data[s.DIMS]['e'] = sum([c.size for c in exp_constr])
        data[s.G], data[s.H] = self.group_coeff_offset(leq_constr + soc_constr + exp_constr)
        return data

    def solve(self, problem, warm_start, verbose, solver_opts):
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
        import ecos
        data = self.get_problem_data(problem)
        global_var = problem.variables()[0]
        results_dict = ecos.solve(data[s.C], data[s.G], data[s.H],
                                  data[s.DIMS], data[s.A], data[s.B],
                                  verbose=verbose,
                                  **solver_opts)

        status = self.STATUS_MAP[results_dict['info']['exitFlag']]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = results_dict["info"]["timing"]["tsolve"]
        attr[s.SETUP_TIME] = results_dict["info"]["timing"]["tsetup"]
        attr[s.NUM_ITERS] = results_dict["info"]["iter"]

        if status in s.SOLUTION_PRESENT:
            primal_val = results_dict['info']['pcost']
            opt_val = primal_val + data[s.OFFSET]
            primal_vars = {global_var.id: results_dict['x']}
            eq_dual = self.get_dual_values(results_dict['y'], problem.constraints, [Zero])
            leq_dual = self.get_dual_values(results_dict['z'], problem.constraints,
                                            [NonPos, SOC, ExpCone])
            eq_dual.update(leq_dual)
            dual_vars = leq_dual
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None
            primal_vars = None
            dual_vars = None

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    @staticmethod
    def get_dual_values(result_vec, constraints, constr_types):
        """Gets the values of the dual variables.

        Parameters
        ----------
        result_vec : array_like
            A vector containing the dual variable values.
        constraints : list
            A list of the constraints in the problem.
        constr_types : type
            A list of constraint types to consider.
        """
        constr_offsets = {}
        offset = 0
        for constr in constraints:
            constr_offsets[constr.constr_id] = offset
            offset += constr.size
        active_constraints = []
        for constr in constraints:
            # Ignore constraints of the wrong type.
            if type(constr) in constr_types:
                active_constraints.append(constr)
        # Store dual values.
        dual_vars = {}
        for constr in active_constraints:
            if constr.id in constr_offsets:
                offset = constr_offsets[constr.id]
                dual_vars[constr.id] = result_vec[offset:offset + constr.size]
                offset += constr.size
        return dual_vars
