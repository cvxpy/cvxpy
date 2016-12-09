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
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solution import Solution
import numpy as np
import scipy.sparse as sp


class ECOS(Reduction):
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

    # Order of exponential cone arguments for solver.
    EXP_CONE_ORDER = [0, 2, 1]

    # Keys for inverse data.
    VAR_ID = 'var_id'
    EQ_CONSTR = 'eq_constr'
    NEQ_CONSTR = 'other_constr'

    def import_solver(self):
        """Imports the solver.
        """
        import ecos
        ecos  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.ECOS

    def is_mat_stuffed(self, expr):
        """Returns whether the expression is reshape(A*x + b).
        """
        # TODO

    def accepts(self, problem):
        """Can ECOS solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos, SOC, ExpCone]:
                return False
            for arg in constr.args:
                if not arg.is_affine():
                    return False
        return True

    @staticmethod
    def get_spacing_matrix(shape, spacing, offset):
        """Returns a sparse matrix that spaces out an expression.

        Parameters
        ----------
        shape : tuple
            (rows in matrix, columns in matrix)
        spacing : int
            The number of rows between each non-zero.
        offset : int
            The number of zero rows at the beginning of the matrix.

        Returns
        -------
        SciPy CSR matrix
            A sparse matrix
        """
        val_arr = []
        row_arr = []
        col_arr = []
        # Selects from each column.
        for var_row in range(shape[1]):
            val_arr.append(np.float32(1.0))
            row_arr.append(spacing*var_row + offset)
            col_arr.append(var_row)
        return sp.coo_matrix((val_arr, (row_arr, col_arr)), shape).tocsr()

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
                if constr.axis == 0:
                    offset[i*gap+1:(i+1)*gap] = offsets[1][i*(gap-1):(i+1)*(gap-1)]
                    mat_arr.append(coeffs[1][i*(gap-1):(i+1)*(gap-1), :])
                else:
                    offset[i*gap+1:(i+1)*gap] = offsets[1][i::gap-1]
                    mat_arr.append(coeffs[1][i::gap-1, :])
            return -sp.vstack(mat_arr), offset
        elif type(constr) == ExpCone:
            for i, coeff in enumerate(coeffs):
                mat = ECOS.get_spacing_matrix((height, coeff.shape[0]),
                                              len(ECOS.EXP_CONE_ORDER),
                                              ECOS.EXP_CONE_ORDER[i])
                offsets[i] = mat*offsets[i]
                coeffs[i] = -mat*coeffs[i]
            return sum(coeffs), sum(offsets)
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

    def apply(self, problem):
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data = {}
        inv_data = {ECOS.VAR_ID: problem.variables()[0].id}
        data[s.C], data[s.OFFSET] = self.get_coeff_offset(problem.objective.args[0])
        data[s.C] = data[s.C].ravel()
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[ECOS.EQ_CONSTR] = constr
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
        data[s.DIMS]['e'] = 0
        for cons in exp_constr:
            data[s.DIMS]['e'] += cons.num_cones()
        other_constr = leq_constr + soc_constr + exp_constr
        inv_data[ECOS.NEQ_CONSTR] = other_constr
        data[s.G], data[s.H] = self.group_coeff_offset(other_constr)
        return data, inv_data

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
        data, inv_data = self.apply(problem)
        solution = ecos.solve(data[s.C], data[s.G], data[s.H],
                              data[s.DIMS], data[s.A], data[s.B],
                              verbose=verbose,
                              **solver_opts)
        return self.invert(solution, inv_data)

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = self.STATUS_MAP[solution['info']['exitFlag']]

        # Timing data
        attr = {}
        attr[s.SOLVE_TIME] = solution["info"]["timing"]["tsolve"]
        attr[s.SETUP_TIME] = solution["info"]["timing"]["tsetup"]
        attr[s.NUM_ITERS] = solution["info"]["iter"]

        if status in s.SOLUTION_PRESENT:
            primal_val = solution['info']['pcost']
            opt_val = primal_val + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[ECOS.VAR_ID]: solution['x']}
            eq_dual = self.get_dual_values(solution['y'], inverse_data[ECOS.EQ_CONSTR])
            leq_dual = self.get_dual_values(solution['z'], inverse_data[ECOS.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
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
    def get_dual_values(result_vec, constraints):
        """Gets the values of the dual variables.

        Parameters
        ----------
        result_vec : array_like
            A vector containing the dual variable values.
        constraints : list
            A list of the constraints in the problem.

        Returns
        -------
           A map of constraint id to dual variable value.
        """
        # Store dual values.
        dual_vars = {}
        offset = 0
        for constr in constraints:
            # TODO reshape based on dual variable size.
            dual_vars[constr.id] = result_vec[offset:offset + constr.size]
            offset += constr.size
        return dual_vars
