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

import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solvers.conic_solvers.scs_conif import (SCS,
                                                              dims_to_solver_dict)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities
import numpy as np


def makeMstart(A, n, ifCol):

    # Construct mstart using nonzero column indices in A
    mstart = np.bincount(A.nonzero()[ifCol])
    mstart = np.concatenate((np.array([0], dtype=np.int64),
                             mstart,
                             np.array([0] * (n - len(mstart)), dtype=np.int64)))
    mstart = np.cumsum(mstart)

    return mstart


class XPRESS(SCS):
    """An interface for the Gurobi solver.
    """
    # Main member of this class: an Xpress problem. Marked with a
    # trailing "_" to denote a member
    prob_ = None
    translate_back_QP_ = False
    solvecount = 0
    version = -1

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    # Map of XPRESS status to CVXPY status.
    STATUS_MAP = {2: s.OPTIMAL,
                  3: s.INFEASIBLE,
                  5: s.UNBOUNDED,
                  4: s.SOLVER_ERROR,
                  6: s.SOLVER_ERROR,
                  7: s.SOLVER_ERROR,
                  8: s.SOLVER_ERROR,
                  9: s.SOLVER_ERROR,
                  10: s.SOLVER_ERROR,
                  11: s.SOLVER_ERROR,
                  12: s.SOLVER_ERROR,
                  13: s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.XPRESS

    def import_solver(self):
        """Imports the solver.
        """
        import xpress
        self.version = xpress.getversion()

    def accepts(self, problem):
        """Can Gurobi solve the problem?
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
        """Returns a new problem and data for inverting the new solution.

        Returns
        -------
        tuple
            (dict of arguments needed for the solver, inverse data)
        """
        data, inv_data = super(XPRESS, self).apply(problem)
        variables = problem.variables()[0]
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[XPRESS.VAR_ID]: solution['primal']}
            if not inverse_data['is_mip']:
                dual_vars = utilities.get_dual_values(
                    solution[s.EQ_DUAL],
                    utilities.extract_dual_value,
                    inverse_data[s.EQ_CONSTR])
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None

        other = {}
        other[s.XPRESS_IIS] = solution[s.XPRESS_IIS]
        other[s.XPRESS_TROW] = solution[s.XPRESS_TROW]
        return Solution(status, opt_val, primal_vars, dual_vars, other)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
        import xpress

        if 'no_qp_reduction' in solver_opts.keys() and solver_opts['no_qp_reduction'] is True:
            self.translate_back_QP_ = True

        c = data[s.C]  # objective coefficients

        dims = dims_to_solver_dict(data[s.DIMS])  # contains number of columns, rows, etc.

        nrowsEQ = dims[s.EQ_DIM]
        nrowsLEQ = dims[s.LEQ_DIM]
        nrows = nrowsEQ + nrowsLEQ

        # linear constraints
        b = data[s.B][:nrows]  # right-hand side
        A = data[s.A][:nrows]  # coefficient matrix

        # Problem
        self.prob_ = xpress.problem()

        mstart = makeMstart(A, len(c), 1)

        varGroups = {}  # If origprob is passed, used to tie IIS to original constraints
        transf2Orig = {}  # Ties transformation constraints to originals via varGroups
        nOrigVar = len(c)

        # Uses flat naming. Warning: this mixes
        # original with auxiliary variables.

        varnames = ['x_{0:05d}'. format(i) for i in range(len(c))]
        linRownames = ['lc_{0:05d}'.format(i) for i in range(len(b))]

        self.prob_.loadproblem("CVXproblem",
                               ['E'] * nrowsEQ + ['L'] * nrowsLEQ,  # qrtypes
                               None,                                # range
                               c,                                   # obj coeff
                               mstart,                              # mstart
                               None,                                # mnel
                               A.indices,                           # row indices
                               A.data,                              # coefficients
                               [-xpress.infinity] * len(c),         # lower bound
                               [xpress.infinity] * len(c),          # upper bound
                               colnames=varnames,                   # column names
                               rownames=linRownames)                # row    names

        x = np.array(self.prob_.getVariable())  # get whole variable vector

        # Set variable types for discrete variables
        self.prob_.chgcoltype(data[s.BOOL_IDX] + data[s.INT_IDX],
                              'B' * len(data[s.BOOL_IDX]) + 'I' * len(data[s.INT_IDX]))

        currow = nrows

        iCone = 0

        auxVars = set(range(nOrigVar, len(c)))

        # Conic constraints
        #
        # Quadratic objective and constraints fall in this category,
        # as all quadratic stuff is converted into a cone via a linear transformation
        for k in dims[s.SOC_DIM]:

            # k is the size of the i-th cone, where i is the index
            # within dims [s.SOC_DIM]. The cone variables in
            # CVXOPT, apparently, are separate variables that are
            # marked as conic but not shown in a cone explicitly.

            A = data[s.A][currow: currow + k].tocsr()
            b = data[s.B][currow: currow + k]
            currow += k

            if self.translate_back_QP_:

                # Conic problem passed by CVXPY is translated back
                # into a QP problem. The problem is passed to us
                # as follows:
                #
                # min c'x
                # s.t. Ax <>= b
                #      y[i] = P[i]' * x + b[i]
                #      ||y[i][1:]||_2 <= y[i][0]
                #
                # where P[i] is a matrix, b[i] is a vector. Get
                # rid of the y variables by explicitly rewriting
                # the conic constraint as quadratic:
                #
                # y[i][1:]' * y[i][1:] <= y[i][0]^2
                #
                # and hence
                #
                # (P[i][1:]' * x + b[i][1:])^2 <= (P[i][0]' * x + b[i][0])^2

                Plhs = A[1:]
                Prhs = A[0]

                indRowL, indColL = Plhs.nonzero()
                indRowR, indColR = Prhs.nonzero()

                coeL = Plhs.data
                coeR = Prhs.data

                lhs = list(b[1:])
                rhs = b[0]

                for i in range(len(coeL)):
                    lhs[indRowL[i]] -= coeL[i] * x[indColL[i]]

                for i in range(len(coeR)):
                    rhs -= coeR[i] * x[indColR[i]]

                self.prob_.addConstraint(xpress.Sum([lhs[i]**2 for i in range(len(lhs))])
                                         <= rhs**2)

            else:

                # Create new (cone) variables and add them to the problem
                conevar = np.array([xpress.var(name='cX{0:d}_{1:d}'.format(iCone, i),
                                               lb=-xpress.infinity if i > 0 else 0)
                                    for i in range(k)])

                self.prob_.addVariable(conevar)

                initrow = self.prob_.attributes.rows

                mstart = makeMstart(A, k, 0)

                trNames = ['linT_qc{0:d}_{1:d}'.format(iCone, i) for i in range(k)]

                # Linear transformation for cone variables <--> original variables
                self.prob_.addrows(['E'] * k,        # qrtypes
                                   b,                # rhs
                                   mstart,           # mstart
                                   A.indices,        # ind
                                   A.data,           # dmatval
                                   names=trNames)  # row names

                self.prob_.chgmcoef([initrow + i for i in range(k)],
                                    conevar, [1] * k)

                conename = 'cone_qc{0:d}'.format(iCone)
                # Real cone on the cone variables (if k == 1 there's no
                # need for this constraint as y**2 >= 0 is redundant)
                if k > 1:
                    self.prob_.addConstraint(
                        xpress.constraint(constraint=xpress.Sum
                                          (conevar[i]**2 for i in range(1, k))
                                          <= conevar[0] ** 2,
                                          name=conename))

                auxInd = list(set(A.indices) & auxVars)

                if len(auxInd) > 0:
                    group = varGroups[varnames[auxInd[0]]]
                    for i in trNames:
                        transf2Orig[i] = group
                    transf2Orig[conename] = group

            iCone += 1

        # Objective. Minimize is by default both here and in CVXOPT
        self.prob_.setObjective(xpress.Sum(c[i] * x[i] for i in range(len(c))))

        # End of the conditional (warm-start vs. no warm-start) code,
        # set options, solve, and report.

        # Set options
        #
        # The parameter solver_opts is a dictionary that contains only
        # one key, 'solver_opt', and its value is a dictionary
        # {'control': value}, matching perfectly the format used by
        # the Xpress Python interface.

        if verbose:
            self.prob_.controls.miplog = 2
            self.prob_.controls.lplog = 1
            self.prob_.controls.outputlog = 1
        else:
            self.prob_.controls.miplog = 0
            self.prob_.controls.lplog = 0
            self.prob_.controls.outputlog = 0

        if 'solver_opts' in solver_opts.keys():
            self.prob_.setControl(solver_opts['solver_opts'])

        self.prob_.setControl({i: solver_opts[i] for i in solver_opts.keys()
                               if i in xpress.controls.__dict__.keys()})

        # Solve
        self.prob_.solve()

        results_dict = {

            'problem':   self.prob_,
            'status':    self.prob_.getProbStatus(),
            'obj_value': self.prob_.getObjVal(),
        }

        status_map_lp, status_map_mip = self.get_status_maps()

        if self.is_mip(data):
            status = status_map_mip[results_dict['status']]
        else:
            status = status_map_lp[results_dict['status']]

        results_dict[s.XPRESS_TROW] = transf2Orig

        results_dict[s.XPRESS_IIS] = None  # Return no IIS if problem is feasible

        if status in s.SOLUTION_PRESENT:
            results_dict['x'] = self.prob_.getSolution()
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                results_dict['y'] = self.prob_.getDual()

        elif status == s.INFEASIBLE:

            # Retrieve all IIS. For LPs there can be more than one,
            # but for QCQPs there is only support for one IIS.

            iisIndex = 0

            self.prob_.iisfirst(0)  # compute all IIS

            row, col, rtype, btype, duals, rdcs, isrows, icols = [], [], [], [], [], [], [], []

            self.prob_.getiisdata(0, row, col, rtype, btype, duals, rdcs, isrows, icols)

            origrow = []
            for iRow in row:
                if iRow.name in transf2Orig.keys():
                    name = transf2Orig[iRow.name]
                else:
                    name = iRow.name

                if name not in origrow:
                    origrow.append(name)

            results_dict[s.XPRESS_IIS] = [{'orig_row': origrow,
                                           'row':      row,
                                           'col':      col,
                                           'rtype':    rtype,
                                           'btype':    btype,
                                           'duals':    duals,
                                           'redcost':  rdcs,
                                           'isolrow':  isrows,
                                           'isolcol':  icols}]

            while self.prob_.iisnext() == 0:
                iisIndex += 1
                self.prob_.getiisdata(iisIndex,
                                      row, col, rtype, btype, duals, rdcs, isrows, icols)
                results_dict[s.XPRESS_IIS].append((
                    row, col, rtype, btype, duals, rdcs, isrows, icols))

        # Generate solution.
        solution = {}

        if results_dict["status"] != s.SOLVER_ERROR:

            self.prob_ = results_dict['problem']

            vartypes = []
            self.prob_.getcoltype(vartypes, 0, len(data[s.C]) - 1)

        status_map_lp, status_map_mip = self.get_status_maps()

        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            solution[s.STATUS] = status_map_mip[results_dict['status']]
        else:
            solution[s.STATUS] = status_map_lp[results_dict['status']]

        if solution[s.STATUS] in s.SOLUTION_PRESENT:

            solution[s.PRIMAL] = results_dict['x']
            solution[s.VALUE] = results_dict['obj_value']

            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                solution[s.EQ_DUAL] = [-v for v in results_dict['y']]

        solution[s.XPRESS_IIS] = results_dict[s.XPRESS_IIS]
        solution[s.XPRESS_TROW] = results_dict[s.XPRESS_TROW]

        return solution

    def get_status_maps(self):

        """
        Create status maps from Xpress to CVXPY
        """

        import xpress

        # Map of Xpress' LP status to CVXPY status.
        status_map_lp = {

            xpress.lp_unstarted:       s.SOLVER_ERROR,
            xpress.lp_optimal:         s.OPTIMAL,
            xpress.lp_infeas:          s.INFEASIBLE,
            xpress.lp_cutoff:          s.OPTIMAL_INACCURATE,
            xpress.lp_unfinished:      s.OPTIMAL_INACCURATE,
            xpress.lp_unbounded:       s.UNBOUNDED,
            xpress.lp_cutoff_in_dual:  s.OPTIMAL_INACCURATE,
            xpress.lp_unsolved:        s.OPTIMAL_INACCURATE,
            xpress.lp_nonconvex:       s.SOLVER_ERROR
        }

        # Same map, for MIPs
        status_map_mip = {

            xpress.mip_not_loaded:     s.SOLVER_ERROR,
            xpress.mip_lp_not_optimal: s.SOLVER_ERROR,
            xpress.mip_lp_optimal:     s.SOLVER_ERROR,
            xpress.mip_no_sol_found:   s.SOLVER_ERROR,
            xpress.mip_solution:       s.OPTIMAL_INACCURATE,
            xpress.mip_infeas:         s.INFEASIBLE,
            xpress.mip_optimal:        s.OPTIMAL,
            xpress.mip_unbounded:      s.UNBOUNDED
        }

        return (status_map_lp, status_map_mip)
