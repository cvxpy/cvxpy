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


def makeMstart(A, n, ifCol: int = 1):
    mstart = np.bincount(A.nonzero()[ifCol])
    mstart = np.concatenate((np.array([0], dtype=np.int64),
                             mstart,
                             np.array([0] * (n - len(mstart)), dtype=np.int64)))
    mstart = np.cumsum(mstart)
    return mstart


class XPRESS(SCS):
    """An interface for the Xpress solver.

    Inherits SCS due to the rich apply() method that extracts A and other data.
    """
    solvecount = 0
    version = -1

    # Solver capabilities.
    MIP_CAPABLE = True
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    def __init__(self) -> None:
        # Main member of this class: an Xpress problem. Marked with a
        # trailing "_" to denote a member
        self.prob_ = None

    def name(self):
        """The name of the solver.
        """
        return s.XPRESS

    def import_solver(self) -> None:
        """Imports the solver.
        """
        import xpress
        self.version = xpress.getversion()

    def accepts(self, problem) -> bool:
        """Can Xpress solve the problem?
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
        variables = problem.x
        data[s.BOOL_IDX] = [int(t[0]) for t in variables.boolean_idx]
        data[s.INT_IDX] = [int(t[0]) for t in variables.integer_idx]
        inv_data['is_mip'] = data[s.BOOL_IDX] or data[s.INT_IDX]

        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution['getObjVal'] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[XPRESS.VAR_ID]: solution['primal']}
            if not inverse_data['is_mip']:
                eq_dual = utilities.get_dual_values(
                    solution['eq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[XPRESS.EQ_CONSTR])
                leq_dual = utilities.get_dual_values(
                    solution['ineq_dual'],
                    utilities.extract_dual_value,
                    inverse_data[XPRESS.NEQ_CONSTR])
                eq_dual.update(leq_dual)
                dual_vars = eq_dual
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
        other[s.SOLVE_TIME] = solution[s.SOLVE_TIME]

        return Solution(status, opt_val, primal_vars, dual_vars, other)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):

        import xpress as xp

        c = data[s.C]  # objective coefficients

        dims = dims_to_solver_dict(data[s.DIMS])  # contains number of columns, rows, etc.

        nrowsEQ = dims[s.EQ_DIM]
        nrowsLEQ = dims[s.LEQ_DIM]
        nrows = nrowsEQ + nrowsLEQ

        # linear constraints
        b = data[s.B][:nrows]  # right-hand side
        A = data[s.A][:nrows]  # coefficient matrix

        # Problem
        self.prob_ = xp.problem()

        mstart = makeMstart(A, len(c), 1)

        varGroups = {}  # If origprob is passed, used to tie IIS to original constraints
        transf2Orig = {}  # Ties transformation constraints to originals via varGroups
        nOrigVar = len(c)

        # Uses flat naming. Warning: this mixes
        # original with auxiliary variables.

        varnames = ['x_{0:05d}'. format(i) for i in range(len(c))]
        linRownames = ['lc_{0:05d}'.format(i) for i in range(len(b))]

        if verbose:
            self.prob_.controls.miplog = 2
            self.prob_.controls.lplog = 1
            self.prob_.controls.outputlog = 1
        else:
            self.prob_.controls.miplog = 0
            self.prob_.controls.lplog = 0
            self.prob_.controls.outputlog = 0
            self.prob_.controls.xslp_log = -1

        self.prob_.loadproblem(probname="CVX_xpress_conic",
                               # constraint types
                               qrtypes=['E'] * nrowsEQ + ['L'] * nrowsLEQ,
                               rhs=b,                               # rhs
                               range=None,                          # range
                               obj=c,                               # obj coeff
                               mstart=mstart,                       # mstart
                               mnel=None,                           # mnel (unused)
                               # linear coefficients
                               mrwind=A.indices[A.data != 0],       # row indices
                               dmatval=A.data[A.data != 0],         # coefficients
                               dlb=[-xp.infinity] * len(c),         # lower bound
                               dub=[xp.infinity] * len(c),          # upper bound
                               colnames=varnames,                   # column names
                               rownames=linRownames)                # row    names

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

            # Create new (cone) variables and add them to the problem
            conevar = np.array([xp.var(name='cX{0:d}_{1:d}'.format(iCone, i),
                                       lb=-xp.infinity if i > 0 else 0)
                                for i in range(k)])

            self.prob_.addVariable(conevar)

            initrow = self.prob_.attributes.rows

            mstart = makeMstart(A, k, 0)

            trNames = ['linT_qc{0:d}_{1:d}'.format(iCone, i) for i in range(k)]

            # Linear transformation for cone variables <--> original variables
            self.prob_.addrows(['E'] * k,        # qrtypes
                               b,                # rhs
                               mstart,           # mstart
                               A.indices[A.data != 0],        # ind
                               A.data[A.data != 0],           # dmatval
                               names=trNames)  # row names

            self.prob_.chgmcoef([initrow + i for i in range(k)],
                                conevar, [1] * k)

            conename = 'cone_qc{0:d}'.format(iCone)
            # Real cone on the cone variables (if k == 1 there's no
            # need for this constraint as y**2 >= 0 is redundant)
            if k > 1:
                self.prob_.addConstraint(
                    xp.constraint(constraint=xp.Sum
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

        # End of the conditional (warm-start vs. no warm-start) code,
        # set options, solve, and report.

        # Set options
        #
        # The parameter solver_opts is a dictionary that contains only
        # one key, 'solver_opt', and its value is a dictionary
        # {'control': value}, matching perfectly the format used by
        # the Xpress Python interface.

        self.prob_.setControl({i: solver_opts[i] for i in solver_opts
                               if i in xp.controls.__dict__})

        if 'bargaptarget' not in solver_opts:
            self.prob_.controls.bargaptarget = 1e-30

        if 'feastol' not in solver_opts:
            self.prob_.controls.feastol = 1e-9

        # If option given, write file before solving
        if 'write_mps' in solver_opts:
            self.prob_.write(solver_opts['write_mps'])

        # Solve
        self.prob_.solve()

        results_dict = {

            'problem':   self.prob_,
            'status':    self.prob_.getProbStatus(),
            'obj_value': self.prob_.getObjVal(),
        }

        status_map_lp, status_map_mip = get_status_maps()

        if 'mip_' in self.prob_.getProbStatusString():
            status = status_map_mip[results_dict['status']]
        else:
            status = status_map_lp[results_dict['status']]

        results_dict[s.XPRESS_TROW] = transf2Orig

        results_dict[s.XPRESS_IIS] = None  # Return no IIS if problem is feasible

        if status in s.SOLUTION_PRESENT:
            results_dict['x'] = self.prob_.getSolution()
            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                results_dict['y'] = - np.array(self.prob_.getDual())

        elif status == s.INFEASIBLE:

            # Retrieve all IIS. For LPs there can be more than one,
            # but for QCQPs there is only support for one IIS.

            iisIndex = 0

            self.prob_.iisfirst(0)  # compute all IIS

            row, col, rtype, btype, duals, rdcs, isrows, icols = [], [], [], [], [], [], [], []

            self.prob_.getiisdata(0, row, col, rtype, btype, duals, rdcs, isrows, icols)

            origrow = []
            for iRow in row:
                if iRow.name in transf2Orig:
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

        status_map_lp, status_map_mip = get_status_maps()

        if data[s.BOOL_IDX] or data[s.INT_IDX]:
            solution[s.STATUS] = status_map_mip[results_dict['status']]
        else:
            solution[s.STATUS] = status_map_lp[results_dict['status']]

        if solution[s.STATUS] in s.SOLUTION_PRESENT:

            solution[s.PRIMAL] = results_dict['x']
            solution[s.VALUE] = results_dict['obj_value']

            if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                solution[s.EQ_DUAL] = results_dict['y'][0:dims[s.EQ_DIM]]
                solution[s.INEQ_DUAL] = results_dict['y'][dims[s.EQ_DIM]:]

        solution[s.XPRESS_IIS] = results_dict[s.XPRESS_IIS]
        solution[s.XPRESS_TROW] = results_dict[s.XPRESS_TROW]

        solution['getObjVal'] = self.prob_.getObjVal()

        solution[s.SOLVE_TIME] = self.prob_.attributes.time

        del self.prob_

        return solution


def get_status_maps():
    """Create status maps from Xpress to CVXPY
    """

    import xpress as xp

    # Map of Xpress' LP status to CVXPY status.
    status_map_lp = {

        xp.lp_unstarted:       s.SOLVER_ERROR,
        xp.lp_optimal:         s.OPTIMAL,
        xp.lp_infeas:          s.INFEASIBLE,
        xp.lp_cutoff:          s.OPTIMAL_INACCURATE,
        xp.lp_unfinished:      s.OPTIMAL_INACCURATE,
        xp.lp_unbounded:       s.UNBOUNDED,
        xp.lp_cutoff_in_dual:  s.OPTIMAL_INACCURATE,
        xp.lp_unsolved:        s.OPTIMAL_INACCURATE,
        xp.lp_nonconvex:       s.SOLVER_ERROR
    }

    # Same map, for MIPs
    status_map_mip = {

        xp.mip_not_loaded:     s.SOLVER_ERROR,
        xp.mip_lp_not_optimal: s.SOLVER_ERROR,
        xp.mip_lp_optimal:     s.SOLVER_ERROR,
        xp.mip_no_sol_found:   s.SOLVER_ERROR,
        xp.mip_solution:       s.OPTIMAL_INACCURATE,
        xp.mip_infeas:         s.INFEASIBLE,
        xp.mip_optimal:        s.OPTIMAL,
        xp.mip_unbounded:      s.UNBOUNDED
    }

    return (status_map_lp, status_map_mip)
