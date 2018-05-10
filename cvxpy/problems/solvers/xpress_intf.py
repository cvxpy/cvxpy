"""
(c) Copyright Fair Isaac Corporation 2017. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

THIS FILE IS DEPRECATED AND MAY BE REMOVED WITHOUT WARNING!
DO NOT CALL THESE FUNCTIONS IN YOUR CODE!
ALL SOLVER INTERFACES ARE BEING MIGRATED TO REDUCTIONS.
"""

import cvxpy.settings as s
import cvxpy.interface as intf

import cvxpy.lin_ops.lin_utils as linutils

from cvxpy.problems.solvers.solver import Solver

import numpy


def makeMstart(A, n, ifCol):

    # Construct mstart using nonzero column indices in A
    mstart = numpy.bincount(A.nonzero()[ifCol])
    mstart = numpy.concatenate((numpy.array([0], dtype=numpy.int64),
                                mstart,
                                numpy.array([0] * (n - len(mstart)), dtype=numpy.int64)))
    mstart = numpy.cumsum(mstart)

    return mstart


class XPRESS(Solver):

    """
    Interface for the FICO Xpress-Optimizer.
    Uses its Python interface to exchange data
    with CVXPY.
    """

    # Main member of this class: an Xpress problem. Marked with a
    # trailing "_" to denote a member
    prob_ = None
    translate_back_QP_ = False

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    MIP_CAPABLE = True

    SDP_CAPABLE = False
    EXP_CAPABLE = False

    solvecount = 0
    version = -1

    def name(self):
        """The name of the solver.
        """
        return s.XPRESS

    def import_solver(self):
        """Imports the solver.
        """
        import xpress
        self.version = xpress.getversion()

    def matrix_intf(self):
        """The interface for matrices passed to the solver.
        """
        return intf.DEFAULT_SPARSE_INTF  # receive a sparse (CSC) matrix

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
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

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
            Should the previous solver result be used to warm_start?
        verbose : bool
            Should the solver print output?
        solver_opts : dict
            Additional arguments for the solver.

        Returns
        -------
        tuple
            (status, optimal value, primal, equality dual, inequality dual)
        """

        import xpress

        verbose = True

        # Get problem data
        data = super(XPRESS, self).get_problem_data(objective, constraints, cached_data)

        origprob = None

        if 'original_problem' in solver_opts.keys():
            origprob = solver_opts['original_problem']

        if 'no_qp_reduction' in solver_opts.keys() and solver_opts['no_qp_reduction'] is True:
            self.translate_back_QP_ = True

        c = data[s.C]  # objective coefficients

        dims = data[s.DIMS]  # contains number of columns, rows, etc.

        nrowsEQ = dims[s.EQ_DIM]
        nrowsLEQ = dims[s.LEQ_DIM]
        nrows = nrowsEQ + nrowsLEQ

        # linear constraints
        b = data[s.B][:nrows]  # right-hand side
        A = data[s.A][:nrows]  # coefficient matrix
        data[s.BOOL_IDX] = solver_opts[s.BOOL_IDX]
        data[s.INT_IDX] = solver_opts[s.INT_IDX]

        n = c.shape[0]  # number of variables

        solver_cache = cached_data[self.name()]

        ###########################################################################################

        # Allow warm start if all dimensions match, i.e., if the
        # modified problem has the same number of rows/column and the
        # same list of cone sizes. Failing that, we can just take the
        # standard route and build the problem from scratch.

        if warm_start and \
           solver_cache.prev_result is not None and \
           n == len(solver_cache.prev_result['obj']) and \
           nrows == len(solver_cache.prev_result['rhs']) and \
           data[s.DIMS][s.SOC_DIM] == solver_cache.prev_result['cone_ind']:

            # We are re-solving a problem that was previously solved

            # Initialize the problem as the same as the previous solve
            self.prob_ = solver_cache.prev_result['problem']

            c0 = solver_cache.prev_result['obj']
            A0 = solver_cache.prev_result['mat']
            b0 = solver_cache.prev_result['rhs']

            vartype0 = solver_cache.prev_result['vartype']

            # If there is a parameter in the objective, it may have changed.
            if len(linutils.get_expr_params(objective)) > 0:
                dci = numpy.where(c != c0)[0]
                self.prob_.chgobj(dci, c[dci])

            # Get equality and inequality constraints.
            sym_data = self.get_sym_data(objective, constraints, cached_data)
            all_constrs, _, _ = self.split_constr(sym_data.constr_map)

            # If there is a parameter in the constraints,
            # A or b may have changed.

            if any(len(linutils.get_expr_params(con.expr)) > 0 for con in constraints):

                dAi = (A != A0).tocoo()  # retrieves row/col nonzeros as a tuple of two arrays
                dbi = numpy.where(b != b0)[0]

                if dAi.getnnz() > 0:
                    self.prob_.chgmcoef(dAi.row, dAi.col,
                                        [A[i, j] for (i, j) in list(zip(dAi.row, dAi.col))])

                if len(dbi) > 0:
                    self.prob_.chgrhs(dbi, b[dbi])

            vartype = []
            self.prob_.getcoltype(vartype, 0, len(data[s.C]) - 1)

            vti = (numpy.array(vartype) != numpy.array(vartype0))

            if any(vti):
                self.prob_.chgcoltype(numpy.arange(len(c))[vti], vartype[vti])

        ############################################################################################

        else:

            # No warm start, create problem from scratch

            # Problem
            self.prob_ = xpress.problem()

            mstart = makeMstart(A, len(c), 1)

            varGroups = {}  # If origprob is passed, used to tie IIS to original constraints
            transf2Orig = {}  # Ties transformation constraints to originals via varGroups
            nOrigVar = len(c)

            # From a summary knowledge of origprob.constraints() and
            # the constraints list, the following seems to hold:
            #
            # 1) origprob.constraints is the list as generated by the
            #    user. origprob.constraints[i].size returns the number
            #    of actual rows in each constraint, while .constr_id
            #    returns its id (not necessarily numbered from 0).
            #
            # 2) constraints is also a list whose every element
            #    contains fields size and constr_id. These correspond
            #    to the items in origprob.constraints, though the list
            #    is in not in order of constr_id. Also, given that it
            #    refers to the transformed problem, it contains extra
            #    constraints deriving from the cone transformations,
            #    all with a constr_id and a size.
            #
            # Given this information, attempt to set names in varnames
            # and linRownames so that they can be easily identified

            # Load linear part of the problem.

            if origprob is not None:

                # The original problem was passed, we can take a
                # better guess at the constraints and variable names.

                nOrigVar = 0
                orig_id = [i.id for i in origprob.constraints]

                varnames = []
                for v in origprob.variables():
                    nOrigVar += v.size[0]
                    if v.size[0] == 1:
                        varnames.append('{0}'. format(v.var_id))
                    else:
                        varnames.extend(['{0}_{1:d}'. format(v.var_id, j)
                                         for j in range(v.size[0])])

                varnames.extend(['aux_{0:d}'.format(i) for i in range(len(varnames), len(c))])

                # Construct constraint name list by checking constr_id for each

                linRownames = []

                for con in constraints:
                    if con.constr_id in orig_id:

                        prefix = ''

                        if type(con.constr_id) == int:
                            prefix = 'row_'

                        if con.size[0] == 1:
                            name = '{0}{1}'.format(prefix, con.constr_id)
                            linRownames.append(name)
                            transf2Orig[name] = con.constr_id

                        else:
                            names = ['{0}{1}_{2:d}'.format(prefix, con.constr_id, j)
                                     for j in range(con.size[0])]
                            linRownames.extend(names)
                            for i in names:
                                transf2Orig[i] = con.constr_id

                # Tie auxiliary variables to constraints. Scan all
                # auxiliary variables in the objective function and in
                # the corresponding columns of A.indices

                iObjQuad = 0  # keeps track of quadratic quantities in the objective

                for i in range(nOrigVar, len(c)):

                    if c[i] != 0:
                        varGroups[varnames[i]] = 'objF_{0}'.format(iObjQuad)
                        iObjQuad += 1

                    if len(A.indices[mstart[i]:mstart[i+1]]) > 0:
                        varGroups[varnames[i]] = linRownames[min(A.indices[mstart[i]:mstart[i+1]])]

            else:

                # fall back to flat naming. Warning: this mixes
                # original with auxiliary variables.

                varnames = ['x_{0:05d}'. format(i) for i in range(len(c))]
                linRownames = ['lc_{0:05d}'.format(i) for i in range(len(b))]

            self.prob_.loadproblem("CVXproblem",
                                   ['E'] * nrowsEQ + ['L'] * nrowsLEQ,  # qrtypes
                                   b,                                   # rhs
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

            x = numpy.array(self.prob_.getVariable())  # get whole variable vector

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
                    conevar = numpy.array([xpress.var(name='cX{0:d}_{1:d}'.format(iCone, i),
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
            if not self.is_mip(data):
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

        return self.format_results(results_dict, data, cached_data)

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

        if results_dict["status"] != s.SOLVER_ERROR:

            solver_cache = cached_data[self.name()]

            self.prob_ = results_dict['problem']

            # Save variable types (continuous, integer, etc.)
            vartypes = []
            self.prob_.getcoltype(vartypes, 0, len(data[s.C]) - 1)

            solver_cache.prev_result = {

                # Save data of current problem so that if
                # warm_start==True in the next call, we check these
                # and decide whether to really do a warmstart.

                'problem':  self.prob_,               # current problem
                'obj':      data[s.C],                # objective coefficients
                'mat':      data[s.A],                # matrix coefficients (+ lin tra for cones)
                'rhs':      data[s.B],                # rhs of constraints  (idem)
                'cone_ind': data[s.DIMS][s.SOC_DIM],  # cone indices (for the cone variables)
                'vartype':  vartypes                  # variable types
            }

        status_map_lp, status_map_mip = self.get_status_maps()

        if self.is_mip(data):
            new_results[s.STATUS] = status_map_mip[results_dict['status']]
        else:
            new_results[s.STATUS] = status_map_lp[results_dict['status']]

        if new_results[s.STATUS] in s.SOLUTION_PRESENT:

            new_results[s.PRIMAL] = results_dict['x']
            new_results[s.VALUE] = results_dict['obj_value']

            if not self.is_mip(data):
                new_results[s.EQ_DUAL] = [-v for v in results_dict['y']]

        new_results[s.XPRESS_IIS] = results_dict[s.XPRESS_IIS]
        new_results[s.XPRESS_TROW] = results_dict[s.XPRESS_TROW]

        return new_results

    def get_problem_data(self, objective, constraints, cached_data):

        data = super(XPRESS, self).get_problem_data(objective, constraints, cached_data)
        data['XPRESSprob'] = self.prob_
        return data
