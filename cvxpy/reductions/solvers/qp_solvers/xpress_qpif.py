import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import (
    makeMstart,
    get_status_maps
)

import numpy as np


class XPRESS(QpSolver):

    """Quadratic interface for the FICO Xpress solver"""

    MIP_CAPABLE = True

    def __init__(self):
        self.prob_ = None

    def name(self):
        return s.XPRESS

    def import_solver(self):

        import xpress
        xpress  # Prevents flake8 warning

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

        return data, inv_data

    def invert(self, results, inverse_data):
        # model = results["model"]
        attr = {}
        if "cputime" in results:
            attr[s.SOLVE_TIME] = results["cputime"]
        attr[s.NUM_ITERS] = \
            int(results['bariter']) \
            if not inverse_data[XPRESS.IS_MIP] \
            else 0

        status_map_lp, status_map_mip = get_status_maps()

        if results['status'] == 'solver_error':
            status = 'solver_error'
        elif 'mip_' in results['getProbStatusString']:
            status = status_map_mip[results['status']]
        else:
            status = status_map_lp[results['status']]

        if status in s.SOLUTION_PRESENT:
            # Get objective value
            opt_val = results['getObjVal'] + inverse_data[s.OFFSET]

            # Get solution
            x = np.array(results['getSolution'])
            primal_vars = {
                XPRESS.VAR_ID:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data[XPRESS.IS_MIP]:
                y = -np.array(results['getDual'])
                dual_vars = {XPRESS.DUAL_VAR_ID: y}

        else:
            primal_vars = None
            dual_vars = None
            opt_val = np.inf
            if status == s.UNBOUNDED:
                opt_val = -np.inf

        return Solution(status, opt_val, primal_vars, dual_vars, attr)

    def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):

        import xpress as xp

        # Objective function: 1/2 x' P x + q'x

        Q = data[s.P]          # objective quadratic coefficients
        q = data[s.Q]          # objective linear coefficient (size n_var)

        # Equations, Ax = b

        A = data[s.A]          # linear coefficient matrix
        b = data[s.B]          # rhs

        # Inequalities, Fx <= g

        F = data[s.F]          # linear coefficient matrix
        g = data[s.G]          # rhs

        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        self.prob_ = xp.problem()

        # qp_solver has the following format:
        #
        #    minimize      1/2 x' P x + q' x
        #    subject to    A x =  b
        #                  F x <= g

        mstartA = makeMstart(A, n_var, 1)
        mstartF = makeMstart(F, n_var, 1)

        # index start is simply the sum of the two mstarts for A and F
        mstart = mstartA + mstartF

        # In order to arrange (A;F) as a single matrix, we have to
        # concatenate all indices. I prefer not to do a vstack of A
        # and F as it might increase memory usage a lot.

        # Begin by creating two vectors of zeros for indices and
        # coefficients
        mrwind = np.zeros(mstart[n_var], dtype=np.int64)
        dmatval = np.zeros(mstart[n_var], dtype=np.float64)

        # Fill in mrwind and dmatval by alternately drawing from
        # indices/coeff of A and F.

        for i in range(n_var):

            nElemA = mstartA[i+1] - mstartA[i]  # number of elements of A in this column
            nElemF = mstartF[i+1] - mstartF[i]  # number of elements of F in this column

            if nElemA:
                mrwind[mstart[i]:mstart[i] + nElemA] = \
                    A.indices[A.data != 0][mstartA[i]:mstartA[i+1]]
                dmatval[mstart[i]:mstart[i] + nElemA] = \
                    A.data[A.data != 0][mstartA[i]:mstartA[i+1]]
            if nElemF:
                mrwind[mstart[i] + nElemA: mstart[i] + nElemA + nElemF] = \
                    n_eq + F.indices[F.data != 0][mstartF[i]:mstartF[i+1]]
                dmatval[mstart[i] + nElemA: mstart[i] + nElemA + nElemF] = \
                    F.data[F.data != 0][mstartF[i]:mstartF[i+1]]

            mstart[i] = mstartA[i] + mstartF[i]

        # The last value of mstart must be the total number of
        # coefficients.
        mstart[n_var] = mstartA[n_var] + mstartF[n_var]

        # Q matrix is input via row/col indices and value, but only
        # for the upper triangle. We just make it symmetric and twice
        # itself, then, just remove all lower-triangular elements.
        Q += Q.transpose()
        Q /= 2
        Q = Q.tocoo()

        mqcol1 = Q.row[Q.row <= Q.col]
        mqcol2 = Q.col[Q.row <= Q.col]
        dqe = Q.data[Q.row <= Q.col]

        colnames = ['x_{0:09d}'.format(i) for i in range(n_var)]
        rownames = ['eq_{0:09d}'.format(i) for i in range(n_eq)] + \
            ['ineq_{0:09d}'.format(i) for i in range(n_ineq)]

        if verbose:
            self.prob_.controls.miplog = 2
            self.prob_.controls.lplog = 1
            self.prob_.controls.outputlog = 1
        else:
            self.prob_.controls.miplog = 0
            self.prob_.controls.lplog = 0
            self.prob_.controls.outputlog = 0
            self.prob_.controls.xslp_log = -1

        self.prob_.loadproblem(probname='CVX_xpress_qp',
                               qrtypes=['E']*n_eq + ['L']*n_ineq,
                               rhs=list(b) + list(g),
                               range=None,
                               obj=q,
                               mstart=mstart,
                               mnel=None,
                               # linear coefficients
                               mrwind=mrwind,
                               dmatval=dmatval,
                               # variable bounds
                               dlb=[-xp.infinity]*n_var,
                               dub=[xp.infinity]*n_var,
                               # quadratic objective (only upper triangle)
                               mqcol1=mqcol1,
                               mqcol2=mqcol2,
                               dqe=dqe,
                               # binary and integer variables
                               qgtype=['B']*len(data[s.BOOL_IDX]) + ['I']*len(data[s.INT_IDX]),
                               mgcols=data[s.BOOL_IDX] + data[s.INT_IDX],
                               # variables' and constraints' names
                               colnames=colnames,
                               rownames=rownames)

        # Set options
        #
        # The parameter solver_opts is a dictionary that contains only
        # one key, 'solver_opt', and its value is a dictionary
        # {'control': value}, matching perfectly the format used by
        # the Xpress Python interface.

        # Set options if compatible with Xpress problem control names

        self.prob_.setControl({i: solver_opts[i] for i in solver_opts
                               if hasattr(xp.controls.__dict__, i)})

        if 'bargaptarget' not in solver_opts.keys():
            self.prob_.controls.bargaptarget = 1e-30

        if 'feastol' not in solver_opts.keys():
            self.prob_.controls.feastol = 1e-9

        # Solve problem
        results_dict = {"model": self.prob_}
        try:

            # If option given, write file before solving
            if 'write_mps' in solver_opts.keys():
                self.prob_.write(solver_opts['write_mps'])

            self.prob_.solve()
            results_dict["cputime"] = self.prob_.attributes.time
        except xp.SolverError:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR
        else:
            results_dict['status'] = self.prob_.getProbStatus()
            results_dict['getProbStatusString'] = self.prob_.getProbStatusString()
            results_dict['obj_value'] = self.prob_.getObjVal()
            try:
                results_dict[s.PRIMAL] = np.array(self.prob_.getSolution())
            except xp.SolverError:
                results_dict[s.PRIMAL] = np.zeros(self.prob_.attributes.ncol)

            status_map_lp, status_map_mip = get_status_maps()

            if results_dict['status'] == 'solver_error':
                status = 'solver_error'
            elif 'mip_' in results_dict['getProbStatusString']:
                status = status_map_mip[results_dict['status']]
            else:
                status = status_map_lp[results_dict['status']]

            results_dict['bariter'] = self.prob_.attributes.bariter
            results_dict['getProbStatusString'] = self.prob_.getProbStatusString()

            if status in s.SOLUTION_PRESENT:
                results_dict['getObjVal'] = self.prob_.getObjVal()
                results_dict['getSolution'] = self.prob_.getSolution()

                if not (data[s.BOOL_IDX] or data[s.INT_IDX]):
                    results_dict['getDual'] = self.prob_.getDual()

        del self.prob_

        return results_dict
