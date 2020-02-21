import cvxpy.settings as s
import cvxpy.interface as intf
from cvxpy.reductions import Solution
from cvxpy.reductions.solvers.qp_solvers.qp_solver import QpSolver
from cvxpy.reductions.solvers.conic_solvers.xpress_conif import (
    get_status,
    #hide_solver_output,
    # set_parameters
)
import numpy as np

# method to get initial indices of each column
from ..conic_solvers.xpress_conif import makeMstart


class XPRESS(QpSolver):

    """Quadratic interface for the FICO Xpress solver"""

    MIP_CAPABLE = True

    def __init__ (self):
        self.prob_ = None

    def name(self):
        return s.XPRESS

    def import_solver(self):
        import xpress
        xpress

    def invert(self, results, inverse_data):
        model = results["model"]
        attr = {}
        if "cputime" in results:
            attr[s.SOLVE_TIME] = results["cputime"]
        attr[s.NUM_ITERS] = \
            int(model.attributes.bariter) \
            if not inverse_data.is_mip \
            else 0

        status = get_status(model)

        if status in s.SOLUTION_PRESENT:
            # Get objective value
            opt_val = model.getObjVal()

            # Get solution
            x = np.array(model.getSolution())
            primal_vars = {
                list(inverse_data.id_map.keys())[0]:
                intf.DEFAULT_INTF.const_to_matrix(np.array(x))
            }

            # Only add duals if not a MIP.
            dual_vars = None
            if not inverse_data.is_mip:
                y = -np.array(model.getDual())
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

        Q = data[s.P].tocoo()       # objective quadratic coefficients
        q = data[s.Q]               # objective linear coefficient (size n_var)

        # Equations, Ax = b

        A = data[s.A]               # linear coefficient matrix
        b = data[s.B]               # rhs

        # Inequalities, Fx <= g

        F = data[s.F]               # linear coefficient matrix
        g = data[s.G]               # rhs

        n_var = data['n_var']
        n_eq = data['n_eq']
        n_ineq = data['n_ineq']

        self.prob_ = xp.problem()

        # setControl() accepts 'option':value dictionaries
        self.prob_.setControl(solver_opts)

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
            nElemF = mstartF[i+1] - mstartF[i]  #                       F

            if nElemA:
                mrwind[mstart[i]:mstart[i] + nElemA] = A.indices[mstartA[i]:mstartA[i+1]]
                dmatval[mstart[i]:mstart[i] + nElemA] = A.data[mstartA[i]:mstartA[i+1]]
            if nElemF:
                mrwind[mstart[i] + nElemA: mstart[i] + nElemA + nElemF] = F.indices[mstartF[i]:mstartF[i+1]]
                dmatval[mstart[i] + nElemA: mstart[i] + nElemA + nElemF] = F.data[mstartF[i]:mstartF[i+1]]

            mstart[i] = mstartA[i] + mstartF[i]

        # The last value of mstart must be the total number of
        # coefficients.
        mstart[n_var] = mstartA[n_var] + mstartF[n_var]

        # Q matrix is input via row/col indices and value, but only
        # for the upper triangle. We just make it symmetric and twice
        # itself, then, just remove all lower-triangular elements.
        Q += Q.transpose()

        mqcol1 = Q.row  [Q.row <= Q.col]
        mqcol2 = Q.col  [Q.row <= Q.col]
        dqe    = Q.data [Q.row <= Q.col]

        colnames = ['x_{0:09d}'.format(i) for i in range(n_var)]
        rownames = ['eq_{0:09d}'.format(i) for i in range(n_eq)] + \
            ['ineq_{0:09d}'.format(i) for i in range(n_ineq)] 

        self.prob_.loadproblem (probname='CVXPY_xpress_qp',
                                qrtypes=['E']*n_eq + ['L']*n_ineq,
                                rhs=b + g,
                                range=None,
                                obj=q,
                                mstart=mstart,
                                mnel=None,
                                # linear coefficients
                                mrwind=mrwind,
                                dmatval=dmatval,
                                # variable bounds
                                dlb=[-xp.infinity]*n_var,
                                dub=[xp.infinity]*nvar,
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

        # Set verbosity
        if not verbose:
            self.prob_.controls.outputlog = 0

        # Solve problem
        results_dict = {}
        try:
            self.prob_.solve()
            results_dict["cputime"] = self.prob_.attributes.time
        except Exception:  # Error in the solution
            results_dict["status"] = s.SOLVER_ERROR

        results_dict["model"] = self.prob_

        return results_dict
