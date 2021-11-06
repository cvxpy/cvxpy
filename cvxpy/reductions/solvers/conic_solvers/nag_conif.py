"""
Copyright, the CVXPY authors

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

import warnings

import numpy as np
import scipy as sp

import cvxpy.settings as s
from cvxpy.constraints import SOC, NonNeg, Zero
from cvxpy.reductions.solution import Solution, failure_solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver


class NAG(ConicSolver):
    """ An interface to the NAG SOCP solver
    """

    MIP_CAPABLE = False
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    # Map of NAG status to CVXPY status
    STATUS_MAP = {0: s.OPTIMAL,
                  20: s.SOLVER_ERROR,
                  22: s.SOLVER_ERROR,
                  23: s.SOLVER_ERROR,
                  24: s.SOLVER_ERROR,
                  50: s.OPTIMAL_INACCURATE,
                  51: s.INFEASIBLE,
                  52: s.UNBOUNDED}

    def import_solver(self) -> None:
        """Imports the solver.
        """
        from naginterfaces.library import opt
        opt  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.NAG

    def accepts(self, problem) -> bool:
        """Can NAG solve the problem?
        """
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
        data = dict()
        inv_data = dict()
        inv_data[self.VAR_ID] = problem.x.id

        if not problem.formatted:
            problem = self.format_constraints(problem, None)
        data[s.PARAM_PROB] = problem
        data[self.DIMS] = problem.cone_dims
        inv_data[self.DIMS] = problem.cone_dims
        constr_map = problem.constr_map

        c, d, A, b = problem.apply_parameters()
        A = -A
        data[s.C] = c.ravel()
        data[s.OBJ_OFFSET] = float(d)
        inv_data[s.OBJ_OFFSET] = float(d)
        inv_data['lin_dim'] = []
        inv_data['soc_dim'] = []
        Gs = list()
        hs = list()
        # Linear inequalities
        num_linear_eq = len(constr_map[Zero])
        num_linear_leq = len(constr_map[NonNeg])
        leq_dim = data[s.DIMS][s.LEQ_DIM]
        eq_dim = data[s.DIMS][s.EQ_DIM]
        if num_linear_leq > 0:
            offset = num_linear_eq
            for con in problem.constraints[offset:offset + num_linear_leq]:
                inv_data['lin_dim'].append((con.id, con.size))
            row_offset = eq_dim
            Gs.append(A[row_offset:row_offset + leq_dim])
            hs.append(b[row_offset:row_offset + leq_dim])
        # Linear equations
        if num_linear_eq > 0:
            for con in problem.constraints[:num_linear_eq]:
                inv_data['lin_dim'].append((con.id, con.size))
            Gs.append(A[:eq_dim])
            hs.append(b[:eq_dim])

        # Second order cones
        num_soc = len(constr_map[SOC])
        soc_dim = sum(data[s.DIMS][s.SOC_DIM])
        if num_soc > 0:
            offset = num_linear_eq + num_linear_leq
            for con in problem.constraints[offset:offset + num_soc]:
                inv_data['soc_dim'].append((con.id, con.size))
            row_offset = leq_dim + eq_dim
            Gs.append(A[row_offset:row_offset + soc_dim])
            hs.append(b[row_offset:row_offset + soc_dim])
        data['nvar'] = len(c) + sum(data[s.DIMS][s.SOC_DIM])
        inv_data['nr'] = len(c)
        if Gs:
            data[s.G] = sp.sparse.vstack(tuple(Gs))
        else:
            data[s.G] = sp.sparse.csc_matrix((0, 0))
        if hs:
            data[s.H] = np.hstack(tuple(hs))
        else:
            data[s.H] = np.array([])
        return (data, inv_data)

    def invert(self, solution, inverse_data):

        status = self.STATUS_MAP[solution['status']]
        sln = solution['sln']

        attr = {}
        if status in s.SOLUTION_PRESENT:
            opt_val = sln.rinfo[0] + inverse_data[s.OBJ_OFFSET]
            nr = inverse_data['nr']
            x = sln.x[0:nr]
            primal_vars = {inverse_data[self.VAR_ID]: x}
            attr[s.SOLVE_TIME] = sln.stats[5]
            attr[s.NUM_ITERS] = sln.stats[0]
            # Recover dual variables
            dual_vars = dict()
            lin_dim = sum(ell for _, ell in inverse_data['lin_dim'])
            if lin_dim > 0:
                lin_dvars = np.zeros(lin_dim)
                idx = 0
                for i in range(lin_dim):
                    lin_dvars[i] = sln.u[idx+1] - sln.u[idx]
                    idx += 2
                idx = 0
                for id, dim in inverse_data['lin_dim']:
                    if dim == 1:
                        dual_vars[id] = lin_dvars[idx]
                    else:
                        dual_vars[id] = np.array(lin_dvars[idx:(idx + dim)])
                    idx += dim
            soc_dim = sum(ell for _, ell in inverse_data['soc_dim'])
            if soc_dim > 0:
                idx = 0
                for id, dim in inverse_data['soc_dim']:
                    if dim == 1:
                        dual_vars[id] = sln.uc[idx]
                    else:
                        dual_vars[id] = np.array(sln.uc[idx:(idx + dim)])
                    idx += dim
            sol = Solution(status, opt_val, primal_vars, dual_vars, attr)
        else:
            sol = failure_solution(status)
        return sol

    def solve_via_data(self, data, warm_start: bool, verbose: bool, solver_opts, solver_cache=None):
        from naginterfaces.base import utils
        from naginterfaces.library import opt

        bigbnd = 1.0e20
        c = data[s.C]
        G = data[s.G]
        h = data[s.H]
        dims = data[s.DIMS]
        nvar = data['nvar']
        soc_dim = nvar - len(c)
        nleq = dims[s.LEQ_DIM]
        neq = dims[s.EQ_DIM]
        m = len(h)
        # declare the NAG problem handle
        handle = opt.handle_init(nvar)

        # define the linear objective
        cvec = np.concatenate((c, np.zeros(soc_dim)))
        opt.handle_set_linobj(handle, cvec)
        # define linear constraints
        rows, cols, vals = sp.sparse.find(G)
        lb = np.zeros(m)
        ub = np.zeros(m)
        lb[0:nleq] = -bigbnd
        lb[nleq:m] = h[nleq:m]
        ub = h
        if nvar > len(c):
            isoc_idx = nleq + neq
            jsoc_idx = len(c)
            rows = np.concatenate((rows, np.arange(isoc_idx, isoc_idx + soc_dim)))
            cols = np.concatenate((cols, np.arange(jsoc_idx, jsoc_idx + soc_dim)))
            vals = np.concatenate((vals, np.ones(soc_dim)))
        rows = rows + 1
        cols = cols + 1
        opt.handle_set_linconstr(handle, lb, ub, rows, cols, vals)

        # define the cones
        idx = len(c)
        size_cdvars = 0
        if soc_dim > 0:
            for size_cone in dims[s.SOC_DIM]:
                opt.handle_set_group(handle, gtype='Q',
                                     group=np.arange(idx+1, idx+size_cone+1),
                                     idgroup=0)
                idx += size_cone
                size_cdvars += size_cone

        # deactivate printing by default
        opt.handle_opt_set(handle, "Print File = -1")
        if verbose:
            opt.handle_opt_set(handle, "Monitoring File = 6")
            opt.handle_opt_set(handle, "Monitoring Level = 2")

        # Set the optional parameters
        kwargs = sorted(solver_opts.keys())
        if "nag_params" in kwargs:
            for option, value in solver_opts["nag_params"].items():
                optstr = option + '=' + str(value)
                opt.handle_opt_set(handle, optstr)
            kwargs.remove("nag_params")
        if kwargs:
            raise ValueError("invalid keyword-argument '{0}'".format(kwargs[0]))

        # Use an explicit I/O manager for abbreviated iteration output:
        iom = utils.FileObjManager(locus_in_output=False)

        # Promote warnings to exceptions
        warnings.simplefilter('error', utils.NagAlgorithmicWarning)
        warnings.simplefilter('error', utils.NagAlgorithmicMajorWarning)

        # Call SOCP interior point solver
        x = np.zeros(nvar)
        status = 0
        u = np.zeros(2*m)
        uc = np.zeros(size_cdvars)
        try:
            if soc_dim > 0:
                sln = opt.handle_solve_socp_ipm(handle, x=x, u=u, uc=uc, io_manager=iom)
            elif soc_dim == 0:
                sln = opt.handle_solve_lp_ipm(handle, x=x, u=u, io_manager=iom)
        except (utils.NagValueError, utils.NagAlgorithmicWarning,
                utils.NagAlgorithmicMajorWarning) as exc:
            status = exc.errno
            sln = exc.return_data
        # Destroy the handle:
        opt.handle_free(handle)

        return {'status': status, 'sln': sln}
