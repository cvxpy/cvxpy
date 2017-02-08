"""
Copyright 2015 Enzo Busseti

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
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
import numpy as np
import scipy.sparse as sp


class MOSEK(Solver):
    """An interface for the MOSEK solver.
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = True
    SDP_CAPABLE = True
    EXP_CAPABLE = False
    MIP_CAPABLE = False

    def import_solver(self):
        """Imports the solver.
        """
        import mosek
        mosek  # For flake8

    def name(self):
        """The name of the solver.
        """
        return s.MOSEK

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

    @staticmethod
    def _handle_mosek_params(task, params):
        if params is None:
            return

        import mosek

        def _handle_str_param(param, value):
            if param.startswith("MSK_DPAR_"):
                task.putnadouparam(param, value)
            elif param.startswith("MSK_IPAR_"):
                task.putnaintparam(param, value)
            elif param.startswith("MSK_SPAR_"):
                task.putnastrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        def _handle_enum_param(param, value):
            if isinstance(param, mosek.dparam):
                task.putdouparam(param, value)
            elif isinstance(param, mosek.iparam):
                task.putintparam(param, value)
            elif isinstance(param, mosek.sparam):
                task.putstrparam(param, value)
            else:
                raise ValueError("Invalid MOSEK parameter '%s'." % param)

        for param, value in params.items():
            if isinstance(param, str):
                _handle_str_param(param.strip(), value)
            else:
                _handle_enum_param(param, value)

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
        import mosek
        with mosek.Env() as env:
            with env.Task(0, 0) as task:
                kwargs = sorted(solver_opts.keys())
                if "mosek_params" in kwargs:
                    self._handle_mosek_params(task, solver_opts["mosek_params"])
                    kwargs.remove("mosek_params")
                if kwargs:
                    raise ValueError("Invalid keyword-argument '%s'" % kwargs[0])

                if verbose:
                    # Define a stream printer to grab output from MOSEK
                    def streamprinter(text):
                        import sys
                        sys.stdout.write(text)
                        sys.stdout.flush()
                    env.set_Stream(mosek.streamtype.log, streamprinter)
                    task.set_Stream(mosek.streamtype.log, streamprinter)

                data = self.get_problem_data(objective, constraints, cached_data)

                A = data[s.A]
                b = data[s.B]
                G = data[s.G]
                h = data[s.H]
                c = data[s.C]
                dims = data[s.DIMS]

                # size of problem
                numvar = len(c) + sum(dims[s.SOC_DIM])
                numcon = len(b) + dims[s.LEQ_DIM] + sum(dims[s.SOC_DIM]) + \
                    sum([el**2 for el in dims[s.SDP_DIM]])

                # otherwise it crashes on empty probl.
                if numvar == 0:
                    result_dict = {s.STATUS: s.OPTIMAL}
                    result_dict[s.PRIMAL] = []
                    result_dict[s.VALUE] = 0. + data[s.OFFSET]
                    result_dict[s.EQ_DUAL] = []
                    result_dict[s.INEQ_DUAL] = []
                    return result_dict

                # objective
                task.appendvars(numvar)
                task.putclist(np.arange(len(c)), c)
                task.putvarboundlist(np.arange(numvar, dtype=int),
                                     [mosek.boundkey.fr]*numvar,
                                     np.zeros(numvar),
                                     np.zeros(numvar))

                # SDP variables
                if sum(dims[s.SDP_DIM]) > 0:
                    task.appendbarvars(dims[s.SDP_DIM])

                # linear equality and linear inequality constraints
                task.appendcons(numcon)
                if A.shape[0] and G.shape[0]:
                    constraints_matrix = sp.bmat([[A], [G]])
                else:
                    constraints_matrix = A if A.shape[0] else G
                coefficients = np.concatenate([b, h])

                row, col, el = sp.find(constraints_matrix)
                task.putaijlist(row, col, el)

                type_constraint = [mosek.boundkey.fx] * len(b)
                type_constraint += [mosek.boundkey.up] * dims[s.LEQ_DIM]
                sdp_total_dims = sum([cdim**2 for cdim in dims[s.SDP_DIM]])
                type_constraint += [mosek.boundkey.fx] * \
                    (sum(dims[s.SOC_DIM]) + sdp_total_dims)

                task.putconboundlist(np.arange(numcon, dtype=int),
                                     type_constraint,
                                     coefficients,
                                     coefficients)

                # cones
                current_var_index = len(c)
                current_con_index = len(b) + dims[s.LEQ_DIM]

                for size_cone in dims[s.SOC_DIM]:
                    row, col, el = sp.find(sp.eye(size_cone))
                    row += current_con_index
                    col += current_var_index
                    task.putaijlist(row, col, el)  # add a identity for each cone
                    # add a cone constraint
                    task.appendcone(mosek.conetype.quad,
                                    0.0,  # unused
                                    np.arange(current_var_index,
                                              current_var_index + size_cone))
                    current_con_index += size_cone
                    current_var_index += size_cone

                # SDP
                for num_sdp_var, size_matrix in enumerate(dims[s.SDP_DIM]):
                    for i_sdp_matrix in range(size_matrix):
                        for j_sdp_matrix in range(size_matrix):
                            coeff = 1. if i_sdp_matrix == j_sdp_matrix else .5
                            task.putbaraij(current_con_index,
                                           num_sdp_var,
                                           [task.appendsparsesymmat(size_matrix,
                                                                    [max(i_sdp_matrix,
                                                                         j_sdp_matrix)],
                                                                    [min(i_sdp_matrix,
                                                                         j_sdp_matrix)],
                                                                    [coeff])],
                                           [1.0])
                            current_con_index += 1

                # solve
                task.putobjsense(mosek.objsense.minimize)
                task.optimize()

                if verbose:
                    task.solutionsummary(mosek.streamtype.msg)

                return self.format_results(task, data, cached_data)

    def format_results(self, task, data, cached_data):
        """Converts the solver output into standard form.

        Parameters
        ----------
        task : mosek.Task
            The solver status interface.
        data : dict
            Information about the problem.
        cached_data : dict
            A map of solver name to cached problem data.

        Returns
        -------
        dict
            The solver output in standard form.
        """

        import mosek
        # Map of MOSEK status to CVXPY status.
        # taken from:
        # http://docs.mosek.com/7.0/pythonapi/Solution_status_keys.html
        STATUS_MAP = {mosek.solsta.optimal: s.OPTIMAL,
                      mosek.solsta.prim_infeas_cer: s.INFEASIBLE,
                      mosek.solsta.dual_infeas_cer: s.UNBOUNDED,
                      mosek.solsta.near_optimal: s.OPTIMAL_INACCURATE,
                      mosek.solsta.near_prim_infeas_cer: s.INFEASIBLE_INACCURATE,
                      mosek.solsta.near_dual_infeas_cer: s.UNBOUNDED_INACCURATE,
                      mosek.solsta.unknown: s.SOLVER_ERROR}

        task.getprosta(mosek.soltype.itr)  # unused
        solsta = task.getsolsta(mosek.soltype.itr)

        result_dict = {s.STATUS: STATUS_MAP[solsta]}

        # Callback data example:
        # http://docs.mosek.com/7.1/pythonapi/The_progress_call-back.html
        # Retrieving double information items:
        # http://docs.mosek.com/7.1/pythonapi/Task_getdouinf_.html#@generated-ID:5ef16e0
        # http://docs.mosek.com/7.1/pythonapi/Double_information_items.html
        result_dict[s.SOLVE_TIME] = task.getdouinf(mosek.dinfitem.optimizer_time)
        result_dict[s.SETUP_TIME] = task.getdouinf(mosek.dinfitem.presolve_time)
        result_dict[s.NUM_ITERS] = task.getintinf(mosek.iinfitem.intpnt_iter)

        if result_dict[s.STATUS] in s.SOLUTION_PRESENT:
            # get primal variables values
            result_dict[s.PRIMAL] = np.zeros(task.getnumvar(), dtype=np.float)
            task.getxx(mosek.soltype.itr, result_dict[s.PRIMAL])
            # get obj value
            result_dict[s.VALUE] = task.getprimalobj(mosek.soltype.itr) + \
                data[s.OFFSET]
            # get dual
            y = np.zeros(task.getnumcon(), dtype=np.float)
            task.gety(mosek.soltype.itr, y)
            # it appears signs are inverted
            result_dict[s.EQ_DUAL] = -y[:len(data[s.B])]
            result_dict[s.INEQ_DUAL] = \
                -y[len(data[s.B]):len(data[s.B])+data[s.DIMS][s.LEQ_DIM]]

        return result_dict
