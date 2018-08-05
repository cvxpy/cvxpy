"""
Copyright 2016, 2018 Sascha-Dominic Schnug

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
WE ARE MOVING ALL SOLVER INTERFACES TO THE REDUCTIONS FOLDER.
"""
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
import numpy as np


class CBC(Solver):
    """ An interface to the CBC solver
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = False
    PSD_CAPABLE = False
    EXP_CAPABLE = False
    MIP_CAPABLE = True

    # Map of GLPK MIP status to CVXPY status.
    STATUS_MAP_MIP = {'solution': s.OPTIMAL,
                      'relaxation infeasible': s.INFEASIBLE,
                      'stopped on user event': s.SOLVER_ERROR}

    STATUS_MAP_LP = {'optimal': s.OPTIMAL,
                     'primal infeasible': s.INFEASIBLE,
                     'stopped due to errors': s.SOLVER_ERROR,
                     'stopped by event handler (virtual int '
                     'ClpEventHandler::event())': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CBC

    def import_solver(self):
        """Imports the solver.
        """
        from cylp.cy import CyClpSimplex
        CyClpSimplex  # For flake8

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
        return (constr_map[s.EQ] + constr_map[s.LEQ], [], [])

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
        # Import basic modelling tools of cylp
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPModel, CyLPArray

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        dims = data[s.DIMS]
        data[s.BOOL_IDX] = solver_opts[s.BOOL_IDX]
        data[s.INT_IDX] = solver_opts[s.INT_IDX]

        n = c.shape[0]

        # Problem
        model = CyLPModel()

        # Variables
        x = model.addVariable('x', n)

        # Constraints
        # eq
        model += A[0:dims[s.EQ_DIM], :] * x == CyLPArray(b[0:dims[s.EQ_DIM]])

        # leq
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        model += A[leq_start:leq_end, :] * x <= CyLPArray(b[leq_start:leq_end])

        # Objective
        model.objective = c

        # Convert model
        model = CyClpSimplex(model)

        # No boolean vars available in Cbc -> model as int + restrict to [0,1]
        if self.is_mip(data):
            # Mark integer- and binary-vars as "integer"
            model.setInteger(x[data[s.BOOL_IDX]])
            model.setInteger(x[data[s.INT_IDX]])

            # Restrict binary vars only
            idxs = data[s.BOOL_IDX]
            n_idxs = len(idxs)

            model.setColumnLowerSubset(np.arange(n_idxs, dtype=np.int32),
                                       np.array(idxs, np.int32),
                                       np.zeros(n_idxs))

            model.setColumnUpperSubset(np.arange(n_idxs, dtype=np.int32),
                                       np.array(idxs, np.int32),
                                       np.ones(n_idxs))

        # Verbosity Clp
        if not verbose:
            model.logLevel = 0

        # Build model & solve
        status = None
        if self.is_mip(data):
            # Convert model
            cbcModel = model.getCbcModel()

            # Verbosity Cbc
            if not verbose:
                cbcModel.logLevel = 0

            # cylp: /cylp/cy/CyCbcModel.pyx#L134
            # Call CbcMain. Solve the problem using the same parameters used by
            # CbcSolver. Equivalent to solving the model from the command line
            # using cbc's binary.
            cbcModel.solve()
            status = cbcModel.status
        else:
            # cylp: /cylp/cy/CyClpSimplex.pyx
            # Run CLP's initialSolve. It does a presolve and uses primal or dual
            # Simplex to solve a problem.
            status = model.initialSolve()

        results_dict = {}
        results_dict["status"] = status

        if self.is_mip(data):
            results_dict["x"] = cbcModel.primalVariableSolution['x']
            results_dict["obj_value"] = cbcModel.objectiveValue
        else:
            results_dict["x"] = model.primalVariableSolution['x']
            results_dict["obj_value"] = model.objectiveValue

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
        status = None
        if self.is_mip(data):
            status = self.STATUS_MAP_MIP[results_dict['status']]
        else:
            status = self.STATUS_MAP_LP[results_dict['status']]
        new_results[s.STATUS] = status
        if new_results[s.STATUS] in s.SOLUTION_PRESENT:
            # No dual variables.
            new_results[s.PRIMAL] = results_dict['x']
            primal_val = results_dict["obj_value"]
            new_results[s.VALUE] = primal_val

        return new_results
