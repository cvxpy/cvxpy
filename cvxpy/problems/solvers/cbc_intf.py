"""
Copyright 2016, 2018 Sascha-Dominic Schnug

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

import importlib
import six
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver
import numpy as np
import scipy.sparse as sp

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

    SUPPORTED_CUT_GENERATORS = {"GomoryCuts": "CyCglGomory",
                                "MIRCuts": "CyCglMixedIntegerRounding",
                                "MIRCuts2": "CyCglMixedIntegerRounding2",
                                "TwoMIRCuts": "CyCglTwomir",
                                "ResidualCapacityCuts": "CyCglResidualCapacity",
                                "KnapsackCuts": "CyCglKnapsackCover",
                                "FlowCoverCuts": "CyCglFlowCover",
                                "CliqueCuts": "CyCglClique",
                                "LiftProjectCuts": "CyCglLiftAndProject",
                                "AllDifferentCuts": "CyCglAllDifferent",
                                "OddHoleCuts": "CyCglOddHole",
                                "RedSplitCuts": "CyCglRedSplit",
                                "LandPCuts": "CyCglLandP",
                                "PreProcessCuts": "CyCglPreProcess",
                                "ProbingCuts": "CyCglProbing",
                                "SimpleRoundingCuts": "CyCglSimpleRounding"}

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
        model = CyClpSimplex()

        # Variables
        x = model.addVariable('x', n)

        # no boolean vars available in cbc -> model as int + restrict to [0,1]
        if self.is_mip(data):
            model.setInteger(x[data[s.BOOL_IDX]])
            model.setInteger(x[data[s.INT_IDX]])
            idxs = data[s.BOOL_IDX]
            n_idxs = len(idxs)

            bin_constrs = sp.coo_matrix((np.ones(n_idxs),
                                        (np.arange(n_idxs), idxs)),
                                         shape=(n_idxs, n))

            model.addConstraint(bin_constrs * x >= 0)
            model.addConstraint(bin_constrs * x <= 1)

        # Constraints
        # eq
        model += A[0:dims[s.EQ_DIM], :] * x == b[0:dims[s.EQ_DIM]]

        # leq
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        model += A[leq_start:leq_end, :] * x <= b[leq_start:leq_end]

        # Objective
        model.objective = c

        # Verbosity Clp
        if not verbose:
            model.logLevel = 0

        # Build model & solve
        status = None
        if self.is_mip(data):
            model.initialSolve()            # see comment in else-branch
            cbcModel = model.getCbcModel()  # after initial LP relaxation:
                                            # need to convert model to use Cbc

            # Verbosity Cbc
            if not verbose:
                cbcModel.logLevel = 0

            # Add cut-generators (optional)
            for cut_name, cut_func in six.iteritems(self.SUPPORTED_CUT_GENERATORS):
                if cut_name in solver_opts and solver_opts[cut_name]:
                    module = importlib.import_module("cylp.cy.CyCgl")
                    funcToCall = getattr(module, cut_func)
                    cut_gen = funcToCall()
                    cbcModel.addCutGenerator(cut_gen, name=cut_name)

            # https://www.coin-or.org/Doxygen/Cbc/classCbcModel.html
            # void CbcModel::branchAndBound(int	doStatistics=0)
            # Invoke the branch & cut algorithm.
            # The method assumes that initialSolve() has been called to solve
            # the LP relaxation. It processes the root node, then proceeds to
            # explore the branch & cut search tree. The search ends when the
            # tree is exhausted or one of several execution limits is reached.
            status = cbcModel.branchAndBound()
        else:
            # https://www.coin-or.org/Doxygen/Clp/classClpSimplex.html
            # int ClpSimplex::initialSolve(ClpSolve& options)
            # General solve algorithm which can do presolve.
            status = model.initialSolve()

        print(status)

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
