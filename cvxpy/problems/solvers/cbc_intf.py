"""
Copyright 2016 Sascha-Dominic Schnug

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

import importlib
import six
import cvxpy.interface as intf
import cvxpy.settings as s
from cvxpy.problems.solvers.solver import Solver


class CBC(Solver):
    """ An interface to the CBC solver
    """

    # Solver capabilities.
    LP_CAPABLE = True
    SOCP_CAPABLE = False
    SDP_CAPABLE = False
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

        n = c.shape[0]

        # Problem
        model = CyClpSimplex()

        # Variables
        x = model.addVariable('x', n)

        if self.is_mip(data):
            for i in data[s.BOOL_IDX]:
                model.setInteger(x[i])
            for i in data[s.INT_IDX]:
                model.setInteger(x[i])

        # Constraints
        # eq
        model += A[0:dims[s.EQ_DIM], :] * x == b[0:dims[s.EQ_DIM]]

        # leq
        leq_start = dims[s.EQ_DIM]
        leq_end = dims[s.EQ_DIM] + dims[s.LEQ_DIM]
        model += A[leq_start:leq_end, :] * x <= b[leq_start:leq_end]

        # no boolean vars available in cbc -> model as int + restrict to [0,1]
        if self.is_mip(data):
            for i in data[s.BOOL_IDX]:
                model += 0 <= x[i] <= 1

        # Objective
        model.objective = c

        # Build model & solve
        status = None
        if self.is_mip(data):
            cbcModel = model.getCbcModel()  # need to convert model
            if not verbose:
                cbcModel.logLevel = 0

            # Add cut-generators (optional)
            for cut_name, cut_func in six.iteritems(self.SUPPORTED_CUT_GENERATORS):
                if cut_name in solver_opts and solver_opts[cut_name]:
                    module = importlib.import_module("cylp.cy.CyCgl")
                    funcToCall = getattr(module, cut_func)
                    cut_gen = funcToCall()
                    cbcModel.addCutGenerator(cut_gen, name=cut_name)

            # solve
            status = cbcModel.branchAndBound()
        else:
            if not verbose:
                model.logLevel = 0
            status = model.primal()  # solve

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
