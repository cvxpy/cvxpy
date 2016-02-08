"""
Copyright 2016 Sascha-Dominic Schnug

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

import cvxpy.interface as intf
import cvxpy.settings as s
import numpy as np
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
                     'stopped by event handler (virtual int ' \
                                    'ClpEventHandler::event())': s.SOLVER_ERROR}

    def name(self):
        """The name of the solver.
        """
        return s.CBC

    def import_solver(self):
        """Imports the solver.
        """
        # Import basic modelling tools of cylp
        from cylp.cy import CyClpSimplex
        from cylp.py.modeling.CyLPModel import CyLPArray
        # Import cut-generator tools of cylp
        from cylp.cy.CyCgl import CyCglGomory, CyCglMixedIntegerRounding
        from cylp.cy.CyCgl import CyCglMixedIntegerRounding2, CyCglResidualCapacity
        from cylp.cy.CyCgl import CyCglKnapsackCover, CyCglFlowCover
        from cylp.cy.CyCgl import CyCglClique, CyCglTwomir
        from cylp.cy.CyCgl import CyCglLiftAndProject, CyCglAllDifferent
        from cylp.cy.CyCgl import CyCglOddHole, CyCglRedSplit
        from cylp.cy.CyCgl import CyCglLandP
        from cylp.cy.CyCgl import CyCglPreProcess, CyCglProbing, CyCglSimpleRounding

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
        from cylp.py.modeling.CyLPModel import CyLPArray

        # Get problem data
        data = self.get_problem_data(objective, constraints, cached_data)

        c = data[s.C]
        b = data[s.B]
        A = data[s.A]
        dims = data[s.DIMS]

        n = c.shape[0]

        solver_cache = cached_data[self.name()]

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
        c_ = CyLPArray(c)
        model.objective = c_ * x

        # Build model & solve
        status = None
        if self.is_mip(data):
            cbcModel = model.getCbcModel()  # need to convert model
            if not verbose:
                cbcModel.logLevel = 0

            # Add cuts if desired
            if "GomoryCuts" in solver_opts["solver_opts"]:
                print 'add gomory'
                from cylp.cy.CyCgl import CyCglGomory
                gom = CyCglGomory()
                cbcModel.addCutGenerator(gom, name="Gomory")
            if "MIRCuts" in solver_opts["solver_opts"]:
                print 'add mir'
                from cylp.cy.CyCgl import CyCglMixedIntegerRounding
                mir = CyCglMixedIntegerRounding()
                cbcModel.addCutGenerator(mir, name="MIR")
            if "MIRCuts2" in solver_opts["solver_opts"]:
                print 'add mir2'
                from cylp.cy.CyCgl import CyCglMixedIntegerRounding2
                mir = CyCglMixedIntegerRounding2()
                cbcModel.addCutGenerator(mir, name="MIR2")
            if "TwoMIRCuts" in solver_opts["solver_opts"]:
                print 'add 2mir'
                from cylp.cy.CyCgl import CyCglTwomir
                mir = CyCglTwomir()
                cbcModel.addCutGenerator(mir, name="Two-MIR")
            if "ResidualCapacityCuts" in solver_opts["solver_opts"]:
                print 'add rescap'
                from cylp.cy.CyCgl import CyCglResidualCapacity
                rca = CyCglResidualCapacity()
                cbcModel.addCutGenerator(rca, name="ResidualCapacity")
            if "KnapsackCuts" in solver_opts["solver_opts"]:
                print 'add kanpsack'
                from cylp.cy.CyCgl import CyCglKnapsackCover
                kna = CyCglKnapsackCover()
                cbcModel.addCutGenerator(kna, name="Knapsack")
            if "FlowCoverCuts" in solver_opts["solver_opts"]:
                print 'add flow-cover'
                from cylp.cy.CyCgl import CyCglFlowCover
                flo = CyCglFlowCover()
                cbcModel.addCutGenerator(flo, name="FlowCover")
            if "CliqueCuts" in solver_opts["solver_opts"]:
                print 'add clique'
                from cylp.cy.CyCgl import CyCglClique
                cli = CyCglClique()
                cbcModel.addCutGenerator(cli, name="Clique")
            if "LiftProjectCuts" in solver_opts["solver_opts"]:
                print 'add lift'
                from cylp.cy.CyCgl import CyCglLiftAndProject
                lap = CyCglLiftAndProject()
                cbcModel.addCutGenerator(lap, name="Lift-and-Project")
            if "AllDifferentCuts" in solver_opts["solver_opts"]:
                print 'add alldiff'
                from cylp.cy.CyCgl import CyCglAllDifferent
                ald = CyCglLiftAndProject()
                cbcModel.addCutGenerator(ald, name="AllDifferent")
            if "OddHoleCuts" in solver_opts["solver_opts"]:
                print 'add oddhole'
                from cylp.cy.CyCgl import CyCglOddHole
                odh = CyCglOddHole()
                cbcModel.addCutGenerator(odh, name="OddHole")
            if "RedSplitCuts" in solver_opts["solver_opts"]:
                print 'add redsplit'
                from cylp.cy.CyCgl import CyCglRedSplit
                res = CyCglRedSplit()
                cbcModel.addCutGenerator(res, name="RedSplit")
            if "LandPCuts" in solver_opts["solver_opts"]:
                print 'add landp'
                from cylp.cy.CyCgl import CyCglLandP
                lnp = CyCglLandP()
                cbcModel.addCutGenerator(lnp, name="LandP")
            if "PreProcessCuts" in solver_opts["solver_opts"]:
                print 'add preprocess'
                from cylp.cy.CyCgl import CyCglPreProcess
                pre = CyCglPreProcess()
                cbcModel.addCutGenerator(pre, name="PreProcess")
            if "ProbingCuts" in solver_opts["solver_opts"]:
                print 'add probing'
                from cylp.cy.CyCgl import CyCglProbing
                pro = CyCglProbing()
                cbcModel.addCutGenerator(pro, name="Probing")
            if "SimpleRoundingCuts" in solver_opts["solver_opts"]:
                print 'add simple rounding'
                from cylp.cy.CyCgl import CyCglSimpleRounding
                sro = CyCglSimpleRounding()
                cbcModel.addCutGenerator(sro, name="SimpleRounding")
            status = cbcModel.branchAndBound()  # solve
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
