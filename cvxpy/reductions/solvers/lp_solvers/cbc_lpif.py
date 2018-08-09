"""
Copyright 2016 Sascha-Dominic Schnug, 2017 Robin Verschueren

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

import numpy as np

import cvxpy.settings as s
from cvxpy.constraints import NonPos, Zero
from cvxpy.problems.problem_data.problem_data import ProblemData
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.solvers.conic_solvers import ConicSolver


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

    def accepts(self, problem):
        """Can CBC solve the problem?
        """
        # TODO check if is matrix stuffed.
        if not problem.objective.args[0].is_affine():
            return False
        for constr in problem.constraints:
            if type(constr) not in [Zero, NonPos]:
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
        data = {}
        inv_data = {self.VAR_ID: problem.variables()[0].id}
        inv_data[s.OFFSET] = data[s.OFFSET][0]

        # Order and group constraints.
        eq_constr = [c for c in problem.constraints if type(c) == Zero]
        inv_data[self.EQ_CONSTR] = eq_constr
        leq_constr = [c for c in problem.constraints if type(c) == NonPos]
        inv_data[self.NEQ_CONSTR] = leq_constr
        return data, inv_data

    def invert(self, solution, inverse_data):
        """Returns the solution to the original problem given the inverse_data.
        """
        status = solution[s.STATUS]

        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE]
            primal_vars = {inverse_data[self.VAR_ID]: solution[s.PRIMAL]}
            eq_dual = ConicSolver.get_dual_values(solution[s.EQ_DUAL], inverse_data[self.EQ_CONSTR])
            leq_dual = ConicSolver.get_dual_values(
                solution[s.INEQ_DUAL],
                inverse_data[self.NEQ_CONSTR])
            eq_dual.update(leq_dual)
            dual_vars = eq_dual
        else:
            if status == s.INFEASIBLE:
                opt_val = np.inf
            elif status == s.UNBOUNDED:
                opt_val = -np.inf
            else:
                opt_val = None
            primal_vars = None
            dual_vars = None

        return Solution(status, opt_val, primal_vars, dual_vars, {})

    def solve(self, problem, warm_start, verbose, solver_opts):
        from cvxpy.problems.solvers.cbc_intf import CBC as CBC_OLD
        solver = CBC_OLD()
        _, inv_data = self.apply(problem)
        objective, _ = problem.objective.canonical_form
        constraints = [con for c in problem.constraints for con in c.canonical_form[1]]
        sol = solver.solve(
            objective,
            constraints,
            {self.name(): ProblemData()},
            warm_start,
            verbose,
            solver_opts)

        return self.invert(sol, inv_data)
