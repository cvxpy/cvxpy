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
import clarabel
import numpy as np

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.expressions.variable import Variable
from cvxpy.problems.problem import Problem
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
from cvxpy.tests.base_test import BaseTest


class ClarabelTest(BaseTest):
    """
    Tests for Clarabel solver interface.
    """

    def setUp(self):
        """
        Sets up a problem as in test_comple.py function test_quad_form.
        """
        np.random.seed(42)
        P = np.random.randn(3, 3) - 1j*np.random.randn(3, 3)
        P = np.conj(P.T).dot(P)
        b = np.arange(3) + 3j*(np.arange(3) + 10)
        x = Variable(3, complex=True)
        self.value = cp.quad_form(b, P).value
        self.prob = Problem(cp.Minimize(cp.quad_form(x, P)), [x == b])

        # A dummy solution object (clarabel.DefaultSolution has no python constructor).
        self.solution: clarabel.DefaultSolution = type("DefaultSolution", (object,), {
            "x": [5.263500536056257e-12, 0.9999999999975893, 1.999999999997496,],
            "s": [0.0, 0.0, 0.0, ],
            "z": [-491.42397588400473, 279.36833425936913, 191.11806896900944, ],
            "status": CLARABEL.SOLVED,
            "obj_val": 21434.491423910207,
            "obj_val_dual": 21434.491423919164,
            "solve_time": 9.79e-5,
            "iterations": 0,
            "r_prim": 5.5338076549567106e-14,
            "r_dual": 1.5829159225003945e-15,
        })

        return super().setUp()
    
    def test_invert_when_solved(self):
        """Tests invert when a solution is present and solver status from clarabel is SOLVED."""
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        solution = solver.invert(self.solution, inverse_data[-1], {})
        self.assertEqual(s.OPTIMAL, solution.status)

    def test_invert_when_insufficient_progress_should_fail(self):
        """
        Tests invert when a solution is present and solver status from clarabel is
        InsufficientProgress.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solution = solver.invert(self.solution, inverse_data[-1], {})
        self.assertEqual(s.SOLVER_ERROR, solution.status)
        
    def test_invert_when_insufficient_progress_but_accept_unknown(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        solution = solver.invert(self.solution, inverse_data[-1], {CLARABEL.ACCEPT_UNKNOWN: True})
        self.assertEqual(s.OPTIMAL_INACCURATE, solution.status)

    def test_invert_when_insufficient_progress_but_accept_unknown_and_no_solution(self):
        """
        Tests invert when a solution is present and solver status from clarabel
        is InsufficientProgress but "accept_unknown" solver option was set to true.
        Nevertheless, clarabel did not return a solution and therefore the resulting
        status should be SolverError.
        """
        solver = CLARABEL()
        _, _, inverse_data = self.prob.get_problem_data("clarabel")
        self.solution.status = CLARABEL.INSUFFICIENT_PROGRESS
        self.solution.x = None
        self.solution.z = None
        solution = solver.invert(self.solution, inverse_data[-1], {"accept_unknown": True})
        self.assertEqual(s.SOLVER_ERROR, solution.status)
        
