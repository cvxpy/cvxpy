"""
Copyright 2017 Robin Verschueren

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

import numpy as np

from cvxpy.tests.base_test import BaseTest
from cvxpy.problems.path_finder import PathFinder
from cvxpy.solver_interface.qp_solvers.qp_solver import QpSolver
from cvxpy.expressions.variables import Variable
from cvxpy.problems.problem import Problem
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.atoms import QuadForm, sum_squares
from cvxpy.problems.problem_type import ProblemType
from cvxpy.reductions import QpMatrixStuffing, ECOS, FlipObjective, ConeMatrixStuffing, Dcp2Cone


class TestPathFinder(BaseTest):
    """Unit tests for reduction path construction"""

    def setUp(self):
        self.x = Variable(2, name='x')
        self.Q = np.eye(2)
        self.c = np.array([1, 0.5])
        self.qp = Problem(Minimize(QuadForm(self.x, self.Q)), [self.x <= -1])
        self.cp = Problem(Minimize(self.c.T * self.x + 1), [self.x >= 0])

    def test_qp_reduction_path(self):
        path = PathFinder().reduction_path(ProblemType(self.qp), [QpSolver])
        self.assertEquals(2, len(path))
        self.assertEquals(path[1], QpMatrixStuffing)

    def test_qp_maximization_reduction_path_qp_solver(self):
        qp_maximization = Problem(Maximize(QuadForm(self.x, -self.Q)), [self.x <= -1])
        path = PathFinder().reduction_path(ProblemType(qp_maximization), [QpSolver])
        self.assertEquals(3, len(path))
        self.assertEquals(path[1], QpMatrixStuffing)
        self.assertEquals(path[2], FlipObjective)

    def test_qp_maximization_reduction_path_ecos(self):
        qp_maximization = Problem(Maximize(-sum_squares(self.x)), [self.x <= -1])
        self.assertTrue(qp_maximization.is_dcp())
        path = PathFinder().reduction_path(ProblemType(qp_maximization), [ECOS])
        self.assertEquals(4, len(path))
        self.assertEquals(path[1], ConeMatrixStuffing)
        self.assertEquals(path[2], Dcp2Cone)
        self.assertEquals(path[3], FlipObjective)

    def test_cone_reduction_path_valid_as_is(self):
        path = PathFinder().reduction_path(ProblemType(self.cp), [ECOS])
        self.assertEquals(1, len(path))
