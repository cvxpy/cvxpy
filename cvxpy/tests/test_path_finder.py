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

from cvxpy.atoms import QuadForm, sum_squares, norm
from cvxpy.constraints.second_order import SOC
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import Minimize, Maximize
from cvxpy.problems.path_finder import PathFinder
from cvxpy.problems.problem import Problem
from cvxpy.problems.problem_type import ProblemType
from cvxpy.reductions.utilities import (QpMatrixStuffing, ECOS, FlipObjective,
                                        ConeMatrixStuffing, Dcp2Cone, Qp2SymbolicQp)
from cvxpy.solver_interface.qp_solvers.qp_solver import QpSolver
from cvxpy.tests.base_test import BaseTest


class TestPathFinder(BaseTest):
    """Unit tests for reduction path construction"""

    def setUp(self):
        self.b = Variable(1, name='b')
        self.x = Variable(2, name='x')
        self.Q = np.eye(2)
        self.c = np.array([1, 0.5])
        self.qp = Problem(Minimize(QuadForm(self.x, self.Q)), [self.x <= -1])
        self.cp = Problem(Minimize(self.c.T * self.x + 1), [SOC(self.b,
            self.x)])

    def test_qp_reduction_path(self):
        path = PathFinder().reduction_path(ProblemType(self.qp), [], QpSolver)
        self.assertEquals(3, len(path))
        self.assertEquals(path[1], QpMatrixStuffing)
        self.assertEquals(path[0], Qp2SymbolicQp)

    def test_qp_maximization_reduction_path_qp_solver(self):
        qp_maximization = Problem(Maximize(QuadForm(self.x, -self.Q)),
            [self.x <= -1])
        path = PathFinder().reduction_path(ProblemType(qp_maximization), [],
            QpSolver)
        self.assertEquals(4, len(path))
        self.assertEquals(path[2], QpMatrixStuffing)
        self.assertEquals(path[1], Qp2SymbolicQp)
        self.assertEquals(path[0], FlipObjective)

    def test_qp_maximization_reduction_path_ecos(self):
        qp_maximization = Problem(Maximize(-sum_squares(self.x)),
            [self.x <= -1])
        self.assertTrue(qp_maximization.is_dcp())
        path = PathFinder().reduction_path(ProblemType(qp_maximization), [], ECOS)
        self.assertEquals(4, len(path))
        self.assertEquals(path[2], ConeMatrixStuffing)
        self.assertEquals(path[1], Dcp2Cone)
        self.assertEquals(path[0], FlipObjective)

    def test_cone_reduction_path(self):
        path = PathFinder().reduction_path(ProblemType(self.cp), [], ECOS)
        self.assertLessEqual(len(path), 3)
        self.assertGreater(len(path), 1)
        if len(path) == 3:
            self.assertEquals(path[2], ECOS)
            self.assertEquals(path[1], ConeMatrixStuffing)
            self.assertEquals(path[0], Dcp2Cone)
        elif len(path) == 2:
            self.assertEquals(path[1], ECOS)
            self.assertEquals(path[0], ConeMatrixStuffing)

    def test_path_nonexistence(self):
        path = PathFinder().reduction_path(ProblemType(self.cp), [], QpSolver)
        self.assertEquals(None, path)
