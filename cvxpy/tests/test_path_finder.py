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
from cvxpy.problems.objective import Minimize
from cvxpy.atoms import QuadForm
from cvxpy.problems.problem_type import ProblemType


class TestPathFinder(BaseTest):
    """Unit tests for reduction path construction"""

    def setUp(self):
        self.x = Variable(2, name='x')
        self.Q = np.eye(2)
        self.c = np.array([1, 0.5])
        self.qp = Problem(Minimize(QuadForm(self.x, self.Q)), [self.x <= -1])
        self.cp = Problem(Minimize(self.c.T * self.x + 1), [self.x >= 0])

    def test_reduction_path_qp(self):
        path = PathFinder().reduction_path(ProblemType(self.qp), [QpSolver('GUROBI')])
        self.assertEquals(2, len(path))
