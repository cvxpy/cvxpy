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
from cvxpy.expressions.variables import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.atoms import QuadForm


class TestTreeCopy(BaseTest):
    """Unit tests for the tree copy utility"""

    def setUp(self):
        self.x = Variable(2, name='x')
        self.Q = np.eye(2)
        self.c = np.array([[1, 0.5], [3, 5]])
        self.quad = QuadForm(self.c * self.x, self.Q)
        self.mul = self.x.T * self.c * self.x + 1

    def test_after_treecopy_variables_not_equal(self):
        new_quad = self.quad.tree_copy()
        for var in self.quad.variables():
            self.assertNotIn()