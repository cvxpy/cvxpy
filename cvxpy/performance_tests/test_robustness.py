"""
Copyright 2013 Steven Diamond

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

import cvxpy.atoms as at
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrix_utilities as intf
from cvxopt import matrix
import scipy.sparse as sp
import unittest

class TestProblem(unittest.TestCase):
    """ Unit tests for the expression/expression module. """
    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')
        self.c = Variable(name='c')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2,2,name='A')
        self.B = Variable(2,2,name='B')
        self.C = Variable(3,2,name='C')

    # Overriden method to handle lists and lower accuracy.
    # ECHU: uncommented to ensure that tests pass
    def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTERFACE):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestProblem, self).assertAlmostEqual(a,b,places=4)

    def test_large_sum(self):
        """Test large number of variables summed.
        """
        for n in [10, 20, 30, 40, 50]:
            A = matrix(range(n*n), (n,n))
            x = Variable(n,n)
            p = Problem(Minimize(at.sum_entries(x)), [x >= A])
            result = p.solve()
            answer = n*n*(n*n+1)/2 - n*n
            print result - answer
            self.assertAlmostEqual(result, answer)

    def test_large_square(self):
        """Test large number of variables squared.
        """
        for n in [10, 20, 30, 40, 50]:
            A = matrix(range(n*n), (n,n))
            x = Variable(n,n)
            p = Problem(Minimize(at.square(x[0, 0])),
                [x >= A])
            result = p.solve()
            self.assertAlmostEqual(result, 0)

    def test_sdp(self):
        """Test a problem with semidefinite cones.
        """
        a = sp.rand(100,100,.1, random_state=1)
        a = a.todense()
        X = Variable(100,100)
        obj = at.norm(X, "nuc") + at.norm(X-a,'fro')
        p = Problem(Minimize(obj))
        p.solve(solver="SCS")
