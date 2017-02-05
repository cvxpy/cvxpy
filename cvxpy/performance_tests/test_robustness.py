"""
Copyright 2017 Steven Diamond

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

import cvxpy.atoms as at
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variables import Variable
from cvxpy.problems.objective import *
from cvxpy.problems.problem import Problem
import cvxpy.interface.matrix_utilities as intf
import numpy as np
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

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTF):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestProblem, self).assertAlmostEqual(a, b, places=3)

    def test_large_sum(self):
        """Test large number of variables summed.
        """
        for n in [10, 20, 30, 40, 50]:
            A = np.arange(n*n)
            A = np.reshape(A, (n, n))
            x = Variable(n, n)
            p = Problem(Minimize(at.sum_entries(x)), [x >= A])
            result = p.solve()
            answer = n*n*(n*n+1)/2 - n*n
            print(result - answer)
            self.assertAlmostEqual(result, answer)

    def test_large_square(self):
        """Test large number of variables squared.
        """
        for n in [10, 20, 30, 40, 50]:
            A = np.arange(n*n)
            A = np.reshape(A, (n, n))
            x = Variable(n, n)
            p = Problem(Minimize(at.square(x[0, 0])),
                        [x >= A])
            result = p.solve()
            self.assertAlmostEqual(result, 0)

    def test_sdp(self):
        """Test a problem with semidefinite cones.
        """
        a = sp.rand(100, 100, .1, random_state=1)
        a = a.todense()
        X = Variable(100, 100)
        obj = at.norm(X, "nuc") + at.norm(X-a, 'fro')
        p = Problem(Minimize(obj))
        p.solve(solver="SCS")

    def test_large_sdp(self):
        """Test for bug where large SDP caused integer overflow in CVXcanon.
        """
        SHAPE = (256, 256)
        rows = SHAPE[0]
        cols = SHAPE[1]
        X = Variable(*SHAPE)
        Z = Variable(rows+cols, rows+cols)
        prob = Problem(Minimize(0.5*at.trace(Z)),
                       [X[0, 0] >= 1, Z[0:rows, rows:rows+cols] == X, Z >> 0, Z == Z.T])
        prob.solve(solver="SCS")
        self.assertAlmostEqual(prob.value, 1.0)
