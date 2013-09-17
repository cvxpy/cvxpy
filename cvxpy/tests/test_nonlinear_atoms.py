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

import cvxpy.atoms.nonlinear.log as cvxlog
from cvxpy.expressions.variable import Variable
import cvxopt.solvers
import cvxopt
# import cvxpy.utilities as u
# import cvxpy.interface.matrix_utilities as intf
import unittest

class TestNonlinearAtoms(unittest.TestCase):
    """ Unit tests for the nonlinear atoms module. """
    # def setUp(self):
    #     self.x = Variable(2, name='x')
    #     self.y = Variable(2, name='y')
    # 
    #     self.A = Variable(2,2,name='A')
    #     self.B = Variable(2,2,name='B')
    #     self.C = Variable(3,2,name='C')

    def test_log(self):
        """ Test that minimize -sum(log(x)) s.t. x <= 1 yields 0.
        
            Rewritten by hand.
            
            neg_log_func implements
            
                t1 - log(t2) <= 0
            
            Implemented as
            
                minimize [-1,-1,0,0] * [t1; t2]
                    t1 - log(t2) <= 0
                    [0 0 -1 0;
                     0 0 0 -1] * [t1; t2] <= [-1; -1]
        """
        F = cvxlog.neg_log_func(2)
        h = cvxopt.matrix([1.,1.])
        G = cvxopt.spmatrix([1.,1.], [0,1], [2,3], (2,4), tc='d')
        sol = cvxopt.solvers.cpl(cvxopt.matrix([-1.0,-1.0,0,0]), F, G, h)
        
        self.assertEqual(sol['status'], 'optimal')
        self.assertAlmostEqual(sol['x'][0], 0.)
        self.assertAlmostEqual(sol['x'][1], 0.)
        self.assertAlmostEqual(sol['x'][2], 1.)
        self.assertAlmostEqual(sol['x'][3], 1.)
        self.assertAlmostEqual(sol['primal objective'], 0.0)
        