import cvxpy as cvx
import numpy as np
from cvxpy.tests.base_test import BaseTest
import unittest


#x = cvx.Variable(5)
#y = cvx.Variable()
#f = cvx.quad_over_lin(x, y)
#
##x = cvx.Variable((3, 5))
##y = cvx.Variable(3)
##f = cvx.quad_over_lin(x, y)
#
##x = cvx.Variable(3)
##y = cvx.Variable(3)
##f =cvx.quad_over_lin(x, y)
#
#constr = []
##constr += [np.array([1, 2, 3]) == y]
#constr += [y == 1]
#constr += [x == 1]
#prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.quad_over_lin(x, y))), constr)
#prob.solve()
#print(prob.value)


class TestQuadOverLin(unittest.TestCase):
#class TestQuadOverLin(BaseTest):
    """ Unit tests for the quad_over_lin atom. """

    def setUp(self):
        self.a = cvx.Variable(name='a')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(2, name='y')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_scalar_y(self):
        obj = cvx.quad_over_lin(self.x, self.a)
        constr = [self.x == 2, self.a == 3]
        p = cvx.Problem(cvx.Minimize(obj), constr)
        result = p.solve()
        self.assertAlmostEqual(result, 8/3)

        obj = cvx.quad_over_lin(self.B, self.a)
        constr = [self.B == 2, self.a == 3]
        p = cvx.Problem(cvx.Minimize(obj), constr)
        result = p.solve()
        self.assertAlmostEqual(result, 16/3)

    def test_vector_y(self):
        obj = cvx.sum(cvx.quad_over_lin(self.x, self.y))
        constr = [self.x == 2, self.y == np.array([1, 3])]
        p = cvx.Problem(cvx.Minimize(obj), constr)
        result = p.solve()
        self.assertAlmostEqual(result, 4 + 4/3)

        obj = cvx.sum(cvx.quad_over_lin(self.A, self.y, axis=1))
        constr = [self.A == np.array([[1, 2], [3, 4]]),
                  self.y == np.array([1, 3])]
        p = cvx.Problem(cvx.Minimize(obj), constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1 + 4 + 9/3 + 16/3)

        obj = cvx.sum(cvx.quad_over_lin(self.C, self.y, axis=0))
        constr = [self.C == np.array([[1, 2], [3, 4], [5, 6]]),
                  self.y == np.array([1, 3])]
        p = cvx.Problem(cvx.Minimize(obj), constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1 + 4/3 + 9 + 16/3 + 25 + 36/3, places=4)






        ##x = cvx.Variable((3, 5))
        ##y = cvx.Variable(3)
        ##f = cvx.quad_over_lin(x, y)

        ##x = cvx.Variable(3)
        ##y = cvx.Variable(3)
        ##f =cvx.quad_over_lin(x, y)

        #constr = []
        ##constr += [np.array([1, 2, 3]) == y]
        #constr += [y == 1]
        #constr += [x == 1]
        #prob = cvx.Problem(cvx.Minimize(cvx.sum(cvx.quad_over_lin(x, y))), constr)
        #prob.solve()
        #print(prob.value)


if __name__ == '__main__':
    unittest.main()
