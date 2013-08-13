from cvxpy import *
from mixed_integer import *
import unittest

class TestVars(unittest.TestCase):
    """ Unit tests for the variable types. """
    def setUp(self):
        pass

    # Overriden method to handle lists and lower accuracy.
    def assertAlmostEqual(self, a, b):
        try:
            a = list(a)
            b = list(b)
            for i in range(len(a)):
                self.assertAlmostEqual(a[i], b[i])
        except Exception:
            super(TestVars, self).assertAlmostEqual(a,b,places=3)

    # Test boolean
    def test_boolean(self):
        x = Variable(5,4)
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == boolean(5,4)])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 20)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)

    # Test choose
    def test_choose(self):
        x = Variable(5,4)
        p = Problem(Minimize(sum(1-x) + sum(x)), [x == choose(5,4,k=4)])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 20)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(sum(x.value), 4)

    # Test card
    def test_card(self):
        x = Variable(5)
        p = Problem(Maximize(sum(x)),
            [x == card(5,k=3), x <= 1, x >= 0])
        result = p.solve(method="admm")
        self.assertAlmostEqual(result, 3)
        for v in x.value:
            self.assertAlmostEqual(v*(1-v), 0)
        self.assertAlmostEqual(sum(x.value), 3)

        # should be equivalent to x == choose
        # x = Variable(5)
        # ones = 5*[[1]]
        # p = Problem(Minimize(ones*(-x)), 
        #     [x == card(5,k=4), x == boolean(5)])
        # result = p.solve(method="admm")
        # self.assertAlmostEqual(result, -4)
        # for v in x.value:
        #     self.assertAlmostEqual(v*(1-v), 0)
        # print x.value
        # self.assertAlmostEqual(sum(x.value), 4)