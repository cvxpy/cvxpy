from cvxpy import *
from cvxpy.tests.base_test import BaseTest

class TestSolvers(BaseTest):
    """ Unit tests for solver specific behavior. """
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

    def test_lp(self):
        """Tests basic LPs.
        """
        if ELEMENTAL in installed_solvers():
            # prob = Problem(Minimize(0), [self.x == 2])
            # prob.solve(verbose=True, solver=ELEMENTAL)
            # self.assertAlmostEqual(prob.value, 0)
            # self.assertItemsAlmostEqual(self.x.value, [2, 2])

            prob = Problem(Minimize(sum_entries(self.x)), [self.x >= 2])
            prob.solve(verbose=True, solver=ELEMENTAL)
            self.assertAlmostEqual(prob.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [2, 2])
