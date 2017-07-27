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

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_lp(self):
        """Tests basic LPs.
        """
        if ELEMENTAL in installed_solvers():
            prob = Problem(Minimize(0), [self.x == 2])
            prob.solve(verbose=False, solver=ELEMENTAL)
            self.assertAlmostEqual(prob.value, 0)
            self.assertItemsAlmostEqual(self.x.value, [2, 2])

            prob = Problem(Minimize(-self.a), [self.a <= 1])
            prob.solve(verbose=False, solver=ELEMENTAL)
            self.assertAlmostEqual(prob.value, -1)
            self.assertAlmostEqual(self.a.value, 1)

    def test_soc(self):
        """Test SOCP representable problems.
        """
        if ELEMENTAL in installed_solvers():
            x = Variable(2, 2)
            prob = Problem(Minimize(huber(x)[0, 0]),
                           [x == [[0.5, -1.5], [4, 0]]])
            prob.solve(verbose=False, solver=ELEMENTAL)
            self.assertAlmostEqual(prob.value, 0.25)

            x = Variable(3)
            prob = Problem(Maximize(pnorm(x, .5)),
                           [x == [1.1, 2, .1]])
            # data = prob.get_problem_data(ELEMENTAL)

            # data['c'], data['b'], data['h'], data['A'], data['G']
            prob.solve(verbose=False, solver=ELEMENTAL)
            self.assertAlmostEqual(prob.value, 7.724231543909264, places=3)

            x = Variable()
            prob = Problem(Minimize(power(x, 1.34)), [x == 7.45])
            prob.solve(solver=ELEMENTAL, verbose=False)
            self.assertAlmostEqual(prob.value, 14.746515290825071, places=3)

            x = Variable(2, 2)
            expr = inv_pos(x)
            prob = Problem(Minimize(expr[1, 0]), [x == [[1, 2], [3, 4]]])
            prob.solve(solver=ELEMENTAL, verbose=False)
            # Constant([[1,1.0/2],[1.0/3,1.0/4]])),
            self.assertAlmostEqual(prob.value, 0.5)

            x = Variable(2, 2)
            expr = sqrt(x)
            constr = [x == [[2, 4], [16, 1]]]
            # Constant([[1.414213562373095,2],[4,1]])),
            prob = Problem(Maximize(expr[0, 0]), constr)
            prob.solve(solver=ELEMENTAL, verbose=False)

            x = Variable(2, 3)
            expr = mixed_norm(x, 1, 1)
            constr = [x == [[1, 2], [3, 4], [5, 6]]]
            prob = Problem(Minimize(expr), constr)
            prob.solve(solver=ELEMENTAL, verbose=False)
            self.assertAlmostEqual(prob.value, 21)
