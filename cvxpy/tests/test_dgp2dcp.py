import cvxpy
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.tests.base_test import BaseTest


class TestDgp2Dcp(BaseTest):
    def test_unconstrained_monomial(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        prod = x * y 
        dgp = cvxpy.Problem(cvxpy.Minimize(prod), [])
        gp2dcp = cvxpy.reductions.Gp2Dcp()

        dcp = gp2dcp.reduce(dgp)
        self.assertIsInstance(dcp.objective.expr, AddExpression)
        self.assertEqual(len(dcp.objective.expr.args), 2)
        self.assertIsInstance(dcp.objective.expr.args[0], cvxpy.Variable)
        self.assertIsInstance(dcp.objective.expr.args[1], cvxpy.Variable)
        opt = dcp.solve()
        # dcp is solved in log-space, so it is unbounded below
        # (since the OPT for dgp is 0 + epsilon). 
        self.assertEqual(opt, -float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")

        dgp = cvxpy.Problem(cvxpy.Maximize(prod), [])
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        self.assertEqual(dcp.solve(), float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertEqual(dgp.value, float("inf"))
        self.assertEqual(dgp.status, "unbounded")

    def test_basic_equality_constraint(self):
        x = cvxpy.Variable(pos=True)
        dgp = cvxpy.Problem(cvxpy.Minimize(x), [x == 1.0])
        gp2dcp = cvxpy.reductions.Gp2Dcp()

        dcp = gp2dcp.reduce(dgp)
        self.assertIsInstance(dcp.objective.expr, cvxpy.Variable)
        opt = dcp.solve()
        self.assertAlmostEqual(opt, 0.0)
        self.assertAlmostEqual(dcp.variables()[0].value, 0.0)

        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertAlmostEquals(dgp.value, 1.0)
        self.assertAlmostEquals(x.value, 1.0)

    def test_maximum_basic(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)

        prod1 = x * y**0.5
        prod2 = 3.0 * x * y**0.5
        obj = cvxpy.Minimize(cvxpy.maximum(prod1, prod2))
        constr = [x == 1.0, y == 4.0]
        
        dgp = cvxpy.Problem(obj, constr)
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        opt = dcp.solve()
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertAlmostEquals(dgp.value, 6.0)
        self.assertAlmostEquals(x.value, 1.0)
        self.assertAlmostEquals(y.value, 4.0)

    def test_sum_largest_basic(self):
        x = cvxpy.Variable((4,), pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 3))
        constr = [x[0] * x[1] * x[2] * x[3] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        dcp.solve()
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        opt = 6.0
        self.assertAlmostEquals(dgp.value, opt)
        self.assertAlmostEquals((x[0] * x[1] * x[2] * x[3]).value, 16, places=2)

        # An unbounded problem.
        x = cvxpy.Variable((4,), pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 3) * y)
        constr = [x[0] * x[1] * x[2] * x[3] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        opt = dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertAlmostEquals(dgp.value, 0.0)
        self.assertAlmostEquals(dgp.status, "unbounded")

        # Another unbounded problem.
        x = cvxpy.Variable(2, pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 1))
        dgp = cvxpy.Problem(obj, [])
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        opt = dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertAlmostEquals(dgp.value, 0.0)
        self.assertAlmostEquals(dgp.status, "unbounded")

        # Composition with posynomials.
        x = cvxpy.Variable((4,), pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(
          cvxpy.hstack([3 * x[0]**0.5 * x[1]**0.5,
                       x[0] * x[1] + 0.5 * x[1] * x[3]**3, x[2]]), 2))
        constr = [x[0] * x[1] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        dcp.solve()
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        # opt = 3 * sqrt(4) * sqrt(4) + (4 * 4 + 0.5 * 4 * epsilon) = 28
        opt = 28.0
        self.assertAlmostEquals(dgp.value, opt, places=2)
        self.assertAlmostEquals((x[0] * x[1]).value, 16.0, places=2)
        self.assertAlmostEquals(x[3].value, 0.0, places=2)
