import cvxpy
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.error import DCPError, DGPError
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
        dgp._clear_solution()
        dgp.solve(gp=True)
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
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEqual(dgp.value, float("inf"))
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
        dgp._clear_solution()
        dgp.solve(gp=True)
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
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEquals(dgp.value, 6.0)
        self.assertAlmostEquals(x.value, 1.0)
        self.assertAlmostEquals(y.value, 4.0)

    def test_sum_largest(self):
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
        self.assertAlmostEquals((x[0] * x[1] * x[2] * x[3]).value, 16,
                                places=2)
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEquals(dgp.value, opt)
        self.assertAlmostEquals((x[0] * x[1] * x[2] * x[3]).value, 16,
                                places=2)

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
        dgp._clear_solution()
        dgp.solve(gp=True)
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
        dgp._clear_solution()
        dgp.solve(gp=True)
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
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEquals(dgp.value, opt, places=2)
        self.assertAlmostEquals((x[0] * x[1]).value, 16.0, places=2)
        self.assertAlmostEquals(x[3].value, 0.0, places=2)

    def test_div(self):
      x = cvxpy.Variable(pos=True)
      y = cvxpy.Variable(pos=True)
      p = cvxpy.Problem(cvxpy.Minimize(x * y), [y / 3  <= x, y >= 1])
      self.assertAlmostEquals(p.solve(gp=True), 1.0 / 3.0)
      self.assertAlmostEquals(y.value, 1.0)
      self.assertAlmostEquals(x.value, 1.0 / 3.0)

    def test_geo_mean(self):
        x = cvxpy.Variable(3, pos=True)
        p = [1, 2, 0.5]
        geo_mean = cvxpy.geo_mean(x, p)
        dgp = cvxpy.Problem(cvxpy.Minimize(geo_mean), [])
        gp2dcp = cvxpy.reductions.Gp2Dcp()
        dcp = gp2dcp.reduce(dgp)
        dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(gp2dcp.retrieve(dcp.solution))
        self.assertEquals(dgp.value, 0.0)
        self.assertEquals(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertEquals(dgp.value, 0.0)
        self.assertEquals(dgp.status, "unbounded")

    def test_solving_non_dgp_problem_raises_error(self):
        problem = cvxpy.Problem(cvxpy.Minimize(-1.0 * cvxpy.Variable()), [])
        with self.assertRaisesRegexp(DGPError, "Problem does not follow "
          "DGP rules. However, the problem does follow DCP rules. "
          "Consider calling this function with `gp=False`."):
            problem.solve(gp=True)
        problem.solve()
        self.assertEqual(problem.status, "unbounded")
        self.assertEqual(problem.value, -float("inf"))

    def test_solving_non_dcp_problem_raises_error(self):
        problem = cvxpy.Problem(
          cvxpy.Minimize(cvxpy.Variable(pos=True) * cvxpy.Variable(pos=True)),
            [])
        with self.assertRaisesRegexp(DCPError, "Problem does not follow "
          "DCP rules. However, the problem does follow DGP rules. "
          "Consider calling this function with `gp=True`."):
            problem.solve()
        problem.solve(gp=True)
        self.assertEqual(problem.status, "unbounded")
        self.assertAlmostEqual(problem.value, 0.0)

    def test_mat_mul(self):
        # TODO(akshayka): Implement canonicalization ...
        # canonicalization might need to collapse the multiplication ...
        pass

    def test_one_minus(self):
        x = cvxpy.Variable(pos=True)
        obj = cvxpy.Maximize(x)
        constr = [cvxpy.one_minus(x) >= 0.5]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(gp=True)
        self.assertAlmostEqual(problem.value, 0.5)
        self.assertAlmostEqual(x.value, 0.5)

    def test_paper_example_sum_largest(self):
        x = cvxpy.Variable((4,), pos=True)
        x0, x1, x2, x3 = (x[0], x[1], x[2], x[3])
        obj = cvxpy.Minimize(cvxpy.sum_largest(
          cvxpy.hstack([
            3 * x0**0.5 * x1**0.5,
            x0 * x1 + 0.5 * x1 * x3**3,
            x2]), 2))
        constr = [x0 * x1 * x2 >= 16]
        p = cvxpy.Problem(obj, constr)
        # smoke test.
        p.solve(gp=True)

    def test_paper_example_one_minus(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(x * y)
        constr = [(y * cvxpy.one_minus(x / y)) ** 2 >= 1, x >= y/3]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(gp=True)
        print(problem.value)
        print(x.value)
        print(y.value)
