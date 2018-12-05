import cvxpy
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.error import DCPError, DGPError
import cvxpy.reductions.dgp2dcp.atom_canonicalizers as dgp_atom_canon
from cvxpy.tests.base_test import BaseTest
import numpy as np


class TestDgp2Dcp(BaseTest):
    def test_unconstrained_monomial(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        prod = x * y 
        dgp = cvxpy.Problem(cvxpy.Minimize(prod), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)

        dcp = dgp2dcp.reduce()
        self.assertIsInstance(dcp.objective.expr, AddExpression)
        self.assertEqual(len(dcp.objective.expr.args), 2)
        self.assertIsInstance(dcp.objective.expr.args[0], cvxpy.Variable)
        self.assertIsInstance(dcp.objective.expr.args[1], cvxpy.Variable)
        opt = dcp.solve()
        # dcp is solved in log-space, so it is unbounded below
        # (since the OPT for dgp is 0 + epsilon). 
        self.assertEqual(opt, -float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")

        dgp = cvxpy.Problem(cvxpy.Maximize(prod), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        self.assertEqual(dcp.solve(), float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertEqual(dgp.value, float("inf"))
        self.assertEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(gp=True)
        self.assertAlmostEqual(dgp.value, float("inf"))
        self.assertEqual(dgp.status, "unbounded")

    def test_basic_equality_constraint(self):
        x = cvxpy.Variable(pos=True)
        dgp = cvxpy.Problem(cvxpy.Minimize(x), [x == 1.0])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)

        dcp = dgp2dcp.reduce()
        self.assertIsInstance(dcp.objective.expr, cvxpy.Variable)
        opt = dcp.solve()
        self.assertAlmostEqual(opt, 0.0)
        self.assertAlmostEqual(dcp.variables()[0].value, 0.0)

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        opt = dcp.solve()
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve()
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        opt = dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        opt = dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve()
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve()
        self.assertEquals(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
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

    def test_add_canon(self):
        X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        Y = cvxpy.Constant(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        Z = X + Y
        canon_matrix, constraints = dgp_atom_canon.add_canon(Z, Z.args)
        self.assertEqual(len(constraints), 0)
        self.assertEqual(canon_matrix.shape, Z.shape)
        expected = np.log(np.exp(X.value) + np.exp(Y.value))
        np.testing.assert_almost_equal(expected, canon_matrix.value)

    def test_matmul_canon(self):
        X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        Y = cvxpy.Constant(np.array([[1.0], [2.0], [3.0]]))
        Z = cvxpy.matmul(X, Y)
        canon_matrix, constraints = dgp_atom_canon.mulexpression_canon(
          Z, Z.args)
        self.assertEqual(len(constraints), 0)
        self.assertEqual(canon_matrix.shape, (2, 1))
        first_entry = np.log(np.exp(2.0) + np.exp(4.0) + np.exp(6.0))
        second_entry = np.log(np.exp(5.0) + np.exp(7.0) + np.exp(9.0))
        self.assertAlmostEqual(first_entry, canon_matrix[0, 0].value)
        self.assertAlmostEqual(second_entry, canon_matrix[1, 0].value)

    def test_trace_canon(self):
        X = cvxpy.Constant(np.array([[1.0, 5.0], [9.0, 14.0]]))
        Y = cvxpy.trace(X)
        canon, constraints = dgp_atom_canon.trace_canon(Y, Y.args)
        self.assertEqual(len(constraints), 0)
        self.assertTrue(canon.is_scalar())
        expected = np.log(np.exp(1.0) + np.exp(14.0))
        self.assertAlmostEqual(expected, canon.value)

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

    def test_paper_example_eye_minus_inv(self):
        X = cvxpy.Variable((2, 2), pos=True)
        obj = cvxpy.Minimize(cvxpy.trace(cvxpy.eye_minus_inv(X)))
        constr = [cvxpy.geo_mean(cvxpy.diag(X)) == 0.1]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(gp=True)

    def test_paper_example_exp_log(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(x * y)
        constr = [cvxpy.exp(y/x) <= cvxpy.log(y)]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(gp=True)

    def test_pf_matrix_completion(self):
        X = cvxpy.Variable((3, 3), pos=True)
        obj = cvxpy.Minimize(cvxpy.pf_eigenvalue(X))
        constr = [
          X[0, 0] == 1.0,
          X[0, 2] == 1.9,
          X[1, 1] == 0.8,
          X[2, 0] == 3.2,
          X[2, 1] == 5.9,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(gp=True)

    def test_rank_one_nmf(self):
        X = cvxpy.Variable((3, 3), pos=True)
        x = cvxpy.Variable((3,), pos=True)
        y = cvxpy.Variable((3,), pos=True)
        xy = cvxpy.vstack([x[0] * y, x[1] * y, x[2] * y])
        R = cvxpy.maximum(
          cvxpy.multiply(X, (xy) ** (-1.0)),
          cvxpy.multiply(X ** (-1.0), xy))
        objective = cvxpy.sum(R)
        constraints = [
          X[0, 0] == 1.0,
          X[0, 2] == 1.9,
          X[1, 1] == 0.8,
          X[2, 0] == 3.2,
          X[2, 1] == 5.9,
          x[0] * x[1] * x[2] == 1.0,
        ]
        # smoke test.
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        prob.solve(gp=True)
