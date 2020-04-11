import warnings

import cvxpy
from cvxpy.atoms.affine.add_expr import AddExpression
import cvxpy.error as error
import cvxpy.reductions.dgp2dcp.atom_canonicalizers as dgp_atom_canon
from cvxpy.reductions import solution
from cvxpy.settings import SOLVER_ERROR
from cvxpy.tests.base_test import BaseTest

import numpy as np


# TODO(akshayka): Changing SOLVER to MOSEK exposes bugs in the mosek interface;
# fix these bugs
SOLVER = cvxpy.ECOS


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
        opt = dcp.solve(SOLVER)
        # dcp is solved in log-space, so it is unbounded below
        # (since the OPT for dgp is 0 + epsilon).
        self.assertEqual(opt, -float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")

        dgp = cvxpy.Problem(cvxpy.Maximize(prod), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        self.assertEqual(dcp.solve(SOLVER), float("inf"))
        self.assertEqual(dcp.status, "unbounded")

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertEqual(dgp.value, float("inf"))
        self.assertEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, float("inf"))
        self.assertEqual(dgp.status, "unbounded")

    def test_basic_equality_constraint(self):
        x = cvxpy.Variable(pos=True)
        dgp = cvxpy.Problem(cvxpy.Minimize(x), [x == 1.0])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)

        dcp = dgp2dcp.reduce()
        self.assertIsInstance(dcp.objective.expr, cvxpy.Variable)
        opt = dcp.solve(SOLVER)
        self.assertAlmostEqual(opt, 0.0)
        self.assertAlmostEqual(dcp.variables()[0].value, 0.0)

        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)

    def test_basic_gp(self):
        x, y, z = cvxpy.Variable((3,), pos=True)
        constraints = [2*x*y + 2*x*z + 2*y*z <= 1.0, x >= 2*y]
        problem = cvxpy.Problem(cvxpy.Minimize(1/(x*y*z)), constraints)
        problem.solve(SOLVER, gp=True)
        self.assertAlmostEqual(15.59, problem.value, places=2)

    def test_maximum(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        prod1 = x * y**0.5
        prod2 = 3.0 * x * y**0.5
        obj = cvxpy.Minimize(cvxpy.maximum(prod1, prod2))
        constr = [x == 1.0, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve(SOLVER)
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 6.0)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)

    def test_prod(self):
        X = np.arange(12).reshape((4, 3))
        np.testing.assert_almost_equal(np.prod(X), cvxpy.prod(X).value)
        np.testing.assert_almost_equal(
          np.prod(X, axis=0), cvxpy.prod(X, axis=0).value)
        np.testing.assert_almost_equal(
          np.prod(X, axis=1), cvxpy.prod(X, axis=1).value)
        np.testing.assert_almost_equal(
          np.prod(X, axis=0, keepdims=True),
          cvxpy.prod(X, axis=0, keepdims=True).value)
        np.testing.assert_almost_equal(
          np.prod(X, axis=1, keepdims=True),
          cvxpy.prod(X, axis=1, keepdims=True).value)

        prod = cvxpy.prod(X)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(np.sum(X), X_canon.value)

        prod = cvxpy.prod(X, axis=0)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(np.sum(X, axis=0), X_canon.value)

        prod = cvxpy.prod(X, axis=1)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(np.sum(X, axis=1), X_canon.value)

        prod = cvxpy.prod(X, axis=0, keepdims=True)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(
          np.sum(X, axis=0, keepdims=True), X_canon.value)

        prod = cvxpy.prod(X, axis=1, keepdims=True)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(
          np.sum(X, axis=1, keepdims=True), X_canon.value)

        X = np.arange(12)
        np.testing.assert_almost_equal(np.prod(X), cvxpy.prod(X).value)
        np.testing.assert_almost_equal(np.prod(X, keepdims=True),
                                       cvxpy.prod(X, keepdims=True).value)

        prod = cvxpy.prod(X)
        X_canon, _ = dgp_atom_canon.prod_canon(prod, prod.args)
        np.testing.assert_almost_equal(np.sum(X), X_canon.value)

        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        posy1 = x * y**0.5 + 3.0 * x * y**0.5
        posy2 = x * y**0.5 + 3.0 * x ** 2 * y**0.5
        self.assertTrue(cvxpy.prod([posy1, posy2]).is_log_log_convex())
        self.assertFalse(cvxpy.prod([posy1, posy2]).is_log_log_concave())
        self.assertFalse(cvxpy.prod([posy1, 1/posy1]).is_dgp())

        m = x * y**0.5
        self.assertTrue(cvxpy.prod([m, m]).is_log_log_affine())
        self.assertTrue(cvxpy.prod([m, 1/posy1]).is_log_log_concave())
        self.assertFalse(cvxpy.prod([m, 1/posy1]).is_log_log_convex())

    def test_max(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        prod1 = x * y**0.5
        prod2 = 3.0 * x * y**0.5
        obj = cvxpy.Minimize(cvxpy.max(cvxpy.hstack([prod1, prod2])))
        constr = [x == 1.0, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve(SOLVER)
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 6.0)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)

    def test_minimum(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        prod1 = x * y**0.5
        prod2 = 3.0 * x * y**0.5
        posy = prod1 + prod2
        obj = cvxpy.Maximize(cvxpy.minimum(prod1, prod2, 1/posy))
        constr = [x == 1.0, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (2.0 + 6.0))
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_min(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        prod1 = x * y**0.5
        prod2 = 3.0 * x * y**0.5
        posy = prod1 + prod2
        obj = cvxpy.Maximize(cvxpy.min(cvxpy.hstack([prod1, prod2, 1/posy])))
        constr = [x == 1.0, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (2.0 + 6.0), places=4)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_sum_largest(self):
        self.skipTest("Enable test once sum_largest is implemented.")
        x = cvxpy.Variable((4,), pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 3))
        constr = [x[0] * x[1] * x[2] * x[3] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve(SOLVER)
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        opt = 6.0
        self.assertAlmostEqual(dgp.value, opt)
        self.assertAlmostEqual((x[0] * x[1] * x[2] * x[3]).value, 16,
                               places=2)
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, opt)
        self.assertAlmostEqual((x[0] * x[1] * x[2] * x[3]).value, 16,
                               places=2)

        # An unbounded problem.
        x = cvxpy.Variable((4,), pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 3) * y)
        constr = [x[0] * x[1] * x[2] * x[3] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        opt = dcp.solve(SOLVER)
        self.assertEqual(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertAlmostEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertAlmostEqual(dgp.status, "unbounded")

        # Another unbounded problem.
        x = cvxpy.Variable(2, pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(x, 1))
        dgp = cvxpy.Problem(obj, [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        opt = dcp.solve(SOLVER)
        self.assertEqual(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertAlmostEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, 0.0)
        self.assertAlmostEqual(dgp.status, "unbounded")

        # Composition with posynomials.
        x = cvxpy.Variable((4,), pos=True)
        obj = cvxpy.Minimize(cvxpy.sum_largest(
          cvxpy.hstack([3 * x[0]**0.5 * x[1]**0.5,
                       x[0] * x[1] + 0.5 * x[1] * x[3]**3, x[2]]), 2))
        constr = [x[0] * x[1] >= 16]
        dgp = cvxpy.Problem(obj, constr)
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve(SOLVER)
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        # opt = 3 * sqrt(4) * sqrt(4) + (4 * 4 + 0.5 * 4 * epsilon) = 28
        opt = 28.0
        self.assertAlmostEqual(dgp.value, opt, places=2)
        self.assertAlmostEqual((x[0] * x[1]).value, 16.0, places=2)
        self.assertAlmostEqual(x[3].value, 0.0, places=2)
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertAlmostEqual(dgp.value, opt, places=2)
        self.assertAlmostEqual((x[0] * x[1]).value, 16.0, places=2)
        self.assertAlmostEqual(x[3].value, 0.0, places=2)

    def test_div(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        p = cvxpy.Problem(cvxpy.Minimize(x * y),
                          [y/3 <= x, y >= 1])
        self.assertAlmostEqual(p.solve(SOLVER, gp=True), 1.0 / 3.0)
        self.assertAlmostEqual(y.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0 / 3.0)

    def test_geo_mean(self):
        x = cvxpy.Variable(3, pos=True)
        p = [1, 2, 0.5]
        geo_mean = cvxpy.geo_mean(x, p)
        dgp = cvxpy.Problem(cvxpy.Minimize(geo_mean), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        dcp.solve(SOLVER)
        self.assertEqual(dcp.value, -float("inf"))
        dgp.unpack(dgp2dcp.retrieve(dcp.solution))
        self.assertEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")
        dgp._clear_solution()
        dgp.solve(SOLVER, gp=True)
        self.assertEqual(dgp.value, 0.0)
        self.assertEqual(dgp.status, "unbounded")

    def test_solving_non_dgp_problem_raises_error(self):
        problem = cvxpy.Problem(cvxpy.Minimize(-1.0 * cvxpy.Variable()), [])
        with self.assertRaisesRegexp(error.DGPError, r"Problem does not follow DGP "
                                     "rules(?s)*.*However, the problem does follow DCP rules.*"):
            problem.solve(SOLVER, gp=True)
        problem.solve(SOLVER)
        self.assertEqual(problem.status, "unbounded")
        self.assertEqual(problem.value, -float("inf"))

    def test_solving_non_dcp_problem_raises_error(self):
        problem = cvxpy.Problem(
          cvxpy.Minimize(cvxpy.Variable(pos=True) * cvxpy.Variable(pos=True)),
        )
        with self.assertRaisesRegexp(error.DCPError, r"Problem does not follow DCP "
                                     "rules(?s)*.*However, the problem does follow DGP rules.*"):
            problem.solve(SOLVER)
        problem.solve(SOLVER, gp=True)
        self.assertEqual(problem.status, "unbounded")
        self.assertAlmostEqual(problem.value, 0.0)

    def test_solving_non_dcp_problems_raises_detailed_error(self):
        x = cvxpy.Variable(3)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(x) - cvxpy.sum_squares(x)))
        with self.assertRaisesRegexp(error.DCPError, r"The objective is not DCP"):
            problem.solve(SOLVER)

        x = cvxpy.Variable(name='x')
        problem = cvxpy.Problem(cvxpy.Minimize(x), [x * x <= 5])
        with self.assertRaisesRegexp(error.DCPError, r"The following constraints are not DCP"):
            problem.solve(SOLVER)

    def test_add_canon(self):
        X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        Y = cvxpy.Constant(np.array([[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]]))
        Z = X + Y
        canon_matrix, constraints = dgp_atom_canon.add_canon(Z, Z.args)
        self.assertEqual(len(constraints), 0)
        self.assertEqual(canon_matrix.shape, Z.shape)
        expected = np.log(np.exp(X.value) + np.exp(Y.value))
        np.testing.assert_almost_equal(expected, canon_matrix.value)

        # Test promotion
        X = cvxpy.Constant(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        y = cvxpy.Constant(2.0)
        Z = X + y
        canon_matrix, constraints = dgp_atom_canon.add_canon(Z, Z.args)
        self.assertEqual(len(constraints), 0)
        self.assertEqual(canon_matrix.shape, Z.shape)
        expected = np.log(np.exp(X.value) + np.exp(y.value))
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

    def test_one_minus_pos(self):
        x = cvxpy.Variable(pos=True)
        obj = cvxpy.Maximize(x)
        constr = [cvxpy.one_minus_pos(x) >= 0.4]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(SOLVER, gp=True)
        self.assertAlmostEqual(problem.value, 0.6)
        self.assertAlmostEqual(x.value, 0.6)

    def test_qp_solver_not_allowed(self):
        x = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(x))
        error_msg = ("When `gp=True`, `solver` must be a conic solver "
                     "(received 'OSQP'); try calling `solve()` with "
                     "`solver=cvxpy.ECOS`.")
        with self.assertRaises(error.SolverError) as err:
            problem.solve(solver="OSQP", gp=True)
            self.assertEqual(error_msg, str(err))

    def test_paper_example_sum_largest(self):
        self.skipTest("Enable test once sum_largest is implemented.")
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
        p.solve(SOLVER, gp=True)

    def test_paper_example_one_minus_pos(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(x * y)
        constr = [(y * cvxpy.one_minus_pos(x / y)) ** 2 >= 1, x >= y/3]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(SOLVER, gp=True)

    def test_paper_example_eye_minus_inv(self):
        X = cvxpy.Variable((2, 2), pos=True)
        obj = cvxpy.Minimize(cvxpy.trace(cvxpy.eye_minus_inv(X)))
        constr = [cvxpy.geo_mean(cvxpy.diag(X)) == 0.1,
                  cvxpy.geo_mean(cvxpy.hstack([X[0, 1], X[1, 0]])) == 0.1]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(gp=True, solver="ECOS")
        np.testing.assert_almost_equal(X.value, 0.1*np.ones((2, 2)), decimal=3)
        self.assertAlmostEqual(problem.value, 2.25)

    def test_simpler_eye_minus_inv(self):
        X = cvxpy.Variable((2, 2), pos=True)
        obj = cvxpy.Minimize(cvxpy.trace(cvxpy.eye_minus_inv(X)))
        constr = [cvxpy.diag(X) == 0.1,
                  cvxpy.hstack([X[0, 1], X[1, 0]]) == 0.1]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(gp=True, solver="ECOS")
        np.testing.assert_almost_equal(X.value, 0.1*np.ones((2, 2)), decimal=3)
        self.assertAlmostEqual(problem.value, 2.25)

    def test_paper_example_exp_log(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        obj = cvxpy.Minimize(x * y)
        constr = [cvxpy.exp(y/x) <= cvxpy.log(y)]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(SOLVER, gp=True)

    def test_pf_matrix_completion(self):
        X = cvxpy.Variable((3, 3), pos=True)
        obj = cvxpy.Minimize(cvxpy.pf_eigenvalue(X))
        known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
        constr = [
          X[known_indices] == [1.0, 1.9, 0.8, 3.2, 5.9],
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(SOLVER, gp=True)

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
        prob.solve(SOLVER, gp=True)

    def test_documentation_prob(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)

        objective_fn = x * y * z
        constraints = [
          4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
        problem = cvxpy.Problem(cvxpy.Maximize(objective_fn), constraints)
        # Smoke test.
        problem.solve(SOLVER, gp=True)

    def test_solver_error(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        prod = x * y
        dgp = cvxpy.Problem(cvxpy.Minimize(prod), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp()
        _, inverse_data = dgp2dcp.apply(dgp)
        soln = solution.Solution(SOLVER_ERROR, None, {}, {}, {})
        dgp_soln = dgp2dcp.invert(soln, inverse_data)
        self.assertEqual(dgp_soln.status, SOLVER_ERROR)

    def test_sum_scalar(self):
        w = cvxpy.Variable(pos=True)
        h = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(h),
                                [w*h >= 10, cvxpy.sum(w) <= 5])
        problem.solve(SOLVER, gp=True)
        np.testing.assert_almost_equal(problem.value, 2)
        np.testing.assert_almost_equal(h.value, 2)
        np.testing.assert_almost_equal(w.value, 5)

    def test_sum_vector(self):
        w = cvxpy.Variable(2, pos=True)
        h = cvxpy.Variable(2, pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(h)),
                                [cvxpy.multiply(w, h) >= 10,
                                cvxpy.sum(w) <= 10])
        problem.solve(SOLVER, gp=True)
        np.testing.assert_almost_equal(problem.value, 4)
        np.testing.assert_almost_equal(h.value, np.array([2, 2]))
        np.testing.assert_almost_equal(w.value, np.array([5, 5]))

    def test_sum_squares_vector(self):
        w = cvxpy.Variable(2, pos=True)
        h = cvxpy.Variable(2, pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum_squares(h)),
                                [cvxpy.multiply(w, h) >= 10,
                                cvxpy.sum(w) <= 10])
        problem.solve(SOLVER, gp=True)
        np.testing.assert_almost_equal(problem.value, 8)
        np.testing.assert_almost_equal(h.value, np.array([2, 2]))
        np.testing.assert_almost_equal(w.value, np.array([5, 5]))

    def test_sum_matrix(self):
        w = cvxpy.Variable((2, 2), pos=True)
        h = cvxpy.Variable((2, 2), pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(h)),
                                [cvxpy.multiply(w, h) >= 10,
                                cvxpy.sum(w) <= 20])
        problem.solve(SOLVER, gp=True)
        np.testing.assert_almost_equal(problem.value, 8)
        np.testing.assert_almost_equal(h.value, np.array([[2, 2], [2, 2]]))
        np.testing.assert_almost_equal(w.value, np.array([[5, 5], [5, 5]]))

    def test_trace(self):
        w = cvxpy.Variable((1, 1), pos=True)
        h = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(h),
                                [w*h >= 10, cvxpy.trace(w) <= 5])
        problem.solve(SOLVER, gp=True)
        np.testing.assert_almost_equal(problem.value, 2)
        np.testing.assert_almost_equal(h.value, 2)
        np.testing.assert_almost_equal(w.value, np.array([[5]]))

    def test_parameter(self):
        param = cvxpy.Parameter(pos=True)
        param.value = 1.0
        dgp = cvxpy.Problem(cvxpy.Minimize(param), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        self.assertAlmostEqual(dcp.parameters()[0].value, np.log(param.value))

        x = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(x), [x == param])
        problem.solve(SOLVER, gp=True)
        self.assertAlmostEqual(problem.value, 1.0)

        param.value = 2.0
        problem.solve(SOLVER, gp=True)
        self.assertAlmostEqual(problem.value, 2.0)

    def test_parameter_name(self):
        param = cvxpy.Parameter(pos=True, name='alpha')
        param.value = 1.0
        dgp = cvxpy.Problem(cvxpy.Minimize(param), [])
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)
        dcp = dgp2dcp.reduce()
        self.assertAlmostEqual(dcp.parameters()[0].name(), 'alpha')

    def test_gmatmul(self):
        x = cvxpy.Variable(2, pos=True)
        A = np.matrix("-5 2; 1 -3").A
        b = np.array([3, 2])
        expr = cvxpy.gmatmul(A, x)
        x.value = b
        self.assertItemsAlmostEqual(expr.value, [3**-5*2**2, 3./8])
        A_par = cvxpy.Parameter((2, 2), value=A)
        self.assertItemsAlmostEqual(cvxpy.gmatmul(A_par, x).value,
                                    [3**-5*2**2, 3./8])
        x.value = None

        prob = cvxpy.Problem(cvxpy.Minimize(1.0), [expr == b])
        prob.solve(gp=True)
        sltn = np.exp(np.linalg.solve(A, np.log(b)))
        self.assertItemsAlmostEqual(x.value, sltn)

class TestDgpUnderDpp(BaseTest):
    def test_basic_equality_constraint(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        x = cvxpy.Variable(pos=True)
        dgp = cvxpy.Problem(cvxpy.Minimize(x), [x == alpha])

        self.assertTrue(dgp.objective.is_dpp('dgp'))
        self.assertTrue(dgp.constraints[0].is_dpp('dgp'))
        self.assertTrue(dgp.is_dpp('dgp'))
        dgp2dcp = cvxpy.reductions.Dgp2Dcp(dgp)

        dcp = dgp2dcp.reduce()
        self.assertTrue(dcp.is_dpp())

        dgp.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)

        alpha.value = 2.0
        dgp.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 2.0)

    def test_nested_power_not_dpp(self):
        alpha = cvxpy.Parameter(value=1.0)
        x = cvxpy.Variable(pos=True)

        pow1 = x**alpha
        self.assertTrue(pow1.is_dpp('dgp'))

        pow2 = pow1**alpha
        self.assertFalse(pow2.is_dpp('dgp'))

    def test_non_dpp_problem_raises_error(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        x = cvxpy.Variable(pos=True)
        dgp = cvxpy.Problem(cvxpy.Minimize((alpha*x)**(alpha)), [x == alpha])
        self.assertTrue(dgp.objective.is_dgp())
        self.assertFalse(dgp.objective.is_dpp('dgp'))

        with self.assertRaises(error.DPPError):
            dgp.solve(gp=True, enforce_dpp=True)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dgp.solve(gp=True, enforce_dpp=False)
            self.assertAlmostEqual(x.value, 1.0)

    def test_basic_monomial(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        beta = cvxpy.Parameter(pos=True, value=2.0)
        x = cvxpy.Variable(pos=True)
        monomial = alpha*beta*x
        problem = cvxpy.Problem(cvxpy.Minimize(monomial), [x == alpha])

        self.assertTrue(problem.is_dgp())
        self.assertTrue(problem.is_dpp('dgp'))
        self.assertFalse(problem.is_dpp('dcp'))

        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(problem.value, 2.0)

        alpha.value = 3.0
        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 3.0)
        # 3 * 2 * 3 == 18
        self.assertAlmostEqual(problem.value, 18.0)

    def test_basic_posynomial(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        beta = cvxpy.Parameter(pos=True, value=2.0)
        kappa = cvxpy.Parameter(pos=True, value=3.0)
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        monomial_one = alpha*beta*x
        monomial_two = beta*kappa*x*y
        posynomial = monomial_one + monomial_two
        problem = cvxpy.Problem(cvxpy.Minimize(posynomial),
                                [x == alpha, y == beta])

        self.assertTrue(problem.is_dgp())
        self.assertTrue(problem.is_dpp('dgp'))
        self.assertFalse(problem.is_dpp('dcp'))

        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 2.0)
        # 1*2*1 + 2*3*1*2 == 2 + 12 == 14
        self.assertAlmostEqual(problem.value, 14.0, places=3)

        alpha.value = 4.0
        beta.value = 5.0
        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(x.value, 4.0)
        self.assertAlmostEqual(y.value, 5.0)
        # 4*5*4 + 5*3*4*5 == 80 + 300 == 380
        self.assertAlmostEqual(problem.value, 380.0, places=3)

    def test_basic_gp(self):
        x, y, z = cvxpy.Variable((3,), pos=True)
        a = cvxpy.Parameter(pos=True, value=2.0)
        b = cvxpy.Parameter(pos=True, value=1.0)
        constraints = [a*x*y + a*x*z + a*y*z <= b, x >= a*y]
        problem = cvxpy.Problem(cvxpy.Minimize(1/(x*y*z)), constraints)
        self.assertTrue(problem.is_dpp('dgp'))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(15.59, problem.value, places=2)

    def test_maximum(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        alpha = cvxpy.Parameter(value=0.5)
        beta = cvxpy.Parameter(pos=True, value=3.0)
        kappa = cvxpy.Parameter(pos=True, value=1.0)
        tau = cvxpy.Parameter(pos=True, value=4.0)

        prod1 = x*y**alpha
        prod2 = beta * x*y**alpha
        obj = cvxpy.Minimize(cvxpy.maximum(prod1, prod2))
        constr = [x == kappa, y == tau]

        problem = cvxpy.Problem(obj, constr)
        self.assertTrue(problem.is_dpp('dgp'))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(1*2, 3*1*2) = 6
        self.assertAlmostEqual(problem.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2
        beta.value = 0.5
        kappa.value = 2.0  # x
        tau.value = 3.0    # y
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(2*9, 0.5*2*9) == 18
        self.assertAlmostEqual(problem.value, 18.0, places=4)
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 3.0)

    def test_max(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        alpha = cvxpy.Parameter(value=0.5)
        beta = cvxpy.Parameter(pos=True, value=3.0)
        kappa = cvxpy.Parameter(pos=True, value=1.0)
        tau = cvxpy.Parameter(pos=True, value=4.0)

        prod1 = x*y**alpha
        prod2 = beta * x*y**alpha
        obj = cvxpy.Minimize(cvxpy.max(cvxpy.hstack([prod1, prod2])))
        constr = [x == kappa, y == tau]

        problem = cvxpy.Problem(obj, constr)
        self.assertTrue(problem.is_dpp('dgp'))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(1*2, 3*1*2) = 6
        self.assertAlmostEqual(problem.value, 6.0, places=4)
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2
        beta.value = 0.5
        kappa.value = 2.0  # x
        tau.value = 3.0    # y
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        # max(2*9, 0.5*2*9) == 18
        self.assertAlmostEqual(problem.value, 18.0, places=4)
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 3.0)

    def test_param_in_exponent_and_elsewhere(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0, name='alpha')
        x = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(x**alpha), [x == alpha])

        self.assertTrue(problem.is_dpp('dgp'))
        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)

        # re-solve (which goes through a separate code path)
        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 1.0)
        self.assertAlmostEqual(x.value, 1.0)

        alpha.value = 3.0
        problem.solve(gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 27.0)
        self.assertAlmostEqual(x.value, 3.0)

    def test_minimum(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        alpha = cvxpy.Parameter(pos=True, value=1.0, name='alpha')
        beta = cvxpy.Parameter(pos=True, value=3.0, name='beta')
        prod1 = x * y**alpha
        prod2 = beta * x * y**alpha
        posy = prod1 + prod2
        obj = cvxpy.Maximize(cvxpy.minimum(prod1, prod2, 1/posy))
        constr = [x == alpha, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        # prod1 = 1*4, prod2 = 3*4 = 12, 1/posy = 1/(3 +12)
        self.assertAlmostEqual(dgp.value, 1.0 / (4.0 + 12.0))
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2.0
        # prod1 = 2*16, prod2 = 3*2*16 = 96, 1/posy = 1/(32 +96)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (32.0 + 96.0))
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_min(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        alpha = cvxpy.Parameter(pos=True, value=1.0, name='alpha')
        beta = cvxpy.Parameter(pos=True, value=3.0, name='beta')
        prod1 = x * y**alpha
        prod2 = beta * x * y**alpha
        posy = prod1 + prod2
        obj = cvxpy.Maximize(cvxpy.min(cvxpy.hstack([prod1, prod2, 1/posy])))
        constr = [x == alpha, y == 4.0]

        dgp = cvxpy.Problem(obj, constr)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        # prod1 = 1*4, prod2 = 3*4 = 12, 1/posy = 1/(3 +12)
        self.assertAlmostEqual(dgp.value, 1.0 / (4.0 + 12.0))
        self.assertAlmostEqual(x.value, 1.0)
        self.assertAlmostEqual(y.value, 4.0)

        alpha.value = 2.0
        # prod1 = 2*16, prod2 = 3*2*16 = 96, 1/posy = 1/(32 +96)
        dgp.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(dgp.value, 1.0 / (32.0 + 96.0))
        self.assertAlmostEqual(x.value, 2.0)
        self.assertAlmostEqual(y.value, 4.0)

    def test_div(self):
        alpha = cvxpy.Parameter(pos=True, value=3.0)
        beta = cvxpy.Parameter(pos=True, value=1.0)
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)

        p = cvxpy.Problem(cvxpy.Minimize(x * y), [y/alpha <= x, y >= beta])
        self.assertAlmostEqual(p.solve(SOLVER, gp=True, enforce_dpp=True),
                               1.0 / 3.0)
        self.assertAlmostEqual(x.value, 1.0 / 3.0)
        self.assertAlmostEqual(y.value, 1.0)

        beta.value = 2.0
        p = cvxpy.Problem(cvxpy.Minimize(x * y), [y/alpha <= x, y >= beta])
        self.assertAlmostEqual(p.solve(SOLVER, gp=True, enforce_dpp=True),
                               4.0 / 3.0)
        self.assertAlmostEqual(x.value, 2.0 / 3.0)
        self.assertAlmostEqual(y.value, 2.0)

    def test_one_minus_pos(self):
        x = cvxpy.Variable(pos=True)
        obj = cvxpy.Maximize(x)
        alpha = cvxpy.Parameter(pos=True, value=0.1)
        constr = [cvxpy.one_minus_pos(alpha + x) >= 0.4]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 0.5)
        self.assertAlmostEqual(x.value, 0.5)

        alpha.value = 0.4
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 0.2)
        self.assertAlmostEqual(x.value, 0.2)

    def test_pf_matrix_completion(self):
        X = cvxpy.Variable((3, 3), pos=True)
        obj = cvxpy.Minimize(cvxpy.pf_eigenvalue(X))
        known_indices = tuple(zip(*[[0, 0], [0, 2], [1, 1], [2, 0], [2, 1]]))
        known_values = np.array([1.0, 1.9, 0.8, 3.2, 5.9])
        constr = [
          X[known_indices] == known_values,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cvxpy.Problem(obj, constr)
        # smoke test.
        problem.solve(SOLVER, gp=True)
        optimal_value = problem.value

        param = cvxpy.Parameter(shape=known_values.shape, pos=True,
                                value=0.5*known_values)
        constr = [
          X[known_indices] == param,
          X[0, 1] * X[1, 0] * X[1, 2] * X[2, 2] == 1.0,
        ]
        problem = cvxpy.Problem(obj, constr)
        problem.solve(SOLVER, gp=True, enforce_dpp=True)

        # now change param to point to known_value, and check we recover
        # the correct optimal value
        param.value = known_values
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, optimal_value)

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
        prob.solve(SOLVER, gp=True)
        optimal_value = prob.value

        param = cvxpy.Parameter(value=-2.0)
        R = cvxpy.maximum(
          cvxpy.multiply(X, (xy) ** (param)),
          cvxpy.multiply(X ** (param), xy))
        objective = cvxpy.sum(R)
        prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        prob.solve(SOLVER, gp=True, enforce_dpp=True)

        # now change param to point to known_value, and check we recover the
        # correct optimal value
        param.value = -1.0
        prob.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(prob.value, optimal_value)

    def test_documentation_prob(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)

        objective_fn = x * y * z
        constraints = [
          4 * x * y * z + 2 * x * z <= 10, x <= 2*y, y <= 2*x, z >= 1]
        problem = cvxpy.Problem(cvxpy.Maximize(objective_fn), constraints)
        # Smoke test.
        problem.solve(SOLVER, gp=True)

    def test_sum_scalar(self):
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        w = cvxpy.Variable(pos=True)
        h = cvxpy.Variable(pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(h),
                                [w*h >= 8, cvxpy.sum(alpha + w) <= 5])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 2)
        self.assertAlmostEqual(h.value, 2)
        self.assertAlmostEqual(w.value, 4)

        alpha.value = 4.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 8)
        self.assertAlmostEqual(h.value, 8)
        self.assertAlmostEqual(w.value, 1)

    def test_sum_vector(self):
        alpha = cvxpy.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
        w = cvxpy.Variable(2, pos=True)
        h = cvxpy.Variable(2, pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(h)),
                                [cvxpy.multiply(w, h) >= 20,
                                cvxpy.sum(alpha + w) <= 10])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 10)
        np.testing.assert_almost_equal(h.value, np.array([5, 5]))
        np.testing.assert_almost_equal(w.value, np.array([4, 4]))

        alpha.value = [4.0, 4.0]
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        self.assertAlmostEqual(problem.value, 40)
        np.testing.assert_almost_equal(h.value, np.array([20, 20]), decimal=3)
        np.testing.assert_almost_equal(w.value, np.array([1, 1]), decimal=3)

    def test_sum_squares_vector(self):
        alpha = cvxpy.Parameter(shape=(2,), pos=True, value=[1.0, 1.0])
        w = cvxpy.Variable(2, pos=True)
        h = cvxpy.Variable(2, pos=True)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum_squares(alpha + h)),
                                [cvxpy.multiply(w, h) >= 20,
                                cvxpy.sum(alpha + w) <= 10])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(w.value, np.array([4, 4]))
        np.testing.assert_almost_equal(h.value, np.array([5, 5]))
        self.assertAlmostEqual(problem.value, 6**2 + 6**2)

        alpha.value = [4.0, 4.0]
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(w.value, np.array([1, 1]), decimal=3)
        np.testing.assert_almost_equal(h.value, np.array([20, 20]), decimal=3)
        np.testing.assert_almost_equal(problem.value, 24**2 + 24**2, decimal=3)

    def test_sum_matrix(self):
        w = cvxpy.Variable((2, 2), pos=True)
        h = cvxpy.Variable((2, 2), pos=True)
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        problem = cvxpy.Problem(cvxpy.Minimize(alpha*cvxpy.sum(h)),
                                [cvxpy.multiply(w, h) >= 10,
                                cvxpy.sum(w) <= 20])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 8)
        np.testing.assert_almost_equal(h.value, np.array([[2, 2], [2, 2]]))
        np.testing.assert_almost_equal(w.value, np.array([[5, 5], [5, 5]]))

        alpha.value = 2.0
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 16)

        w = cvxpy.Variable((2, 2), pos=True)
        h = cvxpy.Parameter((2, 2), pos=True)
        h.value = np.ones((2, 2))
        alpha = cvxpy.Parameter(pos=True, value=1.0)
        problem = cvxpy.Problem(cvxpy.Minimize(cvxpy.sum(h)), [w == h])
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 4.0)

        h.value = 2.0 * np.ones((2, 2))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 8.0)

        h.value = 3.0 * np.ones((2, 2))
        problem.solve(SOLVER, gp=True, enforce_dpp=True)
        np.testing.assert_almost_equal(problem.value, 12.0)
>>>>>>> [DGP] DPP for DGP, all tests passing.
