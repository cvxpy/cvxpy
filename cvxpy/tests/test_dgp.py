import numpy as np
import scipy.sparse as sp

import cvxpy
from cvxpy.tests.base_test import BaseTest


class TestDgp(BaseTest):
    def test_product(self) -> None:
        x = cvxpy.Variable((), pos=True)
        y = cvxpy.Variable((), pos=True)
        prod = x * y
        self.assertTrue(prod.is_dgp())
        self.assertTrue(prod.is_log_log_convex())
        self.assertTrue(prod.is_log_log_concave())

        prod *= prod
        self.assertTrue(prod.is_dgp())
        self.assertTrue(prod.is_log_log_convex())
        self.assertTrue(prod.is_log_log_concave())

        prod *= 5.0
        self.assertTrue(prod.is_dgp())
        self.assertTrue(prod.is_log_log_convex())
        self.assertTrue(prod.is_log_log_concave())

        prod *= -5.0
        self.assertTrue(not prod.is_dgp())
        self.assertTrue(not prod.is_log_log_convex())
        self.assertTrue(not prod.is_log_log_concave())

    def test_product_with_unconstrained_variables_is_not_dgp(self) -> None:
        x = cvxpy.Variable()
        y = cvxpy.Variable()
        prod = x * y
        self.assertTrue(not prod.is_dgp())
        self.assertTrue(not prod.is_log_log_convex())
        self.assertTrue(not prod.is_log_log_concave())

        z = cvxpy.Variable((), pos=True)
        prod = x * z
        self.assertTrue(not prod.is_dgp())
        self.assertTrue(not prod.is_log_log_convex())
        self.assertTrue(not prod.is_log_log_concave())

    def test_division(self) -> None:
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        div = x / y

        self.assertTrue(div.is_log_log_affine())

        posynomial = 5.0 * x * y + 1.2 * y * y
        div = x / y
        self.assertTrue(div.is_log_log_affine())

        div = posynomial / (3.0 * x * y ** (-0.1))
        self.assertTrue(div.is_log_log_convex())
        self.assertFalse(div.is_log_log_concave())
        self.assertTrue(div.is_dgp())

        div = posynomial / (3.0 * x + y)
        self.assertFalse(div.is_log_log_convex())
        self.assertFalse(div.is_log_log_concave())
        self.assertFalse(div.is_dgp())

    def test_add(self) -> None:
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        expr = x + y
        self.assertTrue(expr.is_dgp())
        self.assertTrue(expr.is_log_log_convex())
        self.assertTrue(not expr.is_log_log_concave())

        posynomial = 5.0 * x * y + 1.2 * y * y
        self.assertTrue(posynomial.is_dgp())
        self.assertTrue(posynomial.is_log_log_convex())

    def test_add_with_unconstrained_variables_is_not_dgp(self) -> None:
        x = cvxpy.Variable()
        y = cvxpy.Variable(pos=True)
        expr = x + y
        self.assertTrue(not expr.is_dgp())
        self.assertTrue(not expr.is_log_log_convex())
        self.assertTrue(not expr.is_log_log_concave())

        posynomial = 5.0 * x * y + 1.2 * y * y
        self.assertTrue(not posynomial.is_dgp())
        self.assertTrue(not posynomial.is_log_log_convex())
        self.assertTrue(not posynomial.is_log_log_concave())

    def test_monomials(self) -> None:
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)
        monomial = 5.0 * (x ** 0.1) * y ** (-0.1) * z ** (3)
        self.assertTrue(monomial.is_dgp())
        self.assertTrue(monomial.is_log_log_convex())
        self.assertTrue(monomial.is_log_log_concave())

        monomial *= -1.0
        self.assertTrue(not monomial.is_dgp())
        self.assertTrue(not monomial.is_log_log_convex())
        self.assertTrue(not monomial.is_log_log_concave())

    def test_maximum(self) -> None:
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)
        monomial = 5.0 * (x ** 0.1) * y ** (-0.1) * z ** (3)
        posynomial = 5.0 * x * y + 1.2 * y * y
        another_posynomial = posynomial * posynomial
        expr = cvxpy.maximum(monomial, posynomial, another_posynomial)
        self.assertTrue(expr.is_dgp())
        self.assertTrue(expr.is_log_log_convex())
        self.assertTrue(not expr.is_log_log_concave())

        expr = posynomial * expr
        self.assertTrue(expr.is_dgp())
        self.assertTrue(expr.is_log_log_convex())
        self.assertTrue(not expr.is_log_log_concave())

        expr = posynomial * expr + expr
        self.assertTrue(expr.is_dgp())
        self.assertTrue(expr.is_log_log_convex())

    def test_minimum(self) -> None:
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        z = cvxpy.Variable(pos=True)
        monomial = 5.0 * (x ** 0.1) * y ** (-0.1) * z ** (3)
        posynomial = 5.0 * x * y + 1.2 * y * y
        another_posynomial = posynomial * posynomial
        expr = cvxpy.minimum(monomial, 1 / posynomial, 1 / another_posynomial)
        self.assertTrue(expr.is_dgp())
        self.assertTrue(not expr.is_log_log_convex())
        self.assertTrue(expr.is_log_log_concave())

        expr = (1 / posynomial) * expr
        self.assertTrue(expr.is_dgp())
        self.assertTrue(not expr.is_log_log_convex())
        self.assertTrue(expr.is_log_log_concave())

        expr = expr ** 2
        self.assertTrue(expr.is_dgp())
        self.assertTrue(not expr.is_log_log_convex())
        self.assertTrue(expr.is_log_log_concave())

    def test_constant(self) -> None:
        x = cvxpy.Constant(1.0)
        self.assertTrue(x.is_dgp())
        self.assertFalse((-1.0*x).is_dgp())

    def test_geo_mean(self) -> None:
        x = cvxpy.Variable(3, pos=True)
        p = [1, 2, 0.5]
        geo_mean = cvxpy.geo_mean(x, p)
        self.assertTrue(geo_mean.is_dgp())
        self.assertTrue(geo_mean.is_log_log_affine())
        self.assertTrue(geo_mean.is_log_log_convex())
        self.assertTrue(geo_mean.is_log_log_concave())

    def test_geo_mean_scalar1(self) -> None:
        x = cvxpy.Variable(1, pos=True)
        p = np.array([2])
        geo_mean = cvxpy.geo_mean(x, p)
        self.assertTrue(geo_mean.is_dgp())
        prob = cvxpy.Problem(
            cvxpy.Maximize(geo_mean),
            [x == 2],
        )
        prob.solve()
        self.assertEqual(prob.value, 2)

    def test_geo_mean_scalar2(self) -> None:
        x = cvxpy.Variable(pos=True)
        p = np.array([2])
        geo_mean = cvxpy.geo_mean(x, p)
        self.assertTrue(geo_mean.is_dgp())

    def test_inv_prod(self) -> None:
        x = cvxpy.Variable(2)
        # # test inv_prod with scalar value
        prob1 = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.inv_prod(x[0]) + cvxpy.inv_prod(x[:2])),
            [cvxpy.sum(x) == 2],
        )
        prob1.solve()

        # compare inv_prod with inv_pos
        prob2 = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.inv_prod(x[:1]) + cvxpy.inv_prod(x[:2])),
            [cvxpy.sum(x) == 2],
        )
        prob2.solve()

        prob3 = cvxpy.Problem(
            cvxpy.Minimize(cvxpy.inv_pos(x[0]) + cvxpy.inv_prod(x[:2])),
            [cvxpy.sum(x) == 2],
        )
        prob3.solve()
        self.assertAlmostEqual(prob2.value, prob3.value, 4)

    def test_builtin_sum(self) -> None:
        x = cvxpy.Variable(2, pos=True)
        self.assertTrue(sum(x).is_log_log_convex())

    def test_gmatmul(self) -> None:
        x = cvxpy.Variable(2, pos=True)
        A = cvxpy.Variable((2, 2))
        with self.assertRaises(Exception) as cm:
            cvxpy.gmatmul(A, x)
        self.assertTrue(str(cm.exception) ==
                        "gmatmul(A, X) requires that A be constant.")

        x = cvxpy.Variable(2)
        A = np.ones((4, 2))
        with self.assertRaises(Exception) as cm:
            cvxpy.gmatmul(A, x)
        self.assertTrue(str(cm.exception) ==
                        "gmatmul(A, X) requires that X be positive.")

        x = cvxpy.Variable(3, pos=True)
        A = np.ones((4, 3))
        gmatmul = cvxpy.gmatmul(A, x)
        self.assertTrue(gmatmul.is_dgp())
        self.assertTrue(gmatmul.is_log_log_affine())
        self.assertTrue(gmatmul.is_log_log_convex())
        self.assertTrue(gmatmul.is_log_log_concave())
        self.assertTrue(gmatmul.is_nonneg())
        self.assertTrue(gmatmul.is_incr(0))
        self.assertTrue(cvxpy.gmatmul(-A, x).is_decr(0))

        x = cvxpy.Variable((2, 3), pos=True)
        A = np.array([[2., -1.], [0., 3.]])
        gmatmul = cvxpy.gmatmul(A, x)
        self.assertTrue(gmatmul.is_dgp())
        self.assertTrue(gmatmul.is_log_log_affine())
        self.assertTrue(gmatmul.is_log_log_convex())
        self.assertTrue(gmatmul.is_log_log_concave())
        self.assertFalse(gmatmul.is_incr(0))
        self.assertFalse(gmatmul.is_decr(0))

    def test_power_sign(self) -> None:
        x = cvxpy.Variable(pos=True)
        self.assertTrue((x**1).is_nonneg())
        self.assertFalse((x**1).is_nonpos())

    def test_sparse_constant_not_allowed(self) -> None:
        sparse_matrix = cvxpy.Constant(sp.csc_array(np.array([[1.0, 2.0]])))
        self.assertFalse(sparse_matrix.is_log_log_constant())
