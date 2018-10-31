import cvxpy
from cvxpy.tests.base_test import BaseTest


class TestDgp(BaseTest):
    def test_product(self):
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

    def test_product_with_unconstrained_variables_is_not_dgp(self):
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

    def test_division(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        div = x / y

        self.assertTrue(div.is_log_log_affine())

        posynomial = 5.0 * x * y + 1.2 * y * y
        div = x / y
        self.assertTrue(div.is_log_log_affine())

        div = posynomial / (3.0 * x * y**(-0.1))
        self.assertTrue(div.is_log_log_convex())
        self.assertFalse(div.is_log_log_concave())
        self.assertTrue(div.is_dgp())

        div = posynomial / (3.0 * x + y)
        self.assertFalse(div.is_log_log_convex())
        self.assertFalse(div.is_log_log_concave())
        self.assertFalse(div.is_dgp())

    def test_add(self):
        x = cvxpy.Variable(pos=True)
        y = cvxpy.Variable(pos=True)
        expr = x + y 
        self.assertTrue(expr.is_dgp())
        self.assertTrue(expr.is_log_log_convex())
        self.assertTrue(not expr.is_log_log_concave())

        posynomial = 5.0 * x * y + 1.2 * y * y
        self.assertTrue(posynomial.is_dgp())
        self.assertTrue(posynomial.is_log_log_convex())

    def test_add_with_unconstrained_variables_is_not_dgp(self):
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

    def test_monomials(self):
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

    def test_maximum(self):
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
        self.assertTrue(not expr.is_log_log_concave())

    def test_constant(self):
        x = cvxpy.Constant(1.0)
        self.assertTrue(x.is_dgp())
        self.assertFalse((-1.0*x).is_dgp())

    def test_geo_mean(self):
        x = cvxpy.Variable(3, pos=True)
        p = [1, 2, 0.5]
        geo_mean = cvxpy.geo_mean(x, p)
        self.assertTrue(geo_mean.is_dgp())
        self.assertTrue(geo_mean.is_log_log_affine())
        self.assertTrue(geo_mean.is_log_log_convex())
        self.assertTrue(geo_mean.is_log_log_concave())
