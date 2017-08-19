"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import cvxpy
import cvxpy.settings as s
from cvxpy.atoms import *
from cvxpy.expressions.variables import Variable, NonNegative, Bool, Int
from cvxpy.expressions.constants import Parameter
import cvxpy.transforms.scalarize as scalarize
import cvxpy.utilities as u
import numpy as np
import unittest
from cvxpy import Problem, Minimize
from cvxpy.tests.base_test import BaseTest


class TestTransforms(BaseTest):
    """ Unit tests for the atoms module. """

    def setUp(self):
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_scalarize(self):
        """Test scalarize functions.
        """
        min_objs = [cvxpy.Minimize(self.x[0]**2), cvxpy.Minimize(self.x[1] + 3)]
        mixed_objs = [cvxpy.Minimize(self.x[0]**2), cvxpy.Maximize(self.x[1] + 3)]

        single_obj = scalarize.weighted_sum(min_objs, [2, 1])
        result = Problem(single_obj, [self.x == 2]).solve()
        self.assertAlmostEqual(13, result)

        single_obj = scalarize.weighted_sum(mixed_objs, [2, -1])
        result = Problem(single_obj, [self.x == 2]).solve()
        self.assertAlmostEqual(3, result)

        single_obj = scalarize.max(min_objs, [2, 1])
        result = Problem(single_obj, [self.x == 2]).solve()
        self.assertAlmostEqual(8, result)

        single_obj = scalarize.log_sum_exp(mixed_objs, [2, -4], gamma=1)
        result = Problem(single_obj, [self.x[0] == -2, self.x[1] == -5]).solve()
        self.assertAlmostEqual(8 + np.log(2), result)

        single_obj = scalarize.targets_and_priorities(min_objs, [2, 1], [5, 3], off_target=1e-9)
        result = Problem(single_obj, [self.x == 2]).solve()
        self.assertAlmostEqual(2, result)

        single_obj = scalarize.targets_and_priorities(min_objs, [2, 1], [0, 3], limits=[4, 2], off_target=1e-9)
        result = Problem(single_obj, [self.x[0] == self.x[1]]).solve()
        self.assertItemsAlmostEqual([-1, -1], self.x.value)
        self.assertAlmostEqual(2, result)

        single_obj = scalarize.targets_and_priorities(mixed_objs, [2, -1], [1, 10], off_target=0)
        result = Problem(single_obj, [self.x == 2]).solve()
        self.assertAlmostEqual(1, result)

        single_obj = scalarize.targets_and_priorities(min_objs, [-2, -1], [0, 3], limits=[4, 2], off_target=0)
        result = Problem(single_obj, [self.x[0] == self.x[1]]).solve()
        self.assertItemsAlmostEqual([-1, -1], self.x.value)
        self.assertAlmostEqual(-2, result)


    def test_indicator(self):
        """Test indicator transform.
        """
        cons = [self.a >= 0, self.x == 2]
        obj = cvxpy.Minimize(self.a - sum_entries(self.x))
        expr = cvxpy.indicator(cons)
        assert expr.is_convex()
        assert expr.is_positive()
        prob = Problem(Minimize(expr) + obj)
        result = prob.solve()
        self.assertAlmostEqual(-4, result)
        self.assertAlmostEqual(0, self.a.value)
        self.assertItemsAlmostEqual([2, 2], self.x.value)
        self.assertAlmostEqual(0, expr.value)
        self.a.value = -1
        self.assertAlmostEqual(np.infty, expr.value)

    def test_partial_optimize_dcp(self):
        """Test DCP properties of partial optimize.
        """
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONVEX)

        p2 = Problem(cvxpy.Maximize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEqual(g.curvature, s.CONCAVE)

        p2 = Problem(cvxpy.Maximize(square(t[0])), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        self.assertEquals(g.is_convex(), False)
        self.assertEquals(g.is_concave(), False)

    # Test the partial_optimize atom.
    def test_partial_optimize_eval_1norm(self):
        # Evaluate the 1-norm in the usual way (i.e., in epigraph form).
        dims = 3
        x, t = Variable(dims), Variable(dims)
        xval = [-5]*dims
        p1 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= xval, xval <= t])
        p1.solve()

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, [t], [x])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm using maximize.
        p2 = Problem(cvxpy.Maximize(sum_entries(-t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cvxpy.Maximize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, -p3.value)

        # Try leaving out args.

        # Minimize the 1-norm via partial_optimize.
        p2 = Problem(cvxpy.Minimize(sum_entries(t)), [-t <= x, x <= t])
        g = cvxpy.partial_optimize(p2, opt_vars=[t])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        # Minimize the 1-norm via partial_optimize.
        g = cvxpy.partial_optimize(p2, dont_opt_vars=[x])
        p3 = Problem(cvxpy.Minimize(g), [x == xval])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

        with self.assertRaises(Exception) as cm:
            g = cvxpy.partial_optimize(p2)
        self.assertEqual(str(cm.exception),
                         "partial_optimize called with neither opt_vars nor dont_opt_vars.")

        with self.assertRaises(Exception) as cm:
            g = cvxpy.partial_optimize(p2, [], [x])
        self.assertEqual(str(cm.exception),
                         ("If opt_vars and new_opt_vars are both specified, "
                          "they must contain all variables in the problem.")
                         )

    def test_partial_optimize_min_1norm(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(sum_entries(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = cvxpy.partial_optimize(p1, [t], [x])
        p2 = Problem(Minimize(g))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)

    def test_partial_optimize_simple_problem(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_var(self):
        x, y = Bool(1), Int(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_special_constr(self):
        x, y = Variable(1), Variable(1)

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x + exp(y)), [x+y >= 3, y >= 4, x >= 5])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(exp(y)), [x+y >= 3, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_params(self):
        """Test partial optimize with parameters.
        """
        x, y = Variable(1), Variable(1)
        gamma = Parameter()

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(x+y), [x+y >= gamma, y >= 4, x >= 5])
        gamma.value = 3
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        p2 = Problem(Minimize(y), [x+y >= gamma, y >= 4])
        g = cvxpy.partial_optimize(p2, [y], [x])
        p3 = Problem(Minimize(x+g), [x >= 5])
        p3.solve()
        self.assertAlmostEqual(p1.value, p3.value)

    def test_partial_optimize_numeric_fn(self):
        x, y = Variable(1), Variable(1)
        xval = 4

        # Solve the (simple) two-stage problem by "combining" the two stages (i.e., by solving a single linear program)
        p1 = Problem(Minimize(y), [xval+y >= 3])
        p1.solve()

        # Solve the two-stage problem via partial_optimize
        constr = [y >= -100]
        p2 = Problem(Minimize(y), [x+y >= 3] + constr)
        g = cvxpy.partial_optimize(p2, [y], [x])
        x.value = xval
        y.value = 42
        constr[0].dual_variable.value = 42
        result = g.value
        self.assertAlmostEqual(result, p1.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(constr[0].dual_value, 42)

        # No variables optimized over.
        p2 = Problem(Minimize(y), [x+y >= 3])
        g = cvxpy.partial_optimize(p2, [], [x, y])
        x.value = xval
        y.value = 42
        p2.constraints[0].dual_variable.value = 42
        result = g.value
        self.assertAlmostEqual(result, y.value)
        self.assertAlmostEqual(y.value, 42)
        self.assertAlmostEqual(p2.constraints[0].dual_value, 42)

    def test_partial_optimize_stacked(self):
        # Minimize the 1-norm in the usual way
        dims = 3
        x, t = Variable(dims), Variable(dims)
        p1 = Problem(Minimize(sum_entries(t)), [-t <= x, x <= t])

        # Minimize the 1-norm via partial_optimize
        g = cvxpy.partial_optimize(p1, [t], [x])
        g2 = cvxpy.partial_optimize(Problem(Minimize(g)), [x])
        p2 = Problem(Minimize(g2))
        p2.solve()

        p1.solve()
        self.assertAlmostEqual(p1.value, p2.value)
