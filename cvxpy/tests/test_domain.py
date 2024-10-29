"""
Copyright 2013 Steven Diamond

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

import numpy as np

import cvxpy as cp
from cvxpy import Maximize, Minimize, Problem
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
from cvxpy.transforms.partial_optimize import partial_optimize


class TestDomain(BaseTest):
    """ Unit tests for the domain module. """

    def setUp(self) -> None:
        self.a = Variable(name='a')

        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')
        self.z = Variable(3, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_partial_problem(self) -> None:
        """Test domain for partial minimization/maximization problems.
        """
        for obj in [Minimize((self.a)**-1), Maximize(cp.log(self.a))]:
            orig_prob = Problem(obj, [self.x + self.a >= [5, 8]])
            # Optimize over nothing.
            expr = partial_optimize(orig_prob, dont_opt_vars=[self.x, self.a])
            dom = expr.domain
            constr = [self.a >= -100, self.x >= 0]
            prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
            prob.solve(solver=cp.SCS, eps=1e-6)
            self.assertAlmostEqual(prob.value, 13)
            assert self.a.value >= 0
            assert np.all((self.x + self.a - [5, 8]).value >= -1e-3)

            # Optimize over x.
            expr = partial_optimize(orig_prob, opt_vars=[self.x])
            dom = expr.domain
            constr = [self.a >= -100, self.x >= 0]
            prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
            prob.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(prob.value, 0)
            assert self.a.value >= -1e-3
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

            # Optimize over x and a.
            expr = partial_optimize(orig_prob, opt_vars=[self.x, self.a])
            dom = expr.domain
            constr = [self.a >= -100, self.x >= 0]
            prob = Problem(Minimize(sum(self.x + self.a)), dom + constr)
            prob.solve(solver=cp.CLARABEL)
            self.assertAlmostEqual(self.a.value, -100)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_geo_mean(self) -> None:
        """Test domain for geo_mean
        """
        dom = cp.geo_mean(self.x).domain
        prob = Problem(Minimize(sum(self.x)), dom)
        prob.solve(solver=cp.SCS, eps=1e-5)
        self.assertAlmostEqual(prob.value, 0)

        # No special case for only one weight.
        dom = cp.geo_mean(self.x, [0, 2]).domain
        dom.append(self.x >= -1)
        prob = Problem(Minimize(sum(self.x)), dom)
        prob.solve(solver=cp.SCS, eps=1e-5)
        self.assertItemsAlmostEqual(self.x.value, [-1, 0])

        dom = cp.geo_mean(self.z, [0, 1, 1]).domain
        dom.append(self.z >= -1)
        prob = Problem(Minimize(sum(self.z)), dom)
        prob.solve(solver=cp.SCS, eps=1e-5)
        self.assertItemsAlmostEqual(self.z.value, [-1, 0, 0])

    def test_quad_over_lin(self) -> None:
        """Test domain for quad_over_lin
        """
        dom = cp.quad_over_lin(self.x, self.a).domain
        Problem(Minimize(self.a), dom).solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(self.a.value, 0)

    # Throws Segfault from Eigen
    # def test_lambda_max(self):
    #     """Test domain for lambda_max
    #     """
    #     dom = lambda_max(self.A).domain
    #     A0 = [[1, 2], [3, 4]]
    #     Problem(Minimize(norm2(self.A-A0)), dom).solve(solver=cp.SCS)
    #     self.assertItemsAlmostEqual(self.A.value, np.array([[1, 2.5], [2.5, 4]]))

    def test_pnorm(self) -> None:
        """ Test domain for pnorm.
        """
        dom = cp.pnorm(self.a, -0.5).domain
        prob = Problem(Minimize(self.a), dom)
        prob.solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(prob.value, 0)

    def test_log(self) -> None:
        """Test domain for log.
        """
        dom = cp.log(self.a).domain
        Problem(Minimize(self.a), dom).solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(self.a.value, 0)

    def test_log1p(self) -> None:
        """Test domain for log1p.
        """
        dom = cp.log1p(self.a).domain
        Problem(Minimize(self.a), dom).solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(self.a.value, -1)

    def test_entr(self) -> None:
        """Test domain for entr.
        """
        dom = cp.entr(self.a).domain
        Problem(Minimize(self.a), dom).solve(solver=cp.SCS, eps=1e-6)
        self.assertAlmostEqual(self.a.value, 0)

    def test_kl_div(self) -> None:
        """Test domain for kl_div.
        """
        b = Variable()
        dom = cp.kl_div(self.a, b).domain
        Problem(Minimize(self.a + b), dom).solve(solver=cp.SCS, eps=1e-5)
        self.assertAlmostEqual(self.a.value, 0)
        self.assertAlmostEqual(b.value, 0)

    def test_rel_entr(self) -> None:
        """Test domain for rel_entr.
        """
        b = Variable()
        dom = cp.rel_entr(self.a, b).domain
        Problem(Minimize(self.a + b), dom).solve()
        self.assertAlmostEqual(self.a.value, 0)
        self.assertAlmostEqual(b.value, 0)

    def test_power(self) -> None:
        """Test domain for power.
        """
        opts = {'eps': 1e-8, 'max_iters': 100000}
        dom = cp.sqrt(self.a).domain
        Problem(Minimize(self.a), dom).solve(solver=cp.SCS, **opts)
        self.assertAlmostEqual(self.a.value, 0)

        dom = cp.square(self.a).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve(solver=cp.SCS, **opts)
        self.assertAlmostEqual(self.a.value, -100)

        dom = ((self.a)**-1).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve(solver=cp.SCS, **opts)
        self.assertAlmostEqual(self.a.value, 0)

        dom = ((self.a)**3).domain
        Problem(Minimize(self.a), dom + [self.a >= -100]).solve(solver=cp.SCS, **opts)
        self.assertAlmostEqual(self.a.value, 0)

    def test_log_det(self) -> None:
        """Test domain for log_det.
        """
        dom = cp.log_det(self.A + np.eye(2)).domain
        prob = Problem(Minimize(cp.sum(cp.diag(self.A))), dom)
        prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(prob.value, -2, places=3)

    def test_matrix_frac(self) -> None:
        """Test domain for matrix_frac.
        """
        dom = cp.matrix_frac(self.x, self.A + np.eye(2)).domain
        prob = Problem(Minimize(cp.sum(cp.diag(self.A))), dom)
        prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(prob.value, -2, places=3)
