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

import cvxpy as cp
from cvxpy.atoms.affine.vstack import vstack
from cvxpy.atoms.elementwise.power import power
from cvxpy.expressions.variable import Variable
from cvxpy.tests.base_test import BaseTest
import numpy as np
import warnings


class TestExpressions(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self) -> None:
        pass

    # Test elementwise power
    def test_power(self) -> None:
        x = Variable(3)
        y = Variable(3)
        self.assertFalse(x.is_constant())
        self.assertTrue(x.is_affine())
        self.assertTrue(x.is_quadratic())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = power(x.T @ y, 0)
            self.assertTrue(s.is_constant())
            self.assertTrue(s.is_affine())
            self.assertTrue(s.is_quadratic())

        t = power(x-y, 1)
        self.assertFalse(t.is_constant())
        self.assertTrue(t.is_affine())
        self.assertTrue(t.is_quadratic())

        u = power(x+2*y, 2)
        self.assertFalse(u.is_constant())
        self.assertFalse(u.is_affine())
        self.assertTrue(u.is_quadratic())
        self.assertTrue(u.is_dcp())

        w = (x+2*y)**2
        self.assertFalse(w.is_constant())
        self.assertFalse(w.is_affine())
        self.assertTrue(w.is_quadratic())
        self.assertTrue(w.is_dcp())

    def test_matrix_multiplication(self) -> None:
        x = Variable((3, 5))
        y = Variable((3, 5))
        self.assertFalse(x.is_constant())
        self.assertTrue(x.is_affine())
        self.assertTrue(x.is_quadratic())

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = x.T @ y
        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertTrue(s.is_quadratic())
        self.assertFalse(s.is_dcp())

    def test_quad_over_lin(self) -> None:
        x = Variable((3, 5))
        y = Variable((3, 5))
        z = Variable()
        s = cp.quad_over_lin(x-y, z)
        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertFalse(s.is_quadratic())
        self.assertTrue(s.is_dcp())

        t = cp.quad_over_lin(x+2*y, 5)
        self.assertFalse(t.is_constant())
        self.assertFalse(t.is_affine())
        self.assertTrue(t.is_quadratic())
        self.assertTrue(t.is_dcp())

    def test_matrix_frac(self) -> None:
        x = Variable(5)
        M = np.eye(5)
        P = M.T @ M
        s = cp.matrix_frac(x, P)
        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertTrue(s.is_quadratic())
        self.assertTrue(s.is_dcp())

    def test_quadratic_form(self) -> None:
        x = Variable(5)
        P = np.eye(5) - 2*np.ones((5, 5))
        q = np.ones((5, 1))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = x.T @ P @ x + q.T @ x
        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertTrue(s.is_quadratic())
        self.assertFalse(s.is_dcp())

    def test_sum_squares(self) -> None:
        X = Variable((5, 4))
        P = np.ones((3, 5))
        Q = np.ones((4, 7))
        M = np.ones((3, 7))

        y = P @ X @ Q + M
        self.assertFalse(y.is_constant())
        self.assertTrue(y.is_affine())
        self.assertTrue(y.is_quadratic())
        self.assertTrue(y.is_dcp())

        s = cp.sum_squares(y)
        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertTrue(s.is_quadratic())
        self.assertTrue(s.is_dcp())

        # Frobenius norm squared is indeed quadratic
        # but can't show quadraticity using recursive rules
        t = cp.norm(y, 'fro')**2
        self.assertFalse(t.is_constant())
        self.assertFalse(t.is_affine())
        self.assertFalse(t.is_quadratic())
        self.assertTrue(t.is_dcp())

    def test_indefinite_quadratic(self) -> None:
        x = Variable()
        y = Variable()
        z = Variable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = y*z
            self.assertTrue(s.is_quadratic())
            self.assertFalse(s.is_dcp())

            t = (x+y)**2 - s - z*z
            self.assertTrue(t.is_quadratic())
            self.assertFalse(t.is_dcp())

    def test_non_quadratic(self) -> None:
        x = Variable()
        y = Variable()
        z = Variable()

        s = cp.max(vstack([x, y, z]))**2
        self.assertFalse(s.is_quadratic())

        t = cp.max(vstack([x**2, power(y, 2), z]))
        self.assertFalse(t.is_quadratic())

    def test_affine_prod(self) -> None:
        x = Variable((3, 5))
        y = Variable((5, 4))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s = x @ y

        self.assertFalse(s.is_constant())
        self.assertFalse(s.is_affine())
        self.assertTrue(s.is_quadratic())
        self.assertFalse(s.is_dcp())
