"""
Copyright 2013 Steven Diamond, Eric Chu

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

import math

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """

    def setUp(self) -> None:
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_log_problem(self) -> None:
        # Log in objective.
        obj = cp.Maximize(cp.sum(cp.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = cp.Minimize(cp.sum(self.x))
        constr = [cp.log(self.x) >= 0, self.x <= [1, 1]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = cp.Maximize(cp.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)

        # Scalar log.
        obj = cp.Maximize(cp.log(self.x[1]))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.CLARABEL)
        self.assertAlmostEqual(result, 1)

    def test_entr(self) -> None:
        """Test the entr atom.
        """
        self.assertEqual(cp.entr(0).value, 0)
        assert np.isneginf(cp.entr(-1).value)

    def test_kl_div(self) -> None:
        """Test a problem with kl_div.
        """
        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)

        # Reference distribution
        p_refProb = cp.Parameter(kK, nonneg=True)
        # Distribution to be estimated
        v_prob = cp.Variable(kK)
        objkl = cp.sum(cp.kl_div(v_prob, p_refProb))

        constrs = [cp.sum(v_prob) == 1]
        klprob = cp.Problem(cp.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        klprob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        klprob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)

    def test_rel_entr(self) -> None:
        """Test a problem with rel_entr.
        """
        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, kK)
        npSPriors = npSPriors / sum(npSPriors)

        # Reference distribution
        p_refProb = cp.Parameter(kK, nonneg=True)
        # Distribution to be estimated
        v_prob = cp.Variable(kK)
        obj_rel_entr = cp.sum(cp.rel_entr(v_prob, p_refProb))

        constrs = [cp.sum(v_prob) == 1]
        rel_entr_prob = cp.Problem(cp.Minimize(obj_rel_entr), constrs)
        p_refProb.value = npSPriors
        rel_entr_prob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        rel_entr_prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)

    def test_difference_kl_div_rel_entr(self) -> None:
        """A test showing the difference between kl_div and rel_entr
        """
        x = cp.Variable()
        y = cp.Variable()

        kl_div_prob = cp.Problem(cp.Minimize(cp.kl_div(x, y)), constraints=[x + y <= 1])
        kl_div_prob.solve(solver=cp.CLARABEL)
        self.assertItemsAlmostEqual(x.value, y.value, places=3)
        self.assertItemsAlmostEqual(kl_div_prob.value, 0)

        rel_entr_prob = cp.Problem(cp.Minimize(cp.rel_entr(x, y)), constraints=[x + y <= 1])
        rel_entr_prob.solve(solver=cp.CLARABEL)

        """
        Reference solution computed by passing the following command to Wolfram Alpha:
        minimize x*log(x/y) subject to {x + y <= 1, 0 <= x, 0 <= y}
        """
        self.assertItemsAlmostEqual(x.value, 0.2178117, places=4)
        self.assertItemsAlmostEqual(y.value, 0.7821882, places=4)
        self.assertItemsAlmostEqual(rel_entr_prob.value, -0.278464)

    def test_entr_prob(self) -> None:
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.entr(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)

    def test_exp(self) -> None:
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Minimize(cp.sum(cp.exp(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self) -> None:
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.log(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=2)
