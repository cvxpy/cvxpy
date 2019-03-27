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

import cvxpy as cvx
from cvxpy.tests.base_test import BaseTest
import math
import numpy as np


class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """

    def setUp(self):
        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(2, name='y')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_log_problem(self):
        # Log in objective.
        obj = cvx.Maximize(cvx.sum(cvx.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = cvx.Minimize(cvx.sum(self.x))
        constr = [cvx.log(self.x) >= 0, self.x <= [1, 1]]
        p = cvx.Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = cvx.Maximize(cvx.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)

        # Scalar log.
        obj = cvx.Maximize(cvx.log(self.x[1]))
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)

    def test_entr(self):
        """Test the entr atom.
        """
        self.assertEqual(cvx.entr(0).value, 0)
        assert np.isneginf(cvx.entr(-1).value)

    def test_kl_div(self):
        """Test a problem with kl_div.
        """
        import numpy as np

        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, (kK, 1))
        npSPriors = npSPriors/np.sum(npSPriors)

        # Reference distribution
        p_refProb = cvx.Parameter((kK, 1), nonneg=True)
        # Distribution to be estimated
        v_prob = cvx.Variable((kK, 1))
        objkl = 0.0
        for k in range(kK):
            objkl += cvx.kl_div(v_prob[k, 0], p_refProb[k, 0])

        constrs = [sum(v_prob[k, 0] for k in range(kK)) == 1]
        klprob = cvx.Problem(cvx.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        klprob.solve(solver=cvx.SCS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        klprob.solve(solver=cvx.ECOS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr_prob(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.entr(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.log(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=2)
