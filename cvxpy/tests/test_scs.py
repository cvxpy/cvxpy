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

from cvxpy import *
import cvxpy.atoms.elementwise.log as cvxlog
from cvxpy.tests.base_test import BaseTest
import math
import numpy as np
import sys
if sys.version_info >= (3, 0):
    from functools import reduce


class TestSCS(BaseTest):
    """ Unit tests for SCS. """
    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    # Overriden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_log_problem(self):
        # Log in objective.
        obj = Maximize(sum_entries(log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve(solver=SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = Minimize(sum_entries(self.x))
        constr = [log(self.x) >= 0, self.x <= [1, 1]]
        p = Problem(obj, constr)
        result = p.solve(solver=SCS)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = Maximize(log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve(solver=SCS)
        self.assertAlmostEqual(result, 1)

    def test_sigma_max(self):
        """Test sigma_max.
        """
        const = Constant([[1, 2, 3], [4, 5, 6]])
        constr = [self.C == const]
        prob = Problem(Minimize(norm(self.C, 2)), constr)
        result = prob.solve(solver=SCS)
        self.assertAlmostEqual(result, norm(const, 2).value)
        self.assertItemsAlmostEqual(self.C.value, const.value)

    def test_sdp_var(self):
        """Test sdp var.
        """
        const = Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X = Semidef(3)
        prob = Problem(Minimize(0), [X == const])
        prob.solve(verbose=True, solver=SCS)
        self.assertEqual(prob.status, INFEASIBLE)

    def test_entr(self):
        """Test the entr atom.
        """
        self.assertEqual(entr(0).value, 0)
        assert np.isneginf(entr(-1).value)

    def test_kl_div(self):
        """Test a problem with kl_div.
        """
        import numpy as np
        import cvxpy as cp

        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        #Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, (kK, 1))
        npSPriors = npSPriors/sum(npSPriors)

        #Reference distribution
        p_refProb = cp.Parameter(kK, 1, sign='positive')
        #Distribution to be estimated
        v_prob = cp.Variable(kK, 1)
        objkl = 0.0
        for k in range(kK):
            objkl += cp.kl_div(v_prob[k, 0], p_refProb[k, 0])

        constrs = [sum([v_prob[k, 0] for k in range(kK)]) == 1]
        klprob = cp.Problem(cp.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        result = klprob.solve(solver=SCS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Maximize(sum_entries(entr(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Minimize(sum_entries(exp(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Maximize(sum_entries(log(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_consistency(self):
        """Test case for non-deterministic behavior in cvxopt.
        """
        import cvxpy

        xs = [0, 1, 2, 3]
        ys = [51, 60, 70, 75]

        eta1 = cvxpy.Variable()
        eta2 = cvxpy.Variable()
        eta3 = cvxpy.Variable()
        theta1s = [eta1 + eta3*x for x in xs]
        lin_parts = [theta1 * y + eta2 * y**2 for (theta1, y) in zip(theta1s, ys)]
        g_parts = [-cvxpy.quad_over_lin(theta1, -4*eta2) + 0.5 * cvxpy.log(-2 * eta2)
                   for theta1 in theta1s]
        objective = reduce(lambda x, y: x+y, lin_parts + g_parts)
        problem = cvxpy.Problem(cvxpy.Maximize(objective))
        problem.solve(verbose=True, solver=cvxpy.SCS)
        assert problem.status in [cvxpy.OPTIMAL_INACCURATE, cvxpy.OPTIMAL]
        return [eta1.value, eta2.value, eta3.value]

    def test_warm_start(self):
        """Test warm starting.
        """
        x = Variable(10)
        obj = Minimize(sum_entries(exp(x)))
        prob = Problem(obj, [sum_entries(x) == 1])
        result = prob.solve(solver=SCS)
        assert prob.solve(solver=SCS, verbose=True) == result
        # TODO Probably a bad check. Ought to be the same.
        assert prob.solve(solver=SCS, warm_start=True, verbose=True) != result

    # def test_kl_div(self):
    #     """Test the kl_div atom.
    #     """
    #     self.assertEqual(kl_div(0, 0).value, 0)
    #     self.assertEqual(kl_div(1, 0).value, np.inf)
    #     self.assertEqual(kl_div(0, 1).value, np.inf)
    #     self.assertEqual(kl_div(-1, -1).value, np.inf)
