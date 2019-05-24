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
import scipy.linalg as la


class TestSCS(BaseTest):
    """ Unit tests for SCS. """
    def setUp(self):
        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(2, name='y')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    # Overriden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_log_problem(self):
        # Log in objective.
        obj = cvx.Maximize(cvx.sum(cvx.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = cvx.Minimize(sum(self.x))
        constr = [cvx.log(self.x) >= 0, self.x <= [1, 1]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.SCS)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = cvx.Maximize(cvx.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cvx.Problem(obj, constr)
        result = p.solve(solver=cvx.SCS)

    def test_sigma_max(self):
        """Test sigma_max.
        """
        const = cvx.Constant([[1, 2, 3], [4, 5, 6]])
        constr = [self.C == const]
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.C, 2)), constr)
        result = prob.solve(solver=cvx.SCS)
        self.assertAlmostEqual(result, cvx.norm(const, 2).value)
        self.assertItemsAlmostEqual(self.C.value, const.value)

    def test_sdp_var(self):
        """Test sdp var.
        """
        const = cvx.Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X = cvx.Variable((3, 3), PSD=True)
        prob = cvx.Problem(cvx.Minimize(0), [X == const])
        prob.solve(verbose=True, solver=cvx.SCS)
        self.assertEqual(prob.status, cvx.INFEASIBLE)

    def test_cplx_mats(self):
        """Test complex matrices.
        """
        # Complex-valued matrix
        K = np.array(np.random.rand(2, 2) + 1j * np.random.rand(2, 2))  # example matrix
        n1 = la.svdvals(K).sum()  # trace norm of K

        # Dual Problem
        X = cvx.Variable((2, 2), complex=True)
        Y = cvx.Variable((2, 2), complex=True)
        # X, Y >= 0 so trace is real
        objective = cvx.Minimize(
            cvx.real(0.5 * cvx.trace(X) + 0.5 * cvx.trace(Y))
        )
        constraints = [
            cvx.bmat([[X, -K.conj().T], [-K, Y]]) >> 0,
            X >> 0,
            Y >> 0,
        ]
        problem = cvx.Problem(objective, constraints)

        sol_scs = problem.solve(solver='SCS')
        self.assertEqual(constraints[0].dual_value.shape, (4, 4))
        self.assertEqual(constraints[1].dual_value.shape, (2, 2))
        self.assertEqual(constraints[2].dual_value.shape, (2, 2))
        self.assertAlmostEqual(sol_scs, n1)

    def test_kl_div(self):
        """Test a problem with kl_div.
        """
        import numpy as np

        kK = 50
        kSeed = 10

        prng = np.random.RandomState(kSeed)
        # Generate a random reference distribution
        npSPriors = prng.uniform(0.0, 1.0, (kK, 1))
        npSPriors = npSPriors/sum(npSPriors)

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
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.entr(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.log(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_warm_start(self):
        """Test warm starting.
        """
        x = cvx.Variable(10)
        obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
        prob = cvx.Problem(obj, [cvx.sum(x) == 1])
        result = prob.solve(solver=cvx.SCS, eps=1e-4)
        time = prob.solver_stats.solve_time
        result2 = prob.solve(solver=cvx.SCS, warm_start=True, eps=1e-4)
        time2 = prob.solver_stats.solve_time
        self.assertAlmostEqual(result2, result, places=2)
        print(time > time2)

    def test_psd_constraint(self):
        """Test PSD constraint.
        """
        s = cvx.Variable((2, 2))
        obj = cvx.Maximize(cvx.minimum(s[0, 1], 10))
        const = [s >> 0, cvx.diag(s) == np.ones(2)]
        prob = cvx.Problem(obj, const)
        r = prob.solve(solver=cvx.SCS)
        s = s.value
        print(const[0].residual)
        print("value", r)
        print("s", s)
        print("eigs", np.linalg.eig(s + s.T)[0])
        eigs = np.linalg.eig(s + s.T)[0]
        self.assertEqual(np.all(eigs >= 0), True)
