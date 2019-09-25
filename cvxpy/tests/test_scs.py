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

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
import math
import numpy as np
import scipy.linalg as la


class TestSCS(BaseTest):
    """ Unit tests for SCS. """
    def setUp(self):
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    # Overriden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_log_problem(self):
        # Log in objective.
        obj = cp.Maximize(cp.sum(cp.log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = cp.Minimize(sum(self.x))
        constr = [cp.log(self.x) >= 0, self.x <= [1, 1]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = cp.Maximize(cp.log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = cp.Problem(obj, constr)
        result = p.solve(solver=cp.SCS)

    def test_sigma_max(self):
        """Test sigma_max.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6]])
        constr = [self.C == const]
        prob = cp.Problem(cp.Minimize(cp.norm(self.C, 2)), constr)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, cp.norm(const, 2).value)
        self.assertItemsAlmostEqual(self.C.value, const.value)

    def test_sdp_var(self):
        """Test sdp var.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X = cp.Variable((3, 3), PSD=True)
        prob = cp.Problem(cp.Minimize(0), [X == const])
        prob.solve(solver=cp.SCS)
        self.assertEqual(prob.status, cp.INFEASIBLE)

    def test_cplx_mats(self):
        """Test complex matrices.
        """
        # Complex-valued matrix
        K = np.array(np.random.rand(2, 2) + 1j * np.random.rand(2, 2))  # example matrix
        n1 = la.svdvals(K).sum()  # trace norm of K

        # Dual Problem
        X = cp.Variable((2, 2), complex=True)
        Y = cp.Variable((2, 2), complex=True)
        # X, Y >= 0 so trace is real
        objective = cp.Minimize(
            cp.real(0.5 * cp.trace(X) + 0.5 * cp.trace(Y))
        )
        constraints = [
            cp.bmat([[X, -K.conj().T], [-K, Y]]) >> 0,
            X >> 0,
            Y >> 0,
        ]
        problem = cp.Problem(objective, constraints)

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
        p_refProb = cp.Parameter((kK, 1), nonneg=True)
        # Distribution to be estimated
        v_prob = cp.Variable((kK, 1))
        objkl = 0.0
        for k in range(kK):
            objkl += cp.kl_div(v_prob[k, 0], p_refProb[k, 0])

        constrs = [sum(v_prob[k, 0] for k in range(kK)) == 1]
        klprob = cp.Problem(cp.Minimize(objkl), constrs)
        p_refProb.value = npSPriors
        klprob.solve(solver=cp.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.entr(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Minimize(cp.sum(cp.exp(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.log(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_solve_problem_twice(self):
        """Test a problem with log.
        """
        n = 5
        x = cp.Variable(n)
        obj = cp.Maximize(cp.sum(cp.log(x)))
        p = cp.Problem(obj, [cp.sum(x) == 1])
        p.solve(solver=cp.SCS)
        first_value = x.value
        self.assertItemsAlmostEqual(first_value, n*[1./n])

        p.solve(solver=cp.SCS)
        second_value = x.value
        self.assertItemsAlmostEqual(first_value, second_value)

    def test_warm_start(self):
        """Test warm starting.
        """
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.SCS, eps=1e-4)
        time = prob.solver_stats.solve_time
        result2 = prob.solve(solver=cp.SCS, warm_start=True, eps=1e-4)
        time2 = prob.solver_stats.solve_time
        self.assertAlmostEqual(result2, result, places=2)
        print(time > time2)

    def test_psd_constraint(self):
        """Test PSD constraint.
        """
        s = cp.Variable((2, 2))
        obj = cp.Maximize(cp.minimum(s[0, 1], 10))
        const = [s >> 0, cp.diag(s) == np.ones(2)]
        prob = cp.Problem(obj, const)
        r = prob.solve(solver=cp.SCS)
        s = s.value
        print(const[0].residual)
        print("value", r)
        print("s", s)
        print("eigs", np.linalg.eig(s + s.T)[0])
        eigs = np.linalg.eig(s + s.T)[0]
        self.assertEqual(np.all(eigs >= 0), True)
