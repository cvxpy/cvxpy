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


class TestNonlinearAtoms(BaseTest):
    """ Unit tests for the nonlinear atoms module. """

    def setUp(self):
        self.x = Variable(2, name='x')
        self.y = Variable(2, name='y')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_log_problem(self):
        # Log in objective.
        obj = Maximize(sum_entries(log(self.x)))
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)
        self.assertItemsAlmostEqual(self.x.value, [1, math.e])

        # Log in constraint.
        obj = Minimize(sum_entries(self.x))
        constr = [log(self.x) >= 0, self.x <= [1, 1]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 2)
        self.assertItemsAlmostEqual(self.x.value, [1, 1])

        # Index into log.
        obj = Maximize(log(self.x)[1])
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)

        # Scalar log.
        obj = Maximize(log(self.x[1]))
        constr = [self.x <= [1, math.e]]
        p = Problem(obj, constr)
        result = p.solve()
        self.assertAlmostEqual(result, 1)

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
        npSPriors = npSPriors/np.sum(npSPriors)

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
        if CVXOPT in installed_solvers():
            result = klprob.solve(solver=CVXOPT, verbose=True)
            self.assertItemsAlmostEqual(v_prob.value, npSPriors)
        result = klprob.solve(solver=SCS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors, places=3)
        result = klprob.solve(solver=ECOS, verbose=True)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Maximize(sum_entries(entr(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            p.solve(solver=ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            if CVXOPT in installed_solvers():
                p.solve(solver=CVXOPT, verbose=True)
                self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Minimize(sum_entries(exp(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            if CVXOPT in installed_solvers():
                p.solve(solver=CVXOPT, verbose=True)
                self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=3)
            p.solve(solver=ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = Variable(n)
            obj = Maximize(sum_entries(log(x)))
            p = Problem(obj, [sum_entries(x) == 1])
            if CVXOPT in installed_solvers():
                p.solve(solver=CVXOPT, verbose=True)
                self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=ECOS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n])
            p.solve(solver=SCS, verbose=True)
            self.assertItemsAlmostEqual(x.value, n*[1./n], places=2)

    def test_key_error(self):
        """Test examples that caused key error.
        """
        if CVXOPT in installed_solvers():
            import cvxpy as cvx
            x = cvx.Variable()
            u = -cvx.exp(x)
            prob = cvx.Problem(cvx.Maximize(u), [x == 1])
            prob.solve(verbose=True, solver=cvx.CVXOPT)
            prob.solve(verbose=True, solver=cvx.CVXOPT)

            ###########################################

            import numpy as np
            import cvxopt
            import cvxpy as cp

            kD = 2
            Sk = cp.semidefinite(kD)
            Rsk = cp.Parameter(kD, kD)
            mk = cp.Variable(kD, 1)
            musk = cp.Parameter(kD, 1)

            logpart = -0.5*cp.log_det(Sk)+0.5*cp.matrix_frac(mk, Sk)+(kD/2.)*np.log(2*np.pi)
            linpart = mk.T*musk-0.5*cp.trace(Sk*Rsk)
            obj = logpart-linpart
            prob = cp.Problem(cp.Minimize(obj))
            musk.value = np.ones((2, 1))
            covsk = np.diag([0.3, 0.5])
            Rsk.value = covsk+(musk.value*musk.value.T)
            prob.solve(verbose=True, solver=cp.CVXOPT)
            print("second solve")
            prob.solve(verbose=False, solver=cp.CVXOPT)

    # def test_kl_div(self):
    #     """Test the kl_div atom.
    #     """
    #     self.assertEqual(kl_div(0, 0).value, 0)
    #     self.assertEqual(kl_div(1, 0).value, np.inf)
    #     self.assertEqual(kl_div(0, 1).value, np.inf)
    #     self.assertEqual(kl_div(-1, -1).value, np.inf)
