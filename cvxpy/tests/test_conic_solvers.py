"""
Copyright 2019, the CVXPY developers.

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
import scipy.linalg as la
import cvxpy as cvx
import unittest
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import StandardTestECPs, StandardTestSDPs
from cvxpy.tests.solver_test_helpers import StandardTestSOCPs, StandardTestLPs


class TestECOS(BaseTest):

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_ecos_options(self):
        """Test that all the ECOS solver options work.
        """
        # Test ecos
        # feastol, abstol, reltol, feastol_inacc,
        # abstol_inacc, and reltol_inacc for tolerance values
        # max_iters for the maximum number of iterations,
        EPS = 1e-4
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
        for i in range(2):
            prob.solve(solver=cvx.ECOS, feastol=EPS, abstol=EPS, reltol=EPS,
                       feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                       max_iters=20, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_ecos_bb_options(self):
        """Test that all the ECOS BB solver options work.
        """
        # 'mi_maxiter'
        # maximum number of branch and bound iterations (default: 1000)
        # 'mi_abs_eps'
        # absolute tolerance between upper and lower bounds (default: 1e-6)
        # 'mi_rel_eps'
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0),
                           [self.x == cvx.Variable(2, boolean=True)])
        for i in range(2):
            prob.solve(solver=cvx.ECOS_BB, mi_max_iters=100, mi_abs_eps=1e-6,
                       mi_rel_eps=1e-5, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_ecos_lp_0(self):
        StandardTestLPs.test_lp_0(solver='ECOS')

    def test_ecos_lp_1(self):
        StandardTestLPs.test_lp_1(solver='ECOS')

    def test_ecos_lp_2(self):
        StandardTestLPs.test_lp_2(solver='ECOS')

    def test_ecos_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='ECOS')

    def test_ecos_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='ECOS')

    def test_ecos_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='ECOS')

    def test_ecos_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='ECOS')


class TestSCS(BaseTest):

    """ Unit tests for SCS. """
    def setUp(self):
        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(2, name='y')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=2):
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_scs_options(self):
        """Test that all the SCS solver options work.
        """
        # Test SCS
        # MAX_ITERS, EPS, ALPHA, UNDET_TOL, VERBOSE, and NORMALIZE.
        # If opts is missing, then the algorithm uses default settings.
        # USE_INDIRECT = True
        EPS = 1e-4
        x = cvx.Variable(2, name='x')
        prob = cvx.Problem(cvx.Minimize(cvx.norm(x, 1) + 1.0), [x == 0])
        for i in range(2):
            prob.solve(solver=cvx.SCS, max_iters=50, eps=EPS, alpha=EPS,
                       verbose=True, normalize=True, use_indirect=False)
        self.assertAlmostEqual(prob.value, 1.0, places=2)
        self.assertItemsAlmostEqual(x.value, [0, 0], places=2)

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
        prob.solve(solver=cvx.SCS)
        self.assertEqual(prob.status, cvx.INFEASIBLE)

    def test_complex_matrices(self):
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
        klprob.solve(solver=cvx.SCS)
        self.assertItemsAlmostEqual(v_prob.value, npSPriors)

    def test_entr(self):
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.entr(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_exp(self):
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self):
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cvx.Variable(n)
            obj = cvx.Maximize(cvx.sum(cvx.log(x)))
            p = cvx.Problem(obj, [cvx.sum(x) == 1])
            p.solve(solver=cvx.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_solve_problem_twice(self):
        """Test a problem with log.
        """
        n = 5
        x = cvx.Variable(n)
        obj = cvx.Maximize(cvx.sum(cvx.log(x)))
        p = cvx.Problem(obj, [cvx.sum(x) == 1])
        p.solve(solver=cvx.SCS)
        first_value = x.value
        self.assertItemsAlmostEqual(first_value, n*[1./n])

        p.solve(solver=cvx.SCS)
        second_value = x.value
        self.assertItemsAlmostEqual(first_value, second_value)

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

    def test_warm_start_diffcp(self):
        """Test warm starting in diffcvx.
        """
        try:
            import diffcp
            diffcp  # for flake8
        except ImportError:
            self.skipTest("diffcp not installed.")
        x = cvx.Variable(10)
        obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
        prob = cvx.Problem(obj, [cvx.sum(x) == 1])
        result = prob.solve(solver=cvx.DIFFCP, eps=1e-4)
        result2 = prob.solve(solver=cvx.DIFFCP, warm_start=True, eps=1e-4)
        self.assertAlmostEqual(result2, result, places=2)

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

    def test_scs_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='SCS')

    def test_scs_sdp_1min(self):
        StandardTestSDPs.test_sdp_1min(solver='SCS')

    def test_scs_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='SCS')


@unittest.skipUnless('MOSEK' in cvx.installed_solvers(), 'MOSEK is not installed.')
class TestMosek(unittest.TestCase):

    def test_mosek_lp_0(self):
        StandardTestLPs.test_lp_0(solver='MOSEK')

    def test_mosek_lp_1(self):
        # default settings
        StandardTestLPs.test_lp_1(solver='MOSEK')
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='MOSEK', places=7, bfs=True)

    def test_mosek_lp_2(self):
        StandardTestLPs.test_lp_2(solver='MOSEK')

    def test_mosek_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='MOSEK')

    def test_mosek_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='MOSEK')

    def test_mosek_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='MOSEK')

    def test_mosek_sdp_1(self):
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='MOSEK')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='MOSEK')

    def test_mosek_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='MOSEK')

    def test_mosek_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='MOSEK')

    def test_mosek_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='MOSEK')

    def test_mosek_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='MOSEK')

    def test_mosek_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='MOSEK')

    def test_mosek_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='MOSEK')

    def test_mosek_params(self):
        if cvx.MOSEK in cvx.installed_solvers():
            import mosek
            n = 10
            m = 4
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cvx.Variable(n)
            objective = cvx.Minimize(cvx.norm1(z))
            constraints = [A * z == y]
            problem = cvx.Problem(objective, constraints)

            invalid_mosek_params = {
                "dparam.basis_tol_x": "1e-8"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.MOSEK, mosek_params=invalid_mosek_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.MOSEK, invalid_kwarg=None)

            mosek_params = {
                mosek.dparam.basis_tol_x: 1e-8,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20
            }
            problem.solve(solver=cvx.MOSEK, mosek_params=mosek_params)
        pass


@unittest.skipUnless('SUPER_SCS' in cvx.installed_solvers(), 'SUPER_SCS is not installed.')
class TestSuperSCS(BaseTest):

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

    def test_super_scs_lp_0(self):
        StandardTestLPs.test_lp_0(solver='SUPER_SCS')

    def test_super_scs_lp_1(self):
        StandardTestLPs.test_lp_1(solver='SUPER_SCS')

    def test_super_scs_lp_2(self):
        StandardTestLPs.test_lp_2(solver='SUPER_SCS')

    def test_super_scs_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='SUPER_SCS')

    def test_super_scs_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='SUPER_SCS')

    def test_super_scs_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='SUPER_SCS')

    def test_super_scs_sdp_1(self):
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='SUPER_SCS')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='SUPER_SCS')

    def test_super_scs_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='SUPER_SCS')

    def test_warm_start(self):
        if cvx.SUPER_SCS in cvx.installed_solvers():
            x = cvx.Variable(10)
            obj = cvx.Minimize(cvx.sum(cvx.exp(x)))
            prob = cvx.Problem(obj, [cvx.sum(x) == 1])
            result = prob.solve(solver='SUPER_SCS', eps=1e-4)
            result2 = prob.solve(solver='SUPER_SCS', warm_start=True, eps=1e-4)
            self.assertAlmostEqual(result2, result, places=2)


@unittest.skipUnless('CVXOPT' in cvx.installed_solvers(), 'CVXOPT is not installed.')
class TestCVXOPT(BaseTest):

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_cvxopt_options(self):
        """Test that all the CVXOPT solver options work.
        """
        # TODO race condition when changing these values.
        # 'maxiters'
        # maximum number of iterations (default: 100).
        # 'abstol'
        # absolute accuracy (default: 1e-7).
        # 'reltol'
        # relative accuracy (default: 1e-6).
        # 'feastol'
        # tolerance for feasibility conditions (default: 1e-7).
        # 'refinement'
        # number of iterative refinement steps when solving KKT equations
        # (default: 0 if the problem has no second-order cone
        #  or matrix inequality constraints; 1 otherwise).
        if cvx.CVXOPT in cvx.installed_solvers():
            EPS = 1e-7
            prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
            for i in range(2):
                prob.solve(solver=cvx.CVXOPT, feastol=EPS, abstol=EPS, reltol=EPS,
                           max_iters=20, verbose=True, kktsolver="chol",
                           refinement=2, warm_start=True)
            self.assertAlmostEqual(prob.value, 1.0)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_cvxopt_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CVXOPT')

    def test_cvxopt_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CVXOPT')

    def test_cvxopt_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CVXOPT')

    def test_cvxopt_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='CVXOPT')

    def test_cvxopt_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='CVXOPT')

    def test_cvxopt_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='CVXOPT')

    def test_cvxopt_sdp_1(self):
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='CVXOPT')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='CVXOPT')


@unittest.skipUnless('CBC' in cvx.installed_solvers(), 'CBC is not installed.')
class TestCBC(BaseTest):

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_options(self):
        """Test that all the cvx.CBC solver options work.
        """
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)),
                           [self.x == cvx.Variable(2, boolean=True)])
        if cvx.CBC in cvx.installed_solvers():
            for i in range(2):
                # Some cut-generators seem to be buggy for now -> set to false
                # prob.solve(solver=cvx.CBC, verbose=True, GomoryCuts=True, MIRCuts=True,
                #            MIRCuts2=True, TwoMIRCuts=True, ResidualCapacityCuts=True,
                #            KnapsackCuts=True, FlowCoverCuts=True, CliqueCuts=True,
                #            LiftProjectCuts=True, AllDifferentCuts=False, OddHoleCuts=True,
                #            RedSplitCuts=False, LandPCuts=False, PreProcessCuts=False,
                #            ProbingCuts=True, SimpleRoundingCuts=True)
                prob.solve(solver=cvx.CBC, verbose=True, maximumSeconds=100)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=cvx.CBC)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CBC)

    def test_cbc_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CBC')

    def test_cbc_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CBC')

    def test_cbc_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CBC')

    def test_cbc_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='CBC')

    def test_cbc_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='CBC')

    def test_cbc_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='CBC')


@unittest.skipUnless('CPLEX' in cvx.installed_solvers(), 'CPLEX is not installed.')
class TestCPLEX(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_cplex_warm_start(self):
        """Make sure that warm starting CPLEX behaves as expected
           Note: This only checks output, not whether or not CPLEX is warm starting internally
        """
        if cvx.CPLEX in cvx.installed_solvers():

            A = cvx.Parameter((2, 2))
            b = cvx.Parameter(2)
            h = cvx.Parameter(2)
            c = cvx.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cvx.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A * self.x == b]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.CPLEX, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.CPLEX)

    def test_cplex_params(self):
        if cvx.CPLEX in cvx.installed_solvers():
            n, m = 10, 4
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cvx.Variable(n)
            objective = cvx.Minimize(cvx.norm1(z))
            constraints = [A * z == y]
            problem = cvx.Problem(objective, constraints)

            invalid_cplex_params = {
                "bogus": "foo"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.CPLEX,
                              cplex_params=invalid_cplex_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cvx.CPLEX, invalid_kwarg=None)

            cplex_params = {
                "advance": 0,  # int param
                "simplex.limits.iterations": 1000,  # long param
                "timelimit": 1000.0,  # double param
                "workdir": '"mydir"',  # string param
            }
            problem.solve(solver=cvx.CPLEX, cplex_params=cplex_params)
        pass

    def test_cplex_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CPLEX')

    def test_cplex_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CPLEX')

    def test_cplex_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CPLEX')

    def test_cplex_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='CPLEX')

    def test_cplex_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='CPLEX')

    def test_cplex_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='CPLEX')

    def test_cplex_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='CPLEX')

    def test_cplex_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='CPLEX')

    def test_cplex_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='CPLEX')

    def test_cplex_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='CPLEX')

    def test_cplex_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='CPLEX')


@unittest.skipUnless('GUROBI' in cvx.installed_solvers(), 'GUROBI is not installed.')
class TestGUROBI(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_gurobi_warm_start(self):
        """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
        if cvx.GUROBI in cvx.installed_solvers():
            import numpy as np

            A = cvx.Parameter((2, 2))
            b = cvx.Parameter(2)
            h = cvx.Parameter(2)
            c = cvx.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cvx.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A * self.x == b]
            prob = cvx.Problem(objective, constraints)
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cvx.GUROBI, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cvx.GUROBI)

    def test_gurobi_lp_0(self):
        StandardTestLPs.test_lp_0(solver='GUROBI')

    def test_gurobi_lp_1(self):
        StandardTestLPs.test_lp_1(solver='GUROBI')

    def test_gurobi_lp_2(self):
        StandardTestLPs.test_lp_2(solver='GUROBI')

    def test_gurobi_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='GUROBI')

    def test_gurobi_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='GUROBI')

    def test_gurobi_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='GUROBI')

    def test_gurobi_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='GUROBI')

    def test_gurobi_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='GUROBI')

    def test_gurobi_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='GUROBI')

    def test_gurobi_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='GUROBI')

    def test_gurobi_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='GUROBI')


@unittest.skipUnless('XPRESS' in cvx.installed_solvers(), 'EXPRESS is not installed.')
class TestXPRESS(unittest.TestCase):

    def test_xpress_lp_0(self):
        StandardTestLPs.test_lp_0(solver='XPRESS')

    def test_xpress_lp_1(self):
        StandardTestLPs.test_lp_1(solver='XPRESS')

    def test_xpress_lp_2(self):
        StandardTestLPs.test_lp_2(solver='XPRESS')

    def test_xpress_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='XPRESS')

    def test_xpress_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='XPRESS')

    def test_xpress_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='XPRESS')

    def test_xpress_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='XPRESS')

    def test_xpress_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='XPRESS')

    def test_xpress_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='XPRESS')

    def test_xpress_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='XPRESS')

    def test_xpress_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='XPRESS')


@unittest.skipUnless('NAG' in cvx.installed_solvers(), 'NAG is not installed.')
class TestNAG(unittest.TestCase):

    def test_nag_lp_0(self):
        StandardTestLPs.test_lp_0(solver='NAG')

    def test_nag_lp_1(self):
        StandardTestLPs.test_lp_1(solver='NAG')

    def test_nag_lp_2(self):
        StandardTestLPs.test_lp_2(solver='NAG')

    def test_nag_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='NAG')

    def test_nag_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='NAG')

    def test_nag_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='NAG')


class TestAllSolvers(BaseTest):

    def setUp(self):
        self.a = cvx.Variable(name='a')
        self.b = cvx.Variable(name='b')
        self.c = cvx.Variable(name='c')

        self.x = cvx.Variable(2, name='x')
        self.y = cvx.Variable(3, name='y')
        self.z = cvx.Variable(2, name='z')

        self.A = cvx.Variable((2, 2), name='A')
        self.B = cvx.Variable((2, 2), name='B')
        self.C = cvx.Variable((3, 2), name='C')

    def test_installed_solvers(self):
        """Test the list of installed solvers.
        """
        from cvxpy.reductions.solvers.defines import (SOLVER_MAP_CONIC, SOLVER_MAP_QP,
                                                      INSTALLED_SOLVERS)
        prob = cvx.Problem(cvx.Minimize(cvx.norm(self.x, 1) + 1.0), [self.x == 0])
        for solver in SOLVER_MAP_CONIC.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertAlmostEqual(prob.value, 1.0)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % solver)

        for solver in SOLVER_MAP_QP.keys():
            if solver in INSTALLED_SOLVERS:
                prob.solve(solver=solver)
                self.assertItemsAlmostEqual(self.x.value, [0, 0])
            else:
                with self.assertRaises(Exception) as cm:
                    prob.solve(solver=solver)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % solver)
