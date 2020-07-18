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
import unittest

import numpy as np
import scipy.linalg as la
from nose.tools import assert_raises

import cvxpy as cp
from cvxpy.error import SolverError
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS, INSTALLED_SOLVERS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
    StandardTestECPs,
    StandardTestLPs,
    StandardTestMixedCPs,
    StandardTestSDPs,
    StandardTestSOCPs,
)


class TestECOS(BaseTest):

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_ecos_options(self):
        """Test that all the ECOS solver options work.
        """
        # Test ecos
        # feastol, abstol, reltol, feastol_inacc,
        # abstol_inacc, and reltol_inacc for tolerance values
        # max_iters for the maximum number of iterations,
        EPS = 1e-4
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
        for i in range(2):
            prob.solve(solver=cp.ECOS, feastol=EPS, abstol=EPS, reltol=EPS,
                       feastol_inacc=EPS, abstol_inacc=EPS, reltol_inacc=EPS,
                       max_iters=20, verbose=True, warm_start=True)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

    def test_ecos_lp_0(self):
        StandardTestLPs.test_lp_0(solver='ECOS')

    def test_ecos_lp_1(self):
        StandardTestLPs.test_lp_1(solver='ECOS')

    def test_ecos_lp_2(self):
        StandardTestLPs.test_lp_2(solver='ECOS')

    def test_ecos_lp_3(self):
        StandardTestLPs.test_lp_3(solver='ECOS')

    def test_ecos_lp_4(self):
        StandardTestLPs.test_lp_4(solver='ECOS')

    def test_ecos_lp_5(self):
        StandardTestLPs.test_lp_5(solver='ECOS')

    def test_ecos_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='ECOS')

    def test_ecos_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='ECOS')

    def test_ecos_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='ECOS')

    def test_ecos_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='ECOS')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='ECOS')

    def test_ecos_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='ECOS')

    def test_ecos_exp_soc_1(self):
        StandardTestMixedCPs.test_exp_soc_1(solver='ECOS')


class TestSCS(BaseTest):

    """ Unit tests for SCS. """
    def setUp(self):
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

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
        x = cp.Variable(2, name='x')
        prob = cp.Problem(cp.Minimize(cp.norm(x, 1) + 1.0), [x == 0])
        for i in range(2):
            prob.solve(solver=cp.SCS, max_iters=50, eps=EPS, alpha=EPS,
                       verbose=True, normalize=True, use_indirect=False)
        self.assertAlmostEqual(prob.value, 1.0, places=2)
        self.assertItemsAlmostEqual(x.value, [0, 0], places=2)

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

    def test_complex_matrices(self):
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

    def test_warm_start_diffcp(self):
        """Test warm starting in diffcvx.
        """
        try:
            import diffcp
            diffcp  # for flake8
        except ImportError:
            self.skipTest("diffcp not installed.")
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.DIFFCP, eps=1e-4)
        result2 = prob.solve(solver=cp.DIFFCP, warm_start=True, eps=1e-4)
        self.assertAlmostEqual(result2, result, places=2)

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

    def test_scs_lp_3(self):
        StandardTestLPs.test_lp_3(solver='SCS')

    def test_scs_lp_4(self):
        StandardTestLPs.test_lp_4(solver='SCS')

    def test_scs_lp_5(self):
        StandardTestLPs.test_lp_5(solver='SCS')

    def test_scs_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='SCS')

    def test_scs_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='SCS')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='SCS')

    def test_scs_sdp_1min(self):
        StandardTestSDPs.test_sdp_1min(solver='SCS')

    def test_scs_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='SCS')

    def test_scs_exp_soc_1(self):
        StandardTestMixedCPs.test_exp_soc_1(solver='SCS')


@unittest.skipUnless('MOSEK' in INSTALLED_SOLVERS, 'MOSEK is not installed.')
class TestMosek(unittest.TestCase):

    def test_mosek_lp_0(self):
        StandardTestLPs.test_lp_0(solver='MOSEK')

    def test_mosek_lp_1(self):
        # default settings
        StandardTestLPs.test_lp_1(solver='MOSEK')
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='MOSEK', places=6, bfs=True)

    def test_mosek_lp_2(self):
        StandardTestLPs.test_lp_2(solver='MOSEK')

    def test_mosek_lp_3(self):
        StandardTestLPs.test_lp_3(solver='MOSEK')

    def test_mosek_lp_4(self):
        StandardTestLPs.test_lp_4(solver='MOSEK')

    def test_mosek_lp_5(self):
        StandardTestLPs.test_lp_5(solver='MOSEK')

    def test_mosek_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='MOSEK')

    def test_mosek_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='MOSEK')

    def test_mosek_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='MOSEK')

    def test_mosek_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='MOSEK')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='MOSEK')

    def test_mosek_sdp_1(self):
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='MOSEK')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='MOSEK')

    def test_mosek_expcone_1(self):
        StandardTestECPs.test_expcone_1(solver='MOSEK')

    def test_mosek_exp_soc_1(self):
        StandardTestMixedCPs.test_exp_soc_1(solver='MOSEK')

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
        if cp.MOSEK in INSTALLED_SOLVERS:
            import mosek
            n = 10
            m = 4
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cp.Variable(n)
            objective = cp.Minimize(cp.norm1(z))
            constraints = [A @ z == y]
            problem = cp.Problem(objective, constraints)

            invalid_mosek_params = {
                "dparam.basis_tol_x": "1e-8"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cp.MOSEK, mosek_params=invalid_mosek_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cp.MOSEK, invalid_kwarg=None)

            mosek_params = {
                mosek.dparam.basis_tol_x: 1e-8,
                "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20
            }
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)


@unittest.skipUnless('CVXOPT' in INSTALLED_SOLVERS, 'CVXOPT is not installed.')
class TestCVXOPT(BaseTest):

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_cvxopt_options(self):
        """Test that all the CVXOPT solver options work.
        """
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
        EPS = 1e-7
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
        prob.solve(solver=cp.CVXOPT, feastol=EPS, abstol=EPS, reltol=EPS,
                   max_iters=20, verbose=True, kktsolver='chol', refinement=2)
        self.assertAlmostEqual(prob.value, 1.0)
        self.assertItemsAlmostEqual(self.x.value, [0, 0])

        msg = 'This setup-factor function was called.'

        def setup_dummy_factor(c, G, h, dims, A, b):
            # see cvxpy/reductions/solvers/kktsolver.py for an actual implementation
            # of a setup-factor function.
            raise NotImplementedError(msg)

        with self.assertRaises(NotImplementedError) as nix:
            prob.solve(solver='CVXOPT', kktsolver=setup_dummy_factor)
        self.assertEqual(msg, str(nix.exception))

    def test_cvxopt_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CVXOPT')

    def test_cvxopt_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CVXOPT')

    def test_cvxopt_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CVXOPT')

    def test_cvxopt_lp_3(self):
        StandardTestLPs.test_lp_3(solver='CVXOPT')

    def test_cvxopt_lp_4(self):
        StandardTestLPs.test_lp_4(solver='CVXOPT')

    def test_cvxopt_lp_5(self):
        from cvxpy.reductions.solvers.kktsolver import setup_ldl_factor
        StandardTestLPs.test_lp_5(solver='CVXOPT', kktsolver=setup_ldl_factor)
        StandardTestLPs.test_lp_5(solver='CVXOPT', kktsolver='chol')

    def test_cvxopt_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='CVXOPT')

    def test_cvxopt_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='CVXOPT')

    def test_cvxopt_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='CVXOPT')

    def test_cvxopt_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='CVXOPT')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='CVXOPT')

    def test_cvxopt_sdp_1(self):
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='CVXOPT')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='CVXOPT')


@unittest.skipUnless('CBC' in INSTALLED_SOLVERS, 'CBC is not installed.')
class TestCBC(BaseTest):

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_options(self):
        """Test that all the cvx.CBC solver options work.
        """
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)),
                          [self.x == cp.Variable(2, boolean=True)])
        if cp.CBC in INSTALLED_SOLVERS:
            for i in range(2):
                # Some cut-generators seem to be buggy for now -> set to false
                # prob.solve(solver=cvx.CBC, verbose=True, GomoryCuts=True, MIRCuts=True,
                #            MIRCuts2=True, TwoMIRCuts=True, ResidualCapacityCuts=True,
                #            KnapsackCuts=True, FlowCoverCuts=True, CliqueCuts=True,
                #            LiftProjectCuts=True, AllDifferentCuts=False, OddHoleCuts=True,
                #            RedSplitCuts=False, LandPCuts=False, PreProcessCuts=False,
                #            ProbingCuts=True, SimpleRoundingCuts=True)
                prob.solve(solver=cp.CBC, verbose=True, maximumSeconds=100)
            self.assertItemsAlmostEqual(self.x.value, [0, 0])
        else:
            with self.assertRaises(Exception) as cm:
                prob.solve(solver=cp.CBC)
                self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.CBC)

    def test_cbc_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CBC', duals=False)

    def test_cbc_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CBC', duals=False)

    def test_cbc_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CBC', duals=False)

    def test_cbc_lp_3(self):
        StandardTestLPs.test_lp_3(solver='CBC')

    def test_cbc_lp_4(self):
        StandardTestLPs.test_lp_4(solver='CBC')

    def test_cbc_lp_5(self):
        StandardTestLPs.test_lp_5(solver='CBC')

    def test_cbc_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='CBC')

    def test_cbc_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='CBC')

    def test_cbc_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='CBC')


@unittest.skipUnless('GLPK' in INSTALLED_SOLVERS, 'GLPK is not installed.')
class TestGLPK(unittest.TestCase):

    def test_glpk_lp_0(self):
        StandardTestLPs.test_lp_0(solver='GLPK')

    def test_glpk_lp_1(self):
        StandardTestLPs.test_lp_1(solver='GLPK')

    def test_glpk_lp_2(self):
        StandardTestLPs.test_lp_2(solver='GLPK')

    def test_glpk_lp_3(self):
        StandardTestLPs.test_lp_3(solver='GLPK')

    def test_glpk_lp_4(self):
        StandardTestLPs.test_lp_4(solver='GLPK')

    def test_glpk_lk_5(self):
        StandardTestLPs.test_lp_5(solver='GLPK')

    def test_glpk_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='GLPK_MI')

    def test_glpk_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='GLPK_MI')

    def test_glpk_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='GLPK_MI')


@unittest.skipUnless('CPLEX' in INSTALLED_SOLVERS, 'CPLEX is not installed.')
class TestCPLEX(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_cplex_warm_start(self):
        """Make sure that warm starting CPLEX behaves as expected
           Note: This only checks output, not whether or not CPLEX is warm starting internally
        """
        if cp.CPLEX in INSTALLED_SOLVERS:

            A = cp.Parameter((2, 2))
            b = cp.Parameter(2)
            h = cp.Parameter(2)
            c = cp.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.CPLEX, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.CPLEX)

    def test_cplex_params(self):
        if cp.CPLEX in INSTALLED_SOLVERS:
            n, m = 10, 4
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cp.Variable(n)
            objective = cp.Minimize(cp.norm1(z))
            constraints = [A @ z == y]
            problem = cp.Problem(objective, constraints)

            invalid_cplex_params = {
                "bogus": "foo"
            }
            with self.assertRaises(ValueError):
                problem.solve(solver=cp.CPLEX,
                              cplex_params=invalid_cplex_params)

            with self.assertRaises(ValueError):
                problem.solve(solver=cp.CPLEX, invalid_kwarg=None)

            cplex_params = {
                "advance": 0,  # int param
                "simplex.limits.iterations": 1000,  # long param
                "timelimit": 1000.0,  # double param
                "workdir": '"mydir"',  # string param
            }
            problem.solve(solver=cp.CPLEX, cplex_params=cplex_params)

    def test_cplex_lp_0(self):
        StandardTestLPs.test_lp_0(solver='CPLEX')

    def test_cplex_lp_1(self):
        StandardTestLPs.test_lp_1(solver='CPLEX')

    def test_cplex_lp_2(self):
        StandardTestLPs.test_lp_2(solver='CPLEX')

    def test_cplex_lp_3(self):
        StandardTestLPs.test_lp_3(solver='CPLEX')

    def test_cplex_lp_4(self):
        StandardTestLPs.test_lp_4(solver='CPLEX')

    def test_cplex_lp_5(self):
        StandardTestLPs.test_lp_5(solver='CPLEX')

    def test_cplex_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='CPLEX')

    def test_cplex_socp_1(self):
        # Parameters are set due to a minor dual-variable related
        # presolve bug in CPLEX, which will be fixed in the next
        # CPLEX release.
        StandardTestSOCPs.test_socp_1(solver='CPLEX', places=2,
                                      cplex_params={
                                          "preprocessing.presolve": 0,
                                          "preprocessing.reduce": 2})

    def test_cplex_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='CPLEX')

    def test_cplex_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='CPLEX')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='CPLEX')

    def test_cplex_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='CPLEX')

    def test_cplex_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='CPLEX')

    def test_cplex_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='CPLEX')

    def test_cplex_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='CPLEX', places=3)

    def test_cplex_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='CPLEX')


@unittest.skipUnless('GUROBI' in INSTALLED_SOLVERS, 'GUROBI is not installed.')
class TestGUROBI(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_gurobi_warm_start(self):
        """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import numpy as np

            A = cp.Parameter((2, 2))
            b = cp.Parameter(2)
            h = cp.Parameter(2)
            c = cp.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.GUROBI)

    def test_gurobi_lp_0(self):
        StandardTestLPs.test_lp_0(solver='GUROBI')

    def test_gurobi_lp_1(self):
        StandardTestLPs.test_lp_1(solver='GUROBI')

    def test_gurobi_lp_2(self):
        StandardTestLPs.test_lp_2(solver='GUROBI')

    def test_gurobi_lp_3(self):
        StandardTestLPs.test_lp_3(solver='GUROBI')

    def test_gurobi_lp_4(self):
        StandardTestLPs.test_lp_4(solver='GUROBI')

    def test_gurobi_lp_5(self):
        StandardTestLPs.test_lp_5(solver='GUROBI')

    def test_gurobi_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='GUROBI')

    def test_gurobi_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='GUROBI')

    def test_gurobi_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='GUROBI')

    def test_gurobi_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='GUROBI')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='GUROBI')

    def test_gurobi_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver='GUROBI')

    def test_gurobi_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver='GUROBI')

    def test_gurobi_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver='GUROBI')

    def test_gurobi_mi_socp_1(self):
        StandardTestSOCPs.test_mi_socp_1(solver='GUROBI', places=2)

    def test_gurobi_mi_socp_2(self):
        StandardTestSOCPs.test_mi_socp_2(solver='GUROBI')


@unittest.skipUnless('XPRESS' in INSTALLED_SOLVERS, 'XPRESS is not installed.')
class TestXPRESS(BaseTest):

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_xpress_warm_start(self):
        """Make sure that warm starting Xpress behaves as expected
           Note: Xpress does not have warmstart yet, it will re-solve problem from scratch
        """
        if cp.XPRESS in INSTALLED_SOLVERS:
            import numpy as np

            A = cp.Parameter((2, 2))
            b = cp.Parameter(2)
            h = cp.Parameter(2)
            c = cp.Parameter(2)

            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            objective = cp.Maximize(c[0] * self.x[0] + c[1] * self.x[1])
            constraints = [self.x[0] <= h[0],
                           self.x[1] <= h[1],
                           A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 2)
            self.assertItemsAlmostEqual(self.x.value, [1, 1])

            # Change c from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([2, 2])
            c.value = np.array([2, 1])              # <----- Changed

            # Without setting update_objective = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.XPRESS, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.XPRESS)

    def test_xpress_params(self):
        if cp.XPRESS in INSTALLED_SOLVERS:
            n, m = 10, 4
            A = np.random.randn(m, n)
            x = np.random.randn(n)
            y = A.dot(x)

            # Solve a simple basis pursuit problem for testing purposes.
            z = cp.Variable(n)
            objective = cp.Minimize(cp.norm1(z))
            constraints = [A @ z == y]
            problem = cp.Problem(objective, constraints)

            params = {
                "lpiterlimit": 1000,  # maximum number of simplex iterations
                "maxtime": 1000.0     # time limit
            }
            problem.solve(solver=cp.XPRESS, solver_opts=params)

    def test_xpress_lp_0(self):
        StandardTestLPs.test_lp_0(solver='XPRESS')

    def test_xpress_lp_1(self):
        StandardTestLPs.test_lp_1(solver='XPRESS')

    def test_xpress_lp_2(self):
        StandardTestLPs.test_lp_2(solver='XPRESS')

    def test_xpress_lp_3(self):
        StandardTestLPs.test_lp_3(solver='XPRESS')

    def test_xpress_lp_4(self):
        StandardTestLPs.test_lp_4(solver='XPRESS')

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


@unittest.skipUnless('NAG' in INSTALLED_SOLVERS, 'NAG is not installed.')
class TestNAG(unittest.TestCase):

    def test_nag_lp_0(self):
        StandardTestLPs.test_lp_0(solver='NAG')

    def test_nag_lp_1(self):
        StandardTestLPs.test_lp_1(solver='NAG')

    def test_nag_lp_2(self):
        StandardTestLPs.test_lp_2(solver='NAG')

    def test_nag_lp_3(self):
        StandardTestLPs.test_lp_3(solver='NAG')

    def test_nag_lp_4(self):
        StandardTestLPs.test_lp_4(solver='NAG')

    def test_nag_lp_5(self):
        StandardTestLPs.test_lp_5(solver='NAG')

    def test_nag_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver='NAG')

    def test_nag_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver='NAG')

    def test_nag_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver='NAG')

    def test_nag_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='NAG')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='NAG')


@unittest.skipUnless("SCIP" in INSTALLED_SOLVERS, "SCIP is not installed.")
class TestSCIP(unittest.TestCase):

    def test_scip_lp_0(self):
        StandardTestLPs.test_lp_0(solver="SCIP")

    def test_scip_lp_1(self):
        StandardTestLPs.test_lp_1(solver="SCIP")

    def test_scip_lp_2(self):
        StandardTestLPs.test_lp_2(solver="SCIP", duals=False)

    def test_scip_lp_3(self):
        StandardTestLPs.test_lp_3(solver="SCIP")

    def test_scip_lp_4(self):
        StandardTestLPs.test_lp_4(solver="SCIP")

    def test_scip_socp_0(self):
        StandardTestSOCPs.test_socp_0(solver="SCIP")

    def test_scip_socp_1(self):
        StandardTestSOCPs.test_socp_1(solver="SCIP", places=3, duals=False)

    def test_scip_socp_2(self):
        StandardTestSOCPs.test_socp_2(solver="SCIP", places=2)

    def test_scip_socp_3(self):
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver="SCIP")
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver="SCIP")

    def test_scip_mi_lp_0(self):
        StandardTestLPs.test_mi_lp_0(solver="SCIP")

    def test_scip_mi_lp_1(self):
        StandardTestLPs.test_mi_lp_1(solver="SCIP")

    def test_scip_mi_lp_2(self):
        StandardTestLPs.test_mi_lp_2(solver="SCIP")

    def get_simple_problem(self):
        """Example problem that can be used within additional tests."""
        x = cp.Variable()
        y = cp.Variable()
        constraints = [
            x >= 0,  # x must be positive
            y >= 1,  # y must be greater or equal to 1
            x + y <= 4
        ]
        obj = cp.Maximize(x)
        prob = cp.Problem(obj, constraints)
        return prob

    def test_scip_test_params__no_params_set(self):
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP")
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__valid_params(self):
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP", gp=False)
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__valid_scip_params(self):
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP", scip_params={"lp/fastmip": 1, "limits/gap": 0.1})
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__invalid_params(self):
        prob = self.get_simple_problem()
        # Since an invalid NON-scip param is passed, an error is expected
        # to be raised when calling solve.
        with assert_raises(KeyError) as ke:
            prob.solve(solver="SCIP", a="what?")
            exc = "One or more solver params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc

    def test_scip_test_params__invalid_scip_params(self):
        prob = self.get_simple_problem()
        # Since an invalid SCIP param is passed, an error is expected
        # to be raised when calling solve.
        with assert_raises(KeyError) as ke:
            prob.solve(solver="SCIP", scip_params={"a": "what?"})
            exc = "One or more scip params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc


class TestAllSolvers(BaseTest):

    def setUp(self):
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_installed_solvers(self):
        """Test the list of installed solvers.
        """
        from cvxpy.reductions.solvers.defines import (SOLVER_MAP_CONIC, SOLVER_MAP_QP,
                                                      INSTALLED_SOLVERS)
        prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1) + 1.0), [self.x == 0])
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

    def test_mixed_integer_behavior(self):
        x = cp.Variable(2, name='x', integer=True)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [x >= 0])
        if len(INSTALLED_MI_SOLVERS) == 0:
            try:
                prob.solve()
                assert False
            except SolverError as err:
                msg = str(err)
                self.assertTrue("(a `mixed-integer solver`)" in msg)
        else:
            prob.solve()
            self.assertItemsAlmostEqual(x.value, [0, 0])
