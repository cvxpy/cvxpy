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
import pytest
import scipy.linalg as la
import scipy.stats as st

import cvxpy as cp
import cvxpy.tests.solver_test_helpers as sths
from cvxpy.reductions.solvers.defines import (
    INSTALLED_MI_SOLVERS,
    INSTALLED_SOLVERS,
)
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import (
    StandardTestECPs,
    StandardTestLPs,
    StandardTestMixedCPs,
    StandardTestPCPs,
    StandardTestQPs,
    StandardTestSDPs,
    StandardTestSOCPs,
)
from cvxpy.utilities.versioning import Version


class TestECOS(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_ecos_options(self) -> None:
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

    def test_ecos_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='ECOS')

    def test_ecos_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='ECOS')

    def test_ecos_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='ECOS')

    def test_ecos_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='ECOS')

    def test_ecos_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='ECOS')

    def test_ecos_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='ECOS')

    def test_ecos_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='ECOS')

    def test_ecos_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='ECOS')

    def test_ecos_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='ECOS')

    def test_ecos_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='ECOS')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='ECOS')

    def test_ecos_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='ECOS')

    def test_ecos_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='ECOS')


class TestSCS(BaseTest):

    """ Unit tests for SCS. """
    def setUp(self) -> None:
        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(2, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    # Overridden method to assume lower accuracy.
    def assertItemsAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestSCS, self).assertItemsAlmostEqual(a, b, places=places)

    # Overridden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places: int = 2) -> None:
        super(TestSCS, self).assertAlmostEqual(a, b, places=places)

    def test_scs_retry(self) -> None:
        """Test that SCS retry doesn't trigger a crash.
        """
        n_sec = 20
        np.random.seed(315)
        mu = np.random.random(n_sec)
        random_mat = np.random.rand(n_sec, n_sec)
        C = np.dot(random_mat, random_mat.transpose())

        x = cp.Variable(n_sec)
        prob = cp.Problem(cp.Minimize(cp.QuadForm(x, C)),
                          [cp.sum(x) == 1,
                           0 <= x,
                           x <= 1,
                           x @ mu >= np.max(mu) - 1e-6])
        prob.solve(cp.SCS)
        assert prob.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    def test_scs_options(self) -> None:
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
            prob.solve(solver=cp.SCS, max_iters=50, eps=EPS, alpha=1.2,
                       verbose=True, normalize=True, use_indirect=False)
        self.assertAlmostEqual(prob.value, 1.0, places=2)
        self.assertItemsAlmostEqual(x.value, [0, 0], places=2)

    def test_log_problem(self) -> None:
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

    def test_sigma_max(self) -> None:
        """Test sigma_max.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6]])
        constr = [self.C == const]
        prob = cp.Problem(cp.Minimize(cp.norm(self.C, 2)), constr)
        result = prob.solve(solver=cp.SCS)
        self.assertAlmostEqual(result, cp.norm(const, 2).value)
        self.assertItemsAlmostEqual(self.C.value, const.value)

    def test_sdp_var(self) -> None:
        """Test sdp var.
        """
        const = cp.Constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        X = cp.Variable((3, 3), PSD=True)
        prob = cp.Problem(cp.Minimize(0), [X == const])
        prob.solve(solver=cp.SCS)
        self.assertEqual(prob.status, cp.INFEASIBLE)

    def test_complex_matrices(self) -> None:
        """Test complex matrices.
        """
        # Complex-valued matrix
        np.random.seed(0)
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

    def test_entr(self) -> None:
        """Test a problem with entr.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.entr(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_exp(self) -> None:
        """Test a problem with exp.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Minimize(cp.sum(cp.exp(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_log(self) -> None:
        """Test a problem with log.
        """
        for n in [5, 10, 25]:
            print(n)
            x = cp.Variable(n)
            obj = cp.Maximize(cp.sum(cp.log(x)))
            p = cp.Problem(obj, [cp.sum(x) == 1])
            p.solve(solver=cp.SCS)
            self.assertItemsAlmostEqual(x.value, n*[1./n])

    def test_solve_problem_twice(self) -> None:
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

    def test_warm_start(self) -> None:
        """Test warm starting.
        """
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.SCS)
        time = prob.solver_stats.solve_time
        result2 = prob.solve(solver=cp.SCS, warm_start=True)
        time2 = prob.solver_stats.solve_time
        self.assertAlmostEqual(result2, result, places=2)
        print(time > time2)

    def test_warm_start_diffcp(self) -> None:
        """Test warm starting in diffcvx.
        """
        try:
            import diffcp
            diffcp  # for flake8
        except ModuleNotFoundError:
            self.skipTest("diffcp not installed.")
        x = cp.Variable(10)
        obj = cp.Minimize(cp.sum(cp.exp(x)))
        prob = cp.Problem(obj, [cp.sum(x) == 1])
        result = prob.solve(solver=cp.DIFFCP)
        result2 = prob.solve(solver=cp.DIFFCP, warm_start=True)
        self.assertAlmostEqual(result2, result, places=2)

    def test_psd_constraint(self) -> None:
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

    def test_quad_obj(self) -> None:
        """Test SCS canonicalization with a quadratic objective.
        """
        # Only relevant for SCS >= 3.0.0.
        import scs
        if Version(scs.__version__) >= Version('3.0.0'):
            x = cp.Variable(2)
            expr = cp.sum_squares(x)
            constr = [x >= 1]
            prob = cp.Problem(cp.Minimize(expr), constr)
            data = prob.get_problem_data(solver=cp.SCS)
            self.assertItemsAlmostEqual(data[0]["P"].A, 2*np.eye(2))
            solution1 = prob.solve(solver=cp.SCS)

            # When use_quad_obj = False, the quadratic objective is
            # canonicalized to a SOC constraint.
            prob = cp.Problem(cp.Minimize(expr), constr)
            solver_opts = {"use_quad_obj": False}
            data = prob.get_problem_data(solver=cp.SCS, solver_opts=solver_opts)
            assert "P" not in data[0]
            solution2 = prob.solve(solver=cp.SCS, **solver_opts)

            assert np.isclose(solution1, solution2)

            # Check that there is no P for non-quadratic objectives.
            expr = cp.norm(x, 1)
            prob = cp.Problem(cp.Minimize(expr), constr)
            data = prob.get_problem_data(solver=cp.SCS)
            assert "P" not in data[0]

    def test_quad_obj_with_power(self) -> None:
        """Test a mixed quadratic/power objective.
        """
        # Only relevant for SCS >= 3.0.0.
        import scs
        if Version(scs.__version__) >= Version('3.0.0'):
            # See https://github.com/cvxpy/cvxpy/issues/2059
            x = cp.Variable()
            prob = cp.Problem(cp.Minimize(x**1.6 + x**2), [x >= 1])
            prob.solve(solver=cp.SCS, use_quad_obj=True)
            self.assertAlmostEqual(prob.value, 2)
            self.assertAlmostEqual(x.value, 1)

            # Check problem data.
            data = prob.get_problem_data(solver=cp.SCS, solver_opts={"use_quad_obj": True})
            # Quadratic objective and SOC contraints.
            assert "P" in data[0]
            assert data[0]["dims"].soc

    def test_scs_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SCS')

    def test_scs_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SCS')

    def test_scs_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='SCS', eps=1e-6)

    def test_scs_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='SCS', eps=1e-6)

    def test_scs_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='SCS')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='SCS')

    def test_scs_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='SCS')

    def test_scs_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='SCS', eps=1e-5)

    def test_scs_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='SCS', eps=1e-5)

    def test_scs_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='SCS', eps=1e-5)

    def test_scs_sdp_pcp_1(self):
        StandardTestMixedCPs.test_sdp_pcp_1(solver='SCS')
        
    def test_scs_pcp_1(self) -> None:
        StandardTestPCPs.test_pcp_1(solver='SCS')

    def test_scs_pcp_2(self) -> None:
        StandardTestPCPs.test_pcp_2(solver='SCS')

    def test_scs_pcp_3(self) -> None:
        StandardTestPCPs.test_pcp_3(solver='SCS', eps=1e-12)


@unittest.skipUnless('CLARABEL' in INSTALLED_SOLVERS, 'CLARABEL is not installed.')
class TestClarabel(BaseTest):

    """ Unit tests for Clarabel. """
    def setUp(self) -> None:

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_clarabel_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver=cp.CLARABEL)

    def test_clarabel_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CLARABEL')

    def test_clarabel_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CLARABEL')

    def test_clarabel_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='CLARABEL')

    def test_clarabel_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CLARABEL')

    def test_clarabel_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CLARABEL')

    def test_clarabel_qp_0(self) -> None:
        StandardTestQPs.test_qp_0(solver='CLARABEL')

    def test_clarabel_qp_0_linear_obj(self) -> None:
        StandardTestQPs.test_qp_0(solver='CLARABEL', use_quad_obj=False)

    def test_clarabel_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CLARABEL')

    def test_clarabel_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CLARABEL')

    def test_clarabel_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CLARABEL')

    def test_clarabel_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='CLARABEL')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='CLARABEL')

    def test_clarabel_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='CLARABEL')

    def test_clarabel_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='CLARABEL')

    def test_clarabel_pcp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CLARABEL')

    def test_clarabel_pcp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CLARABEL')

    def test_clarabel_pcp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CLARABEL')

    def test_clarabel_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='CLARABEL')

    def test_clarabel_sdp_2(self) -> None:
        # produces a different optimizer than 
        # the one expected by the standard test
        places = 3
        sth = sths.sdp_2()
        sth.solve('CLARABEL')
        sth.verify_objective(places)
        sth.check_primal_feasibility(places)
        # sth.verify_primal_values(places) # skip
        sth.check_complementarity(places)
        sth.check_dual_domains(places)


@unittest.skipUnless('MOSEK' in INSTALLED_SOLVERS, 'MOSEK is not installed.')
class TestMosek(unittest.TestCase):

    def test_mosek_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='MOSEK')

    def test_mosek_lp_1(self) -> None:
        # default settings
        StandardTestLPs.test_lp_1(solver='MOSEK')
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='MOSEK', places=6, bfs=True)

    def test_mosek_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='MOSEK')

    def test_mosek_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='MOSEK')

    def test_mosek_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='MOSEK')

    def test_mosek_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='MOSEK')

    def test_mosek_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='MOSEK')

    def test_mosek_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='MOSEK')

    def test_mosek_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='MOSEK')

    def test_mosek_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='MOSEK')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='MOSEK')

    def test_mosek_sdp_1(self) -> None:
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='MOSEK')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='MOSEK')

    def test_mosek_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='MOSEK')

    def test_mosek_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='MOSEK')

    def test_mosek_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='MOSEK')

    def test_mosek_pcp_1(self) -> None:
        StandardTestPCPs.test_pcp_1(solver='MOSEK', places=2)

    def test_mosek_pcp_2(self) -> None:
        StandardTestPCPs.test_pcp_2(solver='MOSEK')

    def test_mosek_pcp_3(self) -> None:
        StandardTestPCPs.test_pcp_3(solver='MOSEK')

    def test_mosek_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='MOSEK')

    def test_mosek_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='MOSEK')

    def test_mosek_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='MOSEK')

    def test_mosek_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='MOSEK')

    def test_mosek_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='MOSEK')

    def test_mosek_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='MOSEK', places=3)

    def test_mosek_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='MOSEK')

    def test_mosek_mi_pcp_0(self) -> None:
        StandardTestPCPs.test_mi_pcp_0(solver='MOSEK')

    def test_mosek_params(self) -> None:
        import mosek
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)

        # Solve a simple basis pursuit problem for testing purposes.
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)

        invalid_mosek_params = {
            "MSK_IPAR_NUM_THREADS": "11.3"
        }
        with self.assertRaises(mosek.Error):
            problem.solve(solver=cp.MOSEK, mosek_params=invalid_mosek_params)

        with self.assertRaises(ValueError):
            problem.solve(solver=cp.MOSEK, invalid_kwarg=None)

        mosek_params = {
            mosek.dparam.basis_tol_x: 1e-8,
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 20,
            "MSK_IPAR_NUM_THREADS": "17",
            "MSK_IPAR_PRESOLVE_USE": "MSK_PRESOLVE_MODE_OFF",
            "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-9,
            "MSK_DPAR_INTPNT_CO_TOL_PFEAS": "1e-9"
        }
        with pytest.warns():
            problem.solve(solver=cp.MOSEK, mosek_params=mosek_params)

    def test_mosek_simplex(self) -> None:
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)

        # Solve a simple basis pursuit problem for testing purposes.
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)
        problem.solve(
            solver=cp.MOSEK,
            mosek_params={"MSK_IPAR_OPTIMIZER": "MSK_OPTIMIZER_DUAL_SIMPLEX"}
        )

    def test_mosek_iis(self) -> None:
        """Test IIS feature in Mosek."""
        # Simple infeasible problem.
        n = 2
        x = cp.Variable(n)
        objective = cp.Minimize(cp.sum(x))
        constraints = [x[0] >= 1, x[0] <= -1, x[1] >= 3]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        iis = problem.solver_stats.extra_stats["IIS"]
        assert iis[constraints[0].id] > 0
        assert iis[constraints[1].id] > 0
        assert iis[constraints[2].id] == 0

        # More complicated infeasible problem.
        n = 3
        m = 2
        X = cp.Variable((m, n))
        y = cp.Variable()
        objective = cp.Minimize(cp.sum(X))
        constraints = [y == 2, X >= 3, X[0, 0] + y <= -5]
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        iis = problem.solver_stats.extra_stats["IIS"]
        assert abs(iis[constraints[0].id]) > 0
        dual1 = np.reshape(iis[constraints[1].id], X.shape, order="C")
        assert dual1[0, 0] > 0
        assert dual1[0, 1] == 0
        assert np.all(dual1[1, :] == 0)
        assert iis[constraints[2].id] > 0

    def test_mosek_sdp_power(self) -> None:
        """Test the problem in issue #2128"""
        StandardTestMixedCPs.test_sdp_pcp_1(solver='MOSEK')
        

    def test_power_portfolio(self) -> None:
        """Test the portfolio problem in issue #2042"""
        T, N = 200, 10

        rs = np.random.RandomState(123)
        mean = np.zeros(N) + 1/1000
        cov = rs.rand(N, N) * 1.5 - 0.5
        cov = cov @ cov.T/1000 + np.diag(rs.rand(N) * 0.7 + 0.3)/1000

        Y = st.multivariate_normal.rvs(
            mean=mean,
            cov=cov,
            size=T,
            random_state=rs
        )

        w = cp.Variable((N, 1))
        t = cp.Variable((1, 1))
        z = cp.Variable((1, 1))
        omega = cp.Variable((T, 1))
        psi = cp.Variable((T, 1))
        nu = cp.Variable((T, 1))
        epsilon = cp.Variable((T, 1))
        k = cp.Variable((1, 1))
        b = np.ones((1, N))/N

        X = Y @ w

        h = 0.2
        ones = np.ones((T, 1))
        constraints = [
            cp.constraints.power.PowCone3D(z * (1+h)/(2*h) * ones, psi * (1+h)/h, epsilon, 1/(1+h)),
            cp.constraints.power.PowCone3D(omega/(1-h), nu/h, -z/(2*h) * ones, (1-h)),
            -X - t + epsilon + omega <= 0,
            w >= 0,
            z >= 0,
            ]

        obj = t + z + cp.sum(psi + nu)

        constraints += [cp.sum(w) == k,
                        k >= 0,
                        b @ cp.log(w) >= 1,
                        ]
        objective = cp.Minimize(obj)
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.MOSEK)
        assert prob.status is cp.OPTIMAL

    def test_mosek_accept_unknown(self) -> None:
        mosek_param = {
            "MSK_IPAR_INTPNT_MAX_ITERATIONS": 0
        }
        sth = sths.lp_5()
        sth.solve(solver=cp.MOSEK, accept_unknown=True, mosek_params=mosek_param)
        assert sth.prob.status in {cp.OPTIMAL_INACCURATE, cp.OPTIMAL}

        with pytest.raises(cp.error.SolverError, match="Solver 'MOSEK' failed"):
            sth.solve(solver=cp.MOSEK, mosek_params=mosek_param)

    def test_mosek_number_iters(self) -> None:
        sth = sths.lp_5()
        sth.solve(solver=cp.MOSEK)
        assert sth.prob.solver_stats.num_iters >= 0
        assert sth.prob.solver_stats.extra_stats["mio_intpnt_iter"] == 0
        assert sth.prob.solver_stats.extra_stats["mio_simplex_iter"] == 0

    def test_eps_keyword(self) -> None:
        """Test that the eps keyword is accepted"""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        # This should not raise an exception
        prob.solve(solver=cp.MOSEK, eps=1e-8, mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6})
        assert prob.status is cp.OPTIMAL

        # This exception being raised shows that the eps value is being passed to MOSEK
        import mosek
        with pytest.raises(mosek.Error, match="The parameter value 0.1 is too large"):
            prob.solve(solver=cp.MOSEK,
                       eps=1e-1,
                       mosek_params={'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-6})


        # If parameters are defined explicitly, eps will not overwrite -> no exception
        from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK
        all_params = MOSEK.tolerance_params()
        prob.solve(solver=cp.MOSEK, eps=1e-1, mosek_params={p: 1e-6 for p in all_params})
        assert prob.status is cp.OPTIMAL

        # Fails when used with enums
        with pytest.raises(AssertionError, match="not compatible"):
            prob.solve(solver=cp.MOSEK,
                       eps=1e-1,
                       mosek_params={mosek.dparam.intpnt_co_tol_dfeas: 1e-6})


@unittest.skipUnless('CVXOPT' in INSTALLED_SOLVERS, 'CVXOPT is not installed.')
class TestCVXOPT(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_cvxopt_options(self) -> None:
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

    def test_cvxopt_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='CVXOPT')

    def test_cvxopt_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CVXOPT')

    def test_cvxopt_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CVXOPT')

    def test_cvxopt_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='CVXOPT')

    def test_cvxopt_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CVXOPT')

    def test_cvxopt_lp_5(self) -> None:
        from cvxpy.reductions.solvers.kktsolver import setup_ldl_factor
        StandardTestLPs.test_lp_5(solver='CVXOPT', kktsolver=setup_ldl_factor)
        StandardTestLPs.test_lp_5(solver='CVXOPT', kktsolver='chol')

    def test_cvxopt_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CVXOPT')

    def test_cvxopt_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='CVXOPT')

    def test_cvxopt_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CVXOPT')

    def test_cvxopt_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='CVXOPT')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='CVXOPT')

    def test_cvxopt_sdp_1(self) -> None:
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='CVXOPT')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='CVXOPT')

    def test_cvxopt_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='CVXOPT')


@unittest.skipUnless('SDPA' in INSTALLED_SOLVERS, 'SDPA is not installed.')
class TestSDPA(BaseTest):

    def test_sdpa_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='SDPA')

    def test_sdpa_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='SDPA')

    def test_sdpa_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='SDPA')

    def test_sdpa_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SDPA')

    def test_sdpa_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SDPA')

    def test_sdpa_lp_5(self) -> None:
        # this also tests the ability to pass solver options
        StandardTestLPs.test_lp_5(solver='SDPA',
                                  betaBar=0.1, gammaStar=0.8, epsilonDash=8.0E-6)

    def test_sdpa_lp_7(self) -> None:
        StandardTestLPs.test_lp_7(solver='SDPA')

    def test_sdpa_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='SDPA')

    def test_sdpa_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='SDPA')

    def test_sdpa_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='SDPA')

    def test_sdpa_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='SDPA')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='SDPA')

    def test_sdpa_sdp_1(self) -> None:
        # minimization
        StandardTestSDPs.test_sdp_1min(solver='SDPA')
        # maximization
        StandardTestSDPs.test_sdp_1max(solver='SDPA')

    def test_sdpa_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='SDPA')


def fflush() -> None:
    """
    C code in some solvers uses libc buffering; if we want to capture log output from
    those solvers to use in tests, we must flush the libc buffers before trying to read
    the log contents from python.
    https://github.com/pytest-dev/pytest/issues/8753
    """
    import ctypes
    libc = ctypes.CDLL(None)
    libc.fflush(None)


# We can't inherit from unittest.TestCase since we access some advanced pytest features.
# As a result, we use the pytest skipif decorator instead of unittest.skipUnless.
@pytest.mark.skipif('CBC' not in INSTALLED_SOLVERS, reason='CBC is not installed.')
class TestCBC:

    def _cylp_checks_isProvenInfeasible():
        try:
            # https://github.com/coin-or/CyLP/pull/150
            from cylp.cy.CyCbcModel import problemStatus
            return problemStatus[0] == 'search completed'
        except ImportError:
            return False

    def test_cbc_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='CBC', duals=False)

    def test_cbc_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CBC', duals=False)

    def test_cbc_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CBC', duals=False)

    def test_cbc_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='CBC')

    def test_cbc_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CBC')

    def test_cbc_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CBC', duals=False)

    def test_cbc_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='CBC')

    def test_cbc_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='CBC')

    def test_cbc_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='CBC')

    def test_cbc_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='CBC')

    @pytest.mark.skipif(not _cylp_checks_isProvenInfeasible(),
                        reason='CyLP <= 0.91.4 has no working integer infeasibility detection')
    def test_cbc_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='CBC')

    @pytest.mark.parametrize(
        "opts",
        [
            pytest.param(opts, id=next(iter(opts.keys())))
            for opts in [
                {"dualTolerance": 1.0},
                {"primalTolerance": 1.0},
                {"maxNumIteration": 1},
                {"scaling": 0},
                # {"automaticScaling": True},  # Doesn't work
                # {"infeasibilityCost": 0.000001},  # Doesn't work
                {"optimizationDirection": "max"},
                {"presolve": "off"},
            ]
        ]
    )
    def test_cbc_lp_options(self, opts: dict, capfd: pytest.LogCaptureFixture) -> None:
        """
        Validate that cylp is actually using each option.

        Tentative approach: run model with verbose output with or without the specified
        option; verbose output should be different each way.
        """
        # start by making sure capture buffer is empty to ensure valid results
        fflush()
        capfd.readouterr()
        # run the solver with verbose logging without this option and capture output
        sth = sths.lp_4()
        sth.solve(solver='CBC', logLevel=2)
        fflush()
        base = capfd.readouterr()
        # run the solver with verbose logging *with* the option under test
        try:
            sth.solve(solver='CBC', logLevel=2, **opts)
        except Exception:
            # if setting the option caused the case to fail, that's a pass
            pass
        else:
            # if the case still passes, we at least look for change in the log outputs
            fflush()
            with_opt = capfd.readouterr()
            assert base != with_opt

    def test_cbc_lp_logging(self, capfd: pytest.LogCaptureFixture) -> None:
        """Validate that logLevel parameter is passed to solver"""
        # start by making sure capture buffer is empty to ensure valid results
        fflush()
        capfd.readouterr()

        # for linear problems
        StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=0)
        fflush()
        quiet_output = capfd.readouterr()
        StandardTestLPs.test_lp_0(solver='CBC', duals=False, logLevel=5)
        fflush()
        verbose_output = capfd.readouterr()
        assert len(verbose_output.out) > len(quiet_output.out)

        # for mixed integer problems
        StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=0)
        fflush()
        quiet_output = capfd.readouterr()
        StandardTestLPs.test_mi_lp_0(solver='CBC', logLevel=5)
        fflush()
        verbose_output = capfd.readouterr()
        assert len(verbose_output.out) > len(quiet_output.out)


@unittest.skipUnless('GLPK' in INSTALLED_SOLVERS, 'GLPK is not installed.')
class TestGLPK(unittest.TestCase):

    def test_glpk_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GLPK')

    def test_glpk_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GLPK')

    def test_glpk_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GLPK')

    def test_glpk_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='GLPK')

    def test_glpk_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GLPK')

    def test_glpk_lk_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GLPK')

    def test_glpk_lp_6(self) -> None:
        StandardTestLPs.test_lp_6(solver='GLPK')

    def test_glpk_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='GLPK_MI')

    def test_glpk_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='GLPK_MI')

    def test_glpk_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='GLPK_MI')

    def test_glpk_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='GLPK_MI')

    def test_glpk_mi_lp_4(self) -> None:
        StandardTestLPs.test_mi_lp_4(solver='GLPK_MI')

    def test_glpk_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='GLPK_MI')

    def test_glpk_options(self) -> None:
        sth = sths.lp_1()
        import cvxopt
        assert "tm_lim" not in cvxopt.glpk.options
        sth.solve(solver='GLPK', tm_lim=100)
        assert "tm_lim" not in cvxopt.glpk.options
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)

    def test_glpk_mi_options(self) -> None:
        sth = sths.mi_lp_1()
        import cvxopt
        assert "tm_lim" not in cvxopt.glpk.options
        sth.solve(solver='GLPK_MI', tm_lim=100, verbose=True)
        assert "tm_lim" not in cvxopt.glpk.options
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)


@unittest.skipUnless('GLOP' in INSTALLED_SOLVERS, 'GLOP is not installed.')
class TestGLOP(unittest.TestCase):

    def test_glop_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GLOP')

    def test_glop_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GLOP')

    def test_glop_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GLOP')

    def test_glop_lp_3_no_preprocessing(self) -> None:
        from ortools.glop import parameters_pb2
        params = parameters_pb2.GlopParameters()
        params.use_preprocessing = False
        StandardTestLPs.test_lp_3(solver='GLOP', parameters_proto=params)

    # With preprocessing enabled, Glop internally detects
    # INFEASIBLE_OR_UNBOUNDED. This status is translated to
    # MPSOLVER_INFEASIBLE. See
    # https://github.com/google/or-tools/blob/b37d9c786b69128f3505f15beca09e89bf078a89/ortools/linear_solver/glop_utils.cc#L25-L38.
    @unittest.skip('Known limitation of the GLOP interface.')
    def test_glop_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='GLOP')

    def test_glop_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GLOP')

    def test_glop_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GLOP')

    def test_glop_lp_6_no_preprocessing(self) -> None:
        from ortools.glop import parameters_pb2
        params = parameters_pb2.GlopParameters()
        params.use_preprocessing = False
        StandardTestLPs.test_lp_6(solver='GLOP', parameters_proto=params)

    # Same issue as with test_glop_lp_3.
    @unittest.skip('Known limitation of the GLOP interface.')
    def test_glop_lp_6(self) -> None:
        StandardTestLPs.test_lp_6(solver='GLOP')

    def test_glop_bad_parameters(self) -> None:
        x = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(x), [x <= 1])
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver='GLOP', parameters_proto="not a proto")

    def test_glop_time_limit(self) -> None:
        sth = sths.lp_1()
        # Checks that the option doesn't error. A better test would be to solve
        # a large instance and check that the time limit is hit.
        sth.solve(solver='GLOP', time_limit_sec=1.0)


@unittest.skipUnless('PDLP' in INSTALLED_SOLVERS, 'PDLP is not installed.')
class TestPDLP(unittest.TestCase):

    def test_pdlp_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='PDLP')

    def test_pdlp_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='PDLP')

    def test_pdlp_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='PDLP')

    def test_pdlp_lp_3(self) -> None:
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    # We get the precise status when presolve is disabled.
    def test_pdlp_lp_3_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_3(solver='PDLP', parameters_proto=params)

    def test_pdlp_lp_4(self) -> None:
        sth = sths.lp_4()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    def test_pdlp_lp_4_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_4(solver='PDLP', parameters_proto=params)

    def test_pdlp_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='PDLP')

    def test_pdlp_lp_6(self) -> None:
        sth = sths.lp_6()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='PDLP')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)

    def test_pdlp_lp_6_no_presolve(self) -> None:
        from ortools.pdlp import solvers_pb2
        params = solvers_pb2.PrimalDualHybridGradientParams()
        params.presolve_options.use_glop = False
        StandardTestLPs.test_lp_6(solver='PDLP', parameters_proto=params)

    def test_pdlp_bad_parameters(self) -> None:
        x = cp.Variable(1)
        prob = cp.Problem(cp.Maximize(x), [x <= 1])
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver='PDLP', parameters_proto="not a proto")

    def test_pdlp_time_limit(self) -> None:
        sth = sths.lp_1()
        # Checks that the option doesn't error. A better test would be to solve
        # a large instance and check that the time limit is hit.
        sth.solve(solver='PDLP', time_limit_sec=1.0)


@unittest.skipUnless('CPLEX' in INSTALLED_SOLVERS, 'CPLEX is not installed.')
class TestCPLEX(BaseTest):
    """ Unit tests for solver specific behavior. """

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_cplex_warm_start(self) -> None:
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

    def test_cplex_params(self) -> None:
        if cp.CPLEX in INSTALLED_SOLVERS:
            n, m = 10, 4
            np.random.seed(0)
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

    def test_cplex_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='CPLEX')

    def test_cplex_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='CPLEX')

    def test_cplex_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='CPLEX')

    def test_cplex_lp_3(self) -> None:
        # CPLEX initially produces an INFEASIBLE_OR_UNBOUNDED status
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='CPLEX')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)
        # Determine the precise status with reoptimize=True.
        StandardTestLPs.test_lp_3(solver='CPLEX', reoptimize=True)

    def test_cplex_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='CPLEX')

    def test_cplex_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='CPLEX')

    def test_cplex_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='CPLEX')

    def test_cplex_socp_1(self) -> None:
        # Parameters are set due to a minor dual-variable related
        # presolve bug in CPLEX, which will be fixed in the next
        # CPLEX release.
        StandardTestSOCPs.test_socp_1(solver='CPLEX', places=2,
                                      cplex_params={
                                          "preprocessing.presolve": 0,
                                          "preprocessing.reduce": 2})

    def test_cplex_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='CPLEX')

    def test_cplex_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='CPLEX')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='CPLEX')

    def test_cplex_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='CPLEX')

    def test_cplex_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='CPLEX')

    def test_cplex_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='CPLEX')

    def test_cplex_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='CPLEX')

    def test_cplex_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='CPLEX')

    def test_cplex_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='CPLEX', places=3)

    def test_cplex_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='CPLEX')


@unittest.skipUnless('GUROBI' in INSTALLED_SOLVERS, 'GUROBI is not installed.')
class TestGUROBI(BaseTest):
    """NOTE: solves of LPs (or MILPs) get routed through GUROBI's QP interface!
    So many of these tests are testing the behavior of qurobi_qpif.py"""

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_gurobi_warm_start(self) -> None:
        """Make sure that warm starting Gurobi behaves as expected
           Note: This only checks output, not whether or not Gurobi is warm starting internally
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
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
            constraints = [self.x[0]**2 <= h[0]**2,
                           self.x[1] <= h[1],
                           A @ self.x == b]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])

            # Change A and b from the original values
            A.value = np.array([[0, 0], [0, 1]])   # <----- Changed
            b.value = np.array([0, 1])              # <----- Changed
            h.value = np.array([2, 2])
            c.value = np.array([1, 1])

            # Without setting update_eq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 3)
            self.assertItemsAlmostEqual(self.x.value, [2, 1])

            # Change h from the original values
            A.value = np.array([[1, 0], [0, 0]])
            b.value = np.array([1, 0])
            h.value = np.array([1, 1])              # <----- Changed
            c.value = np.array([1, 1])

            # Without setting update_ineq_constrs = False,
            # the results should change to the correct answer
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertAlmostEqual(result, 2)
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

            # Try creating a new problem and setting x.value.
            init_value = np.array([2, 3])
            self.x.value = init_value
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, 4)
            self.assertItemsAlmostEqual(self.x.value, [1, 2])
            # Check that "start" value was set appropriately.
            model = prob.solver_stats.extra_stats
            model_x = model.getVars()
            for i in range(self.x.size):
                assert init_value[i] == model_x[i].start
                assert np.isclose(self.x.value[i], model_x[i].x)

            # Test with matrix variable.
            z = cp.Variable()
            Y = cp.Variable((3, 2))
            Y_val = np.reshape(np.arange(6), (3, 2))
            Y.value = Y_val + 1
            objective = cp.Maximize(z + cp.sum(Y))
            constraints = [Y <= Y_val,
                           z <= 2]
            prob = cp.Problem(objective, constraints)
            result = prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(result, Y_val.sum() + 2)
            self.assertAlmostEqual(z.value, 2)
            self.assertItemsAlmostEqual(Y.value, Y_val)
            # Check that "start" value was set appropriately.
            model = prob.solver_stats.extra_stats
            model_x = model.getVars()
            assert gurobipy.GRB.UNDEFINED == model_x[0].start
            assert np.isclose(2, model_x[0].x)
            for i in range(1, Y.size + 1):
                row = (i - 1) % Y.shape[0]
                col = (i - 1) // Y.shape[0]
                assert Y_val[row, col] + 1 == model_x[i].start
                assert np.isclose(Y.value[row, col], model_x[i].x)

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, warm_start=True)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.GUROBI)

    def test_gurobi_time_limit_no_solution(self) -> None:
        """Make sure that if Gurobi terminates due to a time limit before finding a solution:
            1) no error is raised,
            2) solver stats are returned.
            The test is skipped if something changes on Gurobi's side so that:
            - a solution is found despite a time limit of zero,
            - a different termination criteria is hit first.
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy
            objective = cp.Minimize(self.x[0])
            constraints = [cp.square(self.x[0]) <= 1]
            prob = cp.Problem(objective, constraints)
            try:
                prob.solve(solver=cp.GUROBI, TimeLimit=0.0)
            except Exception as e:
                self.fail("An exception %s is raised instead of returning a result." % e)

            extra_stats = None
            solver_stats = getattr(prob, "solver_stats", None)
            if solver_stats:
                extra_stats = getattr(solver_stats, "extra_stats", None)
            self.assertTrue(extra_stats, "Solver stats have not been returned.")

            nb_solutions = getattr(extra_stats, "SolCount", None)
            if nb_solutions:
                self.skipTest("Gurobi has found a solution, the test is not relevant anymore.")

            solver_status = getattr(extra_stats, "Status", None)
            if solver_status != gurobipy.StatusConstClass.TIME_LIMIT:
                self.skipTest("Gurobi terminated for a different reason than reaching time limit, "
                              "the test is not relevant anymore.")

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.GUROBI)

    def test_gurobi_environment(self) -> None:
        """Tests that Gurobi environments can be passed to Model.
        Gurobi environments can include licensing and model parameter data.
        """
        if cp.GUROBI in INSTALLED_SOLVERS:
            import gurobipy

            # Set a few parameters to random values close to their defaults
            params = {
                'MIPGap': np.random.random(),  # range {0, INFINITY}
                'AggFill': np.random.randint(10),  # range {-1, MAXINT}
                'PerturbValue': np.random.random(),  # range: {0, INFINITY}
            }

            # Create a custom environment and set some parameters
            custom_env = gurobipy.Env()
            for k, v in params.items():
                custom_env.setParam(k, v)

            # Testing Conic Solver Interface
            sth = StandardTestSOCPs.test_socp_0(solver='GUROBI', env=custom_env)
            model = sth.prob.solver_stats.extra_stats
            for k, v in params.items():
                # https://www.gurobi.com/documentation/9.1/refman/py_model_getparaminfo.html
                name, p_type, p_val, p_min, p_max, p_def = model.getParamInfo(k)
                self.assertEqual(v, p_val)

        else:
            with self.assertRaises(Exception) as cm:
                prob = cp.Problem(cp.Minimize(cp.norm(self.x, 1)), [self.x == 0])
                prob.solve(solver=cp.GUROBI, TimeLimit=0)
            self.assertEqual(str(cm.exception), "The solver %s is not installed." % cp.GUROBI)

    def test_gurobi_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='GUROBI')

    def test_gurobi_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='GUROBI')

    def test_gurobi_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='GUROBI')

    def test_gurobi_lp_3(self) -> None:
        # GUROBI initially produces an INFEASIBLE_OR_UNBOUNDED status
        sth = sths.lp_3()
        with self.assertWarns(Warning):
            sth.prob.solve(solver='GUROBI')
            self.assertEqual(sth.prob.status, cp.settings.INFEASIBLE_OR_UNBOUNDED)
        # The user disables presolve and so makes reoptimization unnecessary
        StandardTestLPs.test_lp_3(solver='GUROBI', InfUnbdInfo=1)
        # The user determines the precise status with reoptimize=True
        StandardTestLPs.test_lp_3(solver='GUROBI', reoptimize=True)

    def test_gurobi_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='GUROBI', reoptimize=True)

    def test_gurobi_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='GUROBI')

    def test_gurobi_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='GUROBI')

    def test_gurobi_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='GUROBI')

    def test_gurobi_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='GUROBI')

    def test_gurobi_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='GUROBI')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='GUROBI')

    def test_gurobi_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='GUROBI')

    def test_gurobi_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='GUROBI')

    def test_gurobi_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='GUROBI')

    def test_gurobi_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='GUROBI')

    def test_gurobi_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='GUROBI')

    def test_gurobi_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='GUROBI', places=2)

    def test_gurobi_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='GUROBI')


@unittest.skipUnless('XPRESS' in INSTALLED_SOLVERS, 'XPRESS is not installed.')
class TestXPRESS(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_xpress_warm_start(self) -> None:
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

    def test_xpress_params(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            n, m = 10, 4
            np.random.seed(0)
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
                "maxtime": 1000       # time limit
            }
            problem.solve(solver=cp.XPRESS, **params)

    def test_xpress_iis_none(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            A = np.array([[2, 1], [1, 2], [-3, -3]])
            b = np.array([2, 2, -5])

            x = cp.Variable(2)
            objective = cp.Minimize(cp.norm2(x))
            constraint = [A @ x <= b]
            problem = cp.Problem(objective, constraint)

            params = {'save_iis': 0}

            problem.solve(solver=cp.XPRESS, **params)

    def test_xpress_iis_full(self) -> None:
        if cp.XPRESS in INSTALLED_SOLVERS:
            A = np.array([[2, 1], [1, 2], [-3, -3]])
            b = np.array([2, 2, -5])

            x = cp.Variable(2)
            objective = cp.Minimize(cp.norm2(x))
            constraint = [A @ x <= b]
            problem = cp.Problem(objective, constraint)

            params = {'save_iis': -1}

            problem.solve(solver=cp.XPRESS, **params)

            assert 'XPRESS_IIS' in problem.solution.attr

    def test_xpress_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='XPRESS')

    def test_xpress_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='XPRESS')

    def test_xpress_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='XPRESS')

    def test_xpress_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='XPRESS')

    def test_xpress_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='XPRESS')

    def test_xpress_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='XPRESS')

    def test_xpress_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='XPRESS')

    def test_xpress_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='XPRESS')

    def test_xpress_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='XPRESS')

    def test_xpress_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='XPRESS')

    def test_xpress_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='XPRESS')

    def test_xpress_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='XPRESS')

    def test_xpress_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='XPRESS')

    def test_xpress_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='XPRESS')

    def test_xpress_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver='XPRESS')


@unittest.skipUnless('NAG' in INSTALLED_SOLVERS, 'NAG is not installed.')
class TestNAG(unittest.TestCase):

    def test_nag_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='NAG')

    def test_nag_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='NAG')

    def test_nag_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='NAG')

    def test_nag_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='NAG')

    def test_nag_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='NAG')

    def test_nag_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='NAG')

    def test_nag_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='NAG')

    def test_nag_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='NAG')

    def test_nag_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='NAG')

    def test_nag_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='NAG')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='NAG')


@unittest.skipUnless("SCIP" in INSTALLED_SOLVERS, "SCIP is not installed.")
class TestSCIP(unittest.TestCase):

    def test_scip_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver="SCIP")

    def test_scip_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver="SCIP")

    def test_scip_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver="SCIP", duals=False)

    def test_scip_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver="SCIP")

    def test_scip_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver="SCIP")

    def test_scip_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver="SCIP")

    def test_scip_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver="SCIP", places=2, duals=False)

    def test_scip_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver="SCIP", places=2, duals=False)

    def test_scip_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver="SCIP", duals=False)
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver="SCIP", duals=False)

    def test_scip_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver="SCIP")

    def test_scip_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver="SCIP")

    def test_scip_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver="SCIP")

    def test_scip_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver="SCIP")

    def test_scip_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver="SCIP")

    def test_scip_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver="SCIP", places=3)

    def test_scip_mi_socp_2(self) -> None:
        StandardTestSOCPs.test_mi_socp_2(solver="SCIP")

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

    def test_scip_test_params__no_params_set(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP")
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__valid_params(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP", gp=False)
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__valid_scip_params(self) -> None:
        prob = self.get_simple_problem()
        prob.solve(solver="SCIP", scip_params={"lp/fastmip": 1, "limits/gap": 0.1})
        # Important that passes without raising an error also check obj.
        assert prob.value == 3

    def test_scip_test_params__invalid_params(self) -> None:
        prob = self.get_simple_problem()
        # Since an invalid NON-scip param is passed, an error is expected
        # to be raised when calling solve.
        with pytest.raises(KeyError) as ke:
            prob.solve(solver="SCIP", a="what?")
            exc = "One or more solver params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc

    def test_scip_test_params__invalid_scip_params(self) -> None:
        prob = self.get_simple_problem()
        # Since an invalid SCIP param is passed, an error is expected
        # to be raised when calling solve.
        with pytest.raises(KeyError) as ke:
            prob.solve(solver="SCIP", scip_params={"a": "what?"})
            exc = "One or more scip params in ['a'] are not valid: 'Not a valid parameter name'"
            assert ke.exception == exc

    def test_scip_time_limit_reached(self) -> None:
        sth = sths.mi_lp_7()

        # TODO doesn't work on windows.
        # run without enough time to find optimum
        # sth.solve(solver="SCIP", scip_params={"limits/time": 0.01})
        # assert sth.prob.status == cp.OPTIMAL_INACCURATE
        # assert all([v.value is not None for v in sth.prob.variables()])

        # run without enough time to do anything
        with pytest.raises(cp.error.SolverError) as se:
            sth.solve(solver="SCIP", scip_params={"limits/time": 0.0})
            exc = "Solver 'SCIP' failed. " \
                  "Try another solver, or solve with verbose=True for more information."
            assert str(se.value) == exc


class TestAllSolvers(BaseTest):

    def setUp(self) -> None:
        self.a = cp.Variable(name='a')
        self.b = cp.Variable(name='b')
        self.c = cp.Variable(name='c')

        self.x = cp.Variable(2, name='x')
        self.y = cp.Variable(3, name='y')
        self.z = cp.Variable(2, name='z')

        self.A = cp.Variable((2, 2), name='A')
        self.B = cp.Variable((2, 2), name='B')
        self.C = cp.Variable((3, 2), name='C')

    def test_installed_solvers(self) -> None:
        """Test the list of installed solvers.
        """
        from cvxpy.reductions.solvers.defines import (
            INSTALLED_SOLVERS,
            SOLVER_MAP_CONIC,
            SOLVER_MAP_QP,
        )
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

    def test_mixed_integer_behavior(self) -> None:
        x = cp.Variable(2, name='x', integer=True)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [x >= 0])
        if INSTALLED_MI_SOLVERS == [cp.ECOS_BB]:
            with pytest.raises(cp.error.SolverError, match="You need a mixed-integer "
                                                           "solver for this model"):
                prob.solve()
        else:
            prob.solve()
            self.assertItemsAlmostEqual(x.value, [0, 0])


class TestECOS_BB(unittest.TestCase):

    def test_ecos_bb_explicit_only(self) -> None:
        """Test that ECOS_BB isn't chosen by default.
        """
        x = cp.Variable(1, name='x', integer=True)
        objective = cp.Minimize(cp.sum(x))
        prob = cp.Problem(objective, [x >= 0])
        if INSTALLED_MI_SOLVERS != [cp.ECOS_BB]:
            prob.solve()
            assert prob.solver_stats.solver_name != cp.ECOS_BB
        else:
            with pytest.raises(cp.error.SolverError, match="You need a mixed-integer "
                                                           "solver for this model"):
                prob.solve()

    def test_ecos_bb_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='ECOS_BB')

    def test_ecos_bb_lp_1(self) -> None:
        # default settings
        StandardTestLPs.test_lp_1(solver='ECOS_BB')
        # require a basic feasible solution
        StandardTestLPs.test_lp_1(solver='ECOS_BB')

    def test_ecos_bb_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='ECOS_BB')

    def test_ecos_bb_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='ECOS_BB')

    def test_ecos_bb_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='ECOS_BB')

    def test_ecos_bb_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='ECOS_BB')

    def test_ecos_bb_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='ECOS_BB')

    def test_ecos_bb_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='ECOS_BB')

    def test_ecos_bb_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='ECOS_BB')

    def test_ecos_bb_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='ECOS_BB')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='ECOS_BB')

    def test_ecos_bb_expcone_1(self) -> None:
        StandardTestECPs.test_expcone_1(solver='ECOS_BB')

    def test_ecos_bb_exp_soc_1(self) -> None:
        StandardTestMixedCPs.test_exp_soc_1(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='ECOS_BB')

    @pytest.mark.skip(reason="Known bug in ECOS BB")
    def test_ecos_bb_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='ECOS_BB')

    def test_ecos_bb_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='ECOS_BB')

    @pytest.mark.skip(reason="Known bug in ECOS BB")
    def test_ecos_bb_mi_socp_1(self) -> None:
        StandardTestSOCPs.test_mi_socp_1(solver='ECOS_BB')


class TestSCIPY(unittest.TestCase):

    def setUp(self):
        import scipy
        self.d = Version(scipy.__version__) >= Version('1.7.0')

    def test_scipy_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='SCIPY', duals=self.d)

    def test_scipy_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='SCIPY', duals=self.d)

    def test_scipy_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='SCIPY', duals=self.d)

    def test_scipy_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='SCIPY')

    def test_scipy_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='SCIPY')

    def test_scipy_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='SCIPY', duals=self.d)

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_4(self) -> None:
        StandardTestLPs.test_mi_lp_4(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='SCIPY')

    @unittest.skipUnless('SCIPY' in INSTALLED_MI_SOLVERS, 'SCIPY version cannot solve MILPs')
    def test_scipy_mi_time_limit_reached(self) -> None:
        sth = sths.mi_lp_7()

        # We only check that the option does not raise an error.
        sth.solve(solver='SCIPY', scipy_options={"time_limit": 0.1})


@unittest.skipUnless('COPT' in INSTALLED_SOLVERS, 'COPT is not installed.')
class TestCOPT(unittest.TestCase):

    def test_copt_lp_0(self) -> None:
        StandardTestLPs.test_lp_0(solver='COPT')

    def test_copt_lp_1(self) -> None:
        StandardTestLPs.test_lp_1(solver='COPT')

    def test_copt_lp_2(self) -> None:
        StandardTestLPs.test_lp_2(solver='COPT')

    def test_copt_lp_3(self) -> None:
        StandardTestLPs.test_lp_3(solver='COPT')

    def test_copt_lp_4(self) -> None:
        StandardTestLPs.test_lp_4(solver='COPT')

    def test_copt_lp_5(self) -> None:
        StandardTestLPs.test_lp_5(solver='COPT')

    def test_copt_socp_0(self) -> None:
        StandardTestSOCPs.test_socp_0(solver='COPT')

    def test_copt_socp_1(self) -> None:
        StandardTestSOCPs.test_socp_1(solver='COPT', places=3)

    def test_copt_socp_2(self) -> None:
        StandardTestSOCPs.test_socp_2(solver='COPT')

    def test_copt_socp_3(self) -> None:
        # axis 0
        StandardTestSOCPs.test_socp_3ax0(solver='COPT')
        # axis 1
        StandardTestSOCPs.test_socp_3ax1(solver='COPT')

    def test_copt_mi_lp_0(self) -> None:
        StandardTestLPs.test_mi_lp_0(solver='COPT')

    def test_copt_mi_lp_1(self) -> None:
        StandardTestLPs.test_mi_lp_1(solver='COPT')

    def test_copt_mi_lp_2(self) -> None:
        StandardTestLPs.test_mi_lp_2(solver='COPT')

    def test_copt_mi_lp_3(self) -> None:
        StandardTestLPs.test_mi_lp_3(solver='COPT')

    def test_copt_mi_lp_5(self) -> None:
        StandardTestLPs.test_mi_lp_5(solver='COPT')

    def test_copt_mi_socp_1(self) -> None:
        # COPT does not support MISOCP.
        with pytest.raises(cp.error.SolverError, match="do not support"):
            StandardTestSOCPs.test_mi_socp_1(solver='COPT')

    def test_copt_sdp_1min(self) -> None:
        StandardTestSDPs.test_sdp_1min(solver='COPT')

    def test_copt_sdp_1max(self) -> None:
        StandardTestSDPs.test_sdp_1max(solver='COPT')

    def test_copt_sdp_2(self) -> None:
        StandardTestSDPs.test_sdp_2(solver='COPT')

    def test_copt_params(self) -> None:
        n = 10
        m = 4
        np.random.seed(0)
        A = np.random.randn(m, n)
        x = np.random.randn(n)
        y = A.dot(x)

        # Solve a simple basis pursuit problem for testing purposes.
        z = cp.Variable(n)
        objective = cp.Minimize(cp.norm1(z))
        constraints = [A @ z == y]
        problem = cp.Problem(objective, constraints)

        with self.assertRaises(AttributeError):
            problem.solve(solver=cp.COPT, invalid_kwarg=None)

        # Valid arg.
        problem.solve(solver=cp.COPT, feastol=1e-9)
