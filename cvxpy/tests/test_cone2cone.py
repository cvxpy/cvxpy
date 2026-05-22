"""
Copyright 2020, the CVXPY developers.

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
import unittest

import numpy as np
import scipy as sp

import cvxpy as cp
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowCone3DApprox, PowConeND
from cvxpy.constraints.psd import PSD, SvecPSD
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.cone2cone.approx import (
    APPROX_CONE_CONVERSIONS,
    ApproxCone2Cone,
)
from cvxpy.reductions.cone2cone.exact import (
    EXACT_CONE_CONVERSIONS,
    ExactCone2Cone,
)
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import (
    CLARABEL as ClarabelSolver,
)
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class TestPowND(BaseTest):

    @staticmethod
    def pcp_3(axis):
        """
        A modification of pcp_2. Reformulate

            max  (x**0.2)*(y**0.8) + z**0.4 - x
            s.t. x + y + z/2 == 2
                 x, y, z >= 0
        Into

            max  x3 + x4 - x0
            s.t. x0 + x1 + x2 / 2 == 2,

                 W := [[x0, x2],
                      [x1, 1.0]]
                 z := [x3, x4]
                 alpha := [[0.2, 0.4],
                          [0.8, 0.6]]
                 (W, z) in PowND(alpha, axis=0)
        """
        x = cp.Variable(shape=(3,))
        expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
        hypos = cp.Variable(shape=(2,))
        expect_hypos = None
        objective = cp.Maximize(cp.sum(hypos) - x[0])
        W = cp.bmat([[x[0], x[2]],
                     [x[1], 1.0]])
        alpha = np.array([[0.2, 0.4],
                          [0.8, 0.6]])
        if axis == 1:
            W = W.T
            alpha = alpha.T
        con_pairs = [
            (x[0] + x[1] + 0.5 * x[2] == 2, None),
            (cp.constraints.PowConeND(W, hypos, alpha, axis=axis), None)
        ]
        obj_pair = (objective, 1.8073406786220672)
        var_pairs = [
            (x, expect_x),
            (hypos, expect_hypos)
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_pcp_3a(self):
        sth = TestPowND.pcp_3(axis=0)
        sth.solve(solver='SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_complementarity(places=3)
        pass

    def test_pcp_3b(self):
        sth = TestPowND.pcp_3(axis=1)
        sth.solve(solver='SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_complementarity(places=3)
        pass

    @staticmethod
    def pcp_4(ceei: bool = True):
        """
        A power cone formulation of a Fisher market equilibrium pricing model.
        ceei = Competitive Equilibrium from Equal Incomes
        """
        # Generate test data
        np.random.seed(0)
        n_buyer = 4
        n_items = 6
        V = np.random.rand(n_buyer, n_items)
        X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
        u = cp.sum(cp.multiply(V, X), axis=1)
        if ceei:
            b = np.ones(n_buyer) / n_buyer
        else:
            b = np.array([0.3, 0.15, 0.2, 0.35])
        log_objective = cp.Maximize(cp.sum(cp.multiply(b, cp.log(u))))
        log_cons = [cp.sum(X, axis=0) <= 1]
        log_prob = cp.Problem(log_objective, log_cons)
        log_prob.solve(solver='SCS', eps=1e-8)
        expect_X = X.value

        z = cp.Variable()
        pow_objective = (cp.Maximize(z), np.exp(log_prob.value))
        pow_cons = [(cp.sum(X, axis=0) <= 1, None),
                    (PowConeND(W=u, z=z, alpha=b), None)]
        pow_vars = [(X, expect_X)]
        sth = STH.SolverTestHelper(pow_objective, pow_vars, pow_cons)
        return sth

    def test_pcp_4a(self):
        sth = TestPowND.pcp_4(ceei=True)
        sth.solve(solver='SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_complementarity(places=3)
        pass

    def test_pcp_4b(self):
        sth = TestPowND.pcp_4(ceei=False)
        sth.solve(solver='SCS', eps=1e-8)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.check_complementarity(places=3)
        pass


class TestRelEntrQuad(BaseTest):

    def expcone_1(self) -> STH.SolverTestHelper:
        """
        min   3 * x[0] + 2 * x[1] + x[2]
        s.t.  0.1 <= x[0] + x[1] + x[2] <= 1
              x >= 0
              and ...
                x[0] >= x[1] * exp(x[2] / x[1])
              equivalently ...
                x[0] / x[1] >= exp(x[2] / x[1])
                log(x[0] / x[1]) >= x[2] / x[1]
                x[1] log(x[1] / x[0]) <= -x[2]
        """
        x = cp.Variable(shape=(3, 1))
        cone_con = ExpCone(x[2], x[1], x[0]).as_quad_approx(5, 5)
        constraints = [cp.sum(x) <= 1.0,
                       cp.sum(x) >= 0.1,
                       x >= 0,
                       cone_con]
        obj = cp.Minimize(3 * x[0] + 2 * x[1] + x[2])
        obj_pair = (obj, 0.23534820622420757)
        expect_exp = [np.array([-1.35348213]), np.array([-0.35348211]), np.array([0.64651792])]
        con_pairs = [(constraints[0], 0),
                     (constraints[1], 2.3534821130067614),
                     (constraints[2], np.zeros(shape=(3, 1))),
                     (constraints[3], expect_exp)]
        expect_x = np.array([[0.05462721], [0.02609378], [0.01927901]])
        var_pairs = [(x, expect_x)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_expcone_1(self):
        sth = self.expcone_1()
        sth.solve(solver='CLARABEL')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)

    def expcone_socp_1(self) -> STH.SolverTestHelper:
        """
        A random risk-parity portfolio optimization problem.
        """
        sigma = np.array([[1.83, 1.79, 3.22],
                          [1.79, 2.18, 3.18],
                          [3.22, 3.18, 8.69]])
        L = np.linalg.cholesky(sigma)
        c = 0.75
        t = cp.Variable(name='t')
        x = cp.Variable(shape=(3,), name='x')
        s = cp.Variable(shape=(3,), name='s')
        e = cp.Constant(np.ones(3, ))
        objective = cp.Minimize(t - c * e @ s)
        con1 = cp.norm(L.T @ x, p=2) <= t
        con2 = ExpCone(s, e, x).as_quad_approx(5, 5)
        # SolverTestHelper data
        obj_pair = (objective, 4.0751197)
        var_pairs = [
            (x, np.array([0.57608346, 0.54315695, 0.28037716])),
            (s, np.array([-0.55150, -0.61036, -1.27161])),
        ]
        con_pairs = [
            (con1, 1.0),
            (con2, [None,
                    None,
                    None]
             )
        ]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_expcone_socp_1(self):
        sth = self.expcone_socp_1()
        sth.solve(solver=cp.SCS)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)


def sdp_ipm_installed():
    viable = {cp.CVXOPT, cp.MOSEK, cp.COPT}.intersection(cp.installed_solvers())
    return len(viable) > 0


@unittest.skipUnless(sdp_ipm_installed(),
                     'First-order solvers are too slow for the accuracy we need.')
class TestOpRelConeQuad(BaseTest):

    def setUp(self, n=3) -> None:
        self.n = n
        self.a = cp.Variable(shape=(n,), pos=True)
        self.b = cp.Variable(shape=(n,), pos=True)
        if hasattr(np.random, 'default_rng'):
            self.rng = np.random.default_rng(0)
        else:
            self.rng = np.random.RandomState(0)
        if hasattr(self.rng, 'random'):
            rand_gen_func = self.rng.random
        else:
            rand_gen_func = self.rng.random_sample

        self.a_lower = np.cumsum(rand_gen_func(n))
        self.a_upper = self.a_lower + 0.05*rand_gen_func(n)
        self.b_lower = np.cumsum(rand_gen_func(n))
        self.b_upper = self.b_lower + 0.05*rand_gen_func(n)
        self.base_cons = [
            self.a_lower <= self.a,
            self.a <= self.a_upper,
            self.b_lower <= self.b,
            self.b <= self.b_upper
        ]
        installed_solvers = cp.installed_solvers()
        if cp.MOSEK in installed_solvers:
            self.solver = cp.MOSEK
        elif cp.CVXOPT in installed_solvers:
            self.solver = cp.CVXOPT
        elif cp.COPT in installed_solvers:
            self.solver = cp.COPT
        else:
            raise RuntimeError('No viable solver installed.')
        pass

    @staticmethod
    def Dop_commute(a: np.ndarray, b: np.ndarray, U: np.ndarray):
        D = np.diag(a * np.log(a/b))
        if np.iscomplexobj(U):
            out = U @ D @ U.conj().T
        else:
            out = U @ D @ U.T
        return out

    @staticmethod
    def sum_rel_entr_approx(a: cp.Expression, b: cp.Expression,
                            apx_m: int, apx_k: int):
        n = a.size
        assert n == b.size
        epi_vec = cp.Variable(shape=n)
        con = cp.constraints.RelEntrConeQuad(a, b, epi_vec, apx_m, apx_k)
        objective = cp.Minimize(cp.sum(epi_vec))
        return objective, con

    def oprelcone_1(self, apx_m, apx_k, real) -> STH.SolverTestHelper:
        """
        These tests construct two matrices that commute (imposing all eigenvectors equal)
        and then use the fact that: T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        i.e. T >> Dop(A, B) for an objective that is an increasing function of the
        eigenvalues (which we here take to be the trace), we compute the reference
        objective value as tr(Dop) whose correctness can be seen by writing out
        tr(T)=tr(T-Dop)+tr(Dop), where tr(T-Dop)>=0 because of PSD-ness of (T-Dop),
        and at optimality we have (T-Dop)=0 (the zero matrix of corresponding size)
        For the case that the input matrices commute, Dop takes on a particularly
        simplified form, i.e.: U @ diag(a * log(a/b)) @ U^{-1} (which is implemented
        in the Dop_commute method above)
        """
        # Compute the expected optimal solution
        temp_obj, temp_con = TestOpRelConeQuad.sum_rel_entr_approx(
            self.a, self.b, apx_m, apx_k
        )
        temp_constraints = [con for con in self.base_cons]
        temp_constraints.append(temp_con)
        temp_prob = cp.Problem(temp_obj, temp_constraints)
        temp_prob.solve()
        expect_a = self.a.value
        expect_b = self.b.value
        expect_objective = temp_obj.value

        # Next: create a matrix representation of the same problem,
        # using operator relative entropy.
        n = self.n
        if real:
            randmat = self.rng.normal(size=(n, n))
            U = sp.linalg.qr(randmat)[0]
            A = cp.symmetric_wrap(U @ cp.diag(self.a) @ U.T)
            B = cp.symmetric_wrap(U @ cp.diag(self.b) @ U.T)
            T = cp.Variable(shape=(n, n), symmetric=True)
        else:
            randmat = 1j * self.rng.normal(size=(n, n))
            randmat += self.rng.normal(size=(n, n))
            U = sp.linalg.qr(randmat)[0]
            A = cp.hermitian_wrap(U @ cp.diag(self.a) @ U.conj().T)
            B = cp.hermitian_wrap(U @ cp.diag(self.b) @ U.conj().T)
            T = cp.Variable(shape=(n, n), hermitian=True)
        main_con = cp.constraints.OpRelEntrConeQuad(A, B, T, apx_m, apx_k)
        obj = cp.Minimize(trace(T))
        expect_T = TestOpRelConeQuad.Dop_commute(expect_a, expect_b, U)

        # Define the SolverTestHelper object
        con_pairs = [(con, None) for con in self.base_cons]
        con_pairs.append((main_con, None))
        obj_pair = (obj, expect_objective)
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_1_m1_k3_real(self):
        sth = self.oprelcone_1(1, 3, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m3_k1_real(self):
        sth = self.oprelcone_1(3, 1, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m4_k4_real(self):
        sth = self.oprelcone_1(4, 4, True)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m1_k3_complex(self):
        sth = self.oprelcone_1(1, 3, False)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def test_oprelcone_1_m3_k1_complex(self):
        sth = self.oprelcone_1(3, 1, False)
        sth.solve(self.solver)
        sth.verify_primal_values(places=3)
        sth.verify_objective(places=3)
        pass

    def oprelcone_2(self) -> STH.SolverTestHelper:
        """
        This test uses the same idea from the tests with commutative matrices,
        instead, here, we make the input matrices to Dop, non-commutative,
        the same condition as before i.e. T=Dop(A, B) for (A, B, T) in OpRelEntrConeQuad
        (for an objective that is an increasing function of the eigenvalues) holds,
        the difference here then, is in how we compute the reference values, which
        has been done by assuming correctness of the original CVXQUAD matlab implementation
        """
        n, m, k = 4, 3, 3
        # generate two sets of linearly orthogonal vectors
        # Each to be set as the eigenvectors of a particular input matrix to Dop
        U1 = np.array([[-0.05878522, -0.78378355, -0.49418311, -0.37149791],
                       [0.67696027, -0.25733435, 0.59263364, -0.35254672],
                       [0.43478177, 0.53648704, -0.54593428, -0.47444939],
                       [0.59096015, -0.17788771, -0.32638042, 0.71595942]])
        U2 = np.array([[-0.42499169, 0.6887562, 0.55846178, 0.18198188],
                       [-0.55478633, -0.7091174, 0.3884544, 0.19613213],
                       [-0.55591804, 0.14358541, -0.72444644, 0.38146522],
                       [0.4500548, -0.04637494, 0.11135968, 0.88481584]])
        a_diag = cp.Variable(shape=(n,), pos=True)
        b_diag = cp.Variable(shape=(n,), pos=True)
        A = U1 @ cp.diag(a_diag) @ U1.T
        B = U2 @ cp.diag(b_diag) @ U2.T
        T = cp.Variable(shape=(n, n), symmetric=True)
        a_lower = np.array([0.40683013, 1.34514597, 1.60057343, 2.13373667])
        a_upper = np.array([1.36158501, 1.61289351, 1.85065805, 3.06140939])
        b_lower = np.array([0.06858235, 0.36798274, 0.95956627, 1.16286541])
        b_upper = np.array([0.70446555, 1.16635299, 1.46126732, 1.81367755])
        con1 = cp.constraints.OpRelEntrConeQuad(A, B, T, m, k)
        con2 = a_lower <= a_diag
        con3 = a_diag <= a_upper
        con4 = b_lower <= b_diag
        con5 = b_diag <= b_upper
        con_pairs = [(con1, None),
                     (con2, None),
                     (con3, None),
                     (con4, None),
                     (con5, None)]
        obj = cp.Minimize(trace(T))

        expect_obj = 1.85476
        expect_T = np.array([[0.49316819, 0.20845265, 0.60474713, -0.5820242],
                             [0.20845265, 0.31084053, 0.2264112, -0.8442255],
                             [0.60474713, 0.2264112, 0.4687153, -0.85667283],
                             [-0.5820242, -0.8442255, -0.85667283, 0.58206723]])

        obj_pair = (obj, expect_obj)
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_2(self):
        sth = self.oprelcone_2()
        sth.solve(self.solver)
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)


class _PSDOnlyClarabel(ClarabelSolver):
    """Test-only Clarabel wrapper exposing only SvecPSD (no SOC)."""
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SvecPSD]

    def name(self):
        return "_PSD_ONLY_CLARABEL"


class _SOCOnlyClarabel(ClarabelSolver):
    """Test-only Clarabel wrapper exposing only SOC (no PowCone3D)."""
    SUPPORTED_CONSTRAINTS = ConicSolver.SUPPORTED_CONSTRAINTS + [SOC]

    def name(self):
        return "_SOC_ONLY_CLARABEL"


class TestExactApproxCone2Cone(BaseTest):
    """Tests for the ExactCone2Cone / ApproxCone2Cone framework."""

    def test_exact_cone_conversions_map(self) -> None:
        """EXACT_CONE_CONVERSIONS contains PowConeND and SOC."""
        self.assertIn(PowConeND, EXACT_CONE_CONVERSIONS)
        self.assertEqual(EXACT_CONE_CONVERSIONS[PowConeND], {PowCone3D})
        self.assertIn(SOC, EXACT_CONE_CONVERSIONS)
        self.assertEqual(EXACT_CONE_CONVERSIONS[SOC], {PSD})
        self.assertNotIn(PowCone3D, EXACT_CONE_CONVERSIONS)

    def test_approx_cone_conversions_map(self) -> None:
        """APPROX_CONE_CONVERSIONS keys on PowCone3DApprox, not PowCone3D."""
        self.assertIn(PowCone3DApprox, APPROX_CONE_CONVERSIONS)
        self.assertEqual(APPROX_CONE_CONVERSIONS[PowCone3DApprox], {SOC})
        self.assertNotIn(PowCone3D, APPROX_CONE_CONVERSIONS)

    def test_exact_cone_conversions_is_dag(self) -> None:
        """EXACT_CONE_CONVERSIONS must be a DAG (no cycles)."""
        # Compute transitive reachability for each source cone.
        # If any cone can reach itself, the graph has a cycle.
        reachable = {src: set(tgts) for src, tgts in EXACT_CONE_CONVERSIONS.items()}
        changed = True
        while changed:
            changed = False
            for src in reachable:
                new = set()
                for t in reachable[src]:
                    if t in reachable:
                        new |= reachable[t]
                if not new.issubset(reachable[src]):
                    reachable[src] |= new
                    changed = True
        for src, targets in reachable.items():
            self.assertNotIn(src, targets,
                             f"Cycle detected: {src.__name__} can reach itself")

    def test_exact_cone2cone_target_cones_filtering(self) -> None:
        """ExactCone2Cone(target_cones={PowConeND}) only converts PowConeND."""
        reduction = ExactCone2Cone(target_cones={PowConeND})
        self.assertIn(PowConeND, reduction.canon_methods)
        self.assertNotIn(SOC, reduction.canon_methods)

    def test_exact_cone2cone_target_cones_soc(self) -> None:
        """ExactCone2Cone(target_cones={SOC}) only converts SOC."""
        reduction = ExactCone2Cone(target_cones={SOC})
        self.assertIn(SOC, reduction.canon_methods)
        self.assertNotIn(PowConeND, reduction.canon_methods)

    def test_approx_cone2cone_target_cones_filtering(self) -> None:
        """ApproxCone2Cone(target_cones=...) filters correctly."""
        reduction = ApproxCone2Cone(target_cones={PowCone3DApprox})
        self.assertIn(PowCone3DApprox, reduction.canon_methods)

    def test_soc_to_psd_via_exact_cone2cone(self) -> None:
        """SOC constraint solved via PSD-only solver uses ExactCone2Cone."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.norm(x, 2) <= 1]
        )
        prob.solve(solver=_PSDOnlyClarabel())
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(prob.value, -np.sqrt(3), places=3)

    def test_soc_to_psd_dual_recovery(self) -> None:
        """Dual variables recovered correctly through SOC→PSD→SvecPSD chain."""
        x = cp.Variable(3)
        t = cp.Variable()
        soc_con = cp.SOC(t, x)
        prob = cp.Problem(cp.Minimize(t), [soc_con, cp.sum(x) == 1])
        prob.solve(solver=_PSDOnlyClarabel())
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        dv = soc_con.dual_value
        self.assertIsNotNone(dv)
        for d in dv:
            self.assertTrue(np.all(np.isfinite(d)))
        # Complementary slackness: <primal_slack, dual> ≈ 0.
        comp = cp.vdot(soc_con.args, soc_con.dual_value).value
        self.assertAlmostEqual(comp, 0, places=3)

    def test_soc_to_psd_packed(self) -> None:
        """Packed SOC constraints via PSD-only solver work correctly."""
        x = cp.Variable(3)
        t = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(t[0] + t[1]),
            [cp.SOC(t, cp.vstack([x[:2], x[1:]]).T, axis=0),
             cp.sum(x) == 1, x >= 0]
        )
        prob.solve(solver=_PSDOnlyClarabel())
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])

    def test_explicit_powcone3d_errors_when_unsupported(self) -> None:
        """Explicit PowCone3D errors when solver doesn't support it."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        prob = cp.Problem(
            cp.Maximize(z),
            [PowCone3D(x, y, z, 0.5), x + y <= 2]
        )
        with self.assertRaises(cp.error.SolverError):
            prob.solve(solver=_SOCOnlyClarabel())

    def test_powcone3d_approx_via_subclass(self) -> None:
        """PowCone3DApprox is approximated when solver lacks PowCone3D."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        prob = cp.Problem(
            cp.Maximize(z),
            [PowCone3DApprox(x, y, z, 0.5), x + y <= 2]
        )
        prob.solve(solver=_SOCOnlyClarabel())
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        self.assertAlmostEqual(z.value, 1.0, places=2)

    def test_approx_cone2cone_dual_recovery(self) -> None:
        """ApproxCone2Cone recovers dual variables for approximated constraints."""
        x = cp.Variable(pos=True)
        y = cp.Variable(pos=True)
        z = cp.Variable()
        pow_con = PowCone3DApprox(x, y, z, 0.5)
        prob = cp.Problem(cp.Maximize(z), [pow_con, x + y <= 2])

        reduction = ApproxCone2Cone(problem=prob, target_cones={PowCone3DApprox})
        reduced_prob, inverse_data = reduction.apply(prob)

        self.assertIn(pow_con.id, inverse_data.cons_id_map)
        canon_id = inverse_data.cons_id_map[pow_con.id]

        mock_dual_value = np.array([1.0])
        mock_solution = Solution(
            status=cp.OPTIMAL,
            opt_val=1.0,
            primal_vars={},
            dual_vars={canon_id: mock_dual_value},
            attr={}
        )

        inverted = reduction.invert(mock_solution, inverse_data)
        self.assertIn(pow_con.id, inverted.dual_vars)
        self.assertEqual(inverted.dual_vars[pow_con.id], mock_dual_value)


class TestPSDUtils(BaseTest):
    """Tests for tri_to_full and psd_format_mat round-trip correctness."""

    def _random_psd(self, n, rng):
        A = rng.standard_normal((n, n))
        return A @ A.T

    def test_tri_to_full_round_trip(self) -> None:
        """psd_format_mat -> tri_to_full recovers the symmetrised matrix."""
        from cvxpy.utilities.psd_utils import TriangleKind, psd_format_mat, tri_to_full
        rng = np.random.default_rng(42)
        n = 4
        X = cp.Variable((n, n), symmetric=True)
        con = PSD(X)
        M_val = self._random_psd(n, rng)
        X.value = M_val

        for tri_kind in TriangleKind:
            for sqrt2 in [True, False]:
                M = psd_format_mat(con, tri_kind, sqrt2)
                svec = (M @ M_val.ravel(order='F'))
                recovered = tri_to_full(svec, n, tri_kind, sqrt2)
                np.testing.assert_allclose(recovered, M_val, atol=1e-12)

    def test_tri_to_full_round_trip_batched(self) -> None:
        """Round-trip works for batched (2, n, n) PSD constraints."""
        from cvxpy.utilities.psd_utils import TriangleKind, psd_format_mat, tri_to_full
        rng = np.random.default_rng(7)
        n = 3
        batch = 2
        X = cp.Variable((batch, n, n), symmetric=True)
        con = PSD(X)
        Ms = np.stack([self._random_psd(n, rng) for _ in range(batch)])
        X.value = Ms

        for tri_kind in TriangleKind:
            for sqrt2 in [True, False]:
                M = psd_format_mat(con, tri_kind, sqrt2)
                svec = M @ Ms.ravel(order='F')
                recovered = tri_to_full(svec, n, tri_kind, sqrt2)
                np.testing.assert_allclose(recovered, Ms, atol=1e-12)

    def test_tri_to_full_n1(self) -> None:
        """1x1 matrix round-trips correctly (edge case: no off-diagonals)."""
        from cvxpy.utilities.psd_utils import TriangleKind, psd_format_mat, tri_to_full
        X = cp.Variable((1, 1), symmetric=True)
        con = PSD(X)
        X.value = np.array([[5.0]])

        for tri_kind in TriangleKind:
            for sqrt2 in [True, False]:
                M = psd_format_mat(con, tri_kind, sqrt2)
                svec = M @ np.array([5.0])
                recovered = tri_to_full(svec, 1, tri_kind, sqrt2)
                np.testing.assert_allclose(recovered, np.array([[5.0]]), atol=1e-12)


class TestMixedPSDSOC(BaseTest):
    """Tests for problems with both PSD and SOC constraints via PSD-only solver."""

    def test_mixed_psd_soc(self) -> None:
        """Problem with both PSD and SOC constraints through _PSDOnlyClarabel."""
        X = cp.Variable((2, 2), symmetric=True)
        y = cp.Variable(2)
        prob = cp.Problem(
            cp.Minimize(cp.trace(X) + cp.sum(y)),
            [X >> 0, X[0, 1] >= 1, cp.norm(y, 2) <= 1]
        )
        prob.solve(solver=_PSDOnlyClarabel())
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        # X optimal: min trace with X[0,1]>=1 and PSD gives trace=2
        # y optimal: min sum(y) with ||y||<=1 gives -sqrt(2)
        self.assertAlmostEqual(prob.value, 2 - np.sqrt(2), places=3)


class TestPowConeNDDCP(BaseTest):

    def test_pow_cone_nd_is_dcp(self) -> None:
        """PowConeND.is_dcp() should check affinity of arguments."""
        x = cp.Variable(2, nonneg=True)
        z = cp.Variable()
        alpha = np.array([0.5, 0.5])
        constr = PowConeND(x, z, alpha)
        self.assertTrue(constr.is_dcp())
        self.assertTrue(constr.args[0].is_affine())
        self.assertTrue(constr.args[1].is_affine())
