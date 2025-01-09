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
import pytest
import scipy as sp

import cvxpy as cp
from cvxpy import settings as s
from cvxpy.atoms.affine.trace import trace
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import INSTALLED_MI_SOLVERS as INSTALLED_MI
from cvxpy.reductions.solvers.defines import MI_SOCP_SOLVERS as MI_SOCP
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.tests.base_test import BaseTest


class TestDualize(BaseTest):

    @staticmethod
    def simulate_chain(in_prob):
        # Get a ParamConeProg object
        reductions = [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()]
        chain = Chain(None, reductions)
        cone_prog, inv_prob2cone = chain.apply(in_prob)

        # Dualize the problem, reconstruct a high-level cvxpy problem for the dual.
        # Solve the problem, invert the dualize reduction.
        cone_prog = ConicSolver.format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
        data, inv_data = a2d.Dualize.apply(cone_prog)
        A, b, c, K_dir = data[s.A], data[s.B], data[s.C], data['K_dir']
        y = cp.Variable(shape=(A.shape[1],))
        constraints = [A @ y == b]
        i = K_dir[a2d.FREE]
        dual_prims = {a2d.FREE: y[:i], a2d.SOC: []}
        if K_dir[a2d.NONNEG]:
            dim = K_dir[a2d.NONNEG]
            dual_prims[a2d.NONNEG] = y[i:i+dim]
            constraints.append(y[i:i+dim] >= 0)
            i += dim
        for dim in K_dir[a2d.SOC]:
            dual_prims[a2d.SOC].append(y[i:i+dim])
            constraints.append(SOC(y[i], y[i+1:i+dim]))
            i += dim
        if K_dir[a2d.DUAL_EXP]:
            exp_len = 3 * K_dir[a2d.DUAL_EXP]
            dual_prims[a2d.DUAL_EXP] = y[i:i+exp_len]
            y_de = cp.reshape(y[i:i+exp_len], (exp_len//3, 3), order='C')  # fill rows first
            constraints.append(ExpCone(-y_de[:, 1], -y_de[:, 0], np.exp(1)*y_de[:, 2]))
            i += exp_len
        if K_dir[a2d.DUAL_POW3D]:
            alpha = np.array(K_dir[a2d.DUAL_POW3D])
            dual_prims[a2d.DUAL_POW3D] = y[i:]
            y_dp = cp.reshape(y[i:], (alpha.size, 3), order='C')  # fill rows first
            pow_con = PowCone3D(y_dp[:, 0] / alpha, y_dp[:, 1] / (1-alpha), y_dp[:, 2], alpha)
            constraints.append(pow_con)
        objective = cp.Maximize(c @ y)
        dual_prob = cp.Problem(objective, constraints)
        dual_prob.solve(solver='SCS', eps=1e-8)
        dual_prims[a2d.FREE] = dual_prims[a2d.FREE].value
        if K_dir[a2d.NONNEG]:
            dual_prims[a2d.NONNEG] = dual_prims[a2d.NONNEG].value
        dual_prims[a2d.SOC] = [expr.value for expr in dual_prims[a2d.SOC]]
        if K_dir[a2d.DUAL_EXP]:
            dual_prims[a2d.DUAL_EXP] = dual_prims[a2d.DUAL_EXP].value
        if K_dir[a2d.DUAL_POW3D]:
            dual_prims[a2d.DUAL_POW3D] = dual_prims[a2d.DUAL_POW3D].value
        dual_duals = {s.EQ_DUAL: constraints[0].dual_value}
        dual_sol = Solution(dual_prob.status, dual_prob.value, dual_prims, dual_duals, dict())
        cone_sol = a2d.Dualize.invert(dual_sol, inv_data)

        # Pass the solution back up the solving chain.
        in_prob_sol = chain.invert(cone_sol, inv_prob2cone)
        in_prob.unpack(in_prob_sol)

    def test_lp_1(self):
        # typical LP
        sth = STH.lp_1()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_2(self):
        # typical LP
        sth = STH.lp_2()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_lp_3(self):
        # unbounded LP
        sth = STH.lp_3()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)

    def test_lp_4(self):
        # infeasible LP
        sth = STH.lp_4()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)

    def test_lp_5(self):
        # LP with redundant constraints
        sth = STH.lp_5()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.check_primal_feasibility(places=4)
        sth.check_complementarity(places=4)
        sth.check_dual_domains(places=4)

    def test_socp_0(self):
        sth = STH.socp_0()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)

    def test_socp_1(self):
        sth = STH.socp_1()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_socp_2(self):
        sth = STH.socp_2()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def _socp_3(self, axis):
        sth = STH.socp_3(axis)
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_socp_3_axis_0(self):
        self._socp_3(0)

    def test_socp_3_axis_1(self):
        self._socp_3(1)

    def test_expcone_1(self):
        sth = STH.expcone_1()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_expcone_socp_1(self):
        sth = STH.expcone_socp_1()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=4)
        sth.verify_primal_values(places=4)
        sth.verify_dual_values(places=4)

    def test_pcp_2(self):
        sth = STH.pcp_2()
        TestDualize.simulate_chain(sth.prob)
        sth.verify_objective(places=3)
        sth.verify_primal_values(places=3)
        sth.verify_dual_values(places=3)


class TestSlacks(BaseTest):

    AFF_LP_CASES = [[a2d.NONNEG], []]
    AFF_SOCP_CASES = [[a2d.NONNEG, a2d.SOC], [a2d.NONNEG], [a2d.SOC], []]
    AFF_EXP_CASES = [[a2d.NONNEG, a2d.EXP], [a2d.NONNEG], [a2d.EXP], []]
    AFF_PCP_CASES = [[a2d.NONNEG], [a2d.POW3D], []]
    AFF_MIXED_CASES = [[a2d.NONNEG], []]

    @staticmethod
    def simulate_chain(in_prob, affine, **solve_kwargs):
        # get a ParamConeProg object
        reductions = [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()]
        chain = Chain(None, reductions)
        cone_prog, inv_prob2cone = chain.apply(in_prob)

        # apply the Slacks reduction, reconstruct a high-level problem,
        # solve the problem, invert the reduction.
        cone_prog = ConicSolver.format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
        data, inv_data = a2d.Slacks.apply(cone_prog, affine)
        G, h, f, K_dir, K_aff = data[s.A], data[s.B], data[s.C], data['K_dir'], data['K_aff']
        G = sp.sparse.csc_matrix(G)
        y = cp.Variable(shape=(G.shape[1],))
        objective = cp.Minimize(f @ y)
        aff_con = TestSlacks.set_affine_constraints(G, h, y, K_aff)
        dir_con = TestSlacks.set_direct_constraints(y, K_dir)
        int_con = TestSlacks.set_integer_constraints(y, data)
        constraints = aff_con + dir_con + int_con
        slack_prob = cp.Problem(objective, constraints)
        slack_prob.solve(**solve_kwargs)
        slack_prims = {a2d.FREE: y[:cone_prog.x.size].value}  # nothing else need be populated.
        slack_sol = Solution(slack_prob.status, slack_prob.value, slack_prims, None, dict())
        cone_sol = a2d.Slacks.invert(slack_sol, inv_data)

        # pass solution up the solving chain
        in_prob_sol = chain.invert(cone_sol, inv_prob2cone)
        in_prob.unpack(in_prob_sol)

    @staticmethod
    def set_affine_constraints(G, h, y, K_aff):
        constraints = []
        i = 0
        if K_aff[a2d.ZERO]:
            dim = K_aff[a2d.ZERO]
            constraints.append(G[i:i+dim, :] @ y == h[i:i+dim])
            i += dim
        if K_aff[a2d.NONNEG]:
            dim = K_aff[a2d.NONNEG]
            constraints.append(G[i:i+dim, :] @ y <= h[i:i+dim])
            i += dim
        for dim in K_aff[a2d.SOC]:
            expr = h[i:i+dim] - G[i:i+dim, :] @ y
            constraints.append(SOC(expr[0], expr[1:]))
            i += dim
        if K_aff[a2d.EXP]:
            dim = 3 * K_aff[a2d.EXP]
            expr = cp.reshape(h[i:i+dim] - G[i:i+dim, :] @ y, (dim//3, 3), order='C')
            constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
            i += dim
        if K_aff[a2d.POW3D]:
            alpha = np.array(K_aff[a2d.POW3D])
            expr = cp.reshape(h[i:] - G[i:, :] @ y, (alpha.size, 3), order='C')
            constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
        return constraints

    @staticmethod
    def set_direct_constraints(y, K_dir):
        constraints = []
        i = K_dir[a2d.FREE]
        if K_dir[a2d.NONNEG]:
            dim = K_dir[a2d.NONNEG]
            constraints.append(y[i:i+dim] >= 0)
            i += dim
        for dim in K_dir[a2d.SOC]:
            constraints.append(SOC(y[i], y[i+1:i+dim]))
            i += dim
        if K_dir[a2d.EXP]:
            dim = 3 * K_dir[a2d.EXP]
            expr = cp.reshape(y[i:i+dim], (dim//3, 3), order='C')
            constraints.append(ExpCone(expr[:, 0], expr[:, 1], expr[:, 2]))
            i += dim
        if K_dir[a2d.POW3D]:
            alpha = np.array(K_dir[a2d.POW3D])
            expr = cp.reshape(y[i:], (alpha.size, 3), order='C')
            constraints.append(PowCone3D(expr[:, 0], expr[:, 1], expr[:, 2], alpha))
        return constraints

    @staticmethod
    def set_integer_constraints(y, data):
        constraints = []
        if data[s.BOOL_IDX]:
            expr = y[data[s.BOOL_IDX]]
            z = cp.Variable(shape=(expr.size,), boolean=True)
            constraints.append(expr == z)
        if data[s.INT_IDX]:
            expr = y[data[s.INT_IDX]]
            z = cp.Variable(shape=(expr.size,), integer=True)
            constraints.append(expr == z)
        return constraints

    def test_lp_2(self):
        # typical LP
        sth = STH.lp_2()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='CLARABEL')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_lp_3(self):
        # unbounded LP
        sth = STH.lp_3()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='CLARABEL')
            sth.verify_objective(places=4)

    def test_lp_4(self):
        # infeasible LP
        sth = STH.lp_4()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='CLARABEL')
            sth.verify_objective(places=4)

    def test_socp_2(self):
        sth = STH.socp_2()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='CLARABEL')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_socp_3(self):
        for axis in [0, 1]:
            sth = STH.socp_3(axis)
            TestSlacks.simulate_chain(sth.prob, [], solver='CLARABEL')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_1(self):
        sth = STH.expcone_1()
        for affine in TestSlacks.AFF_EXP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='CLARABEL')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_socp_1(self):
        sth = STH.expcone_socp_1()
        for affine in TestSlacks.AFF_MIXED_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='SCS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_pcp_1(self):
        sth = STH.pcp_1()
        for affine in TestSlacks.AFF_PCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='SCS', eps=1e-8)
            sth.verify_objective(places=3)
            sth.verify_primal_values(places=3)

    def test_pcp_2(self):
        sth = STH.pcp_2()
        for affine in TestSlacks.AFF_PCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='SCS', eps=1e-8)
            sth.verify_objective(places=3)
            sth.verify_primal_values(places=3)

    @pytest.mark.skipif(
        "HIGHS" not in INSTALLED_MI, 
        reason='HiGHS solver is not installed.'
    )
    def test_mi_lp_1(self):
        sth = STH.mi_lp_1()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver=cp.HIGHS)
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @pytest.mark.skip(reason="Known bug in ECOS BB")
    def test_mi_socp_1(self):
        sth = STH.mi_socp_1()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver=cp.SCIPY)
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @unittest.skipUnless([svr for svr in INSTALLED_MI if svr in MI_SOCP],
                         'No appropriate mixed-integer SOCP solver is installed.')
    def test_mi_socp_2(self):
        sth = STH.mi_socp_2()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine)
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)


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
