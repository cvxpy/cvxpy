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
from cvxpy.constraints.exponential import ExpCone
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.constraints.second_order import SOC
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.defines import (
    INSTALLED_MI_SOLVERS as INSTALLED_MI,)
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
        cone_prog = ConicSolver().format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
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
        dual_sol = cp.Solution(dual_prob.status, dual_prob.value, dual_prims, dual_duals, dict())
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
        cone_prog = ConicSolver().format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
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
        slack_sol = cp.Solution(slack_prob.status, slack_prob.value, slack_prims, None, dict())
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
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_lp_3(self):
        # unbounded LP
        sth = STH.lp_3()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)

    def test_lp_4(self):
        # infeasible LP
        sth = STH.lp_4()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)

    def test_socp_2(self):
        sth = STH.socp_2()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_socp_3(self):
        for axis in [0, 1]:
            sth = STH.socp_3(axis)
            TestSlacks.simulate_chain(sth.prob, [], solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_1(self):
        sth = STH.expcone_1()
        for affine in TestSlacks.AFF_EXP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    def test_expcone_socp_1(self):
        sth = STH.expcone_socp_1()
        for affine in TestSlacks.AFF_MIXED_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS')
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

    def test_mi_lp_1(self):
        sth = STH.mi_lp_1()
        for affine in TestSlacks.AFF_LP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS_BB')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @pytest.mark.skip(reason="Known bug in ECOS BB")
    def test_mi_socp_1(self):
        sth = STH.mi_socp_1()
        for affine in TestSlacks.AFF_SOCP_CASES:
            TestSlacks.simulate_chain(sth.prob, affine, solver='ECOS_BB')
            sth.verify_objective(places=4)
            sth.verify_primal_values(places=4)

    @unittest.skipUnless([svr for svr in INSTALLED_MI if svr in MI_SOCP and svr != 'ECOS_BB'],
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
        sth.solve(solver='ECOS')
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
        sth.solve(solver='ECOS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)


class TestOpRelCone(BaseTest):

    def oprelcone_1(self) -> STH.SolverTestHelper:
        n = 3
        # generates `n` independent, orthonormal vectors
        U = sp.linalg.qr(np.random.randn(n, n), mode='economic')[0]
        a_diag = cp.Variable(shape=(n,), pos=True)
        b_diag = cp.Variable(shape=(n,), pos=True)
        # the below constraint ensures that `A` and `B` are
        # defined in terms of their eigendecomposition and have the same eigenvectors
        A = U @ cp.diag(a_diag) @ U.T
        B = U @ cp.diag(b_diag) @ U.T
        T = cp.Variable(shape=(n, n))
        # constrains A,B,T \in OpRelCone
        con1 = cp.constraints.OpRelCone(A, B, T, 10, 10)
        # imposing some non-trivial constraints to ensure feasibility
        con2 = a_diag <= 5
        con3 = a_diag >= 2.5
        con4 = b_diag <= 1.5
        con5 = b_diag >= 0.5
        con_pairs = [(con1, None),
                     (con2, None),
                     (con3, None),
                     (con4, None),
                     (con5, None)]
        # objective is increasing_func_lambda(T)
        k = 2
        obj = cp.Minimize(cp.atoms.lambda_sum_largest(T, k))
        obj_pair = (obj, 2.5540726360202255)
        expect_T = np.array([[1.27702809e+00, 2.45713885e-14, -5.34214759e-15],
                             [2.45713919e-14, 1.27702809e+00, 1.09987017e-13],
                             [-5.34214762e-15, 1.09987023e-13, 1.27702809e+00]])
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_1(self):
        sth = self.oprelcone_1()
        sth.solve(solver='SCS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)

    def oprelcone_2(self) -> STH.SolverTestHelper:
        n = 4
        U = sp.linalg.qr(np.random.randn(n, n), mode='economic')[0]
        a_diag = cp.Variable(shape=(n,), pos=True)
        b_diag = cp.Variable(shape=(n,), pos=True)
        A = U @ cp.diag(a_diag) @ U.T
        B = U @ cp.diag(b_diag) @ U.T
        T = cp.Variable(shape=(n, n))
        con1 = cp.constraints.OpRelCone(A, B, T, 2, 2)
        con2 = a_diag <= 5
        con3 = b_diag <= 1.5
        con_pairs = [(con1, None),
                     (con2, None),
                     (con3, None)]
        k = 3
        obj = cp.Minimize(cp.atoms.lambda_sum_largest(T, k))
        obj_pair = (obj, -1.6554572363409088)
        expect_T = np.array([[-5.51808473e-01, -3.22477062e-13, -1.53851070e-13, 9.32509529e-14],
                             [-3.22476879e-13, -5.51808473e-01, 5.93445801e-15, 4.76722578e-13],
                             [-1.53851100e-13, 5.93442592e-15, -5.51808473e-01, 4.09022045e-14],
                             [9.32509585e-14, 4.76722648e-13, 4.09021549e-14, -5.51808473e-01]])
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_2(self):
        sth = self.oprelcone_2()
        sth.solve(solver='SCS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)

    def oprelcone_3(self) -> STH.SolverTestHelper:
        n = 5
        # This is where (A, B) do not commute --- generate two sets of
        # orthogonal eigenvectors for each of them
        U1 = sp.linalg.qr(np.random.randn(n, n), mode='economic')[0]
        U2 = sp.linalg.qr(np.random.randn(n, n), mode='economic')[0]
        a_diag = cp.Variable(shape=(n,), pos=True)
        b_diag = cp.Variable(shape=(n,), pos=True)
        A = U1 @ cp.diag(a_diag) @ U1.T
        B = U2 @ cp.diag(b_diag) @ U2.T
        T = cp.Variable(shape=(n, n))
        con1 = cp.constraints.OpRelCone(A, B, T, 10, 10)
        con2 = a_diag <= 5
        con3 = a_diag >= 2.5
        con4 = b_diag <= 1.5
        con5 = b_diag >= 0.5
        con_pairs = [(con1, None),
                     (con2, None),
                     (con3, None),
                     (con4, None),
                     (con5, None)]
        k = 3
        obj = cp.Minimize(cp.atoms.lambda_sum_largest(T, k))
        obj_pair = (obj,  3.8311357528637906)
        expect_T = np.array([[1.27702101e+00, 2.87104297e-11, 6.45992413e-12,
                              1.30762826e-12, 2.02783512e-11],
                             [2.87104297e-11, 1.27702101e+00, 6.13021465e-12,
                              -7.90859723e-12, -6.72410679e-14],
                             [6.45992413e-12, 6.13021464e-12, 1.27702101e+00,
                              1.29246719e-11, -9.42691763e-12],
                             [1.30762826e-12, -7.90859723e-12, 1.29246719e-11,
                              1.27702101e+00, 2.42404676e-12],
                             [2.02783512e-11, -6.72410705e-14, -9.42691763e-12,
                              2.42404676e-12, 1.27702101e+00]])
        var_pairs = [(T, expect_T)]
        sth = STH.SolverTestHelper(obj_pair, var_pairs, con_pairs)
        return sth

    def test_oprelcone_3(self):
        sth = self.oprelcone_3()
        sth.solve(solver='SCS')
        sth.verify_primal_values(places=2)
        sth.verify_objective(places=2)
