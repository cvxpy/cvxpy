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
import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests import solver_test_helpers as STH
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy import settings as s
from cvxpy.reductions.solvers.conic_solvers.scs_conif import SCS
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.exponential import ExpCone


class TestDualize(BaseTest):

    @staticmethod
    def simulate_chain(in_prob):
        # Get a ParamConeProg object
        reductions = [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()]
        chain = Chain(None, reductions)
        cone_prog, inv_prob2cone = chain.apply(in_prob)

        # Dualize the problem, reconstruct a high-level cvxpy problem for the dual.
        # Solve the problem, invert the dualize reduction.
        solver = SCS()
        cone_prog = solver.format_constraints(cone_prog, exp_cone_order=[0, 1, 2])
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
            dual_prims[a2d.DUAL_EXP] = y[i:]
            y_de = cp.reshape(y[i:], ((y.size - i)//3, 3), order='C')  # fill rows first
            constraints.append(ExpCone(-y_de[:, 1], -y_de[:, 0], np.exp(1)*y_de[:, 2]))
        objective = cp.Maximize(c @ y)
        dual_prob = cp.Problem(objective, constraints)
        dual_prob.solve(solver='SCS', eps=1e-8)
        dual_prims[a2d.FREE] = dual_prims[a2d.FREE].value
        if K_dir[a2d.NONNEG]:
            dual_prims[a2d.NONNEG] = dual_prims[a2d.NONNEG].value
        dual_prims[a2d.SOC] = [expr.value for expr in dual_prims[a2d.SOC]]
        if K_dir[a2d.DUAL_EXP]:
            dual_prims[a2d.DUAL_EXP] = dual_prims[a2d.DUAL_EXP].value
        dual_duals = {s.EQ_DUAL: constraints[0].dual_value}
        dual_sol = cp.Solution(dual_prob.status, dual_prob.value, dual_prims, dual_duals, dict())
        cone_sol = a2d.Dualize.invert(dual_sol, inv_data)

        # Pass the solution back up the solving chain.
        in_prob_sol = chain.invert(cone_sol, inv_prob2cone)
        in_prob.unpack(in_prob_sol)
        return cone_prog, dual_prob

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
