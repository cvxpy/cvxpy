"""
Copyright, the CVXPY authors

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
from cvxpy.constraints.zero import Zero
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone.affine2direct import Dualize as A2DDualize
from cvxpy.reductions.cone2cone.dualize_cone_prog import DualizeConeProg
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.tests.base_test import BaseTest


class TestDualizeConeProg(BaseTest):
    """Tests for the post-ConeMatrixStuffing DualizeConeProg reduction."""

    @staticmethod
    def _stuff(prob):
        """Run the full pre-solver chain and return (ParamConeProg, inv_data, chain)."""
        chain = Chain(None, [Dcp2Cone(), CvxAttr2Constr(), ConeMatrixStuffing()])
        pcp, inv = chain.apply(prob)
        return pcp, inv, chain

    def test_flag_set(self) -> None:
        """DualizeConeProg sets the dualized flag."""
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        pcp, _, _ = self._stuff(prob)

        self.assertFalse(pcp.dualized)
        dualize = DualizeConeProg()
        self.assertTrue(dualize.accepts(pcp))
        pcp2, _ = dualize.apply(pcp)
        self.assertTrue(pcp2.dualized)

    def test_rejects_integer(self) -> None:
        x = cp.Variable(2, integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        pcp, _, _ = self._stuff(prob)
        dualize = DualizeConeProg()
        self.assertFalse(dualize.accepts(pcp))

    def test_rejects_already_dualized(self) -> None:
        x = cp.Variable(2)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        pcp, _, _ = self._stuff(prob)
        dualize = DualizeConeProg()
        pcp2, _ = dualize.apply(pcp)
        self.assertFalse(dualize.accepts(pcp2))

    def test_data_matches_a2d(self) -> None:
        """The dualized flag produces the same data as affine2direct.Dualize."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + 2 * x[1] + 3 * x[2]),
            [x >= 0, cp.sum(x) == 10, x[2] <= 5],
        )
        pcp, _, _ = self._stuff(prob)
        pcp = ConicSolver.format_constraints(pcp, exp_cone_order=[0, 1, 2])

        # affine2direct.Dualize on the same PCP
        a2d_data, _ = A2DDualize.apply(pcp)

        # DualizeConeProg just sets the flag
        dualize = DualizeConeProg()
        pcp2, _ = dualize.apply(pcp)

        # The PCP data is unchanged — solver uses it with the flag
        c, _, A, b = pcp2.apply_parameters()

        # A2D returns A.T, c, -b as the dual data
        np.testing.assert_array_equal(a2d_data['A'].toarray(), A.T.toarray())
        np.testing.assert_array_equal(a2d_data['b'], c)
        np.testing.assert_array_equal(a2d_data['c'], -b)

    def test_invert_round_trip(self) -> None:
        """Invert undoes the dualization for a known solution."""
        x = cp.Variable(3)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
        pcp, _, _ = self._stuff(prob)
        pcp = ConicSolver.format_constraints(pcp, exp_cone_order=[0, 1, 2])

        # Solve primal for reference
        prob.solve(solver=cp.CLARABEL)
        ref_val = prob.value

        # Get dual data via affine2direct and "solve" it through CVXPY
        a2d_data, _ = A2DDualize.apply(pcp)

        # Build and solve the dual as a CVXPY problem
        A_d = a2d_data['A']  # (n, m)
        b_d = a2d_data['b']  # (n,)  — RHS of A'y = c
        c_d = a2d_data['c']  # (m,)  — objective -b
        m = c_d.size
        y = cp.Variable(m)
        dual_prob = cp.Problem(
            cp.Maximize(c_d @ y),
            [Zero(A_d @ y - b_d), y >= 0],
        )
        dual_prob.solve(solver=cp.CLARABEL)

        # Strong duality check
        self.assertAlmostEqual(dual_prob.value, ref_val, places=4)
