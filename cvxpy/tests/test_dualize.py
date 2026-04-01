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

import cvxpy as cp
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.cone2cone.affine2direct import DualizeConeProg
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.tests.base_test import BaseTest


class TestDualizeConeProg(BaseTest):
    """Tests for the post-ConeMatrixStuffing DualizeConeProg reduction."""

    @staticmethod
    def _stuff(prob):
        """Run the full pre-solver chain and return (ParamConeProg, inv, chain)."""
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

    def test_data_unchanged(self) -> None:
        """DualizeConeProg does not modify the ParamConeProg data."""
        x = cp.Variable(3)
        prob = cp.Problem(
            cp.Minimize(x[0] + 2 * x[1] + 3 * x[2]),
            [x >= 0, cp.sum(x) == 10, x[2] <= 5],
        )
        pcp, _, _ = self._stuff(prob)
        c_before, d_before, A_before, b_before = pcp.apply_parameters()

        dualize = DualizeConeProg()
        pcp2, _ = dualize.apply(pcp)
        c_after, d_after, A_after, b_after = pcp2.apply_parameters()

        # Data is identical — only the flag changed
        self.assertAlmostEqual(d_before, d_after)
        self.assertItemsAlmostEqual(c_before, c_after)
        self.assertItemsAlmostEqual(b_before, b_after)
        self.assertItemsAlmostEqual(
            A_before.toarray().flatten(), A_after.toarray().flatten())
