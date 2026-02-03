"""
Copyright 2025 CVXPY developers  

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


class TestPowConeND(BaseTest):
    """Unit tests for PowConeND and PowCone3D."""

    def solve_prob(self, prob, solver):
        result = prob.solve(solver=solver, verbose=False)
        self.assertIn(prob.status, [cp.OPTIMAL, cp.OPTIMAL_INACCURATE])
        return result

    def test_pow_cone_nd_3d(self) -> None:
        """
        A variation of test_pcp_3
        Some solvers like clarabel natively support ND power cones, while others
        like SCS only support 3D power cones. In the former case, we can directly
        pass an ND power cone constraint. In the latter case, CVXPY will apply 
        a reduction to convert the ND power cone constraint into a set of 3D power
        cone constraints.

        We check correctness for both axis=0 and axis=1 orientations.
        """
        expect_x = np.array([0.06393515, 0.78320961, 2.30571048])
        for axis in [0, 1]:
            for solver in [cp.CLARABEL, cp.SCS]:
                x = cp.Variable(3, name='x')
                hypos = cp.Variable(2, name='hypos')
                objective = cp.Maximize(cp.sum(hypos) - x[0])
                W = cp.bmat([[x[0], x[2]],
                             [x[1], 1.0]])
                alpha = np.array([[0.2, 0.4],
                                  [0.8, 0.6]])
                if axis == 1:
                    W = W.T
                    alpha = alpha.T

                constraints = [
                    x[0] + x[1] + 0.5 * x[2] == 2,
                    cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
                ]
                prob = cp.Problem(objective, constraints)
                self.solve_prob(prob, solver)
                self.assertItemsAlmostEqual(x.value, expect_x, places=3)

    def test_pow_cone_nd_3d_variable_swap(self) -> None:
        """
        A variation of test_pcp_3 with variable appearing in a different order.
        We expect the same solution as test_pow_cone_nd_3d, but with variables
        reordered.
        Both axis values tested.
        """
        expect_x = np.array([0.06393515, 2.30571048, 0.78320961])
        for axis in [0, 1]:
            x = cp.Variable(3)
            hypos = cp.Variable(2)
            objective = cp.Maximize(cp.sum(hypos) - x[0])
            W = cp.bmat([[x[0], x[1]],
                         [x[2], 1.0]])
            alpha = np.array([[0.2, 0.4],
                              [0.8, 0.6]])
            if axis == 1:
                W = W.T
                alpha = alpha.T
            constraints = [
                x[0] + x[2] + 0.5 * x[1] == 2,
                cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
            ]
            prob = cp.Problem(objective, constraints)
            self.solve_prob(prob, cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, expect_x, places=3)

    def test_pow_cone_nd(self) -> None:
        """
        Solving a PowConeND constraint with more than 3 dimensions.
        Both axis values tested.
        """
        expect_x = np.array([0, 0, 0, 2.28571379, 3.42857186])
        for axis in [0, 1]:
            x = cp.Variable(5)
            hypos = cp.Variable(2)
            objective = cp.Maximize(cp.sum(hypos) - x[0])
            W = cp.bmat([[x[0], x[3]],
                         [x[1], x[4]],
                         [x[2], 1.0]])
            alpha = np.array([[0.2, 0.4],
                              [0.4, 0.3],
                              [0.4, 0.3]])
            if axis == 1:
                W = W.T
                alpha = alpha.T
            constraints = [
                x[0] + x[1] + 0.5 * x[2] + 0.5 * x[3] + 0.25 * x[4] == 2,
                cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
            ]
            prob = cp.Problem(objective, constraints)
            self.solve_prob(prob, cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, expect_x, places=3)
            

    def test_pow_cone_nd_variable_swap(self) -> None:
        """
        A variation of test_pow_cone_nd with variable appearing in a different order.
        We expect the same solution as test_pow_cone_nd, but with variables
        reordered.
        Both axis values tested.
        """
        expect_x = np.array([3.42857186, 0, 0, 2.28571379, 0])
        for axis in [0, 1]:
            x = cp.Variable(5)
            hypos = cp.Variable(2)
            objective = cp.Maximize(cp.sum(hypos) - x[4])
            W = cp.bmat([[x[4], x[3]],
                         [x[1], x[0]],
                         [x[2], 1.0]])
            alpha = np.array([[0.2, 0.4],
                              [0.4, 0.3],
                              [0.4, 0.3]])
            if axis == 1:
                W = W.T
                alpha = alpha.T
            constraints = [
                x[4] + x[1] + 0.5 * x[2] + 0.5 * x[3] + 0.25 * x[0] == 2,
                cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
            ]
            prob = cp.Problem(objective, constraints)
            self.solve_prob(prob, cp.CLARABEL)
            self.assertItemsAlmostEqual(x.value, expect_x, places=3)

    def test_pow_cone_nd_single_cone(self) -> None:
        """
        Solving a PowConeND constraint with only a single cone.
        This check is performed to ensure no variables collapse to lower
        dimensions incorrectly.
        Both axis values tested.
        """
        for axis in [0, 1]:
            for solver in [cp.CLARABEL, cp.SCS]:
                x = cp.Variable(2)
                hypos = cp.Variable(1)
                objective = cp.Maximize(cp.sum(hypos) - x[0])
                W = cp.bmat([[x[0]], [x[1]]])
                alpha = np.array([[0.2], [0.8]])
                if axis == 1:
                    W = W.T
                    alpha = alpha.T
                constraints = [
                    x[0] + x[1] == 2,
                    cp.constraints.PowConeND(W, hypos, alpha, axis=axis)
                ]
                prob = cp.Problem(objective, constraints)
                self.solve_prob(prob, solver)

    def test_3d_pow_cone_scalar_alpha(self) -> None:
        """
        Test PowCone3D with scalar alpha.
        """
        for solver in [cp.CLARABEL, cp.SCS]:
            x = cp.Variable(3)
            constraints = [cp.PowCone3D(x[0], x[1], x[2], 0.75)]
            prob = cp.Problem(cp.Minimize(cp.norm(x)), constraints)
            self.solve_prob(prob, solver)

    def test_pow_cone_nd_balanced_tree_decomposition(self) -> None:
        """Test balanced tree decomposition for various n values."""
        for n in [3, 4, 5, 8, 16, 100]:
            W = cp.Variable(n, pos=True)
            z = cp.Variable()
            alpha = np.ones(n) / n
            con = cp.constraints.PowConeND(W, z, alpha)
            prob = cp.Problem(cp.Maximize(z), [con, W <= 2])
            self.solve_prob(prob, cp.SCS)
            self.assertAlmostEqual(z.value, 2.0, places=3)

    def test_pow_cone_nd_balanced_tree_nonuniform_alpha(self) -> None:
        """Test balanced tree decomposition with non-uniform alpha weights."""
        alpha = np.array([0.05, 0.1, 0.15, 0.2, 0.2, 0.15, 0.1, 0.05])
        bounds = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        expected = np.prod(bounds ** alpha)

        W = cp.Variable(8, pos=True)
        z = cp.Variable()
        con = cp.constraints.PowConeND(W, z, alpha)
        prob = cp.Problem(cp.Maximize(z), [con, W <= bounds])
        self.solve_prob(prob, cp.SCS)
        self.assertAlmostEqual(z.value, expected, places=3)

    def test_pow_cone_nd_balanced_tree_multiple_cones(self) -> None:
        """Test balanced tree decomposition with multiple cones (k > 1)."""
        n, k = 8, 3
        W = cp.Variable((n, k), pos=True)
        z = cp.Variable(k)
        alpha = np.ones((n, k)) / n
        con = cp.constraints.PowConeND(W, z, alpha, axis=0)
        prob = cp.Problem(cp.Maximize(cp.sum(z)), [con, W <= 2])
        self.solve_prob(prob, cp.SCS)
        np.testing.assert_allclose(z.value, [2.0, 2.0, 2.0], rtol=1e-3)

    def test_pow_cone_nd_dual_variables(self) -> None:
        """Test dual variable recovery matches CLARABEL for n=2 and n=4."""
        for n, rtol in [(2, 1e-3), (4, 1e-2)]:
            W = cp.Variable((n, 1), pos=True)
            z = cp.Variable(1)
            alpha = np.ones((n, 1)) / n
            con = cp.constraints.PowConeND(W, z, alpha, axis=0)
            prob = cp.Problem(cp.Maximize(z[0]), [con, W <= 2])

            self.solve_prob(prob, cp.CLARABEL)
            clarabel_w_dual = con.dual_value[0].flatten()
            clarabel_z_dual = con.dual_value[1].flatten()

            self.solve_prob(prob, cp.SCS)
            scs_w_dual = con.dual_value[0].flatten()
            scs_z_dual = con.dual_value[1].flatten()

            np.testing.assert_allclose(scs_w_dual, clarabel_w_dual, rtol=rtol)
            np.testing.assert_allclose(scs_z_dual, clarabel_z_dual, rtol=rtol)

