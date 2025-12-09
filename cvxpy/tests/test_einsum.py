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

import unittest

import numpy as np

import cvxpy as cp
from cvxpy.tests.base_test import BaseTest


class TestEinsum(BaseTest):
    """Unit tests for the einsum atom."""

    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.A_np = rng.normal(size=(4, 4))
        self.B_np = rng.normal(size=(4, 5, 2))
        self.C_np = rng.normal(size=(5,))
        self.D_np = rng.normal(size=(3, 4, 3))
        self.E_np = rng.normal(size=(5, 2, 5))
        self.F_np = rng.normal(size=(2, 3, 2, 4))

        self.x_np = rng.normal(size=(5,))
        self.y_np = rng.normal(size=(5,))
        self.z_np = rng.normal(size=(3,))

    def test_einsum_A_shape(self) -> None:
        """Test einsum with single argument (A)."""
        A = cp.Variable(self.A_np.shape)

        # Test diagonal extraction
        expr = cp.einsum('ii->i', A)
        expected_shape = np.einsum('ii->i', self.A_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test sum over all elements
        expr = cp.einsum('ij->', A)
        expected_shape = np.einsum('ij->', self.A_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test implicit mode
        expr = cp.einsum('ij', A)
        expected_shape = np.einsum('ij', self.A_np).shape
        self.assertEqual(expr.shape, expected_shape)

        expr = cp.einsum('ii', A)
        expected_shape = np.einsum('ii', self.A_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test vector operations
        x = cp.Variable(self.x_np.shape)
        expr = cp.einsum('i->', x)
        expected_shape = np.einsum('i->', self.x_np).shape
        self.assertEqual(expr.shape, expected_shape)

        expr = cp.einsum('i', x)
        expected_shape = np.einsum('i', self.x_np).shape
        self.assertEqual(expr.shape, expected_shape)

    def test_einsum_AB_shape(self) -> None:
        """Test einsum with two arguments (A, B)."""
        A = cp.Variable(self.A_np.shape)
        B = cp.Parameter(self.B_np.shape)

        # Test tensor multiplication
        expr = cp.einsum('ij,jkl->ikl', A, B)
        expected_shape = np.einsum('ij,jkl->ikl', self.A_np, self.B_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test element-wise multiplication
        expr = cp.einsum('ij,ij->ij', A, A)
        expected_shape = np.einsum('ij,ij->ij', self.A_np, self.A_np).shape
        self.assertEqual(expr.shape, expected_shape)

        x = cp.Variable(self.x_np.shape)
        y = cp.Parameter(self.y_np.shape)

        # Test outer product
        expr = cp.einsum('i,j->ij', x, y)
        expected_shape = np.einsum('i,j->ij', self.x_np, self.y_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test vector dot product
        expr = cp.einsum('i,i->', y, y)
        expected_shape = np.einsum('i,i->', self.x_np, self.x_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test implicit mode tensor multiplication
        expr = cp.einsum('ij,jkl', A, B)
        expected_shape = np.einsum('ij,jkl', self.A_np, self.B_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test ellipsis broadcasting
        expr = cp.einsum('...i,ijk->...jk', A, B)
        expected_shape = np.einsum('...i,ijk->...jk', self.A_np, self.B_np).shape
        self.assertEqual(expr.shape, expected_shape)

    def test_einsum_ABC_shape(self) -> None:
        """Test einsum with three arguments (A, B, C)."""
        A = cp.Variable(self.A_np.shape)
        B = cp.Parameter(self.B_np.shape)
        C = cp.Parameter(self.C_np.shape)

        # Test complex contraction
        expr = cp.einsum('ij,jkl,k->il', A, B, C)
        expected_shape = np.einsum('ij,jkl,k->il', self.A_np, self.B_np, self.C_np).shape
        self.assertEqual(expr.shape, expected_shape)

        # Test implicit mode with three tensors
        expr = cp.einsum('ij,jkl,k', A, B, C)
        expected_shape = np.einsum('ij,jkl,k', self.A_np, self.B_np, self.C_np).shape
        self.assertEqual(expr.shape, expected_shape)

    def test_einsum_validation(self) -> None:
        """Test input validation for einsum arguments."""
        # Test wrong number of arguments - too few
        A = cp.Variable(self.A_np.shape)
        B = cp.Parameter(self.B_np.shape)
        C = cp.Parameter(self.C_np.shape)
        D = cp.Variable(self.D_np.shape)
        E = cp.Variable(self.E_np.shape)
        
        with self.assertRaises(ValueError) as cm:
            cp.einsum('ij,jk->ik', A)  # Missing B
        self.assertIn("Number of einsum subscripts must be equal", str(cm.exception))

        # Test wrong number of arguments - too many
        with self.assertRaises(ValueError) as cm:
            cp.einsum('ij,jk->ik', A, B, C)  # Extra C
        self.assertIn("Number of einsum subscripts must be equal", str(cm.exception))

        # Test wrong shape - ndim mismatch
        with self.assertRaises(ValueError) as cm:
            cp.einsum('ij,jkl->ikl', D, B)  # A_3d has 3 dims but pattern expects 2
        self.assertIn(
            "Einstein sum subscript ij does not contain the correct number of indices"
            " for operand 0.",
            str(cm.exception)
        )

        # Test inconsistent dimensions for shared indices
        with self.assertRaises(ValueError) as cm:
            cp.einsum('ij,jkl->ikl', A, E)  # j dimension mismatch
        self.assertIn(
            "Size of label 'j' for operand 1 (4) does not match previous terms (5).", 
            str(cm.exception)
        )

        # Test output index not in inputs
        with self.assertRaises(ValueError) as cm:
            cp.einsum('ij,jkl->im', A, B)  # 'm' not in inputs
        self.assertIn("Output character m did not appear in the input", str(cm.exception))

    def test_einsum_sign(self) -> None:
        """Test sign analysis for einsum arguments."""
        A = cp.Variable(self.A_np.shape)
        B = cp.Variable(self.B_np.shape, pos=True)
        D = cp.Constant(np.ones_like(self.D_np))

        # Test sign analysis
        # Unknown + positive = unknown
        expr1 = cp.einsum('ij,kjl->ik', A, D)
        self.assertEqual(expr1.is_nonneg(), False)
        self.assertEqual(expr1.is_nonpos(), False)

        # Affine case (1 non-constant): convex and concave
        self.assertEqual(expr1.is_convex(), True)
        self.assertEqual(expr1.is_concave(), True)

        # Positive + positive = positive
        expr2 = cp.einsum('ijk,lim->ijkl', B, D)
        self.assertEqual(expr2.is_nonneg(), True)
        self.assertEqual(expr2.is_nonpos(), False)

        # Non-affine case (2 non-constants): neither convex nor concave
        expr3 = cp.einsum('ij,jkl->ik', A, B)
        self.assertEqual(expr3.is_convex(), False)
        self.assertEqual(expr3.is_concave(), False)

    def test_einsum_solve(self) -> None:
        """Test solving einsum problems."""
        # Einsum with only 2d arrays
        A = cp.Variable(self.A_np.shape)
        B = cp.Parameter(self.B_np.shape)
        C = cp.Parameter(self.C_np.shape)
        D = cp.Constant(self.D_np)
        F = cp.Variable(self.F_np.shape)

        B.value = self.B_np
        C.value = self.C_np

        # Test with a single argument and no repeated dimensions
        expr1 = cp.einsum('ij->i', A)
        problem1 = cp.Problem(cp.Minimize(cp.sum(expr1)), [A == self.A_np])
        problem1.solve()
        result1 = expr1.value
        expected_result1 = np.einsum('ij->i', self.A_np)
        self.assertEqual(problem1.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result1, expected_result1)

        # Test with a single argument with repeated dimensions
        D_var = cp.Variable(self.D_np.shape)
        expr2 = cp.einsum('iji->i', D_var) 
        problem2 = cp.Problem(cp.Minimize(cp.sum(expr2)), [D_var == self.D_np])
        problem2.solve()
        result2 = expr2.value
        expected_result2 = np.einsum('iji->i', self.D_np)
        self.assertEqual(problem2.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result2, expected_result2)

        # Test with repeated dimensions and multiple arguments
        expr3 = cp.einsum('ijik,lkl->l', F, D) 
        problem3 = cp.Problem(cp.Minimize(cp.sum(expr3)), [F == self.F_np])
        problem3.solve()
        result3 = expr3.value
        expected_result3 = np.einsum('ijik,lkl->l', self.F_np, self.D_np)
        self.assertEqual(problem3.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result3, expected_result3)

        # Test multi-argument einsum with compatible dimensions
        expr4 = cp.einsum('ij,jkl->ikl', A, B) 
        problem4 = cp.Problem(cp.Minimize(cp.sum(expr4)), [A == self.A_np])
        problem4.solve()
        result4 = expr4.value
        expected_result4 = np.einsum('ij,jkl->ikl', self.A_np, self.B_np)
        self.assertEqual(problem4.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result4, expected_result4)

        # Test einsum with 4 arguments
        expr5 = cp.einsum('ii,ijk,j,lil->ijl', A, B, C, D)
        problem5 = cp.Problem(cp.Minimize(cp.sum(expr5)), [A == self.A_np])
        problem5.solve()
        result5 = expr5.value
        expected_result5 = np.einsum(
            'ii,ijk,j,lil->ijl', self.A_np, self.B_np, self.C_np, self.D_np
        )
        self.assertEqual(problem5.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result5, expected_result5)

        # Test einsum DPP
        C_const = cp.Constant(self.C_np)
        expr6 = cp.einsum('ii,ijk,j,lil->ijl', A, B, C_const, D) 
        problem6 = cp.Problem(cp.Minimize(cp.sum(expr6)), [A == self.A_np])
        self.assertEqual(problem6.is_dpp(), True)
        problem6.solve()
        result6 = expr6.value
        expected_result6 = np.einsum(
            'ii,ijk,j,lil->ijl', self.A_np, self.B_np, self.C_np, self.D_np
        )
        self.assertEqual(problem6.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result6, expected_result6)

        expr7 = cp.einsum('ii,lil->i', A, D)
        problem7 = cp.Problem(cp.Minimize(cp.sum(expr7)), [A == self.A_np])
        problem7.solve()
        result7 = expr7.value
        expected_result7 = np.einsum(
            'ii,lil->i', self.A_np, self.D_np
        )
        self.assertEqual(problem7.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result7, expected_result7)

class TestEinsumDGP(BaseTest):
    """Unit tests for the DGP functionality of the einsum atom."""
    def test_einsum_dgp_solve(self) -> None:
        """Test solving einsum problems."""
        # Einsum with only 2d arrays
        A_pos_np = np.ones((4, 4))
        D_pos_np = np.ones((3,4,3))

        A_pos = cp.Variable(A_pos_np.shape, pos=True)
        D_pos = cp.Variable(D_pos_np.shape, pos=True)

        # Test einsum DGP
        expr1 = cp.einsum('ii,lil->i', A_pos, D_pos)
        problem1 = cp.Problem(cp.Minimize(cp.sum(expr1)), [A_pos == A_pos_np, D_pos == D_pos_np])
        self.assertEqual(problem1.is_dgp(), True)
        problem1.solve(gp=True)
        result1 = expr1.value
        expected_result1 = np.einsum(
            'ii,lil->i', A_pos_np, D_pos_np
        )
        self.assertEqual(problem1.status, cp.OPTIMAL)
        self.assertItemsAlmostEqual(result1, expected_result1)

if __name__ == '__main__':
    unittest.main()
