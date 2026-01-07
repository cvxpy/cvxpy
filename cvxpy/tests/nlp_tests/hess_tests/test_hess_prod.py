import numpy as np

import cvxpy as cp


class TestHessVecProd:

    def test_prod_vector(self):
        """Test hess_vec of prod for a vector."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, 3.0, 4.0])
        vec = np.array([1.0])  # Scalar output, so vec has size 1
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # For prod(x) = x1 * x2 * x3, Hessian H[i,j] = prod / (xi * xj) for i != j
        # H[i,i] = 0
        # prod = 24, so H = [[0, 4, 3], [4, 0, 2], [3, 2, 0]]
        correct_hess = vec[0] * np.array([
            [0.0, 4.0, 3.0],
            [4.0, 0.0, 2.0],
            [3.0, 2.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_vector_with_weight(self):
        """Test hess_vec of prod with non-unit weight."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, 3.0, 4.0])
        vec = np.array([5.0])  # Weight of 5
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # Hessian scaled by vec[0] = 5
        correct_hess = 5.0 * np.array([
            [0.0, 4.0, 3.0],
            [4.0, 0.0, 2.0],
            [3.0, 2.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_vector_with_zero(self):
        """Test hess_vec of prod when one value is zero."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, 0.0, 4.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # With one zero at index 1:
        # H[1, j] = prod of all except 1 and j = nonzero only for j != 1
        # H[1, 0] = 4, H[1, 2] = 2
        # H[i, 1] = same values
        # All other entries are 0 because they still contain the zero
        correct_hess = np.array([
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 2.0],
            [0.0, 2.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_vector_with_two_zeros(self):
        """Test hess_vec of prod when two values are zero."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([0.0, 0.0, 4.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # With two zeros at indices 0 and 1:
        # Only H[0, 1] = H[1, 0] = 4 (prod of all except both zeros)
        correct_hess = np.array([
            [0.0, 4.0, 0.0],
            [4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_vector_with_three_zeros(self):
        """Test hess_vec of prod when all values are zero."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([0.0, 0.0, 0.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # With three zeros, all Hessian entries are 0
        correct_hess = np.zeros((n, n))
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_vector_negative(self):
        """Test hess_vec of prod with negative values."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, -3.0, 4.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # prod = -24
        # H[i,j] = prod / (xi * xj) for i != j
        # H[0,1] = -24 / (2 * -3) = 4
        # H[0,2] = -24 / (2 * 4) = -3
        # H[1,2] = -24 / (-3 * 4) = 2
        correct_hess = np.array([
            [0.0, 4.0, -3.0],
            [4.0, 0.0, 2.0],
            [-3.0, 2.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_matrix(self):
        """Test hess_vec of prod for a matrix (axis=None)."""
        x = cp.Variable((2, 2), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # prod = 24, flattened in F-order: [1, 3, 2, 4]
        # H[i,j] = prod / (xi * xj) for i != j
        vals_flat = np.array([1.0, 3.0, 2.0, 4.0])
        n = 4
        correct_hess = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                if i != j:
                    correct_hess[i, j] = 24.0 / (vals_flat[i] * vals_flat[j])

        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_axis1(self):
        """Test hess_vec of prod with axis=1."""
        x = cp.Variable((2, 3), name='x')
        x.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        # Output shape is (2,), so vec has size 2
        vec = np.array([1.0, 1.0])
        prod_expr = cp.prod(x, axis=1)
        result_dict = prod_expr.hess_vec(vec)

        # prod(x, axis=1) = [6, 120]
        # Each row contributes independently to its output
        # For row 0: [1, 2, 3] -> H entries for indices corresponding to row 0
        # For row 1: [4, 5, 6] -> H entries for indices corresponding to row 1
        # In F-order, input is [1, 4, 2, 5, 3, 6]
        # Row 0 uses indices 0, 2, 4; Row 1 uses indices 1, 3, 5
        n = 6
        correct_hess = np.zeros((n, n))
        # Row 0: vals = [1, 2, 3], prod = 6
        row0_indices = [0, 2, 4]
        row0_vals = [1.0, 2.0, 3.0]
        for i, ii in enumerate(row0_indices):
            for j, jj in enumerate(row0_indices):
                if i != j:
                    correct_hess[ii, jj] = vec[0] * 6.0 / (row0_vals[i] * row0_vals[j])
        # Row 1: vals = [4, 5, 6], prod = 120
        row1_indices = [1, 3, 5]
        row1_vals = [4.0, 5.0, 6.0]
        for i, ii in enumerate(row1_indices):
            for j, jj in enumerate(row1_indices):
                if i != j:
                    correct_hess[ii, jj] = vec[1] * 120.0 / (row1_vals[i] * row1_vals[j])

        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)

    def test_prod_constant(self):
        """Test hess_vec of prod with constant input."""
        x = np.array([1.0, 2.0, 3.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)
        assert result_dict == {}

    def test_prod_two_elements(self):
        """Test hess_vec of prod with only two elements."""
        n = 2
        x = cp.Variable((n,), name='x')
        x.value = np.array([3.0, 5.0])
        vec = np.array([1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.hess_vec(vec)

        # For prod(x) = x1 * x2, H = [[0, 1], [1, 0]]
        correct_hess = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ])
        computed_hess = np.zeros((n, n))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess[rows, cols] = vals
        assert np.allclose(computed_hess, correct_hess)
