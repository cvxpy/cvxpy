import numpy as np

import cvxpy as cp


class TestJacProd:

    def test_prod_vector(self):
        """Test jacobian of prod for a vector."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, 3.0, 4.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # For prod(x) = x1 * x2 * x3, jacobian is [x2*x3, x1*x3, x1*x2]
        # = [12, 8, 6]
        correct_jacobian = np.array([[12.0, 8.0, 6.0]])
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_vector_with_ones(self):
        """Test jacobian of prod when some values are 1."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([1.0, 1.0, 1.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # For x = [1, 1, 1], jacobian is [1, 1, 1]
        correct_jacobian = np.array([[1.0, 1.0, 1.0]])
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_vector_with_zero(self):
        """Test jacobian of prod when one value is zero."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, 0.0, 4.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # For x = [2, 0, 4], jacobian is [0*4, 2*4, 2*0] = [0, 8, 0]
        correct_jacobian = np.array([[0.0, 8.0, 0.0]])
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_vector_with_two_zeros(self):
        """Test jacobian of prod when two values are zero."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([0.0, 0.0, 4.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # For x = [0, 0, 4], all partial derivatives are 0
        correct_jacobian = np.array([[0.0, 0.0, 0.0]])
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_vector_negative(self):
        """Test jacobian of prod with negative values."""
        n = 3
        x = cp.Variable((n,), name='x')
        x.value = np.array([2.0, -3.0, 4.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # For prod(x) = x1 * x2 * x3, jacobian is [x2*x3, x1*x3, x1*x2]
        # = [-12, 8, -6]
        correct_jacobian = np.array([[-12.0, 8.0, -6.0]])
        computed_jacobian = np.zeros((1, n))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_matrix(self):
        """Test jacobian of prod for a matrix (axis=None)."""
        x = cp.Variable((2, 2), name='x')
        x.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()

        # prod = 1*2*3*4 = 24
        # Flattened in F-order: [1, 3, 2, 4]
        # Jacobians: [3*2*4, 1*2*4, 1*3*4, 1*3*2] = [24, 8, 12, 6]
        correct_jacobian = np.array([[24.0, 8.0, 12.0, 6.0]])
        computed_jacobian = np.zeros((1, 4))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_axis0(self):
        """Test jacobian of prod with axis=0."""
        x = cp.Variable((2, 3), name='x')
        x.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        prod_expr = cp.prod(x, axis=0)
        result_dict = prod_expr.jacobian()

        # prod(x, axis=0) = [1*4, 2*5, 3*6] = [4, 10, 18]
        # Output shape: (3,), input shape: (2, 3)
        # d(prod_j)/d(x_ij) = x_{1-i,j}
        # In F-order, input is [1, 4, 2, 5, 3, 6]
        # Output in F-order is [4, 10, 18]
        # Jacobian should be 3x6
        correct_jacobian = np.array([
            [4.0, 1.0, 0.0, 0.0, 0.0, 0.0],  # d(out[0])/d(in)
            [0.0, 0.0, 5.0, 2.0, 0.0, 0.0],  # d(out[1])/d(in)
            [0.0, 0.0, 0.0, 0.0, 6.0, 3.0],  # d(out[2])/d(in)
        ])
        computed_jacobian = np.zeros((3, 6))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_axis1(self):
        """Test jacobian of prod with axis=1."""
        x = cp.Variable((2, 3), name='x')
        x.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        prod_expr = cp.prod(x, axis=1)
        result_dict = prod_expr.jacobian()

        # prod(x, axis=1) = [1*2*3, 4*5*6] = [6, 120]
        # Output shape: (2,), input shape: (2, 3)
        # In F-order, input is [1, 4, 2, 5, 3, 6]
        # Output in F-order is [6, 120]
        # Jacobian should be 2x6
        # d(out[0])/d(x[0,:]) = [2*3, 1*3, 1*2] = [6, 3, 2]
        # d(out[1])/d(x[1,:]) = [5*6, 4*6, 4*5] = [30, 24, 20]
        correct_jacobian = np.array([
            [6.0, 0.0, 3.0, 0.0, 2.0, 0.0],   # d(out[0])/d(in)
            [0.0, 30.0, 0.0, 24.0, 0.0, 20.0],  # d(out[1])/d(in)
        ])
        computed_jacobian = np.zeros((2, 6))
        rows, cols, vals = result_dict[x]
        computed_jacobian[rows, cols] = vals
        assert np.allclose(computed_jacobian, correct_jacobian)

    def test_prod_constant(self):
        """Test jacobian of prod with constant input."""
        x = np.array([1.0, 2.0, 3.0])
        prod_expr = cp.prod(x)
        result_dict = prod_expr.jacobian()
        assert result_dict == {}
