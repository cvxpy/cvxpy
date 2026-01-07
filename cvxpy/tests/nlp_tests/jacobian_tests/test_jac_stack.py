import numpy as np

import cvxpy as cp


class TestJacHstack:
    """Tests for the Jacobian of hstack atom."""

    def test_hstack_vectors(self):
        """Test hstack of two 1D vector variables."""
        x = cp.Variable((3,), name='x')
        y = cp.Variable((2,), name='y')
        x.value = np.array([1.0, 2.0, 3.0])
        y.value = np.array([4.0, 5.0])

        # hstack([log(x), log(y)]) has shape (5,) - 1D concatenation
        expr = cp.hstack([cp.log(x), cp.log(y)])
        assert expr.shape == (5,)
        result_dict = expr.jacobian()

        # Jacobian w.r.t. x: first 3 output elements depend on x
        correct_jac_x = np.zeros((5, 3))
        correct_jac_x[:3, :] = np.diag(1 / x.value)

        computed_jac_x = np.zeros((5, 3))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals
        assert np.allclose(computed_jac_x, correct_jac_x)

        # Jacobian w.r.t. y: last 2 output elements depend on y
        correct_jac_y = np.zeros((5, 2))
        correct_jac_y[3:, :] = np.diag(1 / y.value)

        computed_jac_y = np.zeros((5, 2))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals
        assert np.allclose(computed_jac_y, correct_jac_y)

    def test_hstack_matrices(self):
        """Test hstack of two matrix variables."""
        x = cp.Variable((2, 3), name='x')
        y = cp.Variable((2, 2), name='y')
        x.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y.value = np.array([[7.0, 8.0], [9.0, 10.0]])

        # hstack([log(x), log(y)]) has shape (2, 5)
        expr = cp.hstack([cp.log(x), cp.log(y)])
        assert expr.shape == (2, 5)
        result_dict = expr.jacobian()

        # Output flat order (Fortran): columns 0,1,2 from x, then columns 3,4 from y
        # x columns: [(1,4), (2,5), (3,6)] -> flat indices 0,1,2,3,4,5
        # y columns: [(7,9), (8,10)] -> flat indices 6,7,8,9

        # For x: output flat index = input flat index + 0 (offset=0)
        computed_jac_x = np.zeros((10, 6))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals

        # x at flat index i maps to output flat index i
        correct_jac_x = np.zeros((10, 6))
        for i in range(6):
            correct_jac_x[i, i] = 1 / x.value.flatten('F')[i]
        assert np.allclose(computed_jac_x, correct_jac_x)

        # For y: output flat index = input flat index + 6 (col_offset=3, nrows=2 => 3*2=6)
        computed_jac_y = np.zeros((10, 4))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals

        correct_jac_y = np.zeros((10, 4))
        for i in range(4):
            correct_jac_y[i + 6, i] = 1 / y.value.flatten('F')[i]
        assert np.allclose(computed_jac_y, correct_jac_y)

    def test_hstack_single_variable(self):
        """Test hstack with a single variable split across terms."""
        x = cp.Variable((3,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        # hstack([log(x), exp(x)])
        expr = cp.hstack([cp.log(x), cp.exp(x)])
        result_dict = expr.jacobian()

        # Jacobian w.r.t. x: first 3 rows from log, next 3 from exp
        correct_jac = np.zeros((6, 3))
        correct_jac[:3, :] = np.diag(1 / x.value)  # d(log x)/dx
        correct_jac[3:, :] = np.diag(np.exp(x.value))  # d(exp x)/dx

        computed_jac = np.zeros((6, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals
        assert np.allclose(computed_jac, correct_jac)


class TestJacVstack:
    """Tests for the Jacobian of vstack atom."""

    def test_vstack_vectors(self):
        """Test vstack of two 1D vector variables with same size.

        vstack treats 1D vectors as rows, so they must have the same length.
        """
        x = cp.Variable((3,), name='x')
        y = cp.Variable((3,), name='y')
        x.value = np.array([1.0, 2.0, 3.0])
        y.value = np.array([4.0, 5.0, 6.0])

        # vstack([log(x), log(y)]) has shape (2, 3) - stacks as rows
        expr = cp.vstack([cp.log(x), cp.log(y)])
        assert expr.shape == (2, 3)
        result_dict = expr.jacobian()

        # Output in Fortran order: [(row0,col0), (row1,col0), (row0,col1), ...]
        # = [log(x[0]), log(y[0]), log(x[1]), log(y[1]), log(x[2]), log(y[2])]
        # Output flat indices: x[i] -> 2*i, y[i] -> 2*i+1

        # Jacobian w.r.t. x: output indices 0, 2, 4 depend on x[0], x[1], x[2]
        computed_jac_x = np.zeros((6, 3))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals

        correct_jac_x = np.zeros((6, 3))
        for i in range(3):
            correct_jac_x[2 * i, i] = 1 / x.value[i]
        assert np.allclose(computed_jac_x, correct_jac_x)

        # Jacobian w.r.t. y: output indices 1, 3, 5 depend on y[0], y[1], y[2]
        computed_jac_y = np.zeros((6, 3))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals

        correct_jac_y = np.zeros((6, 3))
        for i in range(3):
            correct_jac_y[2 * i + 1, i] = 1 / y.value[i]
        assert np.allclose(computed_jac_y, correct_jac_y)

    def test_vstack_matrices(self):
        """Test vstack of two matrix variables."""
        x = cp.Variable((2, 3), name='x')
        y = cp.Variable((2, 3), name='y')
        x.value = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y.value = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])

        # vstack([log(x), log(y)]) has shape (4, 3)
        expr = cp.vstack([cp.log(x), cp.log(y)])
        assert expr.shape == (4, 3)
        result_dict = expr.jacobian()

        # Output shape is (4, 3), total size = 12
        # In Fortran order, for output element at (r, c), flat index = r + c * 4
        # For x element at (r, c), input flat index = r + c * 2
        # Output flat index = (r + 0) + c * 4 = r + c * 4

        computed_jac_x = np.zeros((12, 6))
        rows, cols, vals = result_dict[x]
        computed_jac_x[rows, cols] = vals

        # Build correct jacobian for x
        correct_jac_x = np.zeros((12, 6))
        for c in range(3):
            for r in range(2):
                in_idx = r + c * 2
                out_idx = r + c * 4  # row_offset=0
                correct_jac_x[out_idx, in_idx] = 1 / x.value[r, c]
        assert np.allclose(computed_jac_x, correct_jac_x)

        # For y: row_offset = 2, so output row = r + 2
        computed_jac_y = np.zeros((12, 6))
        rows, cols, vals = result_dict[y]
        computed_jac_y[rows, cols] = vals

        correct_jac_y = np.zeros((12, 6))
        for c in range(3):
            for r in range(2):
                in_idx = r + c * 2
                out_idx = (r + 2) + c * 4  # row_offset=2
                correct_jac_y[out_idx, in_idx] = 1 / y.value[r, c]
        assert np.allclose(computed_jac_y, correct_jac_y)

    def test_vstack_single_variable(self):
        """Test vstack with a single variable split across terms.

        vstack([log(x), exp(x)]) where x is (3,) gives shape (2, 3).
        In Fortran order: [log(x[0]), exp(x[0]), log(x[1]), exp(x[1]), log(x[2]), exp(x[2])]
        """
        x = cp.Variable((3,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        # vstack([log(x), exp(x)]) has shape (2, 3)
        expr = cp.vstack([cp.log(x), cp.exp(x)])
        assert expr.shape == (2, 3)
        result_dict = expr.jacobian()

        # Jacobian w.r.t. x: interleaved log/exp rows
        # Output flat indices: log(x[i]) -> 2*i, exp(x[i]) -> 2*i+1
        correct_jac = np.zeros((6, 3))
        for i in range(3):
            correct_jac[2 * i, i] = 1 / x.value[i]  # d(log x[i])/dx[i]
            correct_jac[2 * i + 1, i] = np.exp(x.value[i])  # d(exp x[i])/dx[i]

        computed_jac = np.zeros((6, 3))
        rows, cols, vals = result_dict[x]
        computed_jac[rows, cols] = vals
        assert np.allclose(computed_jac, correct_jac)


class TestJacBmat:
    """Tests for the Jacobian of bmat (block matrix) operation."""

    def test_bmat_2x2(self):
        """Test 2x2 block matrix."""
        A = cp.Variable((2, 2), name='A')
        B = cp.Variable((2, 3), name='B')
        C = cp.Variable((3, 2), name='C')
        D = cp.Variable((3, 3), name='D')

        A.value = np.array([[1.0, 2.0], [3.0, 4.0]])
        B.value = np.array([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        C.value = np.array([[11.0, 12.0], [13.0, 14.0], [15.0, 16.0]])
        D.value = np.array([[17.0, 18.0, 19.0], [20.0, 21.0, 22.0], [23.0, 24.0, 25.0]])

        # bmat([[log(A), log(B)], [log(C), log(D)]]) has shape (5, 5)
        expr = cp.bmat([[cp.log(A), cp.log(B)], [cp.log(C), cp.log(D)]])
        assert expr.shape == (5, 5)
        result_dict = expr.jacobian()

        # Check A's jacobian contribution
        computed_jac_A = np.zeros((25, 4))
        rows, cols, vals = result_dict[A]
        computed_jac_A[rows, cols] = vals

        # A is at top-left 2x2 block
        # In output, A[r,c] is at position (r, c)
        # Output flat index = r + c * 5
        # Input flat index = r + c * 2
        correct_jac_A = np.zeros((25, 4))
        for c in range(2):
            for r in range(2):
                in_idx = r + c * 2
                out_idx = r + c * 5
                correct_jac_A[out_idx, in_idx] = 1 / A.value[r, c]
        assert np.allclose(computed_jac_A, correct_jac_A)

        # Check D's jacobian contribution
        computed_jac_D = np.zeros((25, 9))
        rows, cols, vals = result_dict[D]
        computed_jac_D[rows, cols] = vals

        # D is at bottom-right 3x3 block
        # In output, D[r,c] is at position (r+2, c+2)
        # Output flat index = (r+2) + (c+2) * 5
        correct_jac_D = np.zeros((25, 9))
        for c in range(3):
            for r in range(3):
                in_idx = r + c * 3
                out_idx = (r + 2) + (c + 2) * 5
                correct_jac_D[out_idx, in_idx] = 1 / D.value[r, c]
        assert np.allclose(computed_jac_D, correct_jac_D)


class TestHessStack:
    """Tests for the hess_vec of hstack/vstack/bmat operations."""

    def test_hstack_hess_vec(self):
        """Test hess_vec for hstack with 1D vectors."""
        x = cp.Variable((3,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        # hstack([log(x), exp(x)]) - both have non-trivial Hessians
        # Output shape is (6,)
        expr = cp.hstack([cp.log(x), cp.exp(x)])
        assert expr.shape == (6,)
        vec = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        result_dict = expr.hess_vec(vec)

        # Hessian of log(x_i) is -1/x_i^2, exp(x_i) is exp(x_i)
        # vec[:3] applies to log part, vec[3:] applies to exp part
        expected_diag = -vec[:3] / x.value**2 + vec[3:] * np.exp(x.value)

        # Duplicates are now summed by the implementation
        rows, cols, vals = result_dict[(x, x)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals
        assert np.allclose(np.diag(computed_hess), expected_diag)

    def test_vstack_hess_vec(self):
        """Test hess_vec for vstack with 1D vectors.

        vstack([log(x), exp(x)]) where x is (3,) gives shape (2, 3).
        The output is interleaved: [log(x[0]), exp(x[0]), log(x[1]), exp(x[1]), ...]
        """
        x = cp.Variable((3,), name='x')
        x.value = np.array([1.0, 2.0, 3.0])

        # vstack([log(x), exp(x)]) has shape (2, 3)
        expr = cp.vstack([cp.log(x), cp.exp(x)])
        assert expr.shape == (2, 3)

        # In Fortran order: [log(x[0]), exp(x[0]), log(x[1]), exp(x[1]), log(x[2]), exp(x[2])]
        vec = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 2.0])
        result_dict = expr.hess_vec(vec)

        # vec elements at indices 0, 2, 4 (log part) apply to log
        # vec elements at indices 1, 3, 5 (exp part) apply to exp
        log_vec = vec[::2]  # [1, 1, 1]
        exp_vec = vec[1::2]  # [2, 2, 2]
        expected_diag = -log_vec / x.value**2 + exp_vec * np.exp(x.value)

        # Duplicates are now summed by the implementation
        rows, cols, vals = result_dict[(x, x)]
        computed_hess = np.zeros((3, 3))
        computed_hess[rows, cols] = vals
        assert np.allclose(np.diag(computed_hess), expected_diag)

    def test_bmat_hess_vec(self):
        """Test hess_vec for bmat with different variables.

        bmat([[log(x)], [log(y)]]) is equivalent to vstack([hstack([log(x)]), hstack([log(y)])]).
        With x, y each (2,), the hstacks produce (2,) each, then vstack gives (2, 2).
        In Fortran order: [log(x[0]), log(y[0]), log(x[1]), log(y[1])].
        """
        x = cp.Variable((2,), name='x')
        y = cp.Variable((2,), name='y')
        x.value = np.array([1.0, 2.0])
        y.value = np.array([3.0, 4.0])

        # bmat([[log(x)], [log(y)]]) has shape (2, 2)
        expr = cp.bmat([[cp.log(x)], [cp.log(y)]])
        assert expr.shape == (2, 2)

        # In Fortran order: [log(x[0]), log(y[0]), log(x[1]), log(y[1])]
        vec = np.array([1.0, 2.0, 1.0, 2.0])
        result_dict = expr.hess_vec(vec)

        # For log(x), Hessian diag is -vec_i/x_i^2
        # x[0] gets vec[0]=1, x[1] gets vec[2]=1
        # y[0] gets vec[1]=2, y[1] gets vec[3]=2
        expected_diag_x = -np.array([vec[0], vec[2]]) / x.value**2
        expected_diag_y = -np.array([vec[1], vec[3]]) / y.value**2

        computed_hess_x = np.zeros((2, 2))
        rows, cols, vals = result_dict[(x, x)]
        computed_hess_x[rows, cols] = vals
        assert np.allclose(np.diag(computed_hess_x), expected_diag_x)

        computed_hess_y = np.zeros((2, 2))
        rows, cols, vals = result_dict[(y, y)]
        computed_hess_y[rows, cols] = vals
        assert np.allclose(np.diag(computed_hess_y), expected_diag_y)
