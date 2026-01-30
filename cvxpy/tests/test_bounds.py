"""
Copyright 2013 Steven Diamond

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
import pytest

import cvxpy as cp


class TestBoundsPropagation:
    """Tests for bounds propagation through expression trees."""

    # -------------------- Variable attribute bounds --------------------
    @pytest.mark.parametrize("attr,expected_lb,expected_ub", [
        ({}, -np.inf, np.inf),
        ({"nonneg": True}, 0, np.inf),
        ({"nonpos": True}, -np.inf, 0),
        ({"boolean": True}, 0, 1),
    ])
    def test_variable_attribute_bounds(self, attr, expected_lb, expected_ub) -> None:
        """Test variable bounds from attributes."""
        x = cp.Variable(3, **attr)
        lb, ub = x.get_bounds()
        assert lb.shape == (3,)
        assert ub.shape == (3,)
        assert np.all(lb == expected_lb)
        assert np.all(ub == expected_ub)

    def test_variable_with_bounds(self) -> None:
        """Test variable with explicit bounds (list and tuple)."""
        for bounds in [[0, 1], (0, 1)]:
            x = cp.Variable(3, bounds=bounds)
            lb, ub = x.get_bounds()
            assert np.allclose(lb, 0)
            assert np.allclose(ub, 1)

    def test_constant_bounds(self) -> None:
        """Test constant has exact bounds."""
        c = cp.Constant([1, 2, 3])
        lb, ub = c.get_bounds()
        assert np.allclose(lb, [1, 2, 3])
        assert np.allclose(ub, [1, 2, 3])

    def test_parameter_with_bounds(self) -> None:
        """Test parameter with explicit bounds."""
        p = cp.Parameter(3, bounds=[-1, 1])
        lb, ub = p.get_bounds()
        assert np.allclose(lb, -1)
        assert np.allclose(ub, 1)

    def test_matrix_variable_bounds(self) -> None:
        """Test bounds for matrix variables."""
        X = cp.Variable((2, 3), bounds=[0, 5])
        lb, ub = X.get_bounds()
        assert lb.shape == (2, 3)
        assert ub.shape == (2, 3)
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 5)

    # -------------------- Monotonic atoms --------------------
    @pytest.mark.parametrize("atom_fn,input_bounds,expected_lb,expected_ub", [
        (cp.exp, [0, 1], 1, np.e),
        (cp.log, [1, np.e], 0, 1),
        (cp.sqrt, [1, 4], 1, 2),
    ])
    def test_monotonic_atom_bounds(
        self, atom_fn, input_bounds, expected_lb, expected_ub
    ) -> None:
        """Test bounds propagation through monotonic atoms (exp, log, sqrt)."""
        x = cp.Variable(3, bounds=input_bounds)
        expr = atom_fn(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_power(self) -> None:
        """Test bounds propagation through power."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = x ** 2
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 4)

    # -------------------- Absolute value --------------------
    @pytest.mark.parametrize("bounds,expected_lb,expected_ub", [
        ([1, 2], 1, 2),       # nonneg
        ([-3, -1], 1, 3),     # nonpos
        ([-2, 3], 0, 3),      # spans zero
    ])
    def test_abs_bounds(self, bounds, expected_lb, expected_ub) -> None:
        """Test bounds for abs with various input bound ranges."""
        x = cp.Variable(3, bounds=bounds)
        expr = cp.abs(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_sign_refinement(self) -> None:
        """Test that sign information refines bounds."""
        x = cp.Variable(3, nonneg=True)
        expr = cp.abs(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.all(ub == np.inf)

    # -------------------- Affine operations --------------------
    def test_negation(self) -> None:
        """Test bounds propagation through negation."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = -x
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, -2)
        assert np.allclose(ub, -1)

    def test_addition(self) -> None:
        """Test bounds propagation through addition."""
        x = cp.Variable(3, bounds=[0, 1])
        y = cp.Variable(3, bounds=[2, 3])
        expr = x + y
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 2)
        assert np.allclose(ub, 4)

    @pytest.mark.parametrize("scalar,expected_lb,expected_ub", [
        (2, 2, 4),
        (-2, -4, -2),
    ])
    def test_scalar_multiplication(self, scalar, expected_lb, expected_ub) -> None:
        """Test bounds propagation through scalar multiplication."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = scalar * x
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_elementwise_multiplication(self) -> None:
        """Test bounds propagation through elementwise multiplication."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[3, 4])
        expr = cp.multiply(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 3)
        assert np.allclose(ub, 8)

    def test_multiply_spanning_zero(self) -> None:
        """Test multiplication with intervals spanning zero."""
        x = cp.Variable(3, bounds=[-1, 2])
        y = cp.Variable(3, bounds=[-3, 4])
        expr = cp.multiply(x, y)
        lb, ub = expr.get_bounds()
        # Products: (-1)*(-3)=3, (-1)*4=-4, 2*(-3)=-6, 2*4=8
        assert np.allclose(lb, -6)
        assert np.allclose(ub, 8)

    def test_division_positive(self) -> None:
        """Test bounds for division with positive divisor."""
        x = cp.Variable(3, bounds=[2, 4])
        y = cp.Variable(3, bounds=[1, 2])
        expr = x / y
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)  # 2/2 = 1
        assert np.allclose(ub, 4)  # 4/1 = 4

    # -------------------- Reshape / Transpose / Index --------------------
    def test_reshape(self) -> None:
        """Test bounds propagation through reshape."""
        x = cp.Variable((2, 3), bounds=[1, 6])
        expr = cp.reshape(x, (3, 2))
        lb, ub = expr.get_bounds()
        assert lb.shape == (3, 2)
        assert ub.shape == (3, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 6)

    def test_transpose(self) -> None:
        """Test bounds propagation through transpose."""
        x = cp.Variable((2, 3), bounds=[1, 6])
        expr = x.T
        lb, ub = expr.get_bounds()
        assert lb.shape == (3, 2)
        assert ub.shape == (3, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 6)

    @pytest.mark.parametrize("key,expected_shape", [
        (2, ()),           # single element
        (slice(1, 4), (3,)),  # slice
    ])
    def test_indexing_1d(self, key, expected_shape) -> None:
        """Test bounds propagation through 1D indexing."""
        x = cp.Variable(5, bounds=[0, 10])
        expr = x[key]
        lb, ub = expr.get_bounds()
        assert lb.shape == expected_shape
        assert ub.shape == expected_shape
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 10)

    def test_indexing_2d(self) -> None:
        """Test bounds propagation through 2D indexing."""
        x = cp.Variable((3, 4), bounds=[1, 5])
        expr = x[0:2, 1:3]
        lb, ub = expr.get_bounds()
        assert lb.shape == (2, 2)
        assert ub.shape == (2, 2)
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    def test_promote(self) -> None:
        """Test bounds propagation through promote (scalar to vector)."""
        x = cp.Variable(bounds=[2, 5])
        y = cp.Variable(3, bounds=[0, 0])
        expr = x + y  # This broadcasts x
        lb, ub = expr.get_bounds()
        assert lb.shape == (3,)
        assert np.allclose(lb, 2)
        assert np.allclose(ub, 5)

    # -------------------- Reductions --------------------
    @pytest.mark.parametrize("axis,expected_shape,expected_lb,expected_ub", [
        (None, (), 6, 12),  # full sum: 2*3*1=6, 2*3*2=12
        (0, (3,), 2, 4),    # sum along rows
    ])
    def test_sum(self, axis, expected_shape, expected_lb, expected_ub) -> None:
        """Test bounds propagation through sum."""
        x = cp.Variable((2, 3), bounds=[1, 2])
        expr = cp.sum(x) if axis is None else cp.sum(x, axis=axis)
        lb, ub = expr.get_bounds()
        assert lb.shape == expected_shape
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_max_reduction(self) -> None:
        """Test bounds propagation through max reduction."""
        x = cp.Variable((2, 3), bounds=[1, 5])
        expr = cp.max(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    def test_min_reduction(self) -> None:
        """Test bounds propagation through min reduction."""
        x = cp.Variable((2, 3), bounds=[1, 5])
        expr = cp.min(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 5)

    # -------------------- Maximum / Minimum --------------------
    def test_maximum(self) -> None:
        """Test bounds propagation through maximum."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[0, 3])
        expr = cp.maximum(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 3)

    def test_minimum(self) -> None:
        """Test bounds propagation through minimum."""
        x = cp.Variable(3, bounds=[1, 2])
        y = cp.Variable(3, bounds=[0, 3])
        expr = cp.minimum(x, y)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 2)

    # -------------------- Norms --------------------
    @pytest.mark.parametrize("bounds,expected_lb,expected_ub", [
        ([1, 2], 3, 6),     # nonneg: 3*1=3, 3*2=6
        ([-2, 1], 0, 6),    # spans zero: 0, 3*2=6
    ])
    def test_norm1(self, bounds, expected_lb, expected_ub) -> None:
        """Test bounds propagation through norm1."""
        x = cp.Variable(3, bounds=bounds)
        expr = cp.norm1(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_norm_inf(self) -> None:
        """Test bounds propagation through norm_inf."""
        x = cp.Variable(3, bounds=[1, 2])
        expr = cp.norm_inf(x)
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 2)

    # -------------------- Matrix multiplication --------------------
    def test_matmul_constant_matrix(self) -> None:
        """Test bounds propagation through constant @ variable multiplication."""
        A = np.array([[1, 2, 3], [4, 5, 6]])
        x = cp.Variable(3, bounds=[0, 1])
        expr = A @ x
        lb, ub = expr.get_bounds()
        assert lb.shape == (2,)
        assert ub.shape == (2,)
        assert np.allclose(lb, 0)
        assert np.allclose(ub, [6, 15])

    def test_matmul_variable_intervals(self) -> None:
        """Test bounds for matmul with two variable intervals returns unbounded."""
        A = cp.Variable((2, 3), bounds=[0, 1])
        B = cp.Variable((3, 4), bounds=[0, 1])
        expr = A @ B
        lb, ub = expr.get_bounds()
        assert lb.shape == (2, 4)
        assert ub.shape == (2, 4)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    # -------------------- Composed expressions --------------------
    def test_composed_expression(self) -> None:
        """Test bounds propagation through composed expressions."""
        x = cp.Variable(bounds=[0, 1])
        expr = 2 * x + 1
        lb, ub = expr.get_bounds()
        assert np.allclose(lb, 1)
        assert np.allclose(ub, 3)

    def test_complex_expression(self) -> None:
        """Test bounds propagation through a more complex expression."""
        x = cp.Variable(3, bounds=[0, 1])
        y = cp.Variable(3, bounds=[1, 2])
        expr = cp.abs(x - y)
        lb, ub = expr.get_bounds()
        # x - y has bounds [-2, 0], abs(x - y) has bounds [0, 2]
        assert np.allclose(lb, 0)
        assert np.allclose(ub, 2)

    # -------------------- Scalar bounds broadcast verification --------------------
    @pytest.mark.parametrize("atom_fn,input_bounds", [
        (cp.exp, [0, 1]),
        (cp.log, [1, np.e]),
        (cp.sqrt, [1, 4]),
        (cp.abs, [-1, 1]),
        (lambda x: -x, [-1, 1]),
        (lambda x: x.T, [-1, 1]),
    ])
    def test_scalar_bounds_broadcast(self, atom_fn, input_bounds) -> None:
        """Verify atoms maintain broadcast views with scalar bounds."""
        shape = (100, 100)
        x = cp.Variable(shape, bounds=input_bounds)
        expr = atom_fn(x)
        lb, ub = expr.get_bounds()
        assert lb.shape == expr.shape
        assert ub.shape == expr.shape


class TestBoundsUtilityFunctions:
    """Tests for the bounds utility functions."""

    def test_unbounded(self) -> None:
        """Test unbounded function."""
        from cvxpy.utilities.bounds import unbounded
        lb, ub = unbounded((2, 3))
        assert lb.shape == (2, 3)
        assert ub.shape == (2, 3)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    def test_scalar_bounds(self) -> None:
        """Test scalar_bounds function."""
        from cvxpy.utilities.bounds import scalar_bounds
        lb, ub = scalar_bounds(-1.0, 2.0)
        assert lb.shape == ()
        assert ub.shape == ()
        assert np.allclose(lb, -1.0)
        assert np.allclose(ub, 2.0)

    # -------------------- Arithmetic bounds --------------------
    @pytest.mark.parametrize("fn_name,args,expected_lb,expected_ub", [
        ("add_bounds", ([1, 2], [3, 4], [5, 6], [7, 8]), [6, 8], [10, 12]),
        ("neg_bounds", ([1, 2], [3, 4]), [-3, -4], [-1, -2]),
        ("scale_bounds", ([1, 2], [3, 4], 2.0), [2, 4], [6, 8]),
        ("scale_bounds", ([1, 2], [3, 4], -2.0), [-6, -8], [-2, -4]),
    ])
    def test_arithmetic_bounds(self, fn_name, args, expected_lb, expected_ub) -> None:
        """Test arithmetic bounds utility functions."""
        import cvxpy.utilities.bounds as bounds_mod
        fn = getattr(bounds_mod, fn_name)
        np_args = tuple(np.array(a) if isinstance(a, list) else a for a in args)
        lb, ub = fn(*np_args)
        assert np.allclose(lb, expected_lb)
        assert np.allclose(ub, expected_ub)

    def test_mul_bounds(self) -> None:
        """Test mul_bounds function."""
        from cvxpy.utilities.bounds import mul_bounds
        lb1, ub1 = np.array([1, 2]), np.array([3, 4])
        lb2, ub2 = np.array([5, 6]), np.array([7, 8])
        lb, ub = mul_bounds(lb1, ub1, lb2, ub2)
        assert np.allclose(lb, [5, 12])
        assert np.allclose(ub, [21, 32])

    @pytest.mark.parametrize("lb1,ub1,lb2,ub2,expected_lb,expected_ub,unbounded", [
        ([2, 4], [6, 8], [1, 2], [2, 4], [1, 1], [6, 4], False),
        ([1], [2], [-1], [1], None, None, True),  # divisor spans zero
    ])
    def test_div_bounds(
        self, lb1, ub1, lb2, ub2, expected_lb, expected_ub, unbounded
    ) -> None:
        """Test div_bounds function."""
        from cvxpy.utilities.bounds import div_bounds
        lb, ub = div_bounds(
            np.array(lb1), np.array(ub1), np.array(lb2), np.array(ub2)
        )
        if unbounded:
            assert np.all(lb == -np.inf)
            assert np.all(ub == np.inf)
        else:
            assert np.allclose(lb, expected_lb)
            assert np.allclose(ub, expected_ub)

    # -------------------- Elementwise bounds --------------------
    def test_abs_bounds(self) -> None:
        """Test abs_bounds function."""
        from cvxpy.utilities.bounds import abs_bounds
        lb = np.array([-2, 1])
        ub = np.array([3, 2])
        new_lb, new_ub = abs_bounds(lb, ub)
        assert np.allclose(new_lb, [0, 1])
        assert np.allclose(new_ub, [3, 2])

    @pytest.mark.parametrize("fn_name,lb,ub,expected_lb,expected_ub", [
        ("exp_bounds", [0, 1], [1, 2], [1, np.e], [np.e, np.e**2]),
        ("log_bounds", [1, np.e], [np.e, np.e**2], [0, 1], [1, 2]),
        ("sqrt_bounds", [1, 4], [4, 9], [1, 2], [2, 3]),
    ])
    def test_monotonic_bounds(
        self, fn_name, lb, ub, expected_lb, expected_ub
    ) -> None:
        """Test monotonic bounds utility functions (exp, log, sqrt)."""
        import cvxpy.utilities.bounds as bounds_mod
        fn = getattr(bounds_mod, fn_name)
        new_lb, new_ub = fn(np.array(lb), np.array(ub))
        assert np.allclose(new_lb, expected_lb)
        assert np.allclose(new_ub, expected_ub)

    # -------------------- Power bounds --------------------
    @pytest.mark.parametrize("lb,ub,p,expected_lb,expected_ub", [
        ([2, -3], [4, -1], 2.0, [4, 1], [16, 9]),           # square
        ([-2], [3], 2.0, [0], [9]),                         # square spanning zero
        ([1, 4], [4, 9], 0.5, [1, 2], [2, 3]),              # sqrt
    ])
    def test_power_bounds(self, lb, ub, p, expected_lb, expected_ub) -> None:
        """Test power_bounds function."""
        from cvxpy.utilities.bounds import power_bounds
        new_lb, new_ub = power_bounds(np.array(lb), np.array(ub), p)
        assert np.allclose(new_lb, expected_lb)
        assert np.allclose(new_ub, expected_ub)

    # -------------------- Maximum / Minimum bounds --------------------
    def test_maximum_bounds(self) -> None:
        """Test maximum_bounds function."""
        from cvxpy.utilities.bounds import maximum_bounds
        bounds_list = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([0, 5]), np.array([2, 6])),
        ]
        lb, ub = maximum_bounds(bounds_list)
        assert np.allclose(lb, [1, 5])
        assert np.allclose(ub, [3, 6])

    def test_minimum_bounds(self) -> None:
        """Test minimum_bounds function."""
        from cvxpy.utilities.bounds import minimum_bounds
        bounds_list = [
            (np.array([1, 2]), np.array([3, 4])),
            (np.array([0, 5]), np.array([2, 6])),
        ]
        lb, ub = minimum_bounds(bounds_list)
        assert np.allclose(lb, [0, 2])
        assert np.allclose(ub, [2, 4])

    # -------------------- Reduction bounds --------------------
    @pytest.mark.parametrize("fn_name,expected_lb,expected_ub", [
        ("sum_bounds", 10, 26),
        ("max_reduction_bounds", 4, 8),
        ("min_reduction_bounds", 1, 5),
    ])
    def test_reduction_bounds(self, fn_name, expected_lb, expected_ub) -> None:
        """Test reduction bounds utility functions."""
        import cvxpy.utilities.bounds as bounds_mod
        fn = getattr(bounds_mod, fn_name)
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = fn(lb, ub, axis=None)
        assert np.allclose(new_lb, expected_lb)
        assert np.allclose(new_ub, expected_ub)

    def test_norm1_bounds(self) -> None:
        """Test norm1_bounds function."""
        from cvxpy.utilities.bounds import norm1_bounds
        lb = np.array([1, 2, 3])
        ub = np.array([2, 3, 4])
        new_lb, new_ub = norm1_bounds(lb, ub)
        assert np.allclose(new_lb, 6)
        assert np.allclose(new_ub, 9)

    def test_norm_inf_bounds(self) -> None:
        """Test norm_inf_bounds function."""
        from cvxpy.utilities.bounds import norm_inf_bounds
        lb = np.array([1, 2, 3])
        ub = np.array([2, 5, 4])
        new_lb, new_ub = norm_inf_bounds(lb, ub)
        assert np.allclose(new_lb, 3)
        assert np.allclose(new_ub, 5)

    # -------------------- Shape manipulation bounds --------------------
    def test_broadcast_bounds(self) -> None:
        """Test broadcast_bounds function."""
        from cvxpy.utilities.bounds import broadcast_bounds
        lb, ub = np.array([1]), np.array([2])
        new_lb, new_ub = broadcast_bounds(lb, ub, (3, 4))
        assert new_lb.shape == (3, 4)
        assert new_ub.shape == (3, 4)
        assert np.allclose(new_lb, 1)
        assert np.allclose(new_ub, 2)

    def test_reshape_bounds(self) -> None:
        """Test reshape_bounds function (Fortran order)."""
        from cvxpy.utilities.bounds import reshape_bounds
        lb = np.array([[1, 2], [3, 4]])
        ub = np.array([[5, 6], [7, 8]])
        new_lb, new_ub = reshape_bounds(lb, ub, (4,))
        assert new_lb.shape == (4,)
        assert np.allclose(new_lb, [1, 3, 2, 4])
        assert np.allclose(new_ub, [5, 7, 6, 8])

    def test_transpose_bounds(self) -> None:
        """Test transpose_bounds function."""
        from cvxpy.utilities.bounds import transpose_bounds
        lb = np.array([[1, 2, 3], [4, 5, 6]])
        ub = np.array([[7, 8, 9], [10, 11, 12]])
        new_lb, new_ub = transpose_bounds(lb, ub)
        assert new_lb.shape == (3, 2)
        assert np.allclose(new_lb, [[1, 4], [2, 5], [3, 6]])
        assert np.allclose(new_ub, [[7, 10], [8, 11], [9, 12]])

    def test_index_bounds(self) -> None:
        """Test index_bounds function."""
        from cvxpy.utilities.bounds import index_bounds
        lb = np.array([1, 2, 3, 4, 5])
        ub = np.array([6, 7, 8, 9, 10])
        new_lb, new_ub = index_bounds(lb, ub, slice(1, 4))
        assert new_lb.shape == (3,)
        assert np.allclose(new_lb, [2, 3, 4])
        assert np.allclose(new_ub, [7, 8, 9])

    # -------------------- Matrix multiplication bounds --------------------
    def test_matmul_bounds_constant_left(self) -> None:
        """Test matmul_bounds with constant left operand."""
        from cvxpy.utilities.bounds import matmul_bounds
        A = np.array([[1, 2], [3, 4]])
        lb2 = np.array([[1, 1], [1, 1]])
        ub2 = np.array([[2, 2], [2, 2]])
        lb, ub = matmul_bounds(A, A, lb2, ub2)
        assert lb.shape == (2, 2)
        assert np.allclose(lb, A @ lb2)
        assert np.allclose(ub, A @ ub2)

    def test_matmul_bounds_both_intervals(self) -> None:
        """Test matmul_bounds returns unbounded when both are intervals."""
        from cvxpy.utilities.bounds import matmul_bounds
        lb1 = np.array([[1, 2], [3, 4]])
        ub1 = np.array([[2, 3], [4, 5]])
        lb2 = np.array([[1, 1], [1, 1]])
        ub2 = np.array([[2, 2], [2, 2]])
        lb, ub = matmul_bounds(lb1, ub1, lb2, ub2)
        assert np.all(lb == -np.inf)
        assert np.all(ub == np.inf)

    # -------------------- Sign refinement --------------------
    @pytest.mark.parametrize("is_nonneg,is_nonpos,expected_lb,expected_ub", [
        (True, False, [0, 0], [3, 4]),
        (False, True, [-1, -2], [0, 0]),
    ])
    def test_refine_bounds_from_sign(
        self, is_nonneg, is_nonpos, expected_lb, expected_ub
    ) -> None:
        """Test refine_bounds_from_sign function."""
        from cvxpy.utilities.bounds import refine_bounds_from_sign
        lb = np.array([-1, -2])
        ub = np.array([3, 4])
        new_lb, new_ub = refine_bounds_from_sign(lb, ub, is_nonneg, is_nonpos)
        assert np.allclose(new_lb, expected_lb)
        assert np.allclose(new_ub, expected_ub)


class TestExtractBounds:
    """Tests for extract_lower_bounds / extract_upper_bounds in matrix_stuffing."""

    @pytest.mark.parametrize("attr,bounds,expected_lb,expected_ub", [
        ({"nonneg": True}, [2, 5], 2, 5),   # nonneg, explicit tighter lb
        ({"nonpos": True}, [-5, -2], -5, -2),  # nonpos, explicit tighter ub
    ])
    def test_attribute_with_explicit_bounds(
        self, attr, bounds, expected_lb, expected_ub
    ) -> None:
        """Test that attribute + explicit bounds uses tighter bound."""
        from cvxpy.reductions.matrix_stuffing import (
            extract_lower_bounds,
            extract_upper_bounds,
        )

        x = cp.Variable(3, **attr, bounds=bounds)
        variables = [x]
        var_size = x.size

        lb = extract_lower_bounds(variables, var_size)
        ub = extract_upper_bounds(variables, var_size)

        assert lb is not None
        assert np.allclose(lb, expected_lb)
        assert ub is not None
        assert np.allclose(ub, expected_ub)


class TestBoundsMemoryOptimization:
    """Test memory optimization for bounds propagation."""

    @pytest.mark.parametrize("attr,expected_lb,expected_ub", [
        ({}, -np.inf, np.inf),
        ({"nonneg": True}, 0, np.inf),
        ({"boolean": True}, 0, 1),
        ({"bounds": (-5, 10)}, -5, 10),
    ])
    def test_uniform_bounds_use_broadcast(self, attr, expected_lb, expected_ub) -> None:
        """Test that variables with uniform bounds use broadcast views (0 strides)."""
        x = cp.Variable((100, 100), **attr)
        lb, ub = x.get_bounds()

        assert np.all(lb == expected_lb)
        assert np.all(ub == expected_ub)
        assert lb.strides == (0, 0)
        assert ub.strides == (0, 0)

    def test_array_bounds_do_not_use_broadcast(self) -> None:
        """Test that array bounds don't use broadcast (different values)."""
        lb_values = np.arange(6).reshape(2, 3)
        ub_values = np.arange(6, 12).reshape(2, 3)
        x = cp.Variable((2, 3), bounds=(lb_values, ub_values))
        lb, ub = x.get_bounds()

        assert np.allclose(lb, lb_values)
        assert np.allclose(ub, ub_values)
        assert lb.strides != (0, 0)
        assert ub.strides != (0, 0)
