from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.settings as s
from cvxpy.lin_ops.backends import (
    CooCanonBackend,
    CooTensor,
    PythonCanonBackend,
    SciPyCanonBackend,
    TensorRepresentation,
    get_backend,
)
from cvxpy.lin_ops.backends.coo_backend import (
    _build_interleaved_mul,
    _build_interleaved_rmul,
    _kron_eye_l,
    _kron_eye_r,
    _kron_nd_structure_mul,
    _kron_nd_structure_rmul,
)
from cvxpy.lin_ops.backends.scipy_backend import (
    _apply_nd_kron_structure_mul,
    _apply_nd_kron_structure_rmul,
    _build_interleaved_matrix_mul,
    _build_interleaved_matrix_rmul,
    _expand_parametric_slices_mul,
    _expand_parametric_slices_rmul,
)


@dataclass
class linOpHelper:
    """
    Helper class that allows to access properties of linOps without
    needing to create a full linOps instance
    """

    shape: None | tuple[int, ...] = None
    type: None | str = None
    data: None | int | np.ndarray | list[slice] = None
    args: None | list[linOpHelper] = None


def test_tensor_representation():
    A = TensorRepresentation(
        np.array([10]), np.array([0]), np.array([1]), np.array([0]), shape=(2, 2)
    )
    B = TensorRepresentation(
        np.array([20]), np.array([1]), np.array([1]), np.array([1]), shape=(2, 2)
    )
    combined = TensorRepresentation.combine([A, B])
    assert np.all(combined.data == np.array([10, 20]))
    assert np.all(combined.row == np.array([0, 1]))
    assert np.all(combined.col == np.array([1, 1]))
    assert np.all(combined.parameter_offset == np.array([0, 1]))
    assert combined.shape == (2, 2)
    flattened = combined.flatten_tensor(2)
    assert np.all(flattened.toarray() == np.array([[0, 0], [0, 0], [10, 0], [0, 20]]))


@pytest.mark.parametrize("backend_name", [s.SCIPY_CANON_BACKEND, s.COO_CANON_BACKEND])
def test_build_matrix_order(backend_name):
    """Test that build_matrix respects the order argument for both backends."""
    kwargs = {
        "id_to_col": {1: 0},
        "param_to_size": {-1: 1},
        "param_to_col": {-1: 0},
        "param_size_plus_one": 1,
        "var_length": 2,
    }
    backend = get_backend(backend_name, **kwargs)

    # Simple variable linop of shape (2,) - creates a 2x2 identity in the variable columns
    # Variable id=1 maps to column offset 0, so entries go in columns 0 and 1
    lin_op = linOpHelper(shape=(2,), type="variable", data=1, args=[])

    # Test F order (column-major)
    f_result = backend.build_matrix([lin_op], order='F')

    # Test C order (row-major)
    c_result = backend.build_matrix([lin_op], order='C')

    # Both results should have shape (total_rows * (var_length + 1), param_size_plus_one)
    # total_rows = 2, var_length + 1 = 3, param_size_plus_one = 1
    # So shape is (6, 1)
    assert f_result.shape == (6, 1)
    assert c_result.shape == (6, 1)

    # The underlying tensor has entries at (row=0, col=0) and (row=1, col=1)
    # with shape (2, 3).
    #
    # In F order (column-major): flat_row = col * num_rows + row
    #   (0, 0) -> 0*2 + 0 = 0
    #   (1, 1) -> 1*2 + 1 = 3
    #
    # In C order (row-major): flat_row = col + row * num_cols
    #   (0, 0) -> 0 + 0*3 = 0
    #   (1, 1) -> 1 + 1*3 = 4
    f_dense = f_result.toarray().flatten()
    c_dense = c_result.toarray().flatten()

    # F order: entries at indices 0 and 3
    expected_f = np.array([1., 0., 0., 1., 0., 0.])
    np.testing.assert_array_equal(f_dense, expected_f)

    # C order: entries at indices 0 and 4
    expected_c = np.array([1., 0., 0., 0., 1., 0.])
    np.testing.assert_array_equal(c_dense, expected_c)

    # Test invalid order
    with pytest.raises(ValueError, match="order must be 'F' or 'C'"):
        backend.build_matrix([lin_op], order='INVALID')


class TestBackendInstance:
    def test_get_backend(self):
        args = ({1: 0, 2: 2}, {-1: 1, 3: 1}, {3: 0, -1: 1}, 2, 4)

        backend = get_backend(s.SCIPY_CANON_BACKEND, *args)
        assert isinstance(backend, SciPyCanonBackend)

        backend = get_backend(s.COO_CANON_BACKEND, *args)
        assert isinstance(backend, CooCanonBackend)

        with pytest.raises(KeyError):
            get_backend("notabackend")


backends = [s.SCIPY_CANON_BACKEND, s.COO_CANON_BACKEND]


class TestBackends:
    @staticmethod
    @pytest.fixture(params=backends)
    def backend(request):
        # Not used explicitly in most test cases.
        # Some tests specify other values as needed within the test case.
        kwargs = {
            "id_to_col": {1: 0, 2: 2},
            "param_to_size": {-1: 1, 3: 1},
            "param_to_col": {3: 0, -1: 1},
            "param_size_plus_one": 2,
            "var_length": 4,
        }

        backend = get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_mapping(self, backend):
        func = backend.get_func("sum")
        assert isinstance(func, Callable)
        with pytest.raises(KeyError):
            backend.get_func("notafunc")

    def test_neg(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        neg(x) means we now have
         [[-x11, -x21],
          [-x12, -x22]],

         i.e.,

         x11 x21 x12 x22
        [[-1  0   0   0],
         [0  -1   0   0],
         [0   0  -1   0],
         [0   0   0  -1]]
        """
        empty_view = backend.get_empty_view()
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        neg_lin_op = linOpHelper()
        out_view = backend.neg(neg_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        assert np.all(A == -np.eye(4))

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_transpose(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        transpose(x) means we now have
         [[x11, x21],
          [x12, x22]]

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   0   1   0],
         [0   1   0   0],
         [0   0   0   1]]

        -> It reduces to reordering the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        transpose_lin_op = linOpHelper((2, 2), data=[None], args=[variable_lin_op])
        out_view = backend.transpose(transpose_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_upper_tri(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        upper_tri(x) means we select only x12 (the diagonal itself is not considered).

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[0   0   0   1]]

        -> It reduces to selecting a subset of the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        upper_tri_lin_op = linOpHelper(args=[linOpHelper((2, 2))])
        out_view = backend.upper_tri(upper_tri_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[0, 0, 1, 0]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_index(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        index() returns the subset of rows corresponding to the slicing of variables.

        e.g. x[0:2,0] yields
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0]]

         Passing a single slice only returns the corresponding row of A.
         Note: Passing a single slice does not happen when slicing e.g. x[0], which is expanded to
         the 2d case.

         -> It reduces to selecting a subset of the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        index_2d_lin_op = linOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_2d_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.all(A == expected)

        index_1d_lin_op = linOpHelper(data=[slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_1d_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[1, 0, 0, 0]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_diag_mat(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        diag_mat(x) means we select only the diagonal, i.e., x11 and x22.

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   0   0   1]]

        -> It reduces to selecting a subset of the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        diag_mat_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = backend.diag_mat(diag_mat_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_diag_mat_with_offset(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        diag_mat(x, k=1) means we select only the 1-(super)diagonal, i.e., x12.

        which, when using the same columns as before, now maps to

         x11 x21 x12 x22
        [[0   0   1   0]]

        -> It reduces to selecting a subset of the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        k = 1
        diag_mat_lin_op = linOpHelper(shape=(1, 1), data=k)
        out_view = backend.diag_mat(diag_mat_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[0, 0, 1, 0]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_diag_vec(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        diag_vec(x) means we introduce zero rows as if the vector was the diagonal
        of an n x n matrix, with n the length of x.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[1  0],
         [0  0],
         [0  0],
         [0  1]]
        """

        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        diag_vec_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = backend.diag_vec(diag_vec_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 2)).toarray()
        expected = np.array([[1, 0], [0, 0], [0, 0], [0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_diag_vec_with_offset(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        diag_vec(x, k) means we introduce zero rows as if the vector was the k-diagonal
        of an n+|k| x n+|k| matrix, with n the length of x.

        Thus, for k=1 and using the same columns as before, want to represent
        [[0  x1 0],
        [ 0  0  x2],
        [[0  0  0]]
        i.e., unrolled in column-major order:

         x1  x2
        [[0  0],
        [0  0],
        [0  0],
        [1  0],
        [0  0],
        [0  0],
        [0  0],
        [0  1],
        [0  0]]
        """

        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        k = 1
        diag_vec_lin_op = linOpHelper(shape=(3, 3), data=k)
        out_view = backend.diag_vec(diag_vec_lin_op, view)
        A = out_view.get_tensor_representation(0, 9)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(9, 2)).toarray()
        expected = np.array(
            [[0, 0], [0, 0], [0, 0], [1, 0], [0, 0], [0, 0], [0, 0], [0, 1], [0, 0]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 9) == view.get_tensor_representation(0, 9)

    def test_sum_entries(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along the row axis.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[1  1]]
        """

        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        sum_entries_lin_op = linOpHelper(shape = (2,), data = [None, True], args=[variable_lin_op])
        out_view = backend.sum_entries(sum_entries_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 2)).toarray()
        expected = np.array([[1, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_promote(self, backend):
        """
        define x = Variable((1,)) with
        [x1,]

        x is represented as eye(1) in the A matrix, i.e.,

         x1
        [[1]]

        promote(x) means we repeat the row to match the required dimensionality of n rows.

        Thus, when using the same columns as before and assuming n = 3, we now have

         x1
        [[1],
         [1],
         [1]]
        """

        variable_lin_op = linOpHelper((1,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 1)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(1, 1)).toarray()
        assert np.all(view_A == np.eye(1))

        promote_lin_op = linOpHelper((3,))
        out_view = backend.promote(promote_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(3, 1)).toarray()
        expected = np.array([[1], [1], [1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_broadcast_to_rows(self, backend):
        """
        define x = Variable(3) with
        [x1, x2, x3]

        x is represented as eye(3) in the A matrix, i.e.,
         x1 x2 x3
        [[1  0  0],
         [0  1  0],
         [0  0  1]]
        
        broadcast_to(x, (2, 3)) means we repeat every variable twice along the row axis.

        Thus we expect the following A matrix:
        
         x1 x2 x3
        [[1  0  0],
         [1  0  0],
         [0  1  0],
         [0  1  0]
         [0  0  1],
         [0  0  1]]
        """
        variable_lin_op = linOpHelper((3,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 3)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(3, 3)).toarray()
        assert np.all(view_A == np.eye(3))

        broadcast_lin_op = linOpHelper((2,3), data=(2,3), args=[variable_lin_op])
        out_view = backend.broadcast_to(broadcast_lin_op, view)
        A = out_view.get_tensor_representation(0, 3)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(6, 3)).toarray()
        expected = np.array([[1, 0, 0],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_broadcast_to_cols(self, backend):
        """
        define x = Variable((2,1)) with
        [[x1], 
         [x2]]

        x is represented as eye(2) in the A matrix, i.e.,
         x1 x2
        [[1  0],
         [0  1]]
        
        broadcast_to(x, (2, 3)) means we tile the variables three times along the rows

        Thus we expect the following A matrix:
        
         x1 x2
        [[1  0],
         [0  1],
         [1  0],
         [0  1]
         [1  0],
         [0  1]]
        """
        variable_lin_op = linOpHelper((2,1), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        broadcast_lin_op = linOpHelper((2,3), data=(2,3), args=[variable_lin_op])
        out_view = backend.broadcast_to(broadcast_lin_op, view)
        A = out_view.get_tensor_representation(0, 3)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(6, 2)).toarray()
        expected = np.array([[1, 0],
                             [0, 1],
                             [1, 0],
                             [0, 1],
                             [1, 0],
                             [0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_hstack(self, backend):
        """
        define x,y = Variable((1,)), Variable((1,))

        hstack([x, y]) means the expression should be represented in the A matrix as if it
        was a Variable of shape (2,), i.e.,

          x  y
        [[1  0],
         [0  1]]
        """

        lin_op_x = linOpHelper((1,), type="variable", data=1)
        lin_op_y = linOpHelper((1,), type="variable", data=2)

        hstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        backend.id_to_col = {1: 0, 2: 1}
        out_view = backend.hstack(hstack_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 2)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
        expected = np.eye(2)
        assert np.all(A == expected)

    def test_vstack(self, backend):
        """
        define x,y = Variable((1,2)), Variable((1,2)) with
        [[x1, x2]]
        and
        [[y1, y2]]

        vstack([x, y]) yields

        [[x1, x2],
         [y1, y2]]

        which maps to

         x1   x2  y1  y2
        [[1   0   0   0],
         [0   0   1   0],
         [0   1   0   0],
         [0   0   0   1]]
        """

        lin_op_x = linOpHelper((1, 2), type="variable", data=1)
        lin_op_y = linOpHelper((1, 2), type="variable", data=2)

        vstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        backend.id_to_col = {1: 0, 2: 2}
        out_view = backend.vstack(vstack_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)

    def test_concatenate(self, backend):
        """
        Define x,y = Variable((1,2)), Variable((1,2)) with
        [[x1, x2]]
        and
        [[y1, y2]]

        concatenate([x, y], axis = 0) yields

        [[x1, x2],
         [y1, y2]]

        which maps to

         x1   x2  y1  y2
        [[1   0   0   0],
         [0   0   1   0],
         [0   1   0   0],
         [0   0   0   1]]

        Note that in this case concatenate is equivalent to vstack

        Applying concatenate([x, y], axis=1) yields:

        [[x1, x2, y1, y2]]

        Which is equivalent to hstack([x, y]) in this context.

        The mapping to the matrix A would be:

          x1  x2  y1  y2
         [[1   0   0  0],
          [0   1   0  0],
          [0   0   1  0],
          [0   0   0  1]]

        """
        # See InverseData.get_var_offsets method for references to this map
        backend.id_to_col = {1: 0, 2: 2}

        # Axis = 1
        lin_op_x = linOpHelper((1, 2), type="variable", data=1)
        lin_op_y = linOpHelper((1, 2), type="variable", data=2)

        concatenate_lin_op = linOpHelper(args=[lin_op_x, lin_op_y], data = [1])
        backend.id_to_col = {1: 0, 2: 2}
        out_view = backend.concatenate(concatenate_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.eye(4)
        assert np.all(A == expected)

        # Axis = 0
        concatenate_lin_op = linOpHelper(args=[lin_op_x, lin_op_y], data = [0])
        out_view = backend.concatenate(concatenate_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        assert np.all(A == expected)


    @pytest.mark.parametrize("axis, variable_indices", [
        # Axis 0
        (0, [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]),
        # Axis 1
        (1, [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]),
        # Axis 2
        (2, list(range(16))),
        # Axis None
        (None, list(range(16))),
    ])
    def test_concatenate_nd(self, backend, axis, variable_indices):
        """
        Test the concatenate operation with variables of shape (2, 2, 2)
        along different axes

        Define variables x and y, each of shape (2, 2, 2):

        x = [
            [[x000, x001],
            [x010, x011]],
            [[x100, x101],
            [x110, x111]]
            ]

        y = [
            [[y000, y001],
            [y010, y011]],
            [[y100, y101],
            [y110, y111]]
            ]

        The variables are assigned indices as follows:

        Indices for x:
            x000: 0, x001: 1, x010: 2, x011: 3,
            x100: 4, x101: 5, x110: 6, x111: 7

        Indices for y:
            y000: 8, y001: 9, y010: 10, y011: 11,
            y100: 12, y101: 13, y110: 14, y111: 15

        How we chose the list of variable_indices:

        - For each axis, we perform the concatenation of x and y along that axis.
        - We assign indices to the variables in x and y as per their positions.
        - For each argument, we generate an array of indices from 0 to the number of
        elements minus one, reshaped to the argument's shape with 'F' order
        (column-major order), and offset by the cumulative
        number of elements from previous arguments.

        - We concatenate these indices along the specified axis.
        - We flatten the concatenated indices with 'F' order to obtain the variable_indices,
        which represent the order of variables in the flattened concatenated tensor.

        The expected variable_indices are:

        For axis=0:
            [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]

        For axis=1:
            [0, 1, 2, 3, 8, 9, 10, 11, 4, 5, 6, 7, 12, 13, 14, 15]

        For axis=2:
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        For axis=None:
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

        axis=None follows NumPy; arrays are flattened in 'C' order before concatenating,
        so the resulting array is [x000, x001, x010, x011, x100, x101, x110, x111,
                                y000, y001, y010, y011, y100, y101, y110, y111]
        """
        def get_expected_matrix(variable_indices):
            A = np.zeros((16, 16), dtype=int)
            positions = np.arange(16)
            for pos, var_idx in zip(positions, variable_indices):
                A[pos, var_idx] = 1
            return A

        # Map variable IDs to column indices
        backend.id_to_col = {1: 0, 2: 8}

        # Define lin_op_x and lin_op_y with shape (2, 2, 2)
        lin_op_x = linOpHelper((2, 2, 2), type="variable", data=1)
        lin_op_y = linOpHelper((2, 2, 2), type="variable", data=2)

        # Perform concatenation along the specified axis
        concatenate_lin_op = linOpHelper(args=[lin_op_x, lin_op_y], data = [axis])
        out_view = backend.concatenate(concatenate_lin_op, backend.get_empty_view())
        A = out_view.get_tensor_representation(0, 16)
        # Convert to numpy array
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(16, 16)).toarray()
        expected_A = get_expected_matrix(variable_indices)
        assert np.all(A == expected_A)

    def test_mul(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Multiplying with the constant from the left
        [[1, 2],
         [3, 4]],

         we expect the output to be
        [[  x11 + 2 x21,   x12 + 2 x22],
         [3 x11 + 4 x21, 3 x12 + 4 x22]]

        i.e., when represented in the A matrix (again using column-major order):
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs = linOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))

        mul_lin_op = linOpHelper(data=lhs, args=[variable_lin_op])
        out_view = backend.mul(mul_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 2, 0, 0], [3, 4, 0, 0], [0, 0, 1, 2], [0, 0, 3, 4]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_rmul(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Multiplying with the constant from the right
         (intentionally using 1D vector to cover edge case)
        [1, 2]

         we expect the output to be
         [[x11 + 2 x12],
          [x21 + 2 x22]]

        i.e., when represented in the A matrix:
         x11 x21 x12 x22
        [[1   0   2   0],
         [0   1   0   2]]
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        rhs = linOpHelper((2,), type="dense_const", data=np.array([1, 2]))

        rmul_lin_op = linOpHelper(data=rhs, args=[variable_lin_op])
        out_view = backend.rmul(rmul_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array([[1, 0, 2, 0], [0, 1, 0, 2]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_mul_elementwise(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

         mul_elementwise(x, a) means 'a' is reshaped into a column vector and multiplied by A.
         E.g. for a = (2,3), we obtain

         x1  x2
        [[2  0],
         [0  3]]
        """

        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        lhs = linOpHelper((2,), type="dense_const", data=np.array([2, 3]))

        mul_elementwise_lin_op = linOpHelper(data=lhs)
        out_view = backend.mul_elem(mul_elementwise_lin_op, view)
        A = out_view.get_tensor_representation(0, 2)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
        expected = np.array([[2, 0], [0, 3]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_div(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

         Dividing elementwise with
        [[1, 2],
         [3, 4]],

        we obtain:
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1/3 0   0],
         [0   0   1/2 0],
         [0   0   0   1/4]]
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs = linOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))

        div_lin_op = linOpHelper(data=lhs)
        out_view = backend.div(div_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array([[1, 0, 0, 0], [0, 1 / 3, 0, 0], [0, 0, 1 / 2, 0], [0, 0, 0, 1 / 4]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_trace(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        x is represented as eye(4) in the A matrix (in column-major order), i.e.,

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]

        trace(x) means we sum the diagonal entries of x, i.e.

         x11 x21 x12 x22
        [[1   0   0   1]]
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        trace_lin_op = linOpHelper(args=[variable_lin_op])
        out_view = backend.trace(trace_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array([[1, 0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_conv(self, backend):
        """
        define x = Variable((3,)) with
        [x1, x2, x3]

        having f = [1,2,3], conv(f, x) means we repeat the column vector of f for each column in
        the A matrix, shifting it down by one after each repetition, i.e.,
          x1 x2 x3
        [[1  0  0],
         [2  1  0],
         [3  2  1],
         [0  3  2],
         [0  0  3]]
        """

        variable_lin_op = linOpHelper((3,), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 3)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(3, 3)).toarray()
        assert np.all(view_A == np.eye(3))

        f = linOpHelper((3,), type="dense_const", data=np.array([1, 2, 3]))
        conv_lin_op = linOpHelper(data=f, shape=(5, 1), args=[variable_lin_op])

        out_view = backend.conv(conv_lin_op, view)
        A = out_view.get_tensor_representation(0, 5)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(5, 3)).toarray()
        expected = np.array(
            [[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [3.0, 2.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 3.0]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 5) == view.get_tensor_representation(0, 5)

    def test_kron_r(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1],
             [2]],

        kron(a, x) means we have
        [[x11, x12],
         [x21, x22],
         [2x11, 2x12],
         [2x21, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   2   0],
         [0   0   0   2]]

        However computing kron(a, x) (where x is represented as eye(4))
        directly gives us:
        [[1   0   0   0],
         [2   0   0   0],
         [0   1   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   2   0],
         [0   0   0   1],
         [0   0   0   2]]
        So we must swap the row indices of the resulting matrix.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        a = linOpHelper((2, 1), type="dense_const", data=np.array([[1], [2]]))
        kron_r_lin_op = linOpHelper(data=a, args=[variable_lin_op])

        out_view = backend.kron_r(kron_r_lin_op, view)
        A = out_view.get_tensor_representation(0, 8)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)

    def test_kron_l(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1, 2]],

        kron(x, a) means we have
        [[x11, 2x11, x12, 2x12],
         [x21, 2x21, x22, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   2   0],
         [0   0   0   2]]

         However computing kron(x, a) (where a is reshaped into a column vector
         and x is represented as eye(4)) directly gives us:
        [[1   0   0   0],
         [2   0   0   0],
         [0   1   0   0],
         [0   2   0   0],
         [0   0   1   0],
         [0   0   2   0],
         [0   0   0   1],
         [0   0   0   2]]
        So we must swap the row indices of the resulting matrix.
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        a = linOpHelper((1, 2), type="dense_const", data=np.array([[1, 2]]))
        kron_l_lin_op = linOpHelper(data=a, args=[variable_lin_op])

        out_view = backend.kron_l(kron_l_lin_op, view)
        A = out_view.get_tensor_representation(0, 8)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0, 0.0],
                [0.0, 2.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 2.0],
            ]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)

    def test_get_kron_row_indices(self, backend):
        """
        kron(l,r)
        with
        l = [[x1, x3],  r = [[a],
             [x2, x4]]       [b]]

        yields
        [[ax1, ax3],
         [bx1, bx3],
         [ax2, ax4],
         [bx2, bx4]]

        Which is what we get when we compute kron(l,r) directly,
        as l is represented as eye(4) and r is reshaped into a column vector.

        So we have:
        kron(l,r) =
        [[a, 0, 0, 0],
         [b, 0, 0, 0],
         [0, a, 0, 0],
         [0, b, 0, 0],
         [0, 0, a, 0],
         [0, 0, b, 0],
         [0, 0, 0, a],
         [0, 0, 0, b]].

        Thus, this function should return arange(8).
        """
        indices = backend._get_kron_row_indices((2, 2), (2, 1))
        assert np.all(indices == np.arange(8))

        """
        kron(l,r)
        with
        l = [[x1],  r = [[a, c],
             [x2]]       [b, d]]

        yields
        [[ax1, cx1],
         [bx1, dx1],
         [ax2, cx2],
         [bx2, dx2]]

        Here, we have to swap the row indices of the resulting matrix.
        Immediately applying kron(l,r) gives to eye(2) and r reshaped to
        a column vector gives.

        So we have:
        kron(l,r) =
        [[a, 0],
         [b, 0],
         [c, 0],
         [d, 0],
         [0, a],
         [0, b]
         [0, c],
         [0, d]].

        Thus, we need to return [0, 1, 4, 5, 2, 3, 6, 7].
        """

        indices = backend._get_kron_row_indices((2, 1), (2, 2))
        assert np.all(indices == [0, 1, 4, 5, 2, 3, 6, 7])

        indices = backend._get_kron_row_indices((1, 2), (3, 2))
        assert np.all(indices == np.arange(12))

        indices = backend._get_kron_row_indices((3, 2), (1, 2))
        assert np.all(indices == [0, 2, 4, 1, 3, 5, 6, 8, 10, 7, 9, 11])

        indices = backend._get_kron_row_indices((2, 2), (2, 2))
        expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
        assert np.all(indices == expected)

    def test_tensor_view_combine_potentially_none(self, backend):
        view = backend.get_empty_view()
        assert view.combine_potentially_none(None, None) is None
        a = {"a": [1]}
        b = {"b": [2]}
        assert view.combine_potentially_none(a, None) == a
        assert view.combine_potentially_none(None, a) == a
        assert view.combine_potentially_none(a, b) == view.add_dicts(a, b)


class TestParametrizedBackends:
    @staticmethod
    @pytest.fixture(params=backends)
    def param_backend(request):
        kwargs = {
            "id_to_col": {1: 0},
            "param_to_size": {-1: 1, 2: 2},
            "param_to_col": {2: 0, -1: 2},
            "param_size_plus_one": 3,
            "var_length": 2,
        }

        backend = get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_parametrized_diag_vec(self, param_backend):
        """
        Starting with a parametrized expression
        x1  x2
        [[[1  0],
         [0  0]],

         [[0  0],
         [0  1]]]

        diag_vec(x) means we introduce zero rows as if the vector was the diagonal
        of an n x n matrix, with n the length of x.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[[1  0],
          [0  0],
          [0  0],
          [0  0]]

         [[0  0],
          [0  0],
          [0  0],
          [0  1]]]
        """

        param_lin_op = linOpHelper((2,), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        diag_vec_lin_op = linOpHelper(shape=(2, 2), data=0)
        out_view = param_backend.diag_vec(diag_vec_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(
            0, 4
        )

    def test_parametrized_diag_vec_with_offset(self, param_backend):
        """
        Starting with a parametrized expression
        x1  x2
        [[[1  0],
          [0  0]],

         [[0  0],
          [0  1]]]

        diag_vec(x, k) means we introduce zero rows as if the vector was the k-diagonal
        of an n+|k| x n+|k| matrix, with n the length of x.

        Thus, for k=1 and using the same columns as before, want to represent
        [[0  x1 0],
         [0  0  x2],
         [0  0  0]]
        parametrized across two slices, i.e., unrolled in column-major order:

        slice 0         slice 1
         x1  x2          x1  x2
        [[0  0],        [[0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [1  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  0],
         [0  0],         [0  1],
         [0  0]]         [0  0]]
        """

        param_lin_op = linOpHelper((2,), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        k = 1
        diag_vec_lin_op = linOpHelper(shape=(3, 3), data=k)
        out_view = param_backend.diag_vec(diag_vec_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 9)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
            ]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 9) == param_var_view.get_tensor_representation(
            0, 9
        )

    def test_parametrized_sum_entries(self, param_backend):
        """
        starting with a parametrized expression
        x1  x2
        [[[1  0],
         [0  0]],

         [[0  0],
         [0  1]]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along the row axis.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[[1  0]],

         [[0  1]]]
        """
        param_lin_op = linOpHelper((2,), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 3}
        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        sum_entries_lin_op = linOpHelper(shape=(2,), data=[None, True], args=[variable_lin_op])
        out_view = param_backend.sum_entries(sum_entries_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 1)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 1.0]])
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == param_var_view.get_tensor_representation(
            0, 1
        )

    def test_parametrized_mul(self, param_backend):
        """
        Continuing from the non-parametrized example when the lhs is a parameter,
        instead of multiplying with known values, the matrix is split up into four slices,
        each representing an element of the parameter, i.e. instead of
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]

         we obtain the list of length four, where we have ones at the entries where previously
         we had the 1, 3, 2, and 4 (again flattened in column-major order):

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [0   0   0   0],
             [0   0   1   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [1   0   0   0],
             [0   0   0   0],
             [0   0   1   0]],

            [[0   1   0   0],
             [0   0   0   0],
             [0   0   0   1],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   1   0   0],
             [0   0   0   0],
             [0   0   0   1]]
        ]
        """
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_size_plus_one = 5
        param_backend.var_length = 4
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs_parameter = linOpHelper((2, 2), type="param", data=2)

        mul_lin_op = linOpHelper(data=lhs_parameter, args=[variable_lin_op])
        out_view = param_backend.mul(mul_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 4)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array(
            [[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        # indices are: variable 1, parameter 2, 2 index of the list
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array(
            [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_two == expected_idx_two)

        # indices are: variable 1, parameter 2, 3 index of the list
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_parametrized_rhs_mul(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is multiplied by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of
         x11 x21 x12 x22
        [[1   2   0   0],
         [3   4   0   0],
         [0   0   1   2],
         [0   0   3   4]]

         we obtain the list of length four, where we have the same entries as before
         but each variable maps to a different parameter slice:

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [3   0   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   2   0   0],
             [0   4   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   1   0],
             [0   0   3   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   0   2],
             [0   0   0   4]]
        ]
        """
        param_lin_op = linOpHelper((2, 2), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.var_length = 4
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        lhs = linOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))

        mul_lin_op = linOpHelper(data=lhs, args=[variable_lin_op])
        out_view = param_backend.mul(mul_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array(
            [[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array(
            [[0.0, 2.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 3.0, 0.0]]
        )
        assert np.all(slice_idx_two == expected_idx_two)

        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0], [0.0, 0.0, 0.0, 4.0]]
        )
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(
            0, 4
        )

    def test_parametrized_rmul(self, param_backend):
        """
        Continuing from the non-parametrized example when the rhs is a parameter,
        instead of multiplying with known values, the matrix is split up into two slices,
        each representing an element of the parameter, i.e. instead of
         x11 x21 x12 x22
        [[1   0   2   0],
         [0   1   0   2]]

         we obtain the list of length two, where we have ones at the entries where previously
         we had the 1 and 2:

         x11  x21  x12  x22
        [
         [[1   0   0   0],
          [0   1   0   0]]

         [[0   0   1   0],
          [0   0   0   1]]
        ]
        """

        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        param_backend.var_length = 4
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        rhs_parameter = linOpHelper((2,), type="param", data=2)

        rmul_lin_op = linOpHelper(data=rhs_parameter, args=[variable_lin_op])
        out_view = param_backend.rmul(rmul_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 2)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_parametrized_rhs_rmul(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is multiplied by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of

         x11 x21 x12 x22
        [[1   0   3   0],
         [0   1   0   3],
         [2   0   4   0],
         [0   2   0   4]]

         we obtain the list of length four, where we have the same entries as before
         but each variable maps to a different parameter slice:

         x11  x21  x12  x22
        [
         [[1   0   0   0],
          [0   0   0   0],
          [2   0   0   0],
          [0   0   0   0]]

         [[0   0   0   0],
          [0   1   0   0],
          [0   0   0   0],
          [0   2   0   0]]

         [[0   0   3   0],
          [0   0   0   0],
          [0   0   4   0],
          [0   0   0   0]]

         [[0   0   0   0],
          [0   0   0   3],
          [0   0   0   0],
          [0   0   0   4]]
        ]
        """
        param_lin_op = linOpHelper((2, 2), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.var_length = 4
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        rhs = linOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))

        rmul_lin_op = linOpHelper(data=rhs, args=[variable_lin_op])
        out_view = param_backend.rmul(rmul_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array(
            [[0.0, 0.0, 3.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_two == expected_idx_two)

        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 4.0]]
        )
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(
            0, 4
        )

    def test_mul_elementwise_parametrized(self, param_backend):
        """
        Continuing the non-parametrized example when 'a' is a parameter, instead of multiplying
        with known values, the matrix is split up into two slices, each representing an element
        of the parameter, i.e. instead of
         x1  x2
        [[2  0],
         [0  3]]

         we obtain the list of length two:

          x1  x2
        [
         [[1  0],
          [0  0]],

         [[0  0],
          [0  1]]
        ]
        """

        variable_lin_op = linOpHelper((2,), type="variable", data=1)
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 2)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        lhs_parameter = linOpHelper((2,), type="param", data=2)

        mul_elementwise_lin_op = linOpHelper(data=lhs_parameter)
        out_view = param_backend.mul_elem(mul_elementwise_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 2)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1, 0], [0, 0]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0, 0], [0, 1]])
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 2) == view.get_tensor_representation(0, 2)

    def test_parametrized_div(self, param_backend):
        """
        Continuing from the non-parametrized example when the expression
        that is divided by is parametrized. For a variable that
        was multiplied elementwise by a parameter, instead of
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1/3 0   0],
         [0   0   1/2 0],
         [0   0   0   1/4]]

         we obtain the list of length four, where we have the quotients at the same entries
         but each variable maps to a different parameter slice:

            x11  x21  x12  x22
        [
            [[1   0   0   0],
             [0   0   0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   1/3 0   0],
             [0   0   0   0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   1/2 0],
             [0   0   0   0]],

            [[0   0   0   0],
             [0   0   0   0],
             [0   0   0   0],
             [0   0   0   1/4]]
        ]
        """
        param_lin_op = linOpHelper((2, 2), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.var_length = 4
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        lhs = linOpHelper((2, 2), type="dense_const", data=np.array([[1, 2], [3, 4]]))

        div_lin_op = linOpHelper(data=lhs)
        out_view = param_backend.div(div_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 4)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1 / 3, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1 / 2, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        assert np.all(slice_idx_two == expected_idx_two)

        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1 / 4],
            ]
        )
        assert np.all(slice_idx_three == expected_idx_three)
        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(
            0, 4
        )

    def test_parametrized_trace(self, param_backend):
        """
        Continuing from the non-parametrized example, instead of a pure variable
        input, we take a variable that has been multiplied elementwise by a parameter.

        The trace of this expression is then given by

            x11  x21  x12  x22
        [
            [[1   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   0]],

            [[0   0   0   1]]
        ]
        """
        param_lin_op = linOpHelper((2, 2), type="param", data=2)
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.var_length = 4
        variable_lin_op = linOpHelper((2, 2), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        trace_lin_op = linOpHelper(args=[variable_lin_op])
        out_view = param_backend.trace(trace_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 1)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_one == expected_idx_one)

        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([[0.0, 0.0, 0.0, 0.0]])
        assert np.all(slice_idx_two == expected_idx_two)

        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([[0.0, 0.0, 0.0, 1.0]])
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == param_var_view.get_tensor_representation(
            0, 1
        )

    @pytest.mark.parametrize("backend_name", backends)
    def test_mul_1d_param_1d_var(self, backend_name):
        """1D parameter @ 1D variable = scalar (dot product).

        Bug regression test: 1D parameters must be treated as row vectors
        (1, n) not column vectors (n, 1) for correct matrix dimensions.
        """
        n = 4
        backend = get_backend(
            backend_name,
            id_to_col={1: 0},
            param_to_size={-1: 1, 3: n},
            param_to_col={-1: 0, 3: 1},
            param_size_plus_one=n + 1,
            var_length=n,
        )
        var = linOpHelper((n,), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # param (n,) @ x (n,) -> scalar
        param = linOpHelper((n,), type="param", data=3)
        lin_op = linOpHelper(shape=(), data=param, args=[var])
        out = backend.mul(lin_op, view)

        # Result is 1 row (scalar output)
        total_rows = 1
        tr = out.get_tensor_representation(0, total_rows)

        # Row indices must be within bounds
        assert len(tr.row) > 0, "Should have non-zero entries"
        assert tr.row.max() < total_rows, \
            f"Row index {tr.row.max()} exceeds total_rows {total_rows}"


class TestND_Backends:
    @staticmethod
    @pytest.fixture(params=backends)
    def backend(request):
        # Not used explicitly in most test cases.
        # Some tests specify other values as needed within the test case.
        kwargs = {
            "id_to_col": {1: 0, 2: 2},
            "param_to_size": {-1: 1, 3: 1},
            "param_to_col": {3: 0, -1: 1},
            "param_size_plus_one": 2,
            "var_length": 4,
        }

        backend = get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_nd_sum_entries(self, backend):
        """
        define x = Variable((2,2,2)) with
        [[[x111, x112],
        [x121, x122]],

        [[x211, x212],
        [x221, x222]]]

        x is represented as eye(8) in the A matrix (in column-major order), i.e.,

        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   0   0   0   0   0   0],
         [0   1   0   0   0   0   0   0],
         [0   0   1   0   0   0   0   0],
         [0   0   0   1   0   0   0   0],
         [0   0   0   0   1   0   0   0],
         [0   0   0   0   0   1   0   0],
         [0   0   0   0   0   0   1   0],
         [0   0   0   0   0   0   0   1]]

        sum(x, axis = 0) means we only consider entries in a given axis (axes)

        which, when using the same columns as before, now maps to

        sum(x, axis = 0)
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   1   0   0   0   0   0   0],
         [0   0   1   1   0   0   0   0],
         [0   0   0   0   1   1   0   0],
         [0   0   0   0   0   0   1   1]]

        sum(x, axis = 1)
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   1   0   0   0   0   0],
         [0   1   0   1   0   0   0   0],
         [0   0   0   0   1   0   1   0],
         [0   0   0   0   0   1   0   1]]

        sum(x, axis = 2)
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   0   0   1   0   0   0],
         [0   1   0   0   0   1   0   0],
         [0   0   1   0   0   0   1   0],
         [0   0   0   1   0   0   0   1]]

        To reproduce the outputs above, eliminate the given axis
        and put ones where the remaining axes (axis) match.

        Note: sum(x, keepdims=True) is equivalent to sum(x, keepdims=False)
        with a reshape, which is NO-OP in the backend.
        """

        variable_lin_op = linOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        sum_lin_op = linOpHelper(shape=(2, 2, 2), data=[2, True], args=[variable_lin_op])
        out_view = backend.sum_entries(sum_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 8)).toarray()
        expected = np.array([[1, 0, 0, 0, 1, 0, 0, 0],
                             [0, 1, 0, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 0, 1, 0],
                             [0, 0, 0, 1, 0, 0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    @pytest.mark.parametrize("axes, expected", [((0,1),
                                                [[1, 1, 1, 1, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 1, 1, 1, 1]]),
                                                ((0,2),
                                                [[1, 1, 0, 0, 1, 1, 0, 0],
                                                [0, 0, 1, 1, 0, 0, 1, 1]]),
                                                ((2,1),
                                                [[1, 0, 1, 0, 1, 0, 1, 0],
                                                [0, 1, 0, 1, 0, 1, 0, 1]])])
    def test_nd_sum_entries_multiple_axes(self, backend, axes, expected):
        """
        define x = Variable((2,2,2)) with
        [[[x111, x112],
        [x121, x122]],

        [[x211, x212],
        [x221, x222]]]

        x is represented as eye(8) in the A matrix (in column-major order), i.e.,

        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   0   0   0   0   0   0],
         [0   1   0   0   0   0   0   0],
         [0   0   1   0   0   0   0   0],
         [0   0   0   1   0   0   0   0],
         [0   0   0   0   1   0   0   0],
         [0   0   0   0   0   1   0   0],
         [0   0   0   0   0   0   1   0],
         [0   0   0   0   0   0   0   1]]

        sum(x, axis = (0,1))
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   1   1   1   0   0   0   0],
         [0   0   0   0   1   1   1   1]]

        sum(x, axis = (0,2))
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   1   0   0   1   1   0   0],
         [0   0   1   1   0   0   1   1]]

        sum(x, axis = (1,2))
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   1   0   1   0   1   0],
         [0   1   0   1   0   1   0   1]]

        To reproduce the outputs above, eliminate the given axes
        and put ones where the remaining axes match.
        """

        variable_lin_op = linOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        sum_lin_op = linOpHelper(shape=(2, 2, 2), data=[axes, True], args=[variable_lin_op])
        out_view = backend.sum_entries(sum_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 8)).toarray()
        assert np.all(A == np.array(expected))

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == view.get_tensor_representation(0, 4)

    def test_nd_index(self, backend):
        """
        define x = Variable((2,2,2)) with
        [[[x111, x112],
        [x121, x122]],

        [[x211, x212],
        [x221, x222]]]

        x is represented as eye(8) in the A matrix (in column-major order), i.e.,

        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   0   0   0   0   0   0],
         [0   1   0   0   0   0   0   0],
         [0   0   1   0   0   0   0   0],
         [0   0   0   1   0   0   0   0],
         [0   0   0   0   1   0   0   0],
         [0   0   0   0   0   1   0   0],
         [0   0   0   0   0   0   1   0],
         [0   0   0   0   0   0   0   1]]

        index() returns the subset of rows corresponding to the slicing of variables.

        e.g. x[0:2, 0, 0:2] yields
        x111 x211 x121 x221 x112 x212 x122 x222
        [[1   0   0   0   0   0   0   0],
         [0   1   0   0   0   0   0   0],
         [0   0   0   0   1   0   0   0],
         [0   0   0   0   0   1   0   0]]

         -> It reduces to selecting a subset of the rows of A.
        """

        variable_lin_op = linOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        index_2d_lin_op = linOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1), slice(0, 2, 1)],
                                      args=[variable_lin_op])
        out_view = backend.index(index_2d_lin_op, view)
        A = out_view.get_tensor_representation(0, 4)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 8)).toarray()
        expected = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0]])
        assert np.all(A == expected)

        index_1d_lin_op = linOpHelper(data=[slice(1, 2, 1)], args=[variable_lin_op])
        out_view = backend.index(index_1d_lin_op, view)
        A = out_view.get_tensor_representation(0, 1)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 8)).toarray()
        expected = np.array([[0, 1, 0, 0, 0, 0, 0, 0]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_nd_broadcast_to(self, backend):
        """
        define x = Variable((2,1,2)) with
        [[x11, x12], 
         [x21, x22]]

        x is represented as eye(4) in the A matrix, i.e.,
         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1]]
        
        broadcast_to(x, (2, 3, 2)) means we repeat columns of x three times each subsequently.

        Thus we expect the following A matrix:
        
        x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [1   0   0   0],
         [0   1   0   0],
         [1   0   0   0],
         [0   1   0   0],

         [0   0   1   0],
         [0   0   0   1],
         [0   0   1   0],
         [0   0   0   1],
         [0   0   1   0],
         [0   0   0   1]]
        """
        variable_lin_op = linOpHelper((2,1,2), type="variable", data=1)
        view = backend.process_constraint(variable_lin_op, backend.get_empty_view())

        # cast to numpy
        view_A = view.get_tensor_representation(0, 4)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        broadcast_lin_op = linOpHelper((2,3,2), data=(2,3,2), args=[variable_lin_op])
        out_view = backend.broadcast_to(broadcast_lin_op, view)
        A = out_view.get_tensor_representation(0, 12)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(12, 4)).toarray()
        expected = np.array([[1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [1, 0, 0, 0],
                             [0, 1, 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 1) == view.get_tensor_representation(0, 1)

    def test_nd_mul_3d_var(self, backend):
        """
        Test mul linop with 3D variable: C (m,k) @ X (B,k,n).

        For C @ X where C is 2D (m,k) and X is 3D (B,k,n):
        - vec(C @ X) = (I_n ⊗ C ⊗ I_B) @ vec(X)

        Example: X = Variable((2, 2, 2)) with shape (B=2, k=2, n=2)
        X is represented as eye(8) in column-major (Fortran) order:

        vec(X) index mapping:
        - Index 0: X[0,0,0]
        - Index 1: X[1,0,0]
        - Index 2: X[0,1,0]
        - Index 3: X[1,1,0]
        - Index 4: X[0,0,1]
        - Index 5: X[1,0,1]
        - Index 6: X[0,1,1]
        - Index 7: X[1,1,1]

        For C = [[1, 2], [3, 4]], the result is (B=2, m=2, n=2).
        Each output element is computed as:
        - result[b, i, c] = sum_r C[i, r] * X[b, r, c]

        The resulting A matrix is I_2 ⊗ C ⊗ I_2:

                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [1    0    2    0    0    0    0    0  ]  # C[0,:]@X[0,:,0]
        result[1]  [0    1    0    2    0    0    0    0  ]  # C[0,:]@X[1,:,0]
        result[2]  [3    0    4    0    0    0    0    0  ]  # C[1,:]@X[0,:,0]
        result[3]  [0    3    0    4    0    0    0    0  ]  # C[1,:]@X[1,:,0]
        result[4]  [0    0    0    0    1    0    2    0  ]  # C[0,:]@X[0,:,1]
        result[5]  [0    0    0    0    0    1    0    2  ]  # C[0,:]@X[1,:,1]
        result[6]  [0    0    0    0    3    0    4    0  ]  # C[1,:]@X[0,:,1]
        result[7]  [0    0    0    0    0    3    0    4  ]  # C[1,:]@X[1,:,1]
        """
        # Update backend for 3D variable
        backend.var_length = 8
        backend.id_to_col = {1: 0}

        var = linOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # Verify initial view is identity
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        # Create constant C = [[1, 2], [3, 4]]
        const_data = np.array([[1, 2], [3, 4]])
        const = linOpHelper((2, 2), type="dense_const", data=const_data)

        mul_op = linOpHelper(shape=(2, 2, 2), data=const, args=[var])
        out_view = backend.mul(mul_op, view)
        A = out_view.get_tensor_representation(0, 8)

        # Cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 8)).toarray()

        # Expected: I_2 ⊗ C ⊗ I_2
        # C ⊗ I_2 = [[1,0,2,0], [0,1,0,2], [3,0,4,0], [0,3,0,4]]
        # I_2 ⊗ (C ⊗ I_2) = block_diag(C ⊗ I_2, C ⊗ I_2)
        expected = np.array([
            [1, 0, 2, 0, 0, 0, 0, 0],
            [0, 1, 0, 2, 0, 0, 0, 0],
            [3, 0, 4, 0, 0, 0, 0, 0],
            [0, 3, 0, 4, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0, 1, 0, 2],
            [0, 0, 0, 0, 3, 0, 4, 0],
            [0, 0, 0, 0, 0, 3, 0, 4],
        ])
        assert np.all(A == expected)

        # Note: view is edited in-place
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)

    def test_nd_mul_batch_varying_const(self, backend):
        """
        Test mul linop with batch-varying constant: C (B,m,k) @ X (B,k,n).

        For C @ X where C is 3D (B,m,k) and X is 3D (B,k,n):
        - Each batch b computes: result[b,:,:] = C[b,:,:] @ X[b,:,:]
        - Uses interleaved matrix structure where batch indices alternate

        Example: X = Variable((2, 2, 2)) with shape (B=2, k=2, n=2)
        X is represented as eye(8) in column-major (Fortran) order:

        vec(X) index mapping:
        - Index 0: X[0,0,0]
        - Index 1: X[1,0,0]
        - Index 2: X[0,1,0]
        - Index 3: X[1,1,0]
        - Index 4: X[0,0,1]
        - Index 5: X[1,0,1]
        - Index 6: X[0,1,1]
        - Index 7: X[1,1,1]

        For C with shape (2, 2, 2):
        - C[0] = [[1, 2], [3, 4]]
        - C[1] = [[5, 6], [7, 8]]

        The result shape is (B=2, m=2, n=2). Each output:
        - result[b, i, c] = sum_r C[b, i, r] * X[b, r, c]

        The resulting A matrix has interleaved batch structure:

                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [1    0    2    0    0    0    0    0  ]  # C[0,0,:]@X[0,:,0]
        result[1]  [0    5    0    6    0    0    0    0  ]  # C[1,0,:]@X[1,:,0]
        result[2]  [3    0    4    0    0    0    0    0  ]  # C[0,1,:]@X[0,:,0]
        result[3]  [0    7    0    8    0    0    0    0  ]  # C[1,1,:]@X[1,:,0]
        result[4]  [0    0    0    0    1    0    2    0  ]  # C[0,0,:]@X[0,:,1]
        result[5]  [0    0    0    0    0    5    0    6  ]  # C[1,0,:]@X[1,:,1]
        result[6]  [0    0    0    0    3    0    4    0  ]  # C[0,1,:]@X[0,:,1]
        result[7]  [0    0    0    0    0    7    0    8  ]  # C[1,1,:]@X[1,:,1]

        Note how batch indices are interleaved: rows 0,2,4,6 use C[0], rows 1,3,5,7 use C[1].
        """
        backend.var_length = 8
        backend.id_to_col = {1: 0}

        var = linOpHelper((2, 2, 2), type="variable", data=1)
        view = backend.process_constraint(var, backend.get_empty_view())

        # Verify initial view is identity
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        # Create batch-varying constant C with shape (2, 2, 2)
        # C[0] = [[1, 2], [3, 4]], C[1] = [[5, 6], [7, 8]]
        const_data = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        const = linOpHelper((2, 2, 2), type="dense_const", data=const_data)

        mul_op = linOpHelper(shape=(2, 2, 2), data=const, args=[var])
        out_view = backend.mul(mul_op, view)
        A = out_view.get_tensor_representation(0, 8)

        # Cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 8)).toarray()

        # Expected: interleaved matrix structure
        # Rows 0,2,4,6 use C[0], rows 1,3,5,7 use C[1]
        expected = np.array([
            [1, 0, 2, 0, 0, 0, 0, 0],
            [0, 5, 0, 6, 0, 0, 0, 0],
            [3, 0, 4, 0, 0, 0, 0, 0],
            [0, 7, 0, 8, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 2, 0],
            [0, 0, 0, 0, 0, 5, 0, 6],
            [0, 0, 0, 0, 3, 0, 4, 0],
            [0, 0, 0, 0, 0, 7, 0, 8],
        ])
        assert np.all(A == expected)

        # Note: view is edited in-place
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)


class TestParametrizedND_Backends:
    @staticmethod
    @pytest.fixture(params=backends)
    def param_backend(request):
        kwargs = {
            "id_to_col": {1: 0},
            "param_to_size": {-1: 1, 2: 8},
            "param_to_col": {2: 0, -1: 8},
            "param_size_plus_one": 9,
            "var_length": 8,
        }

        backend = get_backend(request.param, **kwargs)
        assert isinstance(backend, PythonCanonBackend)
        return backend

    def test_parametrized_nd_sum_entries(self, param_backend):
        """
        starting with a (2,2,2) parametrized expression
        x111 x211 x121 x221 x112 x212 x122 x222
        slice(0)
        [[1   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0],
         ...
         [0   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0]]
        slice(1)
        [[0   0   0   0   0   0   0   0],
         [0   1   0   0   0   0   0   0],
         ...
         [0   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0]]
        ...
        slice(7)
        [[0   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0],
         ...
         [0   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   1]]

        sum(x, axis = (0,2)) means we only consider entries in a given axis (axes)

        Thus, when using the same columns as before, we now perform the sum operation
        over each slice individually:

        x111 x211 x121 x221 x112 x212 x122 x222
        slice(0)
        [[1   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0]]
        slice(2)
        [[0   1   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   0]]
        slice(7)
        [[0   0   0   0   0   0   0   0],
         [0   0   0   0   0   0   0   1]]
        """
        param_lin_op = linOpHelper((2,2,2), type="param", data=2)
        variable_lin_op = linOpHelper((2,2,2), type="variable", data=1)
        var_view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())
        mul_elem_lin_op = linOpHelper(data=param_lin_op)
        param_var_view = param_backend.mul_elem(mul_elem_lin_op, var_view)

        sum_entries_lin_op = linOpHelper(shape=(2,2,2), data=[(0,2), True], args=[variable_lin_op])
        out_view = param_backend.sum_entries(sum_entries_lin_op, param_var_view)
        out_repr = out_view.get_tensor_representation(0, 2)

        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.]])
        assert np.all(slice_idx_zero == expected_idx_zero)

        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([[0., 1., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 0.]])
        assert np.all(slice_idx_one == expected_idx_one)

        slice_idx_seven = out_repr.get_param_slice(7).toarray()[:, :-1]
        expected_idx_seven = np.array([[0., 0., 0., 0., 0., 0., 0., 0.],
                                    [0., 0., 0., 0., 0., 0., 0., 1.]])
        assert np.all(slice_idx_seven == expected_idx_seven)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0, 4) == param_var_view.get_tensor_representation(
            0, 4
        )

    def test_parametrized_nd_mul(self, param_backend):
        """
        Test parametrized mul linop with 3D variable: C_param (m,k) @ X (B,k,n).

        For parametrized C @ X where C is a 2D parameter (m,k) and X is 3D (B,k,n):
        - Each element of C becomes a separate parameter slice
        - vec(result) = (I_n ⊗ C ⊗ I_B) @ vec(X) with C parametrized

        Example: X = Variable((2, 2, 2)) with shape (B=2, k=2, n=2)
        C = Parameter((2, 2)) with 4 elements in Fortran order:
        - param_slice 0: C[0,0]
        - param_slice 1: C[1,0]
        - param_slice 2: C[0,1]
        - param_slice 3: C[1,1]

        Each output element: result[b, i, c] = sum_r C[i, r] * X[b, r, c]

        For param_slice 0 (C[0,0]), contributes to result[b, 0, c] from X[b, 0, c]:
                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [1    0    0    0    0    0    0    0  ]
        result[1]  [0    1    0    0    0    0    0    0  ]
        result[2]  [0    0    0    0    0    0    0    0  ]
        result[3]  [0    0    0    0    0    0    0    0  ]
        result[4]  [0    0    0    0    1    0    0    0  ]
        result[5]  [0    0    0    0    0    1    0    0  ]
        result[6]  [0    0    0    0    0    0    0    0  ]
        result[7]  [0    0    0    0    0    0    0    0  ]

        For param_slice 1 (C[1,0]), contributes to result[b, 1, c] from X[b, 0, c]:
                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [0    0    0    0    0    0    0    0  ]
        result[1]  [0    0    0    0    0    0    0    0  ]
        result[2]  [1    0    0    0    0    0    0    0  ]
        result[3]  [0    1    0    0    0    0    0    0  ]
        result[4]  [0    0    0    0    0    0    0    0  ]
        result[5]  [0    0    0    0    0    0    0    0  ]
        result[6]  [0    0    0    0    1    0    0    0  ]
        result[7]  [0    0    0    0    0    1    0    0  ]

        For param_slice 2 (C[0,1]), contributes to result[b, 0, c] from X[b, 1, c]:
                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [0    0    1    0    0    0    0    0  ]
        result[1]  [0    0    0    1    0    0    0    0  ]
        result[2]  [0    0    0    0    0    0    0    0  ]
        result[3]  [0    0    0    0    0    0    0    0  ]
        result[4]  [0    0    0    0    0    0    1    0  ]
        result[5]  [0    0    0    0    0    0    0    1  ]
        result[6]  [0    0    0    0    0    0    0    0  ]
        result[7]  [0    0    0    0    0    0    0    0  ]

        For param_slice 3 (C[1,1]), contributes to result[b, 1, c] from X[b, 1, c]:
                   X000 X100 X010 X110 X001 X101 X011 X111
        result[0]  [0    0    0    0    0    0    0    0  ]
        result[1]  [0    0    0    0    0    0    0    0  ]
        result[2]  [0    0    1    0    0    0    0    0  ]
        result[3]  [0    0    0    1    0    0    0    0  ]
        result[4]  [0    0    0    0    0    0    0    0  ]
        result[5]  [0    0    0    0    0    0    0    0  ]
        result[6]  [0    0    0    0    0    0    1    0  ]
        result[7]  [0    0    0    0    0    0    0    1  ]
        """
        # Reconfigure backend for this test: 4 parameter slices (2x2 parameter)
        param_backend.param_to_size = {-1: 1, 2: 4}
        param_backend.param_to_col = {2: 0, -1: 4}
        param_backend.param_size_plus_one = 5
        param_backend.var_length = 8

        variable_lin_op = linOpHelper((2, 2, 2), type="variable", data=1)
        view = param_backend.process_constraint(variable_lin_op, param_backend.get_empty_view())

        # Verify initial view is identity
        view_A = view.get_tensor_representation(0, 8)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(8, 8)).toarray()
        assert np.all(view_A == np.eye(8))

        # Create parametrized lhs C with shape (2, 2)
        lhs_parameter = linOpHelper((2, 2), type="param", data=2)

        mul_lin_op = linOpHelper(shape=(2, 2, 2), data=lhs_parameter, args=[variable_lin_op])
        out_view = param_backend.mul(mul_lin_op, view)
        out_repr = out_view.get_tensor_representation(0, 8)

        # Verify param_slice 0 (C[0,0]): contributes to result[b,0,c] from X[b,0,c]
        slice_idx_zero = out_repr.get_param_slice(0).toarray()[:, :-1]
        expected_idx_zero = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        assert np.all(slice_idx_zero == expected_idx_zero)

        # Verify param_slice 1 (C[1,0]): contributes to result[b,1,c] from X[b,0,c]
        slice_idx_one = out_repr.get_param_slice(1).toarray()[:, :-1]
        expected_idx_one = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ])
        assert np.all(slice_idx_one == expected_idx_one)

        # Verify param_slice 2 (C[0,1]): contributes to result[b,0,c] from X[b,1,c]
        slice_idx_two = out_repr.get_param_slice(2).toarray()[:, :-1]
        expected_idx_two = np.array([
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ])
        assert np.all(slice_idx_two == expected_idx_two)

        # Verify param_slice 3 (C[1,1]): contributes to result[b,1,c] from X[b,1,c]
        slice_idx_three = out_repr.get_param_slice(3).toarray()[:, :-1]
        expected_idx_three = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place
        assert out_view.get_tensor_representation(0, 8) == view.get_tensor_representation(0, 8)


class TestSciPyBackend:
    @staticmethod
    @pytest.fixture()
    def scipy_backend():
        kwargs = {
            "id_to_col": {1: 0},
            "param_to_size": {-1: 1, 2: 2},
            "param_to_col": {2: 0, -1: 2},
            "param_size_plus_one": 3,
            "var_length": 2,
        }
        backend = get_backend(s.SCIPY_CANON_BACKEND, **kwargs)
        assert isinstance(backend, SciPyCanonBackend)
        return backend

    def test_get_variable_tensor(self, scipy_backend):
        outer = scipy_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, "Should only be in variable with ID 1"
        inner = outer[1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert sp.issparse(tensor), "Should be a scipy sparse matrix or array"
        assert tensor.shape == (2, 2), "Should be a 1*2x2 tensor"
        assert np.all(tensor == np.eye(2)), "Should be eye(2)"

    @pytest.mark.parametrize("data", [np.array([[1, 2], [3, 4]]), sp.eye_array(2) * 4])
    def test_get_data_tensor(self, scipy_backend, data):
        outer = scipy_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert sp.issparse(tensor), "Should be scipy sparse"
        assert tensor.shape == (4, 1), "Should be a 1*4x1 tensor"
        expected = sp.csr_array(data.reshape((-1, 1), order="F"))
        assert (tensor != expected).nnz == 0

    def test_get_param_tensor(self, scipy_backend):
        shape = (2, 2)
        size = np.prod(shape)
        scipy_backend.param_to_size = {-1: 1, 3: 4}
        outer = scipy_backend.get_param_tensor(shape, 3)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {3}, "Should only be the parameter slice of parameter with id 3."
        tensor = inner[3]
        assert sp.issparse(tensor), "Should be scipy sparse"
        assert tensor.shape == (16, 1), "Should be a 4*4x1 tensor"
        assert (
            tensor.reshape((size, size)) != sp.eye_array(size, format="csr")
        ).nnz == 0, "Should be eye(4) when reshaping"

    def test_tensor_view_add_dicts(self, scipy_backend):
        view = scipy_backend.get_empty_view()

        one = sp.eye_array(1)
        two = sp.eye_array(1) * 2
        three = sp.eye_array(1) * 3

        assert view.add_dicts({}, {}) == {}
        assert view.add_dicts({"a": one}, {"a": two}) == {"a": three}
        assert view.add_dicts({"a": one}, {"b": two}) == {"a": one, "b": two}
        assert view.add_dicts({"a": {"c": one}}, {"a": {"c": one}}) == {"a": {"c": two}}
        with pytest.raises(
            ValueError, match=r"Values must either be dicts or \(<class 'scipy.sparse."
        ):
            view.add_dicts({"a": 1}, {"a": 2})

    @staticmethod
    @pytest.mark.parametrize("shape,batch_size,n", [
        ((2, 2), 1, 3),   # 2D case: I_3 ⊗ C
        ((2, 2), 2, 3),   # ND case: I_3 ⊗ C ⊗ I_2
        ((3, 2), 1, 4),   # 2D case with non-square matrix
        ((3, 2), 3, 2),   # ND case with non-square matrix
    ])
    def test_expand_parametric_slices(shape, batch_size, n):
        """
        Test expand_parametric_slices which applies I_n ⊗ C ⊗ I_batch to each param slice.

        For batch_size=1, this reduces to I_n ⊗ C (the 2D case).
        """
        rng = np.random.default_rng(42)
        p = 2  # number of parameter slices
        matrices = [sp.random_array(shape, random_state=rng, density=0.5).tocsc()
                    for _ in range(p)]
        stacked = sp.vstack(matrices, format="csc")
        result = sp.vstack(list(_expand_parametric_slices_mul(stacked, p, batch_size, n)))

        # Expected: apply I_n ⊗ C ⊗ I_batch to each slice, then vstack
        expected = sp.vstack([_apply_nd_kron_structure_mul(m, batch_size, n) for m in matrices])
        assert (expected != result).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape,batch_size,m", [
        ((2, 2), 1, 3),   # 2D case: C.T ⊗ I_m
        ((2, 2), 2, 3),   # ND case: C.T ⊗ I_{batch*m}
        ((3, 2), 1, 4),   # 2D case with non-square matrix
        ((3, 2), 3, 2),   # ND case with non-square matrix
    ])
    def test_expand_parametric_slices_rmul(shape, batch_size, m):
        """
        Test expand_parametric_slices_rmul which applies C.T ⊗ I_{batch*m} to each param slice.

        For batch_size=1, this reduces to C.T ⊗ I_m (the 2D case).
        """
        rng = np.random.default_rng(42)
        p = 2  # number of parameter slices
        matrices = [sp.random_array(shape, random_state=rng, density=0.5).tocsc()
                    for _ in range(p)]
        stacked = sp.vstack(matrices, format="csc")
        result = sp.vstack(list(_expand_parametric_slices_rmul(stacked, p, batch_size, m)))

        # Expected: apply C.T ⊗ I_{batch*m} to each slice, then vstack
        expected = sp.vstack([
            _apply_nd_kron_structure_rmul(mat, batch_size, m) for mat in matrices
        ])
        assert (expected != result).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_stacked_kron_l(shape, scipy_backend):
        p = 2
        reps = 3
        param_id = 2
        matrices = [sp.random_array(shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        repeated = scipy_backend._stacked_kron_l({param_id: stacked}, reps)
        repeated = repeated[param_id]
        expected = sp.vstack([sp.kron(m, sp.eye_array(reps)) for m in matrices])
        assert (expected != repeated).nnz == 0

    @staticmethod
    def test_reshape_single_constant_tensor(scipy_backend):
        a = sp.csc_array(np.tile(np.arange(6), 3).reshape((-1, 1)))
        # param_size=1 for non-parametric data
        reshaped = scipy_backend._reshape_single_constant_tensor(a, (3, 2), param_size=1)
        expected = np.arange(6).reshape((3, 2), order="F")
        expected = sp.csc_array(np.tile(expected, (3, 1)))
        assert (reshaped != expected).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_transpose_stacked(shape, scipy_backend):
        p = 2
        param_id = 2
        matrices = [sp.random_array(shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        transposed = scipy_backend._transpose_stacked(stacked, param_id)
        expected = sp.vstack([m.T for m in matrices])
        assert (expected != transposed).nnz == 0


class TestCooBackend:
    @staticmethod
    @pytest.fixture()
    def coo_backend():
        kwargs = {
            "id_to_col": {1: 0},
            "param_to_size": {-1: 1, 2: 2},
            "param_to_col": {2: 0, -1: 2},
            "param_size_plus_one": 3,
            "var_length": 2,
        }
        backend = get_backend(s.COO_CANON_BACKEND, **kwargs)
        assert isinstance(backend, CooCanonBackend)
        return backend

    def test_coo_tensor_negation(self):
        """Test CooTensor.__neg__ (unary negation)."""
        tensor = CooTensor(
            data=np.array([1.0, 2.0, 3.0]),
            row=np.array([0, 1, 2]),
            col=np.array([0, 1, 2]),
            param_idx=np.array([0, 0, 0]),
            m=3, n=3, param_size=1
        )
        negated = -tensor
        assert np.all(negated.data == np.array([-1.0, -2.0, -3.0]))
        assert np.all(negated.row == tensor.row)
        assert np.all(negated.col == tensor.col)

    def test_coo_tensor_addition(self):
        """Test CooTensor.__add__ (tensor addition)."""
        t1 = CooTensor(
            data=np.array([1.0, 2.0]),
            row=np.array([0, 1]),
            col=np.array([0, 1]),
            param_idx=np.array([0, 0]),
            m=2, n=2, param_size=1
        )
        t2 = CooTensor(
            data=np.array([3.0, 4.0]),
            row=np.array([0, 1]),
            col=np.array([1, 0]),
            param_idx=np.array([0, 0]),
            m=2, n=2, param_size=1
        )
        result = t1 + t2
        assert result.nnz == 4
        assert result.m == 2 and result.n == 2
        # Convert to dense to verify values
        dense = result.toarray()
        expected = np.array([[1.0, 3.0], [4.0, 2.0]])
        assert np.allclose(dense, expected)

    def test_coo_tensor_select_rows(self):
        """Test CooTensor.select_rows for row selection/reordering."""
        # Diagonal matrix with values 1, 2, 3 at positions (0,0), (1,1), (2,2)
        tensor = CooTensor(
            data=np.array([1.0, 2.0, 3.0]),
            row=np.array([0, 1, 2]),
            col=np.array([0, 1, 2]),
            param_idx=np.array([0, 0, 0]),
            m=3, n=3, param_size=1
        )
        # Select rows [2, 0]: new row 0 <- old row 2, new row 1 <- old row 0
        # Columns are preserved, so:
        # - old (2,2)=3 -> new (0,2)=3
        # - old (0,0)=1 -> new (1,0)=1
        selected = tensor.select_rows(np.array([2, 0]))
        assert selected.m == 2
        dense = selected.toarray()
        expected = np.array([[0.0, 0.0, 3.0], [1.0, 0.0, 0.0]])
        assert np.allclose(dense, expected)

    def test_get_variable_tensor(self, coo_backend):
        """Test CooCanonBackend.get_variable_tensor."""
        outer = coo_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, "Should only be in variable with ID 1"
        inner = outer[1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1"
        tensor = inner[-1]
        assert isinstance(tensor, CooTensor)
        assert tensor.m == 2 and tensor.n == 2
        assert np.all(tensor.toarray() == np.eye(2))

    def test_get_data_tensor(self, coo_backend):
        """Test CooCanonBackend.get_data_tensor."""
        data = np.array([[1, 2], [3, 4]])
        outer = coo_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, "Should only be constant variable ID"
        inner = outer[-1]
        assert inner.keys() == {-1}, "Should only be non-parametrized slice"
        tensor = inner[-1]
        assert isinstance(tensor, CooTensor)
        assert tensor.m == 4 and tensor.n == 1
        # Column vector in Fortran order: [1, 3, 2, 4]
        expected = data.flatten(order='F').reshape(-1, 1)
        assert np.allclose(tensor.toarray(), expected)

    @staticmethod
    @pytest.mark.parametrize("shape,reps", [
        ((2, 2), 1),
        ((2, 2), 3),
        ((3, 2), 2),
        ((2, 3), 4),
    ])
    def test_kron_eye_l(shape, reps):
        """Test _kron_eye_l against scipy.sparse.kron(A, I)."""
        rng = np.random.default_rng(42)
        param_size = 2

        # Create random sparse matrices for each param slice
        matrices = [sp.random_array(shape, random_state=rng, density=0.5)
                    for _ in range(param_size)]

        # Build CooTensor from stacked sparse
        stacked = sp.vstack(matrices)
        tensor = CooTensor.from_stacked_sparse(stacked, param_size)

        # Apply _kron_eye_l
        result = _kron_eye_l(tensor, reps)

        # Expected: apply kron(M, I_reps) to each slice, then stack
        expected = sp.vstack([sp.kron(m, sp.eye_array(reps)) for m in matrices])
        assert (expected != result.to_stacked_sparse()).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape,reps", [
        ((2, 2), 1),
        ((2, 2), 3),
        ((3, 2), 2),
        ((2, 3), 4),
    ])
    def test_kron_eye_r(shape, reps):
        """Test _kron_eye_r against scipy.sparse.kron(I, A)."""
        rng = np.random.default_rng(42)
        param_size = 2

        matrices = [sp.random_array(shape, random_state=rng, density=0.5)
                    for _ in range(param_size)]

        stacked = sp.vstack(matrices)
        tensor = CooTensor.from_stacked_sparse(stacked, param_size)

        result = _kron_eye_r(tensor, reps)

        # Expected: apply kron(I_reps, M) to each slice
        expected = sp.vstack([sp.kron(sp.eye_array(reps), m) for m in matrices])
        assert (expected != result.to_stacked_sparse()).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape,batch_size,n", [
        ((2, 2), 1, 1),   # No expansion
        ((2, 2), 1, 3),   # 2D case: I_3 ⊗ C
        ((2, 2), 2, 1),   # Only batch: C ⊗ I_2
        ((2, 2), 2, 3),   # ND case: I_3 ⊗ C ⊗ I_2
        ((3, 2), 1, 4),   # 2D non-square
        ((3, 2), 3, 2),   # ND non-square
    ])
    def test_kron_nd_structure_mul(shape, batch_size, n):
        """Test _kron_nd_structure_mul against scipy-based approach."""
        rng = np.random.default_rng(42)
        param_size = 2

        matrices = [sp.random_array(shape, random_state=rng, density=0.5)
                    for _ in range(param_size)]

        # Use CSR format because _expand_parametric_slices_mul slices the matrix,
        # and COO arrays don't support __getitem__ slicing.
        stacked = sp.vstack(matrices, format="csr")
        tensor = CooTensor.from_stacked_sparse(stacked, param_size)

        # Native COO implementation
        result = _kron_nd_structure_mul(tensor, batch_size, n)

        # Reference: scipy-based implementation
        expected = sp.vstack(list(
            _expand_parametric_slices_mul(stacked, param_size, batch_size, n)
        ))

        assert result.to_stacked_sparse().shape == expected.shape
        assert (expected != result.to_stacked_sparse()).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("const_shape,var_shape", [
        ((2, 3, 4), (2, 4, 5)),     # B=2, m=3, k=4, n=5
        ((3, 2, 2), (3, 2, 3)),     # B=3, m=2, k=2, n=3
        ((2, 2, 3, 4), (2, 2, 4, 2)),  # B=4, m=3, k=4, n=2
    ])
    def test_build_interleaved_mul(const_shape, var_shape):
        """Test _build_interleaved_mul against scipy-based _build_interleaved_matrix_mul."""
        rng = np.random.default_rng(42)
        const_data = rng.random(np.prod(const_shape))

        # Native COO implementation
        result = _build_interleaved_mul(const_data, const_shape, var_shape)

        # Reference: scipy-based implementation
        expected = _build_interleaved_matrix_mul(const_data, const_shape, var_shape)

        assert result.to_stacked_sparse().shape == expected.shape
        assert np.allclose(result.toarray(), expected.toarray())

    @staticmethod
    @pytest.mark.parametrize("shape, batch_size, m", [
        ((2, 2), 1, 1),   # 2D square, no batch
        ((2, 2), 1, 3),   # 2D square, m > 1
        ((2, 2), 2, 1),   # 2D square, batch > 1
        ((2, 2), 2, 3),   # 2D square, both > 1
        ((3, 2), 1, 4),   # 2D non-square
        ((3, 2), 3, 2),   # ND non-square
    ])
    def test_kron_nd_structure_rmul(shape, batch_size, m):
        """Test _kron_nd_structure_rmul against scipy-based approach."""
        rng = np.random.default_rng(42)
        param_size = 2

        matrices = [sp.random_array(shape, random_state=rng, density=0.5)
                    for _ in range(param_size)]

        # Use CSR format for slicing
        stacked = sp.vstack(matrices, format="csr")
        tensor = CooTensor.from_stacked_sparse(stacked, param_size)

        # Native COO implementation
        result = _kron_nd_structure_rmul(tensor, batch_size, m)

        # Reference: scipy-based implementation
        expected = sp.vstack(list(
            _expand_parametric_slices_rmul(stacked, param_size, batch_size, m)
        ))

        assert result.to_stacked_sparse().shape == expected.shape
        assert (expected != result.to_stacked_sparse()).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("var_shape, const_shape", [
        ((2, 3, 4), (2, 4, 5)),     # B=2, m=3, k=4, n=5
        ((3, 2, 2), (3, 2, 3)),     # B=3, m=2, k=2, n=3
        ((2, 2, 3, 4), (2, 2, 4, 2)),  # B=4, m=3, k=4, n=2
    ])
    def test_build_interleaved_rmul(var_shape, const_shape):
        """Test _build_interleaved_rmul against scipy-based _build_interleaved_matrix_rmul."""
        rng = np.random.default_rng(42)
        const_data = rng.random(np.prod(const_shape))

        # Native COO implementation
        result = _build_interleaved_rmul(const_data, const_shape, var_shape)

        # Reference: scipy-based implementation
        expected = _build_interleaved_matrix_rmul(const_data, const_shape, var_shape)

        assert result.to_stacked_sparse().shape == expected.shape
        assert np.allclose(result.toarray(), expected.toarray())

    @staticmethod
    def test_coo_reshape_vs_reshape_parametric_constant():
        """
        Test that coo_reshape and reshape_parametric_constant behave differently.

        - coo_reshape: Uses linear index reshaping, preserves all entries.
          Used by the 'reshape' linop for general reshape operations.
        - reshape_parametric_constant: Deduplicates based on param_idx for
          parametric tensors. Used for reshaping constant data in matmul.

        This is a regression test for an issue where using parametric reshape
        logic in coo_reshape caused DGP tests to fail with index out of bounds
        errors, because DGP generates tensors where param_idx doesn't map
        directly to positions in the target matrix.
        """
        from cvxpy.lin_ops.backends.coo_backend import (
            coo_reshape,
            reshape_parametric_constant,
        )

        # Create a parametric tensor with duplicated param_idx entries
        # (simulating what happens after broadcast_to)
        tensor = CooTensor(
            data=np.array([1.0, 1.0, 1.0, 1.0]),  # Duplicated entries
            row=np.array([0, 1, 2, 3]),
            col=np.array([0, 0, 0, 0]),
            param_idx=np.array([0, 0, 1, 1]),  # param_idx 0 and 1 appear twice
            m=4, n=1, param_size=2
        )

        # Reshape from column (4, 1) to (2, 2)
        new_m, new_n = 2, 2

        # coo_reshape: preserves all entries, uses linear index reshaping
        result_linear = coo_reshape(tensor, new_m, new_n)
        assert result_linear.nnz == 4, "coo_reshape should preserve all entries"
        # Linear indices: col*m + row = 0*4+0=0, 0*4+1=1, 0*4+2=2, 0*4+3=3
        # In (2,2): 0->(0,0), 1->(1,0), 2->(0,1), 3->(1,1)
        assert np.array_equal(result_linear.row, np.array([0, 1, 0, 1]))
        assert np.array_equal(result_linear.col, np.array([0, 0, 1, 1]))

        # reshape_parametric_constant: deduplicates based on param_idx
        result_param = reshape_parametric_constant(tensor, new_m, new_n)
        assert result_param.nnz == 2, "reshape_parametric_constant should deduplicate"
        # param_idx 0 -> position (0,0), param_idx 1 -> position (1,0)
        assert np.array_equal(result_param.param_idx, np.array([0, 1]))

    @staticmethod
    def test_get_constant_data_shape_for_broadcast_param():
        """
        Test that get_constant_data returns correct matrix structure for broadcast parameter.

        This is an intermediate check that catches reshape bugs in ND matmul.
        """
        np.random.seed(42)
        B, m, k, n = 2, 3, 4, 5
        P = cp.Parameter((m, k))
        P.value = np.random.randn(m, k)
        X = cp.Variable((B, k, n))
        expr = P @ X

        obj, _ = expr.canonical_form
        const_linop = obj.data  # broadcast_to

        # Set up backend
        prob = cp.Problem(cp.Minimize(cp.sum_squares(expr)))
        variables = prob.variables()
        parameters = prob.parameters()

        var_length = sum(int(np.prod(v.shape)) for v in variables)
        id_to_col = {variables[0].id: 0}
        param_to_size = {p.id: int(np.prod(p.shape)) for p in parameters}
        param_to_col = {p.id: 0 for p in parameters}
        param_size = sum(param_to_size.values())

        backend = CooCanonBackend(
            param_to_size=param_to_size,
            param_to_col=param_to_col,
            param_size_plus_one=param_size + 1,
            var_length=var_length,
            id_to_col=id_to_col
        )

        empty_view = backend.get_empty_view()
        lhs_data, is_param_free = backend.get_constant_data(
            const_linop,
            empty_view,
            target_shape=(m, k)
        )

        assert not is_param_free, "Parameter expression should not be param_free"

        # Check that reshaped tensor has correct matrix structure
        for param_id, tensor in lhs_data.items():
            assert tensor.m == m, f"Expected m={m}, got {tensor.m}"
            assert tensor.n == k, f"Expected n={k}, got {tensor.n}"
            assert tensor.nnz == m * k, f"Expected nnz={m * k}, got {tensor.nnz}"
            # Each param_idx should appear exactly once (no broadcast duplication)
            unique_params = np.unique(tensor.param_idx)
            assert len(unique_params) == param_to_size[param_id], \
                f"Expected {param_to_size[param_id]} unique param_idx, got {len(unique_params)}"
