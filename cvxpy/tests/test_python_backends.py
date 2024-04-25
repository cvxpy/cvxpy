from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
    CanonBackend,
    NumPyCanonBackend,
    PythonCanonBackend,
    SciPyCanonBackend,
    TensorRepresentation,
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


class TestBackendInstance:
    def test_get_backend(self):
        args = ({1: 0, 2: 2}, {-1: 1, 3: 1}, {3: 0, -1: 1}, 2, 4)

        backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, *args)
        assert isinstance(backend, SciPyCanonBackend)

        backend = CanonBackend.get_backend(s.NUMPY_CANON_BACKEND, *args)
        assert isinstance(backend, NumPyCanonBackend)

        with pytest.raises(KeyError):
            CanonBackend.get_backend("notabackend")


backends = [s.SCIPY_CANON_BACKEND, s.NUMPY_CANON_BACKEND]


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

        backend = CanonBackend.get_backend(request.param, **kwargs)
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

        transpose_lin_op = linOpHelper((2, 2))
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

        sum_entries_lin_op = linOpHelper()
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

        Thus, we need to to return [0, 1, 4, 5, 2, 3, 6, 7].
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

        backend = CanonBackend.get_backend(request.param, **kwargs)
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

        sum_entries_lin_op = linOpHelper()
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


class TestNumPyBackend:
    @staticmethod
    @pytest.fixture()
    def numpy_backend():
        kwargs = {
            "id_to_col": {1: 0},
            "param_to_size": {-1: 1, 2: 2},
            "param_to_col": {2: 0, -1: 2},
            "param_size_plus_one": 3,
            "var_length": 2,
        }
        backend = CanonBackend.get_backend(s.NUMPY_CANON_BACKEND, **kwargs)
        assert isinstance(backend, NumPyCanonBackend)
        return backend

    def test_get_variable_tensor(self, numpy_backend):
        outer = numpy_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, "Should only be in variable with ID 1"
        inner = outer[1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert isinstance(tensor, np.ndarray), "Should be a numpy array"
        assert tensor.shape == (1, 2, 2), "Should be a 1x2x2 tensor"
        assert np.all(tensor[0] == np.eye(2)), "Should be eye(2)"

    @pytest.mark.parametrize("data", [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
    def test_get_data_tensor(self, numpy_backend, data):
        outer = numpy_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert isinstance(tensor, np.ndarray), "Should be a numpy array"
        assert isinstance(tensor[0], np.ndarray), "Inner matrix should also be a numpy array"
        assert tensor.shape == (1, 4, 1), "Should be a 1x4x1 tensor"
        expected = numpy_backend._to_dense(data).reshape((-1, 1), order="F")
        assert np.all(tensor[0] == expected)

    def test_get_param_tensor(self, numpy_backend):
        shape = (2, 2)
        size = np.prod(shape)
        outer = numpy_backend.get_param_tensor(shape, 3)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {3}, "Should only be the parameter slice of parameter with id 3."
        tensor = inner[3]
        assert isinstance(tensor, np.ndarray), "Should be a numpy array"
        assert tensor.shape == (4, 4, 1), "Should be a 4x4x1 tensor"
        assert np.all(tensor[:, :, 0] == np.eye(size)), "Should be eye(4) along axes 1 and 2"

    def test_tensor_view_add_dicts(self, numpy_backend):
        view = numpy_backend.get_empty_view()

        one = np.array([1])
        two = np.array([2])
        three = np.array([3])

        assert view.add_dicts({}, {}) == {}
        assert view.add_dicts({"a": one}, {"a": two}) == {"a": three}
        assert view.add_dicts({"a": one}, {"b": two}) == {"a": one, "b": two}
        assert view.add_dicts({"a": {"c": one}}, {"a": {"c": one}}) == {"a": {"c": two}}
        with pytest.raises(
            ValueError, match="Values must either be dicts or <class 'numpy.ndarray'>"
        ):
            view.add_dicts({"a": 1}, {"a": 2})


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
        backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, **kwargs)
        assert isinstance(backend, SciPyCanonBackend)
        return backend

    def test_get_variable_tensor(self, scipy_backend):
        outer = scipy_backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, "Should only be in variable with ID 1"
        inner = outer[1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert isinstance(tensor, sp.spmatrix), "Should be a scipy sparse matrix"
        assert tensor.shape == (2, 2), "Should be a 1*2x2 tensor"
        assert np.all(tensor == np.eye(2)), "Should be eye(2)"

    @pytest.mark.parametrize("data", [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
    def test_get_data_tensor(self, scipy_backend, data):
        outer = scipy_backend.get_data_tensor(data)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensor = inner[-1]
        assert isinstance(tensor, sp.spmatrix), "Should be a scipy sparse matrix"
        assert tensor.shape == (4, 1), "Should be a 1*4x1 tensor"
        expected = sp.csr_matrix(data.reshape((-1, 1), order="F"))
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
        assert isinstance(tensor, sp.spmatrix), "Should be a scipy sparse matrix"
        assert tensor.shape == (16, 1), "Should be a 4*4x1 tensor"
        assert (
            tensor.reshape((size, size)) != sp.eye(size, format="csr")
        ).nnz == 0, "Should be eye(4) when reshaping"

    def test_tensor_view_add_dicts(self, scipy_backend):
        view = scipy_backend.get_empty_view()

        one = sp.eye(1)
        two = sp.eye(1) * 2
        three = sp.eye(1) * 3

        assert view.add_dicts({}, {}) == {}
        assert view.add_dicts({"a": one}, {"a": two}) == {"a": three}
        assert view.add_dicts({"a": one}, {"b": two}) == {"a": one, "b": two}
        assert view.add_dicts({"a": {"c": one}}, {"a": {"c": one}}) == {"a": {"c": two}}
        with pytest.raises(
            ValueError, match="Values must either be dicts or " "<class 'scipy.sparse."
        ):
            view.add_dicts({"a": 1}, {"a": 2})

    @staticmethod
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_stacked_kron_r(shape, scipy_backend):
        p = 2
        reps = 3
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        repeated = scipy_backend._stacked_kron_r({param_id: stacked}, reps)
        repeated = repeated[param_id]
        expected = sp.vstack([sp.kron(sp.eye(reps), m) for m in matrices])
        assert (expected != repeated).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 3), (4, 4)])
    def test_stacked_kron_l(shape, scipy_backend):
        p = 2
        reps = 3
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        repeated = scipy_backend._stacked_kron_l({param_id: stacked}, reps)
        repeated = repeated[param_id]
        expected = sp.vstack([sp.kron(m, sp.eye(reps)) for m in matrices])
        assert (expected != repeated).nnz == 0

    @staticmethod
    def test_reshape_single_constant_tensor(scipy_backend):
        a = sp.csc_matrix(np.tile(np.arange(6), 3).reshape((-1, 1)))
        reshaped = scipy_backend._reshape_single_constant_tensor(a, (3, 2))
        expected = np.arange(6).reshape((3, 2), order="F")
        expected = sp.csc_matrix(np.tile(expected, (3, 1)))
        assert (reshaped != expected).nnz == 0

    @staticmethod
    @pytest.mark.parametrize("shape", [(1, 1), (2, 2), (3, 2), (2, 3)])
    def test_transpose_stacked(shape, scipy_backend):
        p = 2
        param_id = 2
        matrices = [sp.random(*shape, random_state=i, density=0.5) for i in range(p)]
        stacked = sp.vstack(matrices)
        transposed = scipy_backend._transpose_stacked(stacked, param_id)
        expected = sp.vstack([m.T for m in matrices])
        assert (expected != transposed).nnz == 0
