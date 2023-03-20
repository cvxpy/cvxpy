from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.lin_ops.canon_backend import (
    CanonBackend,
    ScipyCanonBackend,
    ScipyTensorView,
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
    A = TensorRepresentation(np.array([10]), np.array([0]), np.array([1]), np.array([0]))
    B = TensorRepresentation(np.array([20]), np.array([1]), np.array([1]), np.array([1]))
    combined = TensorRepresentation.combine([A, B])
    assert np.all(combined.data == np.array([10, 20]))
    assert np.all(combined.row == np.array([0, 1]))
    assert np.all(combined.col == np.array([1, 1]))
    assert np.all(combined.parameter_offset == np.array([0, 1]))


def test_scipy_tensor_view_combine_potentially_none():
    assert ScipyTensorView.combine_potentially_none(None, None) is None
    a = {"a": [1]}
    b = {"b": [2]}
    assert ScipyTensorView.combine_potentially_none(a, None) == a
    assert ScipyTensorView.combine_potentially_none(None, a) == a
    assert ScipyTensorView.combine_potentially_none(a, b) == ScipyTensorView.add_dicts(a, b)


def test_scipy_tensor_view_add_dicts():
    assert ScipyTensorView.add_dicts({}, {}) == {}
    assert ScipyTensorView.add_dicts({"a": [1]}, {"a": [2]}) == {"a": [3]}
    assert ScipyTensorView.add_dicts({"a": [1]}, {"b": [2]}) == {"a": [1], "b": [2]}
    assert ScipyTensorView.add_dicts({"a": {"c": [1]}}, {"a": {"c": [1]}}) == {'a': {'c': [2]}}
    with pytest.raises(ValueError, match="Values must either be dicts or lists"):
        ScipyTensorView.add_dicts({"a": 1}, {"a": 2})


class TestBackend:

    def test_get_backend(self):
        args = ({1: 0, 2: 2}, {-1: 1, 3: 1}, {3: 0, -1: 1}, 2, 4)
        backend = CanonBackend.get_backend(s.SCIPY_CANON_BACKEND, *args)
        assert isinstance(backend, ScipyCanonBackend)

        with pytest.raises(KeyError):
            CanonBackend.get_backend('notabackend')


class TestScipyBackend:
    # Not used explicitly in most test cases.
    # Some tests specify other values as needed within the test case.
    param_size_plus_one = 2
    id_to_col = {1: 0, 2: 2}
    param_to_size = {-1: 1, 3: 1}
    param_to_col = {3: 0, -1: 1}
    var_length = 4

    @pytest.fixture
    def backend(self):
        return ScipyCanonBackend(self.id_to_col, self.param_to_size, self.param_to_col,
                                 self.param_size_plus_one, self.var_length)

    def test_mapping(self, backend):
        func = backend.get_func('sum')
        assert isinstance(func, Callable)
        with pytest.raises(KeyError):
            backend.get_func('notafunc')

    def test_gettensor(self, backend):
        outer = backend.get_variable_tensor((2,), 1)
        assert outer.keys() == {1}, "Should only be in variable with ID 1"
        inner = outer[1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensors = inner[-1]
        assert isinstance(tensors, list), "Should be list of tensors"
        assert len(tensors) == 1, "Should be a single tensor"
        assert (tensors[0] != sp.eye(2, format='csr')).nnz == 0, "Should be eye(2)"

    @pytest.mark.parametrize('data', [np.array([[1, 2], [3, 4]]), sp.eye(2) * 4])
    def test_get_data_tensor(self, backend, data):
        outer = backend.get_data_tensor(data)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {-1}, "Should only be in parameter slice -1, i.e. non parametrized."
        tensors = inner[-1]
        assert isinstance(tensors, list), "Should be list of tensors"
        assert len(tensors) == 1, "Should be a single tensor"
        expected = sp.csr_matrix(data.reshape((-1, 1), order="F"))
        assert (tensors[0] != expected).nnz == 0

    def test_get_param_tensor(self, backend):
        shape = (2, 2)
        size = np.prod(shape)
        outer = backend.get_param_tensor(shape, 3)
        assert outer.keys() == {-1}, "Should only be constant variable ID."
        inner = outer[-1]
        assert inner.keys() == {3}, "Should only be the parameter slice of parameter with id 3."
        tensors = inner[3]
        assert isinstance(tensors, list), "Should be list of tensors"
        assert len(tensors) == size, "Should be a tensor for each element of the parameter"
        assert (sp.hstack(tensors) != sp.eye(size, format='csr')).nnz == 0, \
            'Should be eye(4) along axes 1 and 2'

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        neg_lin_op = linOpHelper()
        out_view = backend.neg(neg_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        assert np.all(A == -np.eye(4))

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        transpose_lin_op = linOpHelper((2, 2))
        out_view = backend.transpose(transpose_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        upper_tri_lin_op = linOpHelper(args=[linOpHelper((2, 2))])
        out_view = backend.upper_tri(upper_tri_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array(
            [[0, 0, 1, 0]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        index_2d_lin_op = linOpHelper(data=[slice(0, 2, 1), slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_2d_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        assert np.all(A == expected)

        index_1d_lin_op = linOpHelper(data=[slice(0, 1, 1)], args=[variable_lin_op])
        out_view = backend.index(index_1d_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        diag_mat_lin_op = linOpHelper(shape=(2, 2))
        out_view = backend.diag_mat(diag_mat_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0],
             [0, 0, 0, 1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        diag_vec_lin_op = linOpHelper(shape=(2, 2))
        out_view = backend.diag_vec(diag_vec_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 2)).toarray()
        expected = np.array(
            [[1, 0],
             [0, 0],
             [0, 0],
             [0, 1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_sum_entries(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

        sum_entries(x) means we consider the entries in all rows, i.e., we sum along axis 0.

        Thus, when using the same columns as before, we now have

         x1  x2
        [[1  1]]
        """

        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        sum_entries_lin_op = linOpHelper()
        out_view = backend.sum_entries(sum_entries_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 2)).toarray()
        expected = np.array(
            [[1, 1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((1,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)
        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(1, 1)).toarray()
        assert np.all(view_A == np.eye(1))

        promote_lin_op = linOpHelper((3,))
        out_view = backend.promote(promote_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(3, 1)).toarray()
        expected = np.array(
            [[1],
             [1],
             [1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_hstack(self, backend):
        """
        define x,y = Variable((1,)), Variable((1,))

        hstack([x, y]) means the expression should be represented in the A matrix as if it
        was a Variable of shape (2,), i.e.,

          x  y
        [[1  0],
         [0  1]]
        """

        lin_op_x = linOpHelper((1,), type='variable', data=1)
        lin_op_y = linOpHelper((1,), type='variable', data=2)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, {1: 0, 2: 1},
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        hstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        out_view = backend.hstack(hstack_lin_op, empty_view)
        A = out_view.get_tensor_representation(0)

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

        lin_op_x = linOpHelper((1, 2), type='variable', data=1)
        lin_op_y = linOpHelper((1, 2), type='variable', data=2)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, {1: 0, 2: 2},
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        vstack_lin_op = linOpHelper(args=[lin_op_x, lin_op_y])
        out_view = backend.vstack(vstack_lin_op, empty_view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0],
             [0, 0, 1, 0],
             [0, 1, 0, 0],
             [0, 0, 0, 1]]
        )
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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))

        mul_lin_op = linOpHelper(data=lhs, args=[variable_lin_op])
        out_view = backend.mul(mul_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array(
            [[1, 2, 0, 0],
             [3, 4, 0, 0],
             [0, 0, 1, 2],
             [0, 0, 3, 4]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_parametrized_mul(self, backend):
        """
        Continuing the previous example when the lhs is a parameter, instead of multiplying with
        known values, the matrix is split up into four slices, each representing an element of the
        parameter, i.e. instead of
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

        param_size_plus_one = 5
        id_to_col = {1: 0}
        param_to_size = {-1: 1, 2: 4}
        param_to_col = {2: 0, -1: 4}
        var_length = 4

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(param_size_plus_one, id_to_col,
                                                    param_to_size, param_to_col,
                                                    var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs_parameter = linOpHelper((2, 2), type='param', data=2)

        mul_lin_op = linOpHelper(data=lhs_parameter, args=[variable_lin_op])
        out_view = backend.mul(mul_lin_op, view)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_view.tensor[1][2][0].toarray()
        expected_idx_zero = np.array(
            [[1., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 0.]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_view.tensor[1][2][1].toarray()
        expected_idx_one = np.array(
            [[0., 0., 0., 0.],
             [1., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 1., 0.]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        # indices are: variable 1, parameter 2, 2 index of the list
        slice_idx_two = out_view.tensor[1][2][2].toarray()
        expected_idx_two = np.array(
            [[0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 1.],
             [0., 0., 0., 0.]]
        )
        assert np.all(slice_idx_two == expected_idx_two)

        # indices are: variable 1, parameter 2, 3 index of the list
        slice_idx_three = out_view.tensor[1][2][3].toarray()
        expected_idx_three = np.array(
            [[0., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 1.]]
        )
        assert np.all(slice_idx_three == expected_idx_three)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        rhs = linOpHelper((2,), type='dense_const', data=np.array([1, 2]))

        rmul_lin_op = linOpHelper(data=rhs, args=[variable_lin_op])
        out_view = backend.rmul(rmul_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 4)).toarray()
        expected = np.array(
            [[1, 0, 2, 0],
             [0, 1, 0, 2]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_parametrized_rmul(self, backend):
        """
        Continuing the previous example when the rhs is a parameter, instead of multiplying with
        known values, the matrix is split up into two slices, each representing an element of the
        parameter, i.e. instead of
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

        param_size_plus_one = 3
        id_to_col = {1: 0}
        param_to_size = {-1: 1, 2: 2}
        param_to_col = {2: 0, -1: 2}
        var_length = 4

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(param_size_plus_one, id_to_col,
                                                    param_to_size, param_to_col,
                                                    var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        rhs_parameter = linOpHelper((2,), type='param', data=2)

        rmul_lin_op = linOpHelper(data=rhs_parameter, args=[variable_lin_op])
        out_view = backend.rmul(rmul_lin_op, view)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_view.tensor[1][2][0].toarray()
        expected_idx_zero = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_view.tensor[1][2][1].toarray()
        expected_idx_one = np.array(
            [[0, 0, 1, 0],
             [0, 0, 0, 1]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_mul_elementwise(self, backend):
        """
        define x = Variable((2,)) with
        [x1, x2]

        x is represented as eye(2) in the A matrix, i.e.,

         x1  x2
        [[1  0],
         [0  1]]

         mul_elementwise(x, a) means 'a' is reshaped into a column vector and multiplied with A.
         E.g. for a = (2,3), we obtain

         x1  x2
        [[2  0],
         [0  3]]
        """

        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        lhs = linOpHelper((2,), type='dense_const', data=np.array([2, 3]))

        mul_elementwise_lin_op = linOpHelper(data=lhs)
        out_view = backend.mul_elem(mul_elementwise_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(2, 2)).toarray()
        expected = np.array(
            [[2, 0],
             [0, 3]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_mul_elementwise_parametrized(self, backend):
        """
        Continuing the previous example when 'a' is a parameter, instead of multiplying with known
        values, the matrix is split up into two slices, each representing an element of the
        parameter, i.e. instead of
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

        param_size_plus_one = 3
        id_to_col = {1: 0}
        param_to_size = {-1: 1, 2: 2}
        param_to_col = {2: 0, -1: 2}
        var_length = 2

        variable_lin_op = linOpHelper((2,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(param_size_plus_one, id_to_col,
                                                    param_to_size, param_to_col,
                                                    var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(2, 2)).toarray()
        assert np.all(view_A == np.eye(2))

        lhs_parameter = linOpHelper((2,), type='param', data=2)

        mul_elementwise_lin_op = linOpHelper(data=lhs_parameter)
        out_view = backend.mul_elem(mul_elementwise_lin_op, view)

        # indices are: variable 1, parameter 2, 0 index of the list
        slice_idx_zero = out_view.tensor[1][2][0].toarray()
        expected_idx_zero = np.array(
            [[1, 0],
             [0, 0]]
        )
        assert np.all(slice_idx_zero == expected_idx_zero)

        # indices are: variable 1, parameter 2, 1 index of the list
        slice_idx_one = out_view.tensor[1][2][1].toarray()
        expected_idx_one = np.array(
            [[0, 0],
             [0, 1]]
        )
        assert np.all(slice_idx_one == expected_idx_one)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        lhs = linOpHelper((2, 2), type='dense_const', data=np.array([[1, 2], [3, 4]]))

        div_lin_op = linOpHelper(data=lhs)
        out_view = backend.div(div_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(4, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 0],
             [0, 1/3, 0, 0],
             [0, 0, 1/2, 0],
             [0, 0, 0, 1/4]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        trace_lin_op = linOpHelper(args=[variable_lin_op])
        out_view = backend.trace(trace_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(1, 4)).toarray()
        expected = np.array(
            [[1, 0, 0, 1]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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

        variable_lin_op = linOpHelper((3,), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(3, 3)).toarray()
        assert np.all(view_A == np.eye(3))

        f = linOpHelper((3,), type='dense_const', data=np.array([1, 2, 3]))
        conv_lin_op = linOpHelper(data=f, shape=(5, 1), args=[variable_lin_op])

        out_view = backend.conv(conv_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(5, 3)).toarray()
        expected = np.array(
            [[1., 0., 0.],
             [2., 1., 0.],
             [3., 2., 1.],
             [0., 3., 2.],
             [0., 0., 3.]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

    def test_kron_r(self, backend):
        """
        define x = Variable((2,2)) with
        [[x11, x12],
         [x21, x22]]

        and
        a = [[1, 2]],

        kron(a, x) means we have
        [[x11, x12, 2x11, 2x12],
         [x21, x22, 2x21, 2x22]]

        i.e. as represented in the A matrix (again in column-major order)

         x11 x21 x12 x22
        [[1   0   0   0],
         [0   1   0   0],
         [0   0   1   0],
         [0   0   0   1],
         [2   0   0   0],
         [0   2   0   0],
         [0   0   2   0],
         [0   0   0   2]]
        """

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        a = linOpHelper((1, 2), type='dense_const', data=np.array([[1, 2]]))
        kron_r_lin_op = linOpHelper(data=a, args=[variable_lin_op])

        out_view = backend.kron_r(kron_r_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.],
             [2., 0., 0., 0.],
             [0., 2., 0., 0.],
             [0., 0., 2., 0.],
             [0., 0., 0., 2.]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)

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
        """

        variable_lin_op = linOpHelper((2, 2), type='variable', data=1)
        empty_view = ScipyTensorView.get_empty_view(self.param_size_plus_one, self.id_to_col,
                                                    self.param_to_size, self.param_to_col,
                                                    self.var_length)

        view = backend.process_constraint(variable_lin_op, empty_view)

        # cast to numpy
        view_A = view.get_tensor_representation(0)
        view_A = sp.coo_matrix((view_A.data, (view_A.row, view_A.col)), shape=(4, 4)).toarray()
        assert np.all(view_A == np.eye(4))

        a = linOpHelper((1, 2), type='dense_const', data=np.array([[1, 2]]))
        kron_l_lin_op = linOpHelper(data=a, args=[variable_lin_op])

        out_view = backend.kron_l(kron_l_lin_op, view)
        A = out_view.get_tensor_representation(0)

        # cast to numpy
        A = sp.coo_matrix((A.data, (A.row, A.col)), shape=(8, 4)).toarray()
        expected = np.array(
            [[1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [2., 0., 0., 0.],
             [0., 2., 0., 0.],
             [0., 0., 1., 0.],
             [0., 0., 0., 1.],
             [0., 0., 2., 0.],
             [0., 0., 0., 2.]]
        )
        assert np.all(A == expected)

        # Note: view is edited in-place:
        assert out_view.get_tensor_representation(0) == view.get_tensor_representation(0)
