import numpy as np
import pytest

from cvxpy.lin_ops.canon_backend import ScipyTensorView, TensorRepresentation


def test_tensor_representation():
    A = TensorRepresentation(np.array([0]), np.array([0]), np.array([1]), np.array([10]))
    B = TensorRepresentation(np.array([1]), np.array([1]), np.array([1]), np.array([20]))
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
    with pytest.raises(ValueError, match="Values must either be dicts or lists"):
        ScipyTensorView.add_dicts({"a": 1}, {"a": 2})
