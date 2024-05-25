from dataclasses import dataclass

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor


@dataclass
class MockeInverseData:
    var_offsets: dict
    x_length: int
    var_shapes: dict
    param_shapes: dict
    param_to_size: dict
    param_id_map: dict


@pytest.fixture
def coeff_extractor():
    inverset_data = MockeInverseData(
        var_offsets={1: 0},
        x_length=2,
        var_shapes={1: (2,)},
        param_shapes={2: (), 3: ()},
        param_to_size={-1: 1, 2: 1, 3: 1},
        param_id_map={2: 0, 3: 1, -1: 2},
    )
    backend = cp.CPP_CANON_BACKEND
    return CoeffExtractor(inverset_data, backend)


def test_issue_2402_scalar_parameter():
    """
    This is the problem reported in #2402, failing to solve when two parameters
    are used on quadratic forms with the same variable.
    """

    r =  np.array([-0.48,  0.11,  0.09, -0.39,  0.03])
    Sigma = np.array([
        [2.4e-04, 1.3e-04, 2.0e-04, 1.6e-04, 2.0e-04],
        [1.3e-04, 2.8e-04, 2.1e-04, 1.7e-04, 1.5e-04],
        [2.0e-04, 2.1e-04, 5.8e-04, 3.3e-04, 2.3e-04],
        [1.6e-04, 1.7e-04, 3.3e-04, 6.9e-04, 2.1e-04],
        [2.0e-04, 1.5e-04, 2.3e-04, 2.1e-04, 3.6e-04]])

    w = cp.Variable(5)
    risk_aversion = cp.Parameter(value=1., nonneg=True)
    ridge_coef = cp.Parameter(value=0., nonneg=True)
    
    obj_func = r @ w - risk_aversion * cp.quad_form(w, Sigma) -  ridge_coef * cp.sum_squares(w)
    objective = cp.Maximize(obj_func)
    fixed_w = np.array([10, 11, 12, 13, 14])
    constraints = [w == fixed_w]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    expected_value = r @ fixed_w - risk_aversion.value * np.dot(fixed_w, np.dot(Sigma, fixed_w)) - \
        ridge_coef.value * np.sum(np.square(fixed_w))
    
    assert np.isclose(prob.value, expected_value)
    assert np.allclose(w.value, fixed_w)


def test_issue_2402_scalar_constant():
    """
    This slight modification uses a constant instead of a parameter,
    which was a separate issue in the same problem.
    """

    r =  np.array([-0.48,  0.11,  0.09, -0.39,  0.03])
    Sigma = np.array([
        [2.4e-04, 1.3e-04, 2.0e-04, 1.6e-04, 2.0e-04],
        [1.3e-04, 2.8e-04, 2.1e-04, 1.7e-04, 1.5e-04],
        [2.0e-04, 2.1e-04, 5.8e-04, 3.3e-04, 2.3e-04],
        [1.6e-04, 1.7e-04, 3.3e-04, 6.9e-04, 2.1e-04],
        [2.0e-04, 1.5e-04, 2.3e-04, 2.1e-04, 3.6e-04]])

    w = cp.Variable(5)
    risk_aversion = cp.Parameter(value=1., nonneg=True)
    ridge_coef = 0
    
    obj_func = r @ w - risk_aversion * cp.quad_form(w, Sigma) -  ridge_coef * cp.sum_squares(w)
    objective = cp.Maximize(obj_func)
    fixed_w = np.array([10, 11, 12, 13, 14])
    constraints = [w == fixed_w]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    expected_value = r @ fixed_w - risk_aversion.value * np.dot(fixed_w, np.dot(Sigma, fixed_w)) - \
        ridge_coef * np.sum(np.square(fixed_w))
    
    assert np.isclose(prob.value, expected_value)
    assert np.allclose(w.value, fixed_w)


def test_issue_2402_vector():
    """
    This slight modification with the ridge_coef as a vector also failed
    with a different error due to a dimension mismatch.
    """

    r =  np.array([-0.48,  0.11,  0.09, -0.39,  0.03])
    Sigma = np.array([
        [2.4e-04, 1.3e-04, 2.0e-04, 1.6e-04, 2.0e-04],
        [1.3e-04, 2.8e-04, 2.1e-04, 1.7e-04, 1.5e-04],
        [2.0e-04, 2.1e-04, 5.8e-04, 3.3e-04, 2.3e-04],
        [1.6e-04, 1.7e-04, 3.3e-04, 6.9e-04, 2.1e-04],
        [2.0e-04, 1.5e-04, 2.3e-04, 2.1e-04, 3.6e-04]
    ])

    w = cp.Variable(5)
    risk_aversion = cp.Parameter(value=2., nonneg=True)
    ridge_coef = cp.Parameter((5), value=np.arange(5), nonneg=True)

    obj_func = r @ w - risk_aversion * cp.quad_form(w, Sigma) - \
        cp.sum(cp.multiply(cp.multiply(ridge_coef, np.array([5,6,7,8,9])), cp.square(w)))

    objective = cp.Maximize(obj_func)
    fixed_w = np.array([10, 11, 12, 13, 14])
    constraints = [w == fixed_w]
    prob = cp.Problem(objective, constraints)
    prob.solve()
    
    expected_value = r @ fixed_w - risk_aversion.value * np.dot(fixed_w, np.dot(Sigma, fixed_w)) - \
        np.sum(ridge_coef.value * np.array([5,6,7,8,9]) * np.square(fixed_w))

    assert np.isclose(prob.value, expected_value)
    assert np.allclose(w.value, fixed_w)


def test_problem_end_to_end():
    """
    This is a MWE / regression test for the issue reported in #2402.
    """
    x = cp.Variable(2)
    p1 = cp.Parameter(value=1.0, nonneg=True)
    p2 = cp.Parameter(value=0.0, nonneg=True)

    P = np.eye(2)

    objective = cp.Minimize(p1 * cp.quad_form(x, P) + p2 * cp.sum_squares(x))
    problem = cp.Problem(objective, constraints=[cp.sum(x) == 1])
    problem.solve()
    assert np.isclose(problem.value, 0.5)
    assert np.allclose(x.value, [0.5, 0.5])


def test_coeff_extractor(coeff_extractor):
    """
    This is a unit test for the same problem.
    The variable and parameter namings are derived from the problem above.
    """
    x1 = cp.Variable(2, var_id=1)
    x14 = cp.Variable((1, 1), var_id=14)
    x16 = cp.Variable(var_id=16)

    p2 = cp.Parameter(value=1.0, nonneg=True, id=2)
    p3 = cp.Parameter(value=0.0, nonneg=True, id=3)

    affine_expr = p2 * x14 + p3 * x16

    quad_forms = {
        x14.id: (
            p2 * x14,
            1,
            SymbolicQuadForm(x1, cp.Constant(np.eye(2)), cp.quad_form(x1, np.eye(2))),
        ),
        x16.id: (
            p3 * x16,
            1,
            SymbolicQuadForm(x1, cp.Constant(np.eye(2)), cp.quad_over_lin(x1, 1.0)),
        ),
    }
    coeffs, constant = coeff_extractor.extract_quadratic_coeffs(affine_expr, quad_forms)
    
    assert len(coeffs) == 1
    assert np.allclose(coeffs[1]["q"].toarray(), np.zeros((2, 3)))
    P = coeffs[1]["P"]
    assert isinstance(P, TensorRepresentation)
    assert np.allclose(P.data, np.ones((4)))
    assert np.allclose(P.row, np.array([0, 1, 0, 1]))
    assert np.allclose(P.col, np.array([0, 1, 0, 1]))
    assert P.shape == (2, 2)
    assert np.allclose(P.parameter_offset, np.array([0, 0, 1, 1]))
    assert np.allclose(constant.toarray(), np.zeros((3)))
