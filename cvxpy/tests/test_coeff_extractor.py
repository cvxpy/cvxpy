from dataclasses import dataclass

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
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
    assert np.allclose(coeffs[1]["q"], np.zeros((2, 3)))
    P = coeffs[1]["P"]
    assert len(P) == 3
    assert np.allclose(P[0], np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0]]))
    assert len(P[1]) == 2
    assert np.allclose(P[1][0], np.array([0, 1, 0, 1]))
    assert np.allclose(P[1][1], np.array([0, 1, 0, 1]))
    assert P[2] == (2, 2)
    assert np.allclose(constant.toarray(), np.zeros((3)))
