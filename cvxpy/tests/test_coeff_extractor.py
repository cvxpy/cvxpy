from dataclasses import dataclass

import numpy as np
import pytest

import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.lin_ops.canon_backend import TensorRepresentation
from cvxpy.utilities.coeff_extractor import CoeffExtractor


@dataclass
class MockInverseData:
    var_offsets: dict
    x_length: int
    var_shapes: dict
    param_shapes: dict
    param_to_size: dict
    param_id_map: dict


@pytest.fixture
def coeff_extractor():
    inverset_data = MockInverseData(
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


def test_issue_2437():
    """
    This is a MWE / regression test for the issue reported in #2437.
    """

    N = 3

    t_cost = np.array([0.01, 0.02, 0.03])
    alpha = np.array([0.04, 0.05, 0.06])
    ivol = np.array([0.07, 0.08, 0.09])

    w = cp.Variable(N, name="w")

    risk = (cp.multiply(w, ivol) ** 2).sum()
    U = w @ alpha - risk - cp.abs(w) @ t_cost
    problem = cp.Problem(cp.Maximize(U), [])

    assert np.isclose(
        problem.solve(solver=cp.CLARABEL, use_quad_obj=True),
        problem.solve(solver=cp.CLARABEL, use_quad_obj=False),
        rtol=0,
        atol=1e-3,
    )


class TestBlockQuadExtraction:
    """Tests for block-structured quadratic form extraction."""

    @pytest.fixture
    def block_extractor(self):
        """Create an extractor for a size-6 variable."""
        inverse_data = MockInverseData(
            var_offsets={1: 0},
            x_length=6,
            var_shapes={1: (6,)},
            param_shapes={},
            param_to_size={-1: 1},
            param_id_map={-1: 0},
        )
        backend = cp.CPP_CANON_BACKEND
        return CoeffExtractor(inverse_data, backend)

    def test_block_extraction_contiguous_identity(self, block_extractor):
        """Test contiguous blocks with identity P."""
        import scipy.sparse as sp

        # Setup: 2 blocks of size 3, P = I_6
        N = 6
        P_coo = sp.eye(N, format='coo')

        # block_indices for 2 output elements, each using 3 contiguous inputs
        block_indices = [np.arange(0, 3), np.arange(3, 6)]

        # c_part: coefficients [1.0, 2.0] for 2 outputs, 1 param
        c_part = np.array([[1.0], [2.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Expected: 6 entries total (diagonal), first 3 scaled by 1.0, last 3 by 2.0
        assert len(result.data) == 6
        assert result.shape == (6, 6)

        # Check that diagonal entries are correct
        # Entries 0,1,2 should have value 1.0; entries 3,4,5 should have value 2.0
        sorted_indices = np.argsort(result.row)
        sorted_data = result.data[sorted_indices]
        sorted_rows = result.row[sorted_indices]
        sorted_cols = result.col[sorted_indices]

        np.testing.assert_array_equal(sorted_rows, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_equal(sorted_cols, [0, 1, 2, 3, 4, 5])
        np.testing.assert_array_almost_equal(sorted_data, [1.0, 1.0, 1.0, 2.0, 2.0, 2.0])

    def test_block_extraction_noncontiguous(self, block_extractor):
        """Test non-contiguous blocks (e.g., axis=1 reduction with column-major)."""
        import scipy.sparse as sp

        # P = I_6, simulating x shape (3, 2) flattened column-major
        # For axis=1 reduction, each row of x becomes an output
        # Row 0: indices [0, 3], Row 1: indices [1, 4], Row 2: indices [2, 5]
        N = 6
        P_coo = sp.eye(N, format='coo')

        # Non-contiguous blocks
        block_indices = [
            np.array([0, 3]),  # Output 0: x[0,0] and x[0,1]
            np.array([1, 4]),  # Output 1: x[1,0] and x[1,1]
            np.array([2, 5]),  # Output 2: x[2,0] and x[2,1]
        ]

        # c_part: coefficients [1.0, 2.0, 3.0] for 3 outputs, 1 param
        c_part = np.array([[1.0], [2.0], [3.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Expected: 6 entries (diagonal), scaled by respective coefficients
        assert len(result.data) == 6
        assert result.shape == (6, 6)

        # Check diagonal entries match expected scaling
        # Index 0 -> coef 1.0, Index 1 -> coef 2.0, Index 2 -> coef 3.0
        # Index 3 -> coef 1.0, Index 4 -> coef 2.0, Index 5 -> coef 3.0
        expected_data = {0: 1.0, 1: 2.0, 2: 3.0, 3: 1.0, 4: 2.0, 5: 3.0}
        for i in range(len(result.data)):
            row, col, data = result.row[i], result.col[i], result.data[i]
            assert row == col  # Diagonal
            assert np.isclose(data, expected_data[row])

    def test_block_extraction_sparse_blocks(self, block_extractor):
        """Test with sparse (non-identity) P blocks."""
        import scipy.sparse as sp

        # P has off-diagonal entries within blocks
        # Block 0: entries at (0,0), (0,1), (1,0), (1,1), (2,2)
        # Block 1: entries at (3,3), (4,4), (5,5)
        row = np.array([0, 0, 1, 1, 2, 3, 4, 5])
        col = np.array([0, 1, 0, 1, 2, 3, 4, 5])
        data = np.array([1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0])
        P_coo = sp.coo_matrix((data, (row, col)), shape=(6, 6))

        block_indices = [np.arange(0, 3), np.arange(3, 6)]
        c_part = np.array([[2.0], [3.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Block 0: 5 entries scaled by 2.0
        # Block 1: 3 entries scaled by 3.0
        assert len(result.data) == 8

        # Check scaling
        for i in range(len(result.data)):
            row_idx = result.row[i]
            expected_coef = 2.0 if row_idx < 3 else 3.0
            orig_data = data[np.where((P_coo.row == row_idx) & (P_coo.col == result.col[i]))[0][0]]
            assert np.isclose(result.data[i], orig_data * expected_coef)

    def test_block_extraction_with_parameters(self, block_extractor):
        """Test block extraction with multiple parameter columns."""
        import scipy.sparse as sp

        # P = I_4, 2 blocks of size 2
        N = 4
        P_coo = sp.eye(N, format='coo')

        block_indices = [np.arange(0, 2), np.arange(2, 4)]

        # c_part: 2 outputs x 2 params
        # Output 0: coefs [1.0, 2.0], Output 1: coefs [3.0, 4.0]
        c_part = np.array([[1.0, 2.0], [3.0, 4.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 2)

        # Each block has 2 diagonal entries, each scaled by 2 param coefficients
        # Total: 2 blocks * 2 entries * 2 params = 8 entries
        assert len(result.data) == 8

        # Check parameter offsets are present
        assert set(result.parameter_offset) == {0, 1}

    def test_block_extraction_empty_blocks(self, block_extractor):
        """Test that empty blocks are handled correctly."""
        import scipy.sparse as sp

        # P has entries only in the first 3 indices
        row = np.array([0, 1, 2])
        col = np.array([0, 1, 2])
        data = np.array([1.0, 1.0, 1.0])
        P_coo = sp.coo_matrix((data, (row, col)), shape=(6, 6))

        # Block 1 has no P entries
        block_indices = [np.arange(0, 3), np.arange(3, 6)]
        c_part = np.array([[1.0], [2.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Only 3 entries from first block
        assert len(result.data) == 3
        np.testing.assert_array_almost_equal(result.data, [1.0, 1.0, 1.0])

    def test_block_extraction_zero_coefficients(self, block_extractor):
        """Test that blocks with zero coefficients are skipped."""
        import scipy.sparse as sp

        N = 4
        P_coo = sp.eye(N, format='coo')

        block_indices = [np.arange(0, 2), np.arange(2, 4)]
        # Second block has zero coefficient
        c_part = np.array([[1.0], [0.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Only 2 entries from first block
        assert len(result.data) == 2
        assert all(r < 2 for r in result.row)  # All from first block

    def test_block_extraction_all_empty(self, block_extractor):
        """Test that completely empty extraction returns empty TensorRepresentation."""
        import scipy.sparse as sp

        N = 4
        P_coo = sp.eye(N, format='coo')

        block_indices = [np.arange(0, 2), np.arange(2, 4)]
        # All blocks have zero coefficients
        c_part = np.array([[0.0], [0.0]])

        result = block_extractor._extract_block_quad(P_coo, c_part, block_indices, 1)

        # Should return empty TensorRepresentation with correct shape
        assert len(result.data) == 0
        assert result.shape == (4, 4)

    def test_symbolic_quad_form_with_block_indices(self):
        """Test that SymbolicQuadForm correctly stores block_indices."""
        x = cp.Variable(6)
        P = np.eye(6)
        expr = cp.quad_form(x, P)  # Just for the original_expression

        block_indices = [np.arange(0, 3), np.arange(3, 6)]
        sqf = SymbolicQuadForm(x, cp.Constant(P), expr, block_indices=block_indices)

        assert sqf.block_indices is block_indices
        data = sqf.get_data()
        assert len(data) == 2
        assert data[1] is block_indices

    def test_symbolic_quad_form_without_block_indices(self):
        """Test that SymbolicQuadForm works without block_indices (backward compat)."""
        x = cp.Variable(2)
        P = np.eye(2)
        expr = cp.quad_form(x, P)

        sqf = SymbolicQuadForm(x, cp.Constant(P), expr)

        assert sqf.block_indices is None
        data = sqf.get_data()
        assert len(data) == 2
        assert data[1] is None
