import sys

import numpy as np
import pytest
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.psd import PSD
from cvxpy.constraints.zero import Equality
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr


class TestAttributes:

    @pytest.mark.parametrize("sparsity", [[np.array([0, 0]), np.array([0, 1])], [(0, 1), (0, 2)]])
    def test_sparsity_pattern(self, sparsity):
        X = cp.Variable((3, 3), sparsity=sparsity)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X.value, z)

    def test_sparsity_condition(self):
        if tuple(int(s) for s in sys.version.split('.')[:2]) < (3, 13):
            return
        data = np.arange(8).reshape((2,2,2))
        mask = np.where(data % 2 == 0)
        X = cp.Variable((2,2,2), sparsity=mask)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = np.zeros((2,2,2))
        z[mask] = -1
        assert np.allclose(X.value, z)

    def test_sparsity_invalid_input(self):
        with pytest.raises(ValueError, match="mismatching number of index"
                           " arrays for shape; got 3, expected 2"):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1), (0, 1)])

    def test_sparsity_incorrect_dim(self):
        with pytest.raises(
            ValueError, match="all index and data arrays must have the same length"
        ):
            cp.Variable((3, 3), sparsity=[(0, 1), (0, 1, 2)])

    def test_sparsity_out_of_bounds(self):
        with pytest.raises(
            ValueError, match="axis 1 index 5 exceeds matrix dimension 3"
        ):
            cp.Variable((3, 3), sparsity=[(0, 1, 2), (3, 4, 5)])

    def test_sparsity_0D_variable(self):
        with pytest.raises(ValueError, match="Indices should have 0 dimensions."):
            cp.Variable(sparsity=[(0, 1)])

    def test_sparsity_reduces_num_var(self):
        X = cp.Variable((3, 3), sparsity=[(0, 1), (0, 2)])
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 2

        X = cp.Variable((3, 3))
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        assert prob.get_problem_data(cp.CLARABEL)[0]['A'].shape[1] == 9
        
    def test_sparsity_assign_value(self):
        X = cp.Variable((10, 10))
        sparsity = [(0, 2, 1, 2), (0, 1, 2, 2)]
        A = cp.Parameter((10, 10), sparsity=sparsity)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= A])
        A_value = np.zeros((10, 10))
        A_value[sparsity[0], sparsity[1]] = -1
        with pytest.warns(
            RuntimeWarning,
            match='Writing to a sparse CVXPY expression via `.value` is discouraged.'
                  ' Use `.value_sparse` instead'
        ):
            A.value = A_value
        
        prob.solve()
        z = np.zeros((10, 10))
        z[A.sparse_idx] = -1
        assert np.allclose(X.value, z)
        
        A.value_sparse = sp.coo_array((-np.ones(4), sparsity), (10, 10))
        prob.solve()
        assert np.allclose(X.value, z)
        
        z = sp.coo_array(([-1, -3, -2, -4], [(0, 1, 2, 2), (0, 2, 1, 2)]), shape=(10, 10))
        z1 = sp.coo_array(([-1, -4, -2, -3], [(0, 2, 2, 1), (0, 2, 1, 2)]), shape=(10, 10))
        A.value_sparse = z
        prob.solve()
        assert np.allclose(z.toarray(), z1.toarray())
        assert np.allclose(X.value, z1.toarray())
        assert np.allclose(X.value, z.toarray())
        A.value_sparse = z1
        prob.solve()
        assert np.allclose(z.toarray(), z1.toarray())
        assert np.allclose(X.value, z1.toarray())
        assert np.allclose(X.value, z.toarray())

    def test_sparsity_incorrect_pattern(self):
        A = cp.Parameter((3, 3), sparsity=[(0, 2, 1, 2), (0, 1, 2, 2)])
        with pytest.raises(
            ValueError, match="Parameter value must be zero outside of sparsity pattern."
        ):
            A.value = np.ones((3, 3))
        with pytest.raises(
            ValueError,
            match='Invalid sparsity pattern '
                  r'\(array\(\[0\](, dtype=int32)?\), array\(\[0\](, dtype=int32)?\)\)'
                  ' for Parameter value.'
        ):
            A.value_sparse = sp.coo_array(([1], ([0], [0])), (3, 3))
            
    def test_sparsity_read_value(self):
        sparsity = [(0, 2, 1, 2), (0, 1, 2, 2)]
        X = cp.Variable((3, 3), sparsity=sparsity)
        assert X.value is None
        
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1])
        prob.solve()
        with pytest.warns(
            RuntimeWarning,
            match='Reading from a sparse CVXPY expression via `.value` is discouraged.'
                  ' Use `.value_sparse` instead'
        ):
            X_value = X.value
        
        z = np.zeros((3, 3))
        z[X.sparse_idx] = -1
        assert np.allclose(X_value, z)
        
        X_value_sparse = X.value_sparse
        assert np.allclose(X_value_sparse.toarray(), z)

    def test_infeasible_sparse(self):
        # Create a sparse variable 
        x = cp.Variable(100, sparsity=(np.array([1, 15, 45, 67, 89]),))
        objective = cp.Minimize(cp.sum_squares(x))

        # Create infeasible constraints
        constraints = [x[1] >= 10, x[1] <= 1]
        problem = cp.Problem(objective, constraints)
        problem.solve()
        assert problem.status == "infeasible"

    def test_diag_value_sparse(self):
        X = cp.Variable((3, 3), diag=True)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= -1, X <= 1])
        prob.solve()
        z = -np.eye(3)
        assert sp.issparse(X.value) and X.value.format == "dia"
        assert np.allclose(X.value.toarray(), z)

    def test_variable_bounds(self):
        # Valid bounds: Scalars promoted to arrays
        x = cp.Variable((2, 2), name="x", bounds=[0, 10])
        assert np.array_equal(x.bounds[0], np.zeros((2, 2)))
        assert np.array_equal(x.bounds[1], np.full((2, 2), 10))

        # Valid bounds: Arrays with matching shape
        bounds = [np.zeros((2, 2)), np.ones((2, 2)) * 5]
        x = cp.Variable((2, 2), name="x", bounds=bounds)
        assert np.array_equal(x.bounds[0], np.zeros((2, 2)))
        assert np.array_equal(x.bounds[1], np.ones((2, 2)) * 5)

        # Valid bounds: One bound is None
        bounds = [None, 5]
        x = cp.Variable((2, 2), name="x", bounds=bounds)
        assert np.array_equal(x.bounds[0], np.full((2, 2), -np.inf))
        assert np.array_equal(x.bounds[1], np.full((2, 2), 5))

        # Invalid bounds: Length not equal to 2
        bounds = [0]  # Only one item
        with pytest.raises(ValueError, match="Bounds should be a list of two items."):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Non-iterable type
        bounds = 10  # Not iterable
        with pytest.raises(ValueError, match="Bounds should be a list of two items."):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Arrays with non-matching shape
        bounds = [np.zeros((3, 3)), np.ones((3, 3))]
        with pytest.raises(
            ValueError,
            match="Bounds should be None, scalars, or arrays with the same dimensions "
                "as the variable/parameter.",
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Lower bound > Upper bound
        bounds = [5, 0]
        with pytest.raises(
            ValueError,
            match="Invalid bounds: some upper bounds are less than "
                "corresponding lower bounds.",
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: NaN in bounds
        bounds = [np.nan, 10]
        with pytest.raises(
            ValueError, match="np.nan is not feasible as lower or upper bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Upper bound is -inf
        bounds = [0, -np.inf]
        with pytest.raises(
            ValueError, match="-np.inf is not feasible as an upper bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

        # Invalid bounds: Lower bound is inf
        bounds = [np.inf, 10]
        with pytest.raises(
            ValueError, match="np.inf is not feasible as a lower bound."
        ):
            x = cp.Variable((2, 2), name="x", bounds=bounds)

    def test_scalar_bool(self):
        x = cp.Variable(nonpos=True)
        n = cp.Variable(boolean=True)
        cp.Problem(cp.Maximize(x), [n == 1]).solve()

    def test_scalar_int(self):
        x = cp.Variable(nonpos=True)
        n = cp.Variable(integer=True)
        cp.Problem(cp.Maximize(x), [n == 1]).solve()

    def test_boolean_var_value(self):
        # ensure that boolean variables can be assigned values
        # https://github.com/cvxpy/cvxpy/issues/2879
        x = cp.Variable(2, boolean=True)
        val = np.array([True, False])
        x.value = val
        np.testing.assert_array_equal(x.value, val.astype(float), strict=True)

    def test_integer_var_value(self):
        x = cp.Variable(2, integer=True)
        val = np.array([1, 2], dtype=int)
        x.value = val
        np.testing.assert_array_equal(x.value, val.astype(float), strict=True)

class TestMultipleAttributes:

    def test_multiple_attributes(self) -> None:
        x = cp.Variable(shape=(2,2), symmetric=True, nonneg=True, integer=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == target])
        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[1]) is Equality

        prob.solve()
        assert np.allclose(x.value, target)

    def test_nonneg_PSD(self) -> None:
        x = cp.Variable(shape=(2,2), PSD=True, nonneg=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == target])
        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is PSD
        assert type(new_prob.apply(prob)[0].constraints[1]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[2]) is Equality

        prob.solve()
        assert np.allclose(x.value, target)

    def test_nonpos_NSD(self) -> None:
        x = cp.Variable(shape=(2,2), NSD=True, nonpos=True)
        target = np.array(np.eye(2) * 5)
        prob = cp.Problem(cp.Minimize(0), [x == -target])

        new_prob = CvxAttr2Constr(prob, reduce_bounds=True)
        assert type(new_prob.apply(prob)[0].constraints[0]) is PSD
        assert type(new_prob.apply(prob)[0].constraints[1]) is Inequality
        assert type(new_prob.apply(prob)[0].constraints[2]) is Equality

        prob.solve()
        assert np.allclose(x.value, -target)

    def test_integer_bounds(self) -> None:
        x = cp.Variable(shape=(2,2), integer=True, bounds=[-1.5, 2])
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [])
        prob.solve()
        assert prob.value == -4
        assert np.allclose(x.value, np.ones((2,2)) * -1)

    def test_nonpos_nonneg_variable(self) -> None:
        x = cp.Variable(shape=(2,2), nonpos=True, nonneg=True)
        target = np.zeros((2,2))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [])
        prob.solve()
        assert np.allclose(prob.value, 0)
        assert np.allclose(x.value, target)
    
    def test_sparse_symmetric_variable(self) -> None:
        with pytest.raises(
            ValueError, 
            match="A CVXPY Variable cannot have more than one of the following attributes"
        ):
            cp.Variable(shape=(2, 2), symmetric=True, sparsity=[(0, 1), (0, 1)])

    def test_sparse_bounded_variable(self) -> None:
        x = cp.Variable(shape=(2,2), sparsity=[(0,1),(0,1)],
                        bounds=[np.array([[-1.5, -4], [-3, -2.5]]), 10])
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [])
        prob.solve()
        assert np.allclose(prob.value, -4)
        assert np.allclose(x.value, np.array([[-1.5, 0], [0, -2.5]]))

    def test_sparse_integer_variable(self) -> None:
        x = cp.Variable(shape=(2,2), sparsity=[(0,1),(0,1)], integer=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x>=-5.5])
        prob.solve()
        assert np.allclose(prob.value, -10)
        assert np.allclose(x.value, np.eye(2) * -5)
        
    def test_parameter_multiple_attributes(self) -> None:
        """Test parameters with multiple attributes."""
        # Test parameter with nonpos and integer attributes
        p = cp.Parameter(shape=(2, 2), nonpos=True, integer=True)
        p.value = -np.ones((2, 2))
        x = cp.Variable(shape=(2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        prob.solve()
        assert np.allclose(x.value, -np.ones((2, 2)))

        # TODO make parameter validation work for multiple attributes.
        # # Invalid assignment should raise ValueError
        # with pytest.raises(ValueError, match="Parameter value must be nonpositive."):
        #     p.value = np.ones((2, 2))
        
        # # Non-integer value should raise ValueError
        # with pytest.raises(ValueError, match="Parameter value must be integer."):
        #     p.value = -np.ones((2, 2)) * 0.5

    def test_parameter_bounds_and_attributes(self) -> None:
        """Test parameters with bounds and other attributes."""
        # Parameter with bounds and nonneg
        p = cp.Parameter(shape=(2, 2), nonneg=True, bounds=[0, 10])
        p.value = np.ones((2, 2)) * 5
        x = cp.Variable(shape=(2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        prob.solve()
        assert np.allclose(x.value, np.ones((2, 2)) * 5)

        # TODO make parameter validation work for multiple attributes.
        # # Test values outside bounds
        # with pytest.raises(ValueError, match="Parameter value must be nonnegative."):
        #     p.value = -np.ones((2, 2))
        
        # with pytest.raises(ValueError, 
        #  match="Parameter value must be less than or equal to upper bound."):
        #     p.value = np.ones((2, 2)) * 15
    
    def test_parameter_sparsity_and_attributes(self) -> None:
        """Test parameters with sparsity and other attributes."""
        sparsity = [(0, 1), (0, 1)]
        p = cp.Parameter(shape=(2, 2), sparsity=sparsity, nonneg=True)
        
        # Valid value assignment
        p_value = sp.eye_array(2).tocoo()
        p.value_sparse = p_value
        
        x = cp.Variable(shape=(2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        prob.solve()
        assert np.allclose(x.value, p_value.todense())
        
        # TODO make parameter validation work for multiple attributes.
        # # Invalid value assignment (negative and in sparsity pattern)
        # p_value = np.zeros((2, 2))
        # p_value[sparsity[0], sparsity[1]] = -1
        # with pytest.raises(ValueError, match="Parameter value must be nonnegative."):
        #     p.value = p_value
            
        # # Value out of sparsity pattern
        # p_value = np.ones((2, 2))
        # with pytest.raises(ValueError, 
        #  match="Parameter value must be zero outside of sparsity pattern."):
        #     p.value = p_value
    
    def test_sparse_complex_variable(self) -> None:
        """Test sparse complex variables."""
        sparsity = [(0, 1), (0, 1)]
        x = cp.Variable(shape=(2, 2), complex=True, sparsity=sparsity)
        
        # Check that x has correct attributes
        assert x.is_complex()
        assert hasattr(x, 'sparse_idx')
        
        # Create target value with only nonzeros at sparsity locations
        target = np.zeros((2, 2), dtype=complex)
        target[0, 0] = 1+1j
        target[1, 1] = 2-2j
        
        # Set up and solve problem
        prob = cp.Problem(
            cp.Minimize(cp.norm(x, 'fro')), 
            [x[0, 0] == target[0, 0], 
            x[1, 1] == target[1, 1]]
        )
        prob.solve(solver=cp.CLARABEL)
        
        # Check sparse value access
        sparse_value = x.value_sparse
        assert sp.issparse(sparse_value)
        assert np.allclose(sparse_value.todense(), target)
    
    def test_parameter_complex_sparsity(self) -> None:
        """Test complex parameters with sparsity."""
        sparsity = [(0, 1), (0, 1)]
        p = cp.Parameter(shape=(2, 2), complex=True, sparsity=sparsity)
        
        # Valid value assignment - sparse complex matrix
        complex_values = np.array([1+2j, 3-4j])
        sparse_matrix = sp.coo_array((complex_values, sparsity), shape=(2, 2))
        p.value_sparse = sparse_matrix
        
        # Use in optimization problem
        x = cp.Variable(shape=(2, 2), complex=True)
        prob = cp.Problem(cp.Minimize(cp.norm(x - p, 'fro')))
        prob.solve(solver=cp.CLARABEL)
        
        # Verify results
        assert np.allclose(x.value, p.value)
        assert np.allclose(
            x.value,
            p.value_sparse.todense()
        )
    
    def test_parameter_psd_and_attributes(self) -> None:
        """Test parameters with PSD and other attributes."""
        p = cp.Parameter(shape=(2, 2), PSD=True, nonneg=True)
        
        # Valid PSD and nonneg value
        p.value = np.array([[2, 0], [0, 3]])
        x = cp.Variable(shape=(2, 2))
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= p])
        prob.solve()
        assert np.allclose(x.value, np.array([[2, 0], [0, 3]]))
        
        # TODO make parameter validation work for multiple attributes.
        # # Invalid: Not PSD
        # with pytest.raises(ValueError, match="Parameter value must be positive semidefinite."):
        #     p.value = np.array([[1, 2], [2, 1]])
        
        # # Invalid: Not nonneg
        # with pytest.raises(ValueError, match="Parameter value must be nonnegative."):
        #     p.value = np.array([[-1, 0], [0, 1]])

    def test_bool_int_variable(self):
        """Test boolean and integer attributes on a variable."""
        # Create a variable with both boolean and integer attributes
        x = cp.Variable(shape=(2, 2), boolean=[(0, 0), (0, 1)], integer=[(1, 1), (0, 1)])
        
        # Set up a simple problem
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= np.array([[0.5, -0.5], [2.3, 3.2]])])
        prob.solve()
        
        # Check that the solution is feasible
        assert (x.value == np.array([[1, 0], [3, 4]])).all()

    def test_variable_repr(self):
        # test boolean attributes
        x = cp.Variable((10, 10), name="x", nonneg=True)
        assert x.__repr__() == "Variable((10, 10), x, nonneg=True)"

        # test bounds representation
        y = cp.Variable((10, 10), name="y", bounds=[0, 10])
        assert y.__repr__() == (
            "Variable((10, 10), y, bounds=([[0 0 ... 0 0]\n"
            " [0 0 ... 0 0]\n"
            " ...\n"
            " [0 0 ... 0 0]\n"
            " [0 0 ... 0 0]], [[10 10 ... 10 10]\n"
            " [10 10 ... 10 10]\n"
            " ...\n"
            " [10 10 ... 10 10]\n"
            " [10 10 ... 10 10]]))"
        )

        # test sparse, mixed-integer/boolean representation
        z = cp.Variable((10, 10), name="z", sparsity=[(0, 1), (0, 2)])
        assert z.__repr__() == (
            "Variable((10, 10), z, sparsity=[(0, 1), (0, 2)])"
        )
