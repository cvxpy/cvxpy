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
        X = cp.Variable((3, 3))
        sparsity = [(0, 2, 1, 2), (0, 1, 2, 2)]
        A = cp.Parameter((3, 3), sparsity=sparsity)
        prob = cp.Problem(cp.Minimize(cp.sum(X)), [X >= A])
        A_value = np.zeros((3, 3))
        A_value[sparsity[0], sparsity[1]] = -1
        with pytest.warns(
            RuntimeWarning,
            match='Writing to a sparse CVXPY expression via `.value` is discouraged.'
                  ' Use `.value_sparse` instead'
        ):
            A.value = A_value
        
        prob.solve()
        z = np.zeros((3, 3))
        z[A.sparse_idx] = -1
        assert np.allclose(X.value, z)
        
        A.value_sparse = sp.coo_array((-np.ones(4), sparsity))
        prob.solve()
        assert np.allclose(X.value, z)
        
        z = sp.coo_array(([-1, -3, -2, -4], [(0, 1, 2, 2), (0, 2, 1, 2)]))
        z1 = sp.coo_array(([-1, -4, -2, -3], [(0, 2, 2, 1), (0, 2, 1, 2)]))
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
            match="A CVXPY Variable cannot have more than one of the following attributes be true"
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
