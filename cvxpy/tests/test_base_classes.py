import inspect

import numpy as np
import pytest

from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.atoms.atom import Atom
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.leaf import Leaf
from cvxpy.interface.base_matrix_interface import BaseMatrixInterface
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.utilities.canonical import Canonical


@pytest.mark.parametrize("expected_abc", [
    Canonical,
    Expression, Atom, AffAtom, Leaf,
    Constraint,
    Reduction, Solver, ConicSolver,
    ParamProb,
    BaseMatrixInterface,
])
def test_is_abstract(expected_abc):
    assert inspect.isabstract(expected_abc)


class _DenseMatrixInterface(BaseMatrixInterface):
    def const_to_matrix(self, value, convert_scalars: bool = False):
        return np.asarray(value)

    def identity(self, size):
        return np.eye(size)

    def shape(self, matrix):
        return np.asarray(matrix).shape

    def scalar_value(self, matrix):
        return np.asarray(matrix).item()

    def scalar_matrix(self, value, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        return np.full(shape, value)

    def reshape(self, matrix, shape):
        return np.reshape(matrix, shape, order="F")


def test_base_matrix_interface_block_formatting_branches():
    iface = _DenseMatrixInterface()

    assert iface.size(np.zeros((2, 3))) == 6
    np.testing.assert_array_equal(iface.zeros((2, 2)), np.zeros((2, 2)))
    np.testing.assert_array_equal(iface.ones((1, 3)), np.ones((1, 3)))
    assert iface.index(np.array([[4]]), (0, 0)) == 4

    mat = np.zeros((2, 3))
    iface.block_add(mat, np.array([[2]]), 0, 0, 2, 2)
    np.testing.assert_array_equal(mat[:, :2], 2 * np.ones((2, 2)))

    mat = np.zeros((2, 3))
    iface.block_add(mat, np.array([[1], [2], [3], [4]]), 0, 0, 2, 2)
    np.testing.assert_array_equal(mat[:, :2], np.array([[1, 3], [2, 4]]))

    mat = np.zeros((4, 1))
    iface.block_add(mat, np.array([[1, 3], [2, 4]]), 0, 0, 4, 1)
    np.testing.assert_array_equal(mat[:, 0], np.array([1, 2, 3, 4]))

    mat = np.zeros((1, 2))
    iface.block_add(mat, [[5, 6]], 0, 0, 1, 2)
    np.testing.assert_array_equal(mat, np.array([[5, 6]]))
