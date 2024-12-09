import inspect

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
