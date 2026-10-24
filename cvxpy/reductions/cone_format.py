"""
Copyright, the CVXPY authors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import numpy as np

from cvxpy.constraints import PSD, SOC, ExpCone, NonNeg, PowCone3D, PowConeND, SvecPSD, Zero
from cvxpy.problems.param_prob import ParamProb
from cvxpy.reductions.reduction import Reduction


def _spacing_rows(spacing, streak, num_blocks, offset):
    """Rows taken by ``num_blocks`` runs of ``streak`` rows, ``spacing`` apart.

    Block ``b`` occupies rows ``offset + b*(streak + spacing)`` onwards, for
    ``streak`` rows; the ``spacing`` rows after it are left to another
    argument of the same constraint. Returned flattened, block by block.
    """
    return (np.arange(num_blocks * (streak + spacing))
            .reshape(num_blocks, streak + spacing)[:, :streak].ravel() + offset)


def _identity_rows(constr, exp_cone_order):
    """Zero, NonNeg, PSD and SvecPSD keep every row where it is."""
    return np.arange(constr.size)


def _soc_rows(constr, exp_cone_order):
    """Group each t row with the X rows of its own cone."""
    assert constr.axis == 0, 'SOC must be lowered to axis == 0'
    # Handle scalar X (shape is empty tuple)
    x_dim = constr.args[1].shape[0] if constr.args[1].shape else 1
    num_cones = constr.args[0].size
    return np.concatenate([_spacing_rows(x_dim, 1, num_cones, 0),
                           _spacing_rows(1, x_dim, num_cones, 1)])


def _exp_rows(constr, exp_cone_order):
    """Interleave (x, y, z) per cone, in the solver's argument order."""
    return np.concatenate([
        _spacing_rows(len(exp_cone_order) - 1, 1, arg.size, exp_cone_order[i])
        for i, arg in enumerate(constr.args)])


def _pow3d_rows(constr, exp_cone_order):
    """Interleave (x, y, z) per cone."""
    return np.concatenate([_spacing_rows(2, 1, arg.size, i)
                           for i, arg in enumerate(constr.args)])


def _pownd_rows(constr, exp_cone_order):
    """Each cone is a column of W followed by its entry of z."""
    W = constr.args[0]
    m, n = (W.shape[0], 1) if W.ndim == 1 else W.shape
    assert constr.args[1].size == n
    return np.concatenate([_spacing_rows(0, 1, m, (m + 1) * j) for j in range(n)]
                          + [_spacing_rows(m, 1, n, m)])


# Where each constraint type's rows land within its own block.
_ROW_LAYOUT = {
    Zero: _identity_rows,
    NonNeg: _identity_rows,
    PSD: _identity_rows,
    SvecPSD: _identity_rows,
    SOC: _soc_rows,
    ExpCone: _exp_rows,
    PowCone3D: _pow3d_rows,
    PowConeND: _pownd_rows,
}


def restruct_permutation(constraints, exp_cone_order):
    """Where each stuffed constraint row moves, and its sign.

    The cone-restructuring matrix R is always a signed permutation: every
    block is either +-I (Zero contributes -I; NonNeg, PSD and SvecPSD
    contribute I) or a pure interleaving of the constraint's arguments
    (SOC and the exponential/power cones). So R never has to be built, let
    alone multiplied by -- restructuring is a row remap plus a sign flip.

    Purely structural: derived from the constraint types and shapes plus
    ``exp_cone_order``, never from coefficient or parameter values.

    Returns ``(new_row, sign)``, both indexed by the *old* row: row ``i``
    becomes row ``new_row[i]``, scaled by ``sign[i]``. ``None`` when there are
    no constraints, i.e. nothing to reorder.
    """
    if not constraints:
        return None
    new_row, sign, base = [], [], 0
    for constr in constraints:
        layout = _ROW_LAYOUT.get(type(constr))
        if layout is None:
            raise ValueError("Unsupported constraint type.")
        rows = layout(constr, exp_cone_order)
        new_row.append(rows + base)
        # Only the zero cone is negated; it enters as -I.
        sign.append(np.full(rows.size, -1.0 if type(constr) == Zero else 1.0))
        base += rows.size
    return np.concatenate(new_row), np.concatenate(sign)


def format_cone_prog(problem, exp_cone_order):
    """``problem`` with its constraint rows in a solver's cone layout.

    The layout is derived here, because it is structural; applying it is left
    to the program, because that depends on how the program stores its data --
    a stuffed parameter tensor is scattered, concrete matrices are gathered.

    Guarding on ``formatted`` here rather than in the reduction keeps a second
    call harmless: an out-of-tree interface that still formats in its own
    ``apply`` gets a no-op rather than a silently twice-permuted program.
    """
    if problem.formatted:
        return problem
    return problem.with_row_layout(
        restruct_permutation(problem.constraints, exp_cone_order))


class ConeFormat(Reduction):
    """Apply a conic solver's cone row layout as an explicit chain step.

    Interfaces used to each open by calling ``format_constraints``
    themselves, guarded on ``problem.formatted`` -- eleven copies of a step
    every new interface had to remember, and one that ``Dualize`` already
    assumed had run. An already-formatted program is now a precondition of
    ``ConicSolver.apply`` rather than something each interface repairs.

    The layout itself is derived here and applied by the program, via
    ``with_row_layout``, so a program that knows a cheaper way to restructure
    itself can say so rather than be rebuilt.

    ``ExtractDirectCones`` is the one reduction that still formats on its own
    (``cone2cone/extract_direct_cones.py``). It has to: it scans rows in
    per-cone order to find its cones, which is earlier than this step runs.
    It leaves ``formatted`` set, so this reduction is then a no-op.
    """

    def __init__(self, solver) -> None:
        super().__init__()
        self.solver = solver

    def accepts(self, problem) -> bool:
        return isinstance(problem, ParamProb)

    def apply(self, problem):
        return format_cone_prog(problem, self.solver.EXP_CONE_ORDER), None

    def invert(self, solution, inverse_data):
        return solution
