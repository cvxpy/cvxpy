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
from __future__ import annotations

from typing import TYPE_CHECKING

from cvxpy.atoms import (
    EXP_ATOMS,
    NONPOS_ATOMS,
    POWCONE_ATOMS,
    POWCONE_ND_ATOMS,
    PSD_ATOMS,
    SOC_ATOMS,
)
from cvxpy.atoms.elementwise.power import Power
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.constraints import (
    PSD,
    SOC,
    Equality,
    ExpCone,
    Inequality,
    NonNeg,
    NonPos,
    PowCone3D,
    PowConeND,
    Zero,
)
from cvxpy.expressions.leaf import Leaf
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
from cvxpy.reductions.dcp2cone.canonicalizers.quad import QUAD_CANON_METHODS

if TYPE_CHECKING:
    from cvxpy.expressions.expression import Expression
    from cvxpy.problems.problem import Problem
    from cvxpy.reductions.solvers.solver import Solver


def _objective_cone_atoms(expr: Expression, affine_above: bool = True) -> list[type]:
    """Collect atom types that need conic canonicalization under quad_obj=True.

    Mirrors the Dcp2Cone.canonicalize_tree walk: atoms in the affine head
    of the objective that have a quadratic canonicalization are handled by the
    QP path and do NOT produce conic constraints. Atoms below that head, or
    atoms without a quad canon, still need cone canonicalization.
    """
    if isinstance(expr, Leaf):
        return []
    if expr.is_constant() and not expr.parameters():
        return []

    is_affine = type(expr) not in CANON_METHODS
    child_affine_above = is_affine and affine_above
    result: list[type] = []
    for arg in expr.args:
        result += _objective_cone_atoms(arg, child_affine_above)

    if affine_above and type(expr) in QUAD_CANON_METHODS:
        # Power with non-quadratic exponent falls back to cone canon.
        if isinstance(expr, Power) and not expr._quadratic_power():
            result.append(type(expr))
        # quad_over_lin needs a constant denominator for the QP path.
        elif type(expr) == quad_over_lin and not expr.args[1].is_constant():
            result.append(type(expr))
        # Otherwise QP handles it — no cone needed.
    elif type(expr) in CANON_METHODS:
        result.append(type(expr))

    return result


class ProblemForm:
    """Analyzes a CVXPY Problem to determine its structural properties.

    Attributes like quadratic objectives, required cones, integer constraints,
    and whether the problem has constraints are computed lazily and cached.
    """

    def __init__(self, problem: Problem) -> None:
        self._problem = problem
        self._has_quadratic_objective: bool | None = None
        self._is_mixed_integer: bool | None = None
        self._cones_full: set[type] | None = None
        self._cones_quad: set[type] | None = None

    def has_quadratic_objective(self) -> bool:
        """Whether the objective contains a quadratic term."""
        if self._has_quadratic_objective is None:
            self._has_quadratic_objective = (
                self._problem.objective.expr.has_quadratic_term()
            )
        return self._has_quadratic_objective

    def cones(self, quad_obj: bool = False) -> set[type]:
        """Set of cone constraint types required after canonicalization.

        Parameters
        ----------
        quad_obj : bool
            If True, exclude objective atoms handled by the QP path when
            the objective is quadratic. If False (default), return the
            conservative full cone set.
        """
        if quad_obj:
            if self._cones_quad is None:
                self._cones_quad = self._compute_cones(quad_obj=True)
            return self._cones_quad
        else:
            if self._cones_full is None:
                self._cones_full = self._compute_cones(quad_obj=False)
            return self._cones_full

    def _compute_cones(self, quad_obj: bool) -> set[type]:
        """Compute the set of cones required by the problem.

        Parameters
        ----------
        quad_obj : bool
            If True, exclude objective atoms handled by the QP path.
        """
        from cvxpy.reductions.cone2cone.approx import APPROX_CONE_CONVERSIONS

        problem = self._problem
        constr_types = {type(c) for c in problem.constraints}
        cones: set[type] = set()

        # Collect atoms from constraints (always use full atom set).
        constr_atoms: list[type] = []
        for constr in problem.constraints:
            constr_atoms += constr.atoms()

        # Collect atoms from the objective. When quad_obj is True and the
        # objective is quadratic, use _objective_cone_atoms to exclude atoms
        # handled by the QP path.
        if quad_obj and self.has_quadratic_objective():
            obj_atoms = _objective_cone_atoms(problem.objective.expr)
        else:
            obj_atoms = problem.objective.expr.atoms()

        atoms = obj_atoms + constr_atoms

        if SOC in constr_types or any(atom in SOC_ATOMS for atom in atoms):
            cones.add(SOC)
        if ExpCone in constr_types or any(atom in EXP_ATOMS for atom in atoms):
            cones.add(ExpCone)
        if any(t in constr_types for t in [Inequality, NonPos, NonNeg]) \
                or any(atom in NONPOS_ATOMS for atom in atoms):
            cones.add(NonNeg)
        if Equality in constr_types or Zero in constr_types:
            cones.add(Zero)
        if PSD in constr_types \
                or any(atom in PSD_ATOMS for atom in atoms) \
                or any(v.is_psd() or v.is_nsd() for v in problem.variables()):
            cones.add(PSD)
        if PowCone3D in constr_types or any(atom in POWCONE_ATOMS for atom in atoms):
            cones.add(PowCone3D)
        if PowConeND in constr_types or any(atom in POWCONE_ND_ATOMS for atom in atoms):
            cones.add(PowConeND)

        # Include specialized constraint types that need cone conversions.
        for ct in constr_types:
            if ct in APPROX_CONE_CONVERSIONS:
                cones.add(ct)

        return cones

    def is_mixed_integer(self) -> bool:
        """Whether the problem has integer or boolean variables."""
        if self._is_mixed_integer is None:
            self._is_mixed_integer = self._problem.is_mixed_integer()
        return self._is_mixed_integer

    def has_constraints(self) -> bool:
        """Whether the problem will have constraints after canonicalization."""
        return (len(self.cones()) > 0
                or len(self._problem.constraints) > 0
                or any(var.domain for var in self._problem.variables()))


_QP_CONES = frozenset([NonNeg, Zero])


def pick_default_solver(problem_form: ProblemForm) -> Solver | None:
    """Pick the default solver for a problem based on its structure.

    Checks premium solvers first (MOSEK, MOREAU, GUROBI), then falls back
    to open-source defaults based on problem type.

    Parameters
    ----------
    problem_form : ProblemForm
        Pre-canonicalization structural analysis of the problem.

    Returns
    -------
    Solver or None
        A solver instance, or None if no suitable installed solver is found.
    """
    import cvxpy.settings as s
    from cvxpy.reductions.solvers import defines as slv_def

    # 1-3: Premium solvers — use if installed and capable.
    for solver_name in (s.MOSEK, s.MOREAU, s.GUROBI):
        solver = slv_def.SOLVER_MAP_CONIC.get(solver_name)
        if solver is not None and solver.is_installed() \
                and solver.can_solve(problem_form):
            return solver

    # 4: Mixed-integer → HIGHS.
    if problem_form.is_mixed_integer():
        solver = slv_def.SOLVER_MAP_CONIC.get(s.HIGHS)
        if solver is not None and solver.is_installed():
            return solver
        return None

    # 5: LP → Clarabel.
    if problem_form.cones() <= _QP_CONES:
        solver = slv_def.SOLVER_MAP_CONIC.get(s.CLARABEL)
        if solver is not None and solver.is_installed():
            return solver
        return None

    # 6: QP → OSQP.
    if problem_form.has_quadratic_objective() \
            and problem_form.cones(quad_obj=True) <= _QP_CONES:
        solver = slv_def.SOLVER_MAP_QP.get(s.OSQP)
        if solver is not None and solver.is_installed():
            return solver
        return None

    # 7: SDP → SCS.
    if PSD in problem_form.cones():
        solver = slv_def.SOLVER_MAP_CONIC.get(s.SCS)
        if solver is not None and solver.is_installed():
            return solver
        return None

    # 8: Everything else (SOCP, ExpCone, PowCone, etc.) → Clarabel.
    solver = slv_def.SOLVER_MAP_CONIC.get(s.CLARABEL)
    if solver is not None and solver.is_installed():
        return solver
    return None
