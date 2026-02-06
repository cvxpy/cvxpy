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

import cvxpy.settings as s
from cvxpy.atoms import (
    EXP_ATOMS,
    GP_EXP_ATOMS,
    GP_NONPOS_ATOMS,
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
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.leaf import Leaf
from cvxpy.reductions.cone2cone.approx import APPROX_CONE_CONVERSIONS
from cvxpy.reductions.dcp2cone.canonicalizers import CANON_METHODS
from cvxpy.reductions.dcp2cone.canonicalizers.quad import QUAD_CANON_METHODS
from cvxpy.reductions.solvers import defines as slv_def

if TYPE_CHECKING:
    from cvxpy.expressions.expression import Expression
    from cvxpy.problems.problem import Problem
    from cvxpy.reductions.solvers.solver import Solver


def _objective_cone_atoms(
    expr: Expression, affine_above: bool = True, eval_params: bool = False,
) -> list[type]:
    """Collect atom types that need conic canonicalization under quad_obj=True.

    Mirrors the Dcp2Cone.canonicalize_tree walk: atoms in the affine head
    of the objective that have a quadratic canonicalization are handled by the
    QP path and do NOT produce conic constraints. Atoms below that head, or
    atoms without a quad canon, still need cone canonicalization.

    Parameters
    ----------
    eval_params : bool
        When True, treat parameter-only sub-expressions as constant
        (they will be evaluated by EvalParams before canonicalization).
    """
    if isinstance(expr, Leaf):
        return []
    # Constraints (e.g. Equality) may appear as args of atoms like indicator.
    # They aren't atoms themselves, but their expression args can contain
    # atoms that need conic canonicalization.
    if isinstance(expr, Constraint):
        result: list[type] = []
        for arg in expr.args:
            result += _objective_cone_atoms(arg, False, eval_params)
        return result
    if expr.is_constant() and (eval_params or not expr.parameters()):
        return []

    is_affine = type(expr) not in CANON_METHODS
    child_affine_above = is_affine and affine_above
    result: list[type] = []
    for arg in expr.args:
        result += _objective_cone_atoms(arg, child_affine_above, eval_params)

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


def _expr_cone_atoms(expr: Expression, eval_params: bool = False) -> list[type]:
    """Collect atom types needing conic canonicalization from an expression.

    Unlike ``_objective_cone_atoms``, this does not apply QP-path filtering.
    Used for constraint expressions and for the objective when the QP path
    is not active.

    Parameters
    ----------
    eval_params : bool
        When True, treat parameter-only sub-expressions as constant.
    """
    if isinstance(expr, Leaf):
        return []
    if isinstance(expr, Constraint):
        result: list[type] = []
        for arg in expr.args:
            result += _expr_cone_atoms(arg, eval_params)
        return result
    if expr.is_constant() and (eval_params or not expr.parameters()):
        return []
    result: list[type] = []
    for arg in expr.args:
        result += _expr_cone_atoms(arg, eval_params)
    if type(expr) in CANON_METHODS:
        result.append(type(expr))
    return result


class ProblemForm:
    """Analyzes a CVXPY Problem to determine its structural properties.

    Attributes like quadratic objectives, required cones, integer constraints,
    and whether the problem has constraints are computed lazily and cached.
    """

    def __init__(
        self, problem: Problem, gp: bool = False, eval_params: bool = False,
    ) -> None:
        self._problem = problem
        self._gp = gp
        self._eval_params = eval_params
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
        if self._cones_quad is None:
            self._compute_cones()
        if quad_obj:
            return self._cones_quad
        return self._cones_full

    def _compute_cones(self) -> None:
        """Compute both the QP-filtered and full cone sets.

        All QUAD_CANON_METHODS atoms canonicalize to SOC, so the only
        possible difference between the two sets is the presence of SOC.
        We compute the QP-filtered set as the base and derive the full
        set by checking if unfiltered objective atoms add SOC.
        """
        if self._gp:
            self._compute_gp_cones()
            return

        problem = self._problem
        eval_params = self._eval_params

        # When eval_params is set, constraints that are entirely
        # parameter-determined (no variables) will become trivial after
        # EvalParams runs, so exclude them from cone analysis.
        if eval_params:
            relevant_constrs = [c for c in problem.constraints if c.variables()]
        else:
            relevant_constrs = problem.constraints

        constr_types = {type(c) for c in relevant_constrs}
        cones: set[type] = set()

        # Collect atoms from constraints.
        constr_atoms: list[type] = []
        for constr in relevant_constrs:
            if eval_params:
                constr_atoms += _expr_cone_atoms(constr, eval_params=True)
            else:
                constr_atoms += constr.atoms()

        # Use QP-filtered objective atoms as the base: when the objective
        # is quadratic, _objective_cone_atoms excludes atoms handled by the
        # QP path (all of which map to SOC).
        if self.has_quadratic_objective():
            base_obj_atoms = _objective_cone_atoms(
                problem.objective.expr, eval_params=eval_params)
        else:
            if eval_params:
                base_obj_atoms = _expr_cone_atoms(
                    problem.objective.expr, eval_params=True)
            else:
                base_obj_atoms = problem.objective.expr.atoms()

        atoms = base_obj_atoms + constr_atoms

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

        # This is the QP-filtered set (base).
        self._cones_quad = cones

        # Full set: if SOC is absent and the unfiltered objective would
        # add it, create a new set with SOC included.
        if SOC not in cones and self.has_quadratic_objective():
            full_obj_atoms = problem.objective.expr.atoms()
            if any(atom in SOC_ATOMS for atom in full_obj_atoms):
                self._cones_full = cones | {SOC}
                return

        # No difference — share the same object.
        self._cones_full = cones

    def _compute_gp_cones(self) -> None:
        """Compute cone sets for a DGP problem.

        DGP atoms map to different DCP atoms than their standard DCP
        canonicalization. For example, norm1 needs NonNeg in DCP but
        ExpCone in GP (via log_sum_exp). There is no QP path for GP,
        so _cones_quad and _cones_full are identical.
        """
        problem = self._problem
        constr_types = {type(c) for c in problem.constraints}
        cones: set[type] = set()

        atoms = problem.atoms()

        if any(atom in GP_EXP_ATOMS for atom in atoms):
            cones.add(ExpCone)
        if any(t in constr_types for t in [Inequality, NonPos, NonNeg]) \
                or any(atom in GP_NONPOS_ATOMS for atom in atoms):
            cones.add(NonNeg)
        if Equality in constr_types or Zero in constr_types:
            cones.add(Zero)
        if PSD in constr_types \
                or any(v.is_psd() or v.is_nsd() for v in problem.variables()):
            cones.add(PSD)

        for ct in constr_types:
            if ct in APPROX_CONE_CONVERSIONS:
                cones.add(ct)

        self._cones_quad = cones
        self._cones_full = cones

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
