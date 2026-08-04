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

from typing import TYPE_CHECKING, Literal

from cvxpy.expressions import cvxtypes

if TYPE_CHECKING:
    from cvxpy.constraints.constraint import Constraint
    from cvxpy.expressions.expression import Expression
    from cvxpy.problems.objective import Objective
    from cvxpy.problems.problem import Problem

DCP = 'DCP'
DGP = 'DGP'
# Max number of nodes a reasonable expression should have.
MAX_NODES = 10_000

DisciplineName = Literal["DCP", "DGP"]
PropName = Literal["is_dcp", "is_dgp"]
AtomCurvature = Literal["convex", "concave"]


def node_count(expr) -> int:
    """Return node count for the expression/constraint.

    ``args`` may contain nested lists (e.g. a ``Problem`` stores its
    constraints as a list), so lists are walked explicitly.
    """
    if isinstance(expr, list):
        return sum(node_count(arg) for arg in expr)
    return 1 + sum(node_count(arg) for arg in expr.args)


def monotonicity_word(atom: Expression, idx: int) -> str:
    """Short monotonicity label for diagnostic messages."""
    incr = atom.is_incr(idx)
    decr = atom.is_decr(idx)
    if incr and decr:
        return "constant"
    if incr:
        return "nondecreasing"
    if decr:
        return "nonincreasing"
    return "nonmonotonic"


def explain_composition_failure(
    expr: Expression, atom_curvature: AtomCurvature
) -> str | None:
    """Explain the first DCP composition-rule failure for an atom curvature.

    Mirrors the argument checks in ``Atom.is_convex`` / ``Atom.is_concave``,
    but reports which argument failed and why.
    """
    for idx, arg in enumerate(expr.args):
        if atom_curvature == "convex":
            ok = (
                arg.is_affine()
                or (arg.is_convex() and expr.is_incr(idx))
                or (arg.is_concave() and expr.is_decr(idx))
            )
        else:
            ok = (
                arg.is_affine()
                or (arg.is_concave() and expr.is_incr(idx))
                or (arg.is_convex() and expr.is_decr(idx))
            )
        if ok:
            continue

        atom_name = expr.atom_name()
        mono = monotonicity_word(expr, idx)
        arg_curv = arg.curvature.lower()
        pretty_arg = arg.format_labeled()
        if len(expr.args) == 1:
            arg_phrase = f"Its argument {pretty_arg} is {arg_curv}"
        else:
            arg_phrase = f"Argument {idx} ({pretty_arg}) is {arg_curv}"
        return (
            f"{atom_name} is {atom_curvature} and {mono}. "
            f"{arg_phrase}. "
            f"DCP does not allow a {atom_curvature} {mono} atom "
            f"to be applied to a {arg_curv} argument."
        )
    return None


def _format_violations(violations: list[tuple[str, str | None]], indent: str = "") -> str:
    msg = ""
    for expr_label, reason in violations:
        msg += f"\n{indent}{expr_label}"
        if reason is not None:
            msg += f"\n{indent}    Reason: {reason}"
    return msg


def _passes_discipline(expr, prop_name: PropName) -> bool:
    """Whether ``expr`` satisfies the named DCP/DGP predicate.

    Only DCP and DGP are supported here; DQCP/DNLP use other diagnostics.
    """
    if prop_name == "is_dcp":
        return expr.is_dcp()
    if prop_name == "is_dgp":
        return expr.is_dgp()
    raise ValueError(f"Unsupported discipline property: {prop_name}")


def _is_minimal_violator(expr, prop_name: PropName) -> bool:
    """True if ``expr`` fails ``prop_name`` but every child passes.

    These are the nodes we report: the failure starts here, not in a child.
    For leaves, ``all([])`` is true, so a failing leaf counts as minimal.
    """
    return (
        not _passes_discipline(expr, prop_name)
        and all(_passes_discipline(child, prop_name) for child in expr.args)
    )


def _find_non_prop_nodes(
    expr,
    prop_name: PropName,
    discipline_type: DisciplineName,
    res: list[tuple[str, str | None]] | None = None,
) -> list[tuple[str, str | None]]:
    """Collect minimal violating nodes under ``expr`` (see ``_is_minimal_violator``)."""
    if res is None:
        res = []

    if _is_minimal_violator(expr, prop_name):
        res.append(_violation_entry(expr, discipline_type))
    for child in expr.args:
        res = _find_non_prop_nodes(child, prop_name, discipline_type, res)
    return res


def _violation_entry(
    expr, discipline_type: DisciplineName
) -> tuple[str, str | None]:
    if discipline_type == DCP:
        return (expr.format_labeled(), expr.dcp_failure_reason())
    if discipline_type == DGP:
        expr_label = str(expr)
        reason = None
        if isinstance(expr, cvxtypes.variable()) and not expr.is_pos():
            expr_label += " <-- needs to be declared positive (pos=True)"
        return (expr_label, reason)
    raise ValueError(f"Unknown discipline type: {discipline_type}")


def explain_expression_dcp(expr: Expression) -> str:
    """Explain why an expression fails DCP."""
    if expr.is_dcp():
        return "Expression follows DCP rules."
    violations = _find_non_prop_nodes(expr, "is_dcp", DCP)
    if not violations:
        # For a normal expression tree this should not happen: if ``is_dcp`` is
        # false, some node fails while its children pass. (Contrast objectives,
        # where empty violations means a sense mismatch — see
        # ``explain_objective_dcp``.)
        return "Expression does not follow DCP rules."
    return "The following subexpressions are not DCP:" + _format_violations(violations)


def explain_objective_dcp(objective: Objective) -> str:
    """Explain why an objective fails DCP."""
    if objective.is_dcp():
        return "Objective follows DCP rules."
    violations = _find_non_prop_nodes(objective.expr, "is_dcp", DCP)
    if violations:
        msg = "The objective is not DCP. Its following subexpressions are not:"
        return msg + _format_violations(violations)
    required = "convex" if objective.NAME == "minimize" else "concave"
    got = "convex" if objective.args[0].is_convex() else "concave"
    return (
        "The objective is not DCP, even though each sub-expression is.\n"
        f"Reason: {objective.NAME.capitalize()}(...) requires a {required} "
        f"objective, but the objective is {got}."
    )


def explain_constraint_dcp(constraint: Constraint) -> str:
    """Explain why a constraint fails DCP."""
    if constraint.is_dcp():
        return "Constraint follows DCP rules."
    violations = _find_non_prop_nodes(constraint, "is_dcp", DCP)
    pretty = constraint.format_labeled()
    msg = (
        f"The following constraint is not DCP:\n{pretty} , "
        "because the following subexpressions are not:"
    )
    return msg + _format_violations(violations, indent="|--  ")


def explain_problem_dcp(problem: Problem) -> str:
    """Explain why a problem fails DCP."""
    if problem.is_dcp():
        return "Problem follows DCP rules."
    return build_non_disciplined_error_msg(problem, DCP)


def build_non_disciplined_error_msg(problem, discipline_type: DisciplineName) -> str:
    if discipline_type == DCP:
        prop_name: PropName = "is_dcp"
        prefix_conv = ""
    elif discipline_type == DGP:
        prop_name = "is_dgp"
        prefix_conv = "log_log_"
    else:
        raise ValueError("Unknown discipline type")

    if not _passes_discipline(problem.objective, prop_name):
        if discipline_type == DCP:
            return explain_objective_dcp(problem.objective)
        non_disciplined_nodes = _find_non_prop_nodes(
            problem.objective.expr, prop_name, discipline_type
        )
        if len(non_disciplined_nodes) > 0:
            msg = "The objective is not {}. Its following subexpressions are not:".format(
                discipline_type
            )
            msg += _format_violations(non_disciplined_nodes)
        else:
            convex_str = "{}{}".format(prefix_conv, "convex")
            concave_str = "{}{}".format(prefix_conv, "concave")
            fun_attr_check = getattr(problem.objective.args[0], "is_{}".format(convex_str))()
            got_curvature = convex_str if fun_attr_check else concave_str
            msg = ("The objective is not {}, even though each sub-expression is.\n"
                   "You are trying to {} a function that is {}.").format(
                        discipline_type,
                        problem.objective.NAME,
                        got_curvature,
                    )
        return msg
    not_disciplined_constraints = [
        expr for expr in problem.constraints if not _passes_discipline(expr, prop_name)
    ]
    msg = "The following constraints are not {}:".format(discipline_type)
    for expr in not_disciplined_constraints:
        msg += '\n%s , because the following subexpressions are not:' % (
            expr.format_labeled(),)
        non_disciplined_nodes = _find_non_prop_nodes(expr, prop_name, discipline_type)
        msg += _format_violations(non_disciplined_nodes, indent="|--  ")
    return msg
