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


def node_count(expr) -> int:
    """Return node count for the expression/constraint."""
    return 1 + sum(node_count(arg) for arg in expr.args)


def curvature_word(expr: Expression) -> str:
    """Short curvature label for diagnostic messages."""
    if expr.is_affine():
        return "affine"
    if expr.is_convex():
        return "convex"
    if expr.is_concave():
        return "concave"
    return "unknown"


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


def explain_composition_failure(expr: Expression, atom_curvature: str) -> str | None:
    """Explain the first DCP composition-rule failure for an atom curvature."""
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
        arg_curv = curvature_word(arg)
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


def explain_dcp_violation(node: Expression | Constraint) -> str | None:
    """Explain why a minimal DCP-violating node fails DCP."""
    return node.dcp_failure_reason()


def _format_violations(violations: list[tuple[str, str | None]], indent: str = "") -> str:
    msg = ""
    for str_expr, reason in violations:
        msg += f"\n{indent}{str_expr}"
        if reason is not None:
            msg += f"\n{indent}    Reason: {reason}"
    return msg


def _passes_discipline(expr, prop_name: str) -> bool:
    if prop_name == "is_dcp":
        return expr.is_dcp()
    return expr.is_dgp()


def _find_non_prop_leaves(
    expr,
    prop_name: str,
    discipline_type: str,
    res: list[tuple[str, str | None]] | None = None,
) -> list[tuple[str, str | None]]:
    """Collect minimal nodes that fail ``prop_name`` while all children pass."""
    if res is None:
        res = []
    if len(expr.args) == 0 and _passes_discipline(expr, prop_name):
        return res
    if (not _passes_discipline(expr, prop_name) and
            all(_passes_discipline(child, prop_name) for child in expr.args)):
        if discipline_type == DCP:
            str_expr = expr.format_labeled()
            reason = explain_dcp_violation(expr)
        else:
            str_expr = str(expr)
            reason = None
            if isinstance(expr, cvxtypes.variable()):
                str_expr += " <-- needs to be declared positive"
        res.append((str_expr, reason))
    for child in expr.args:
        res = _find_non_prop_leaves(child, prop_name, discipline_type, res)
    return res


def explain_expression_dcp(expr: Expression) -> str:
    """Explain why an expression fails DCP."""
    if expr.is_dcp():
        return "Expression follows DCP rules."
    violations = _find_non_prop_leaves(expr, "is_dcp", DCP)
    if not violations:
        return "Expression does not follow DCP rules."
    return "The following subexpressions are not DCP:" + _format_violations(violations)


def explain_objective_dcp(objective: Objective) -> str:
    """Explain why an objective fails DCP."""
    if objective.is_dcp():
        return "Objective follows DCP rules."
    violations = _find_non_prop_leaves(objective.expr, "is_dcp", DCP)
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
    violations = _find_non_prop_leaves(constraint, "is_dcp", DCP)
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


def build_non_disciplined_error_msg(problem, discipline_type) -> str:
    if discipline_type == DCP:
        prop_name = "is_dcp"
        prefix_conv = ""
    elif discipline_type == DGP:
        prop_name = "is_dgp"
        prefix_conv = "log_log_"
    else:
        raise ValueError("Unknown discipline type")

    if not _passes_discipline(problem.objective, prop_name):
        if discipline_type == DCP:
            return explain_objective_dcp(problem.objective)
        non_disciplined_leaves = _find_non_prop_leaves(
            problem.objective.expr, prop_name, discipline_type
        )
        if len(non_disciplined_leaves) > 0:
            msg = "The objective is not {}. Its following subexpressions are not:".format(
                discipline_type
            )
            msg += _format_violations(non_disciplined_leaves)
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
        non_disciplined_leaves = _find_non_prop_leaves(expr, prop_name, discipline_type)
        msg += _format_violations(non_disciplined_leaves, indent="|--  ")
    return msg
