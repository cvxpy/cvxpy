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

from cvxpy.expressions.variable import Variable

DCP = 'DCP'
DGP = 'DGP'
# Max number of nodes a reasonable expression should have.
MAX_NODES = 10_000


def node_count(expr) -> int:
    """Return node count for the expression/constraint."""
    return 1 + sum(node_count(arg) for arg in getattr(expr, 'args', []))


def _curvature_word(expr) -> str:
    """Short curvature label for diagnostic messages."""
    if expr.is_affine():
        return "affine"
    if expr.is_convex():
        return "convex"
    if expr.is_concave():
        return "concave"
    return "unknown"


def _monotonicity_word(atom, idx: int) -> str:
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


def _atom_display_name(expr) -> str:
    """Human-readable atom name for diagnostics."""
    name = type(expr).__name__
    # Power / PowerApprox are the implementation of power, sqrt, and square.
    if name in ("Power", "PowerApprox") and hasattr(expr, "p"):
        try:
            p = float(expr.p.value)
        except (TypeError, ValueError, AttributeError):
            return "power"
        if p == 0.5:
            return "sqrt"
        if p == 2.0:
            return "square"
        return "power"
    return name


def format_expr_for_diagnostics(expr) -> str:
    """Pretty-print an expression for DCP diagnostic messages.

    Uses user labels when set, and renders ``power(., 0.5)`` / ``power(., 2)``
    as ``sqrt`` / ``square`` so messages match how users typically write code.
    """
    if getattr(expr, "_label", None) is not None:
        return expr._label

    cls_name = type(expr).__name__
    args = getattr(expr, "args", [])

    if cls_name in ("Power", "PowerApprox"):
        inner = format_expr_for_diagnostics(args[0])
        try:
            p = float(expr.p.value)
        except (TypeError, ValueError, AttributeError):
            return f"power({inner}, {expr.p.value})"
        if p == 0.5:
            return f"sqrt({inner})"
        if p == 2.0:
            return f"square({inner})"
        return f"power({inner}, {expr.p.value})"

    if not args:
        return expr.name() if hasattr(expr, "name") else str(expr)

    formatted_args = [format_expr_for_diagnostics(a) for a in args]

    if cls_name == "AddExpression":
        return " + ".join(formatted_args)

    if cls_name == "NegExpression":
        if args[0].args:
            return f"-({formatted_args[0]})"
        return f"-{formatted_args[0]}"

    op_name = getattr(type(expr), "OP_NAME", None)
    if op_name and op_name != "BINARY_OP" and len(args) == 2:
        left, right = formatted_args
        if type(args[0]).__name__ in ("AddExpression", "DivExpression"):
            left = f"({left})"
        if type(args[1]).__name__ in ("AddExpression", "DivExpression"):
            right = f"({right})"
        elif (cls_name == "DivExpression"
              and type(args[1]).__name__ in ("MulExpression", "multiply")):
            right = f"({right})"
        return f"{left} {op_name} {right}"

    data = []
    if hasattr(expr, "get_data") and expr.get_data() is not None:
        data = [str(elem) for elem in expr.get_data()]
    parts = formatted_args + data
    return f"{cls_name}({', '.join(parts)})"


def format_constraint_for_diagnostics(constraint) -> str:
    """Pretty-print a constraint for DCP diagnostic messages."""
    cls = type(constraint).__name__
    args = getattr(constraint, "args", [])
    if cls == "Equality" and len(args) == 2:
        lhs = format_expr_for_diagnostics(args[0])
        rhs = format_expr_for_diagnostics(args[1])
        return f"{lhs} == {rhs}"
    if cls == "Inequality" and len(args) == 2:
        lhs = format_expr_for_diagnostics(args[0])
        rhs = format_expr_for_diagnostics(args[1])
        return f"{lhs} <= {rhs}"
    if cls == "Zero" and len(args) == 1:
        return f"{format_expr_for_diagnostics(args[0])} == 0"
    if cls == "NonPos" and len(args) == 1:
        return f"{format_expr_for_diagnostics(args[0])} <= 0"
    if cls == "NonNeg" and len(args) == 1:
        return f"{format_expr_for_diagnostics(args[0])} >= 0"
    if cls == "PSD" and len(args) == 1:
        return f"{format_expr_for_diagnostics(args[0])} >> 0"
    return str(constraint)


def explain_dcp_violation(expr) -> str | None:
    """Explain why a minimal DCP-violating subexpression fails DCP.

    Parameters
    ----------
    expr : Expression
        A node that fails ``is_dcp()`` while every child passes (or a leaf).

    Returns
    -------
    str or None
        A reason string, or None if no specific explanation is available.
    """
    # Constraints are handled separately.
    from cvxpy.constraints.constraint import Constraint
    if isinstance(expr, Constraint):
        return explain_constraint_dcp_violation(expr)

    if expr.is_dcp():
        return None

    # Intrinsic atom curvature (not composed).
    try:
        atom_convex = expr.is_atom_convex()
        atom_concave = expr.is_atom_concave()
    except (AttributeError, NotImplementedError):
        return None

    atom_name = _atom_display_name(expr)

    if not atom_convex and not atom_concave:
        pretty = format_expr_for_diagnostics(expr)
        return (
            f"{atom_name} is neither a convex nor a concave atom, "
            f"so DCP cannot verify the curvature of {pretty}."
        )

    reasons: list[str] = []

    # Explain composition failures under each intrinsic curvature the atom has.
    if atom_convex:
        reason = _explain_composition_failure(expr, atom_curvature="convex")
        if reason is not None:
            reasons.append(reason)
    if atom_concave:
        reason = _explain_composition_failure(expr, atom_curvature="concave")
        if reason is not None:
            reasons.append(reason)

    if not reasons:
        return None
    # Prefer a single reason when only one intrinsic curvature applies.
    if len(reasons) == 1:
        return reasons[0]
    return " ".join(reasons)


def explain_constraint_dcp_violation(constraint) -> str | None:
    """Explain why a constraint fails DCP when its arguments are themselves DCP.

    Parameters
    ----------
    constraint : Constraint
        A constraint that fails ``is_dcp()``.

    Returns
    -------
    str or None
        A reason string describing the constraint's curvature requirement.
    """
    cls = type(constraint).__name__

    if cls in ("Equality", "Zero"):
        expr = constraint.expr
        if expr.is_affine():
            return None
        if cls == "Equality" and len(constraint.args) == 2:
            pretty = (
                f"{format_expr_for_diagnostics(constraint.args[0])} - "
                f"{format_expr_for_diagnostics(constraint.args[1])}"
            )
        else:
            pretty = format_expr_for_diagnostics(constraint.args[0])
        return (
            f"Equality constraints require an affine expression, "
            f"but {pretty} is {_curvature_word(expr)}."
        )

    if cls == "Inequality":
        expr = constraint.expr
        if expr.is_convex():
            return None
        lhs = format_expr_for_diagnostics(constraint.args[0])
        rhs = format_expr_for_diagnostics(constraint.args[1])
        return (
            f"The inequality {lhs} <= {rhs} requires (lhs - rhs) to be convex, "
            f"but it is {_curvature_word(expr)}."
        )

    if cls == "NonPos":
        expr = constraint.args[0]
        if expr.is_convex():
            return None
        pretty = format_expr_for_diagnostics(expr)
        return (
            f"The constraint {pretty} <= 0 requires a convex expression, "
            f"but {pretty} is {_curvature_word(expr)}."
        )

    if cls == "NonNeg":
        expr = constraint.args[0]
        if expr.is_concave():
            return None
        pretty = format_expr_for_diagnostics(expr)
        return (
            f"The constraint {pretty} >= 0 requires a concave expression, "
            f"but {pretty} is {_curvature_word(expr)}."
        )

    if cls == "PSD":
        expr = constraint.args[0]
        if expr.is_affine():
            return None
        pretty = format_expr_for_diagnostics(expr)
        return (
            f"PSD constraints require an affine expression, "
            f"but {pretty} is {_curvature_word(expr)}."
        )

    if cls in ("SOC", "RSOC"):
        bad = [a for a in constraint.args if not a.is_affine()]
        if not bad:
            return None
        pretty_bad = ", ".join(format_expr_for_diagnostics(a) for a in bad)
        return (
            f"{cls} constraints require affine arguments, "
            f"but these arguments are not affine: {pretty_bad}."
        )

    return None


def _explain_composition_failure(expr, atom_curvature: str) -> str | None:
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

        atom_name = _atom_display_name(expr)
        mono = _monotonicity_word(expr, idx)
        arg_curv = _curvature_word(arg)
        pretty_arg = format_expr_for_diagnostics(arg)
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


def _format_violations(violations, indent: str = "") -> str:
    msg = ""
    for str_expr, reason in violations:
        msg += f"\n{indent}{str_expr}"
        if reason is not None:
            msg += f"\n{indent}    Reason: {reason}"
    return msg


def _find_non_prop_leaves(expr, prop_name: str, discipline_type: str, res=None):
    """Collect minimal nodes that fail ``prop_name`` while all children pass."""
    if res is None:
        res = []
    if (len(expr.args) == 0 and getattr(expr, prop_name)()):
        return res
    if ((not getattr(expr, prop_name)()) and
            all(getattr(child, prop_name)() for child in expr.args)):
        if discipline_type == DCP:
            from cvxpy.constraints.constraint import Constraint
            if isinstance(expr, Constraint):
                str_expr = format_constraint_for_diagnostics(expr)
            else:
                str_expr = format_expr_for_diagnostics(expr)
            reason = explain_dcp_violation(expr)
        else:
            str_expr = str(expr)
            reason = None
            if isinstance(expr, Variable):
                str_expr += " <-- needs to be declared positive"
        res.append((str_expr, reason))
    for child in expr.args:
        res = _find_non_prop_leaves(child, prop_name, discipline_type, res)
    return res


def _explain_dcp_objective(objective) -> str:
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


def _explain_dcp_constraint(constraint) -> str:
    """Explain why a constraint fails DCP."""
    if constraint.is_dcp():
        return "Constraint follows DCP rules."
    violations = _find_non_prop_leaves(constraint, "is_dcp", DCP)
    pretty = format_constraint_for_diagnostics(constraint)
    msg = (
        f"The following constraint is not DCP:\n{pretty} , "
        "because the following subexpressions are not:"
    )
    return msg + _format_violations(violations, indent="|--  ")


def _explain_dcp_expression(expr) -> str:
    """Explain why an expression fails DCP."""
    if expr.is_dcp():
        return "Expression follows DCP rules."
    violations = _find_non_prop_leaves(expr, "is_dcp", DCP)
    if not violations:
        return "Expression does not follow DCP rules."
    msg = "The following subexpressions are not DCP:"
    return msg + _format_violations(violations)


def explain_dcp(obj) -> str:
    """Explain DCP violations in a problem, objective, constraint, or expression.

    Parameters
    ----------
    obj : Problem or Objective or Constraint or Expression
        The object to diagnose.

    Returns
    -------
    str
        A human-readable explanation. If ``obj`` follows DCP, a short
        success message is returned.

    Notes
    -----
    Covers composition-rule failures, non-DCP atoms, minimize/maximize
    curvature mismatches, and constraint-type curvature requirements
    (equalities/PSD/SOC need affine arguments; inequalities need the
    appropriate convex/concave side). Does not suggest equivalent DCP
    rewrites of convex-but-not-DCP expressions.
    """
    # Lazy imports avoid circular dependencies at module load time.
    from cvxpy.constraints.constraint import Constraint
    from cvxpy.expressions.expression import Expression
    from cvxpy.problems.objective import Objective
    from cvxpy.problems.problem import Problem

    if isinstance(obj, Problem):
        if obj.is_dcp():
            return "Problem follows DCP rules."
        return build_non_disciplined_error_msg(obj, DCP)
    if isinstance(obj, Objective):
        return _explain_dcp_objective(obj)
    if isinstance(obj, Constraint):
        return _explain_dcp_constraint(obj)
    if isinstance(obj, Expression):
        return _explain_dcp_expression(obj)
    raise TypeError(
        "explain_dcp() expects a Problem, Objective, Constraint, or Expression, "
        f"got {type(obj).__name__}."
    )


def build_non_disciplined_error_msg(problem, discipline_type) -> str:
    prop_name = None
    prefix_conv = ""
    if discipline_type == DCP:
        prop_name = "is_dcp"
    elif discipline_type == DGP:
        prop_name = "is_dgp"
        prefix_conv = "log_log_"
    else:
        raise ValueError("Unknown discipline type")

    if not getattr(problem.objective, prop_name)():
        if discipline_type == DCP:
            return _explain_dcp_objective(problem.objective)
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
        expr for expr in problem.constraints if not getattr(expr, prop_name)()
    ]
    msg = "The following constraints are not {}:".format(discipline_type)
    for expr in not_disciplined_constraints:
        if discipline_type == DCP:
            pretty = format_constraint_for_diagnostics(expr)
        else:
            pretty = str(expr)
        msg += '\n%s , because the following subexpressions are not:' % (pretty,)
        non_disciplined_leaves = _find_non_prop_leaves(expr, prop_name, discipline_type)
        msg += _format_violations(non_disciplined_leaves, indent="|--  ")
    return msg
