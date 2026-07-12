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

import cvxpy as cp
from cvxpy.error import DCPError
from cvxpy.tests.base_test import BaseTest
from cvxpy.utilities.debug_tools import (
    build_non_disciplined_error_msg,
    explain_dcp_violation,
)


class TestDcpDiagnostics(BaseTest):
    """Tests for DCP violation explanations."""

    def test_explain_concave_atom_on_convex_arg(self) -> None:
        # Convex but not DCP: sqrt(1 + x^2).
        x = cp.Variable()
        expr = cp.sqrt(1 + cp.square(x))
        self.assertFalse(expr.is_dcp())
        reason = explain_dcp_violation(expr)
        self.assertIsNotNone(reason)
        self.assertIn("concave", reason)
        self.assertIn("nondecreasing", reason)
        self.assertIn("convex", reason)
        self.assertIn(
            "DCP does not allow a concave nondecreasing atom "
            "to be applied to a convex argument",
            reason,
        )

    def test_explain_convex_atom_on_concave_arg(self) -> None:
        x = cp.Variable()
        expr = cp.exp(-cp.square(x))
        self.assertFalse(expr.is_dcp())
        reason = explain_dcp_violation(expr)
        self.assertIsNotNone(reason)
        self.assertIn("convex", reason)
        self.assertIn("nondecreasing", reason)
        self.assertIn("concave", reason)
        self.assertIn(
            "DCP does not allow a convex nondecreasing atom "
            "to be applied to a concave argument",
            reason,
        )

    def test_explain_neither_convex_nor_concave_atom(self) -> None:
        x = cp.Variable()
        y = cp.Variable()
        expr = cp.multiply(x, y)
        self.assertFalse(expr.is_dcp())
        reason = explain_dcp_violation(expr)
        self.assertIsNotNone(reason)
        self.assertIn("neither a convex nor a concave atom", reason)

    def test_error_msg_includes_composition_reason(self) -> None:
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sqrt(1 + cp.square(x))))
        msg = build_non_disciplined_error_msg(prob, "DCP")
        self.assertIn("The objective is not DCP", msg)
        self.assertIn("sqrt(1.0 + square(", msg)
        self.assertIn("Reason:", msg)
        self.assertIn(
            "DCP does not allow a concave nondecreasing atom "
            "to be applied to a convex argument",
            msg,
        )
        self.assertNotIn("PowerApprox", msg)

    def test_power_name_uses_sqrt_and_square(self) -> None:
        x = cp.Variable(name="x")
        expr = cp.sqrt(1 + cp.square(x))
        self.assertEqual(expr.format_labeled(), "sqrt(1.0 + square(x))")
        self.assertEqual(expr.atom_name(), "sqrt")
        self.assertEqual(cp.square(x).atom_name(), "square")

    def test_error_msg_minimize_concave_objective(self) -> None:
        # Expression is DCP (concave), but Minimize requires convex.
        x = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sqrt(x)))
        self.assertTrue(cp.sqrt(x).is_dcp())
        self.assertFalse(prob.is_dcp())
        msg = build_non_disciplined_error_msg(prob, "DCP")
        self.assertIn("The objective is not DCP, even though each sub-expression is", msg)
        self.assertIn(
            "Reason: Minimize(...) requires a convex objective, but the objective is concave",
            msg,
        )

    def test_error_msg_maximize_convex_objective(self) -> None:
        x = cp.Variable()
        prob = cp.Problem(cp.Maximize(cp.square(x)))
        msg = build_non_disciplined_error_msg(prob, "DCP")
        self.assertIn(
            "Reason: Maximize(...) requires a concave objective, but the objective is convex",
            msg,
        )

    def test_solve_raises_dcp_error_with_reason(self) -> None:
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(cp.sqrt(1 + cp.square(x))))
        with self.assertRaises(DCPError) as cm:
            prob.solve(solver=cp.CLARABEL)
        self.assertIn("Reason:", str(cm.exception))
        self.assertIn("concave nondecreasing atom", str(cm.exception))

    def test_constraint_violation_includes_reason(self) -> None:
        x = cp.Variable()
        # log(square(x)) is not DCP; used in an inequality.
        prob = cp.Problem(cp.Minimize(0), [cp.log(cp.square(x)) <= 0])
        msg = build_non_disciplined_error_msg(prob, "DCP")
        self.assertIn("The following constraints are not DCP", msg)
        self.assertIn("Reason:", msg)
        self.assertIn("concave nondecreasing atom", msg)

    def test_equality_requires_affine(self) -> None:
        x = cp.Variable(name="x")
        msg = (cp.square(x) == 0).explain_dcp()
        self.assertIn("square(x) == 0", msg)
        self.assertIn("Equality constraints require an affine expression", msg)
        self.assertIn("is convex", msg)

    def test_inequality_requires_convex_difference(self) -> None:
        x = cp.Variable(name="x")
        msg = (cp.sqrt(x) <= 2).explain_dcp()
        self.assertIn("sqrt(x) <= 2", msg)
        self.assertIn("requires (lhs - rhs) to be convex", msg)
        self.assertIn("concave", msg)

    def test_psd_requires_affine(self) -> None:
        X = cp.Variable((2, 2), name="X")
        msg = (cp.square(X) >> 0).explain_dcp()
        self.assertIn("PSD constraints require an affine expression", msg)

    def test_explain_dcp_on_expression(self) -> None:
        x = cp.Variable(name="x")
        expr = cp.sqrt(1 + cp.square(x))
        msg = expr.explain_dcp()
        self.assertIn("sqrt(1.0 + square(x))", msg)
        self.assertIn("Reason:", msg)
        self.assertIn("concave nondecreasing atom", msg)

        ok = (x + 1).explain_dcp()
        self.assertEqual(ok, "Expression follows DCP rules.")

    def test_explain_dcp_on_problem(self) -> None:
        x = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sqrt(x)))
        msg = prob.explain_dcp()
        self.assertIn("Minimize(...) requires a convex objective", msg)

        ok = cp.Problem(cp.Minimize(cp.square(x)))
        self.assertEqual(ok.explain_dcp(), "Problem follows DCP rules.")

    def test_explain_dcp_on_objective_and_constraint(self) -> None:
        x = cp.Variable(name="x")
        obj = cp.Minimize(cp.sqrt(1 + cp.square(x)))
        self.assertIn("sqrt(1.0 + square(x))", obj.explain_dcp())

        constr = cp.log(cp.square(x)) <= 0
        msg = constr.explain_dcp()
        self.assertIn("constraint is not DCP", msg)
        self.assertIn("Reason:", msg)
