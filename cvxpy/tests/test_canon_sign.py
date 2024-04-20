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
from cvxpy.tests.base_test import BaseTest
from cvxpy.reductions.eliminate_pwl.canonicalizers.maximum_canon import maximum_canon

from typing import Iterable


class TestCanonSign(BaseTest):
    @staticmethod
    def exprs() -> Iterable[cp.Expression]:
        """List of test expressions"""
        expr_nonneg = cp.Variable(pos=True)
        expr_nonpos = cp.Variable(neg=True)
        expr_zero = cp.Constant(0)

        return [expr_nonneg, expr_nonpos, expr_zero]

    def expr_sign_assertions(self, a: cp.Expression, b: cp.Expression) -> None:
        """Asserting sign methods are preserved across a & b (a function of a)"""
        self.assertEqual(a.is_nonpos(), b.is_nonpos())
        self.assertEqual(a.is_nonneg(), b.is_nonneg())

    def test_maximum_sign(self):
        """Iterate over test expressions,
        check that sign is preserved for composition of all with maximum_canon"""
        args = [0]
        for expr in self.exprs():
            canon_expr, canon_cons = maximum_canon(expr, args)
            self.expr_sign_assertions(expr, canon_expr)
