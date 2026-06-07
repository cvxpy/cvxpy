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
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters as conv
from cvxpy.reductions.inverse_data import InverseData
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import convert_expr
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.helpers import build_var_dict
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


class TestConstantMatmulOperand:
    """A pure-constant matmul operand is consumed by convert_matmul as raw matrix data;
    convert_expr must not eagerly convert it (which would densify a sparse constant into
    a parameter node that convert_matmul then discards).
    """

    def test_sparse_constant_operand_not_densified(self):
        # S @ x with S a sparse diagonal: converting S would call to_dense_float and
        # densify the n x n diagonal. The fix skips converting the constant operand.
        n = 300
        S = sp.diags(np.arange(1, n + 1, dtype=float))
        x = cp.Variable(n)
        inverse_data = InverseData(cp.Problem(cp.Minimize(cp.sum(x))))
        var_dict, n_vars = build_var_dict(inverse_data)

        calls = []
        orig = conv.to_dense_float
        conv.to_dense_float = lambda v: calls.append(getattr(v, "shape", None)) or orig(v)
        try:
            convert_expr(S @ x, var_dict, n_vars, {})
        finally:
            conv.to_dense_float = orig
        # The n x n sparse constant must never be densified.
        assert all(shape != (n, n) for shape in calls)

    def test_sparse_constant_matmul_derivative_check(self):
        # sum_squares(S @ x) with S a sparse diagonal solves/derivative-checks correctly
        # through the diffengine after the constant operand is no longer densified.
        n = 50
        rng = np.random.default_rng(0)
        S = sp.diags(rng.uniform(0.5, 2.0, size=n))
        b = rng.standard_normal(n)
        x = cp.Variable(n)
        x.value = np.zeros(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(S @ x - b)))
        DerivativeChecker(prob).run_and_assert()
