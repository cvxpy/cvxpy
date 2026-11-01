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
import pytest

import cvxpy as cp
from cvxpy.reductions.algebra.matmul_reassoc import _normalize_matmul
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker
from cvxpy.tests.test_matmul_reassoc import reassoc_cases


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestReassociatedChainDerivatives:

    def test_derivatives_match_on_reassociated_chains(self):
        rng = np.random.default_rng(0)
        X, cases = reassoc_cases(rng)
        for expr in cases:
            rewritten = _normalize_matmul(expr)
            X.value = np.zeros((4, 3))
            prob = cp.Problem(cp.Minimize(cp.sum_squares(
                rewritten - rng.standard_normal(expr.shape)) + cp.sum_squares(X)))
            DerivativeChecker(prob).run_and_assert()
