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
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestNormalCdf:

    def test_normcdf(self):
        np.random.seed(0)
        x = cp.Variable()
        obj = cp.normcdf((x - 1) ** 2)
        prob = cp.Problem(cp.Minimize(obj))
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        assert np.allclose(x.value, 1.0)
        assert np.allclose(prob.value, 0.5)

    def test_normcdf_with_quadratic(self):
        x = cp.Variable()
        lmbda = 1.0
        obj = cp.normcdf(x) - lmbda * x ** 2
        prob = cp.Problem(cp.Maximize(obj))
        prob.solve(nlp=True, verbose=True)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        grad = np.exp(-0.5 * x.value ** 2) / np.sqrt(2 * np.pi) - 2 * lmbda * x.value
        assert np.allclose(grad, 0.0)
