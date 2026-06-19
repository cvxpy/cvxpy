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
import numpy.linalg as LA
import pytest

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestStressMLE():

    def test_zero_mean(self):
        np.random.seed(1234)
        TOL = 1e-3
        METHODS = [1, 2, 3]
        all_n = np.arange(2, 100, 5)
        scaling_factors = [1e0]

        for n in all_n:
            np.random.seed(n)
            for factor in scaling_factors:
                data = factor*np.random.randn(n)
                sigma_opt = (1 / np.sqrt(n)) * LA.norm(data)
                res = LA.norm(data) ** 2
                for method in METHODS:
                    if method == 1:
                        sigma = cp.Variable((1, ), nonneg=True)
                        obj = (n / 2) * cp.log(2*np.pi*cp.square(sigma)) + \
                                (1 / (2 * cp.square(sigma))) * res
                        constraints = []
                    elif method == 2:
                        sigma2 = cp.Variable((1, ), nonneg=True)
                        obj = (n / 2) * cp.log( 2 * np.pi * sigma2) + (1 / (2 * sigma2)) * res
                        constraints = []
                        sigma = cp.sqrt(sigma2)
                    elif method == 3:
                        sigma = cp.Variable((1, ), nonneg=True)
                        obj = n  * cp.log(np.sqrt(2*np.pi)*sigma) + \
                                (1 / (2 * cp.square(sigma))) * res
                        constraints = []

                    problem = cp.Problem(cp.Minimize(obj), constraints)
                    problem.solve(solver=cp.IPOPT, nlp=True)
                    DerivativeChecker(problem).run_and_assert()
                    assert(np.abs(sigma.value - sigma_opt) / np.max([1, np.abs(sigma_opt)]) <= TOL)

    def test_nonzero_mean(self):
        np.random.seed(1234)
        TOL = 1e-3
        METHODS = [1, 2, 3]
        all_n = np.arange(2, 100, 5)
        scaling_factors = [1e0]
        mu = cp.Variable((1, ), name="mu")

        for n in all_n:
            np.random.seed(n)
            for factor in scaling_factors:
                data = factor*np.random.randn(n)
                sigma_opt = (1 / np.sqrt(n)) * LA.norm(data - np.mean(data))
                mu_opt = np.mean(data)
                for method in METHODS:
                    mu.value = None
                    print("Method, n, scale factor: ", method, n, factor)
                    if method == 1:
                        sigma = cp.Variable((1, ), nonneg=True)
                        obj = (n / 2) * cp.log(2*np.pi*cp.square(sigma)) + \
                                (1 / (2 * cp.square(sigma))) * cp.sum(cp.square(data-mu))
                        constraints = []
                    elif method == 2:
                        sigma2 = cp.Variable((1, ), nonneg=True)
                        obj = (n / 2) * cp.log( 2 * np.pi * sigma2) + \
                                (1 / (2 * sigma2)) * cp.sum(cp.square(data-mu))
                        constraints = []
                        sigma = cp.sqrt(sigma2)
                    elif method == 3:
                        sigma = cp.Variable((1, ), nonneg=True)
                        obj = n  * cp.log(np.sqrt(2*np.pi)*sigma) + \
                                (1 / (2 * cp.square(sigma))) * cp.sum(cp.square(data-mu))
                        constraints = []

                    problem = cp.Problem(cp.Minimize(obj), constraints)
                    problem.solve(solver=cp.IPOPT, nlp=True, verbose=True)
                    DerivativeChecker(problem).run_and_assert()
                    assert(np.abs(sigma.value - sigma_opt) / np.max([1, np.abs(sigma_opt)]) <= TOL)
                    assert(np.abs(mu.value - mu_opt) / np.max([1, np.abs(mu_opt)]) <= TOL)
