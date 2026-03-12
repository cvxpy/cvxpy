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
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.nlp_tests.derivative_checker import DerivativeChecker


@pytest.mark.skipif('IPOPT' not in INSTALLED_SOLVERS, reason='IPOPT is not installed.')
class TestMultiplyDifferentFormats:
    
    def test_dense_sparse_sparse(self):
        np.random.seed(0)
        n = 5
       
        # dense
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = np.random.rand(n, n) - 0.5
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0)], -2)
        assert np.allclose(x.value[(A < 0)], 2)

        # CSR
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = sp.csr_matrix(A)
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0).todense()], -2)
        assert np.allclose(x.value[(A < 0).todense()], 2)

        # CSC
        x = cp.Variable((n, n), bounds=[-2, 2])
        A = sp.csc_matrix(A)
        obj = cp.Minimize(cp.sum(cp.multiply(A, x)))
        prob = cp.Problem(obj)
        x.value = np.random.rand(n, n)
        checker = DerivativeChecker(prob)
        checker.run_and_assert()
        prob.solve(nlp=True, verbose=False)
        assert np.allclose(x.value[(A > 0).todense()], -2)
        assert np.allclose(x.value[(A < 0).todense()], 2)
            
