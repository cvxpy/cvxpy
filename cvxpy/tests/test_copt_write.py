"""
Copyright 2022, the CVXPY authors

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
import os
import unittest

import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS
from cvxpy.tests.base_test import BaseTest


@unittest.skipUnless('COPT' in INSTALLED_SOLVERS, 'COPT is not installed.')
class TestCOPTWrite(BaseTest):
    def test_write(self) -> None:
        """Test the COPT model.write().
        """
        import py

        tmpdir = py.path.local()
        tmpfile = tmpdir.mkdir("outfiles").join("copt_model.bin")
        tmpname = str(tmpfile)

        m = 20
        n = 15
        np.random.seed(0)
        A = np.random.randn(m, n)
        b = np.random.randn(m)

        x = cp.Variable(n)
        cost = cp.sum_squares(A @ x - b)
        prob = cp.Problem(cp.Minimize(cost))
        prob.solve(solver=cp.COPT,
                   verbose=True,
                   save_file=tmpname)

        assert os.path.exists(tmpname)
