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

import os
import unittest

import numpy as np

import cvxpy as cp
from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS


@unittest.skipUnless('GUROBI' in INSTALLED_SOLVERS, 'GUROBI is not installed.')
def test_write(tmpdir):
    """Test the Gurobi model.write().
    """

    filename = "gurobi_model.lp"
    path = os.path.join(tmpdir, filename)

    m = 20
    n = 15
    np.random.seed(0)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    x = cp.Variable(n)
    cost = cp.sum_squares(A @ x - b)
    prob = cp.Problem(cp.Minimize(cost))
    prob.solve(solver=cp.GUROBI,
                verbose=True,
                save_file=path)

    assert os.path.exists(path)
