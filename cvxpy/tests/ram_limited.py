"""
Copyright 2020, the CVXPY developers.

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

import sys
import unittest

import numpy as np

import cvxpy as cp

"""
This file contains tests which require massive amounts of RAM.

The tests might be for benchmarking CVXPY compile time,
or checking correctness of canonicalization.

The tests here are run on a case-by-case basis by CVXPY developers.
"""


def issue826() -> None:
    # In GitHub issue #826, it was discovered that cvxcore's C++
    # implementation implicitly limited problem data (such as a
    # constraint matrix) to have at most 2^(32)-1 nonzero entries.
    #
    # This test is for checking that #826 is resolved.
    n = 2**8
    m = int(2 ** 32 / n) + 1

    vals = np.arange(m*n, dtype=np.double) / 1000.0
    A = vals.reshape(n, m)
    x = cp.Variable(shape=(m,))
    cons = [A @ x >= 0]
    prob = cp.Problem(cp.Maximize(0), cons)
    data = prob.get_problem_data(solver='SCS')
    vals_canon = data[0]['A'].data

    tester = unittest.TestCase()
    diff = vals - vals_canon
    err = np.abs(diff)
    tester.assertLessEqual(err, 1e-3)
    print('\t issue826 test finished')


if __name__ == '__main__':
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == 'issue826':
                print('Start issue826 test')
                issue826()
            else:
                print('Unknown argument:\n\t' + arg)
    else:
        print('Start issue826 test')
        issue826()
