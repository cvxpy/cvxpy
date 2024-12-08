"""
Copyright 2024 the CVXPY developers

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

from cvxpy.atoms.affine.bmat import bmat
from cvxpy.atoms.affine.hstack import hstack
from cvxpy.atoms.affine.promote import promote
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.log_sum_exp import log_sum_exp


def add_canon(expr, args):
    if expr.is_scalar():
        return log_sum_exp(hstack(args)), []

    rows = []
    summands = [
       promote(s, expr.shape) if s.is_scalar() else s for s in args]
    if len(expr.shape) == 1:
        for i in range(expr.shape[0]):
            row = []
            row.append(
              log_sum_exp(hstack([summand[i] for summand in summands])))
            rows.append(row)
        return reshape(bmat(rows), expr.shape, order='F'), []
    else:
        for i in range(expr.shape[0]):
            row = []
            for j in range(expr.shape[1]):
                row.append(
                  log_sum_exp(hstack([summand[i, j] for summand in summands])))
            rows.append(row)
        return reshape(bmat(rows), expr.shape, order='F'), []
