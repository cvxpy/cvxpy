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
from cvxpy import (Variable, lambda_sum_largest, entr, trace,)
from cvxpy.atoms.affine.sum import sum

def vn_entr_canon(expr, args):
    N = args[0]
    n = N.shape[0]
    x = Variable(shape= (n,))
    t = Variable()
    constr1 = [lambda_sum_largest(N, r) <= sum(x[:r]) for r in range(1,n+1)]
    constr2 = trace(N) === sum(x)
    constr3 = x[1:] >= x[:(n-1)]
    constr4 = sum(entr(x)) >= t
    constraints= constr1 + [constr2, constr3, constr4]
    return t, constraints
