"""
Copyright 2013 Steven Diamond

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

from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.atom_canonicalizers.exp_canon import exp_canon


def logistic_canon(expr, args):
    x = args[0]
    shape = expr.shape
    # log(1 + exp(x)) <= t <=> exp(-t) + exp(x - t) <= 1
    t0 = Variable(shape)
    t1, constr1 = exp_canon(expr, [-t0])
    t2, constr2 = exp_canon(expr, [x - t0])
    ones = Constant(np.ones(shape))
    constraints = constr1 + constr2 + [t1 + t2 <= ones]
    return t0, constraints
