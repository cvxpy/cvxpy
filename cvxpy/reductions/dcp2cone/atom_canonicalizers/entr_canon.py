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

from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
import numpy as np


def entr_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(shape)
    # -x\log(x) >= t <=> x\exp(t/x) <= 1
    # TODO(akshayka): ExpCone requires each of its inputs to be a Variable;
    # is this something that we want to change?
    ones = Constant(np.ones(shape))
    constraints = [ExpCone(t, x, ones)]
    return t, constraints
