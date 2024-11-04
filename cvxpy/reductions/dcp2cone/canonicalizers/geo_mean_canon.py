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

from cvxpy.expressions.variable import Variable
from cvxpy.utilities.power_tools import gm_constrs


def geo_mean_canon(expr, args):
    x = args[0]
    w = expr.w
    shape = expr.shape
    t = Variable(shape)

    if x.shape == ():
        x_list = [x]
    else:
        x_list = [x[i] for i in range(len(w))]

    # todo: catch cases where we have (0, 0, 1)?
    # todo: what about curvature case (should be affine) in trivial
    #       case of (0, 0 , 1)?
    # should this behavior match with what we do in power?
    return t, gm_constrs(t, x_list, w)
