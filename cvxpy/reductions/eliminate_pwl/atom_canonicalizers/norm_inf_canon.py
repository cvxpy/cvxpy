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

from cvxpy.atoms.affine.promote import promote
from cvxpy.expressions.variable import Variable


def norm_inf_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(shape)

    promoted_t = promote(t, x.shape)
    return t, [x <= promoted_t, x + promoted_t >= 0]
