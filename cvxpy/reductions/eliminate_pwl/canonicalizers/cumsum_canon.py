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


def cumsum_canon(expr, args):
    """Cumulative sum.
    """
    X = args[0]
    axis = expr.axis
    # Implicit O(n) definition:
    # X = Y[1:,:] - Y[:-1, :]
    Y = Variable(expr.shape)
    if axis == 0:
        if (expr.shape[0] == 1):
            return X, []
        else:
            constr = [X[1:] == Y[1:] - Y[:-1], Y[0] == X[0]]
    else:
        if (expr.shape[1] == 1):
            return X, []
        else:
            constr = [X[:, 1:] == Y[:, 1:] - Y[:, :-1], Y[:, 0] == X[:, 0]]
    return Y, constr
