"""
Copyright 2017 Steven Diamond

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

from cvxpy.expressions.expression import Expression
from cvxpy.atoms.lambda_max import lambda_max


def lambda_min(X):
    """ Miximum eigenvalue; :math:`\lambda_{\min}(A)`.

    """
    X = Expression.cast_to_const(X)
    return -lambda_max(-X)
