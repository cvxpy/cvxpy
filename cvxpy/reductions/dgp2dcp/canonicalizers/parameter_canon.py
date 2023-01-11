"""
Copyright 2018 Akshay Agrawal

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

from cvxpy.expressions.constants.parameter import Parameter


def parameter_canon(expr, args):
    del args
    # NB: we do _not_ reuse the original parameter's id. This is important,
    # because we want to distinguish between parameters in the DGP problem
    # and parameters in the DCP problem (for differentiation)
    param = Parameter(expr.shape, name=expr.name())
    param.value = np.log(expr.value)
    return param, []
