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

import warnings

from scipy import sparse

from cvxpy.expressions import expression as exp

SPARSE_MATRIX_CLASSES = [
    sparse.csc_matrix, 
    sparse.csr_matrix, 
    sparse.coo_matrix, 
    sparse.bsr_matrix, 
    sparse.lil_matrix, 
    sparse.dia_matrix,
    sparse.dok_matrix,
]
BIN_OPS = ["__div__", "__mul__", "__add__", "__sub__",
           "__le__", "__eq__", "__lt__", "__gt__"]

SCIPY_WRAPPER_DEPRECATION_MESSAGE = """
Your CVXPY program is using a deprecated feature of our SciPy interface.

We believed it was impossible to hit this warning; please inform us of how you
reached this warning at https://github.com/cvxpy/cvxpy/discussions/2187 so we can
ensure that we correct this issue without causing breakage.
"""

def wrap_bin_op(method):
    """Factory for wrapping binary operators.
    """
    def new_method(self, other):
        output = method(self, other)
        if isinstance(other, exp.Expression) and output is not NotImplemented:
            warnings.warn(SCIPY_WRAPPER_DEPRECATION_MESSAGE,
                          category=FutureWarning)
            return NotImplemented
        else:
            return output
    return new_method

for cls in SPARSE_MATRIX_CLASSES:
    for method_name in BIN_OPS:
        method = getattr(cls, method_name)
        new_method = wrap_bin_op(method)
        setattr(cls, method_name, new_method)
