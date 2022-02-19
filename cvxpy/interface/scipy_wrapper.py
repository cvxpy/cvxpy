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

from scipy.sparse import spmatrix

from cvxpy.expressions import expression as exp

BIN_OPS = ["__div__", "__mul__", "__add__", "__sub__",
           "__le__", "__eq__", "__lt__", "__gt__"]


def wrap_bin_op(method):
    """Factory for wrapping binary operators.
    """
    def new_method(self, other):
        if isinstance(other, exp.Expression):
            return NotImplemented
        else:
            return method(self, other)
    return new_method


for method_name in BIN_OPS:
    method = getattr(spmatrix, method_name)
    new_method = wrap_bin_op(method)
    setattr(spmatrix, method_name, new_method)
