"""
Copyright 2013 CVXPY Developers

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


from cvxpy.atoms.affine.wraps import nonneg_wrap
from cvxpy.atoms.max import max as cvxpy_max
from cvxpy.atoms.min import min as cvxpy_min


def ptp(x, axis=None, keepdims=False):
    """
    Range of values (maximum - minimum) along an axis.

    The name of the function comes from the acronym for ‘peak to peak’.
    """
    return nonneg_wrap(cvxpy_max(x, axis, keepdims) - cvxpy_min(x, axis, keepdims))
