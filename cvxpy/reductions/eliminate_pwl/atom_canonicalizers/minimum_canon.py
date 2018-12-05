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

from cvxpy.atoms.elementwise.maximum import maximum
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers.maximum_canon import maximum_canon


def minimum_canon(expr, args):
    tmp = maximum(*[-arg for arg in args])
    canon, constr = maximum_canon(tmp, tmp.args)
    return -canon, constr
