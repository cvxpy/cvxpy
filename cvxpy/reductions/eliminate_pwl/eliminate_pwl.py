"""
Copyright 2017 Robin Verschueren

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

from cvxpy.atoms import abs, maximum, sum_largest, max, norm1, norm_inf
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.eliminate_pwl.atom_canonicalizers import (
    CANON_METHODS as elim_pwl_methods)


class EliminatePwl(Canonicalization):
    """Eliminates piecewise linear atoms."""

    def __init__(self, problem=None) -> None:
        super(EliminatePwl, self).__init__(
          problem=problem, canon_methods=elim_pwl_methods)

    def accepts(self, problem) -> bool:
        atom_types = [type(atom) for atom in problem.atoms()]
        pwl_types = [abs, maximum, sum_largest, max, norm1, norm_inf]
        return any(atom in pwl_types for atom in atom_types)

    def apply(self, problem):
        if not self.accepts(problem):
            raise ValueError("Cannot canonicalize pwl atoms.")
        return super(EliminatePwl, self).apply(problem)
