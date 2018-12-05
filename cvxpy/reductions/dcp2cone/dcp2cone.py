"""
Copyright 2013 Steven Diamond, 2017 Akshay Agrawal, 2017 Robin Verschueren

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

from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers import (CANON_METHODS as
                                                           cone_canon_methods)


class Dcp2Cone(Canonicalization):
    """Reduce DCP problems to a conic form.

    This reduction takes as input (minimization) DCP problems and converts
    them into problems with affine objectives and conic constraints whose
    arguments are affine.
    """
    def __init__(self, problem=None):
        super(Dcp2Cone, self).__init__(
          problem=problem, canon_methods=cone_canon_methods)

    def accepts(self, problem):
        """A problem is accepted if it is a minimization and is DCP.
        """
        return type(problem.objective) == Minimize and problem.is_dcp()

    def apply(self, problem):
        """Converts a DCP problem to a conic form.
        """
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to cone program")
        return super(Dcp2Cone, self).apply(problem)
