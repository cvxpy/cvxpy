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

from cvxpy.constraints import Equality, Inequality, NonPos, Zero
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.atom_canonicalizers import (
    CANON_METHODS as qp_canon_methods)
from cvxpy.reductions.utilities import are_args_affine


def accepts(problem):
    """
    Problems with quadratic, piecewise affine objectives,
    piecewise-linear constraints inequality constraints, and
    affine equality constraints are accepted by the reduction.
    """
    return (problem.objective.expr.is_qpwa()
            and not set(['PSD', 'NSD']).intersection(convex_attributes(
                                                     problem.variables()))
            and all((type(c) in (Inequality, NonPos) and c.expr.is_pwl()) or
                    (type(c) in (Equality, Zero) and are_args_affine([c]))
                    for c in problem.constraints))


class Qp2SymbolicQp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine
    expressions and symbolic quadratic forms.
    """
    def __init__(self, problem=None):
        super(Qp2SymbolicQp, self).__init__(
          problem=problem, canon_methods=qp_canon_methods)

    def accepts(self, problem):
        """
        Problems with quadratic, piecewise affine objectives,
        piecewise-linear constraints inequality constraints, and
        affine equality constraints are accepted.
        """
        return accepts(problem)

    def apply(self, problem):
        """Converts a QP to an even more symbolic form."""
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        return super(Qp2SymbolicQp, self).apply(problem)
