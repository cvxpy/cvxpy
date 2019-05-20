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
import contextlib


from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.atom_canonicalizers import (CANON_METHODS as
                                                           cone_canon_methods)


@contextlib.contextmanager
def _param_vexity_scope(expr):
    """Treats parameters as affine, not constants."""
    for p in expr.parameters():
        p._is_constant = False
    expr._check_is_constant(recompute=True)

    yield

    for p in expr.parameters():
        p._is_constant = True
    expr._check_is_constant(recompute=True)


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

    def canonicalize_tree(self, expr):
        # Only allow param * var (not var * param). Associate right to left.
        if isinstance(expr, MulExpression) and expr.parameters():
            op_type = type(expr)
            lhs = expr.args[0]
            rhs = expr.args[1]
            if lhs.variables():
                with _param_vexity_scope(rhs):
                    assert rhs.is_affine()
                canon_lhs, c = self.canonicalize_tree(lhs)
                if canon_lhs.parameters():
                    t = Variable(canon_lhs.shape)
                    c += [t == canon_lhs]
                    canon_lhs = t
                return op_type(canon_lhs, rhs), c
            elif rhs.variables():
                with _param_vexity_scope(lhs):
                    assert lhs.is_affine()
                canon_rhs, c = self.canonicalize_tree(rhs)
                if canon_rhs.parameters():
                    t = Variable(canon_rhs.shape)
                    c += [t == canon_rhs]
                    canon_rhs = t
                return op_type(lhs, canon_rhs), c

            # Neither side has variables. One side must be affine in parameters.
            lhs_affine = False
            rhs_affine = False
            with _param_vexity_scope(lhs):
                lhs_affine = lhs.is_affine()
            with _param_vexity_scope(rhs):
                rhs_affine = rhs.is_affine()
            assert lhs_affine or rhs_affine

            if lhs_affine:
                canon_rhs, c = self.canonicalize_tree(rhs)
                t = Variable(canon_rhs.shape)
                return lhs * t, c + [t == canon_rhs]
            else:
                canon_lhs, c = self.canonicalize_tree(lhs)
                t = Variable(canon_lhs.shape)
                return canon_lhs * rhs, c + [t == canon_lhs]
        else:
            return super(Dcp2Cone, self).canonicalize_tree(expr)

    def apply(self, problem):
        """Converts a DCP problem to a conic form.
        """
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to cone program")
        return super(Dcp2Cone, self).apply(problem)
