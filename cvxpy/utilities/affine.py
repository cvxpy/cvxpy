"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

import abc
from ..expressions import types

class Affine(object):
    """ Interface for affine objects. """
    __metaclass__ = abc.ABCMeta

    # Returns a dict of variable id to coefficient.
    @abc.abstractmethod
    def coefficients(self):
        return NotImplemented

    # Returns a list of variables in the expression.
    @abc.abstractmethod
    def variables(self):
        return NotImplemented

    # Casts expression as an AffExpression.
    @staticmethod
    def cast_as_affine(expr):
        if isinstance(expr, types.aff_expr()):
            return expr
        elif isinstance(expr, types.expression()):
            obj,constr = expr.canonical_form()
            if len(constr) > 0:
                raise Exception("Non-affine argument '%s'." % expr.name())
            return obj
        else:
            return Affine.cast_as_affine(types.constant()(expr))