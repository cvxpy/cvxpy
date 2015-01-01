"""
Copyright 2013 Steven Diamond, Eric Chu

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
from cvxpy.expressions.variables.variable import Variable
from cvxpy.constraints.int_constr import IntConstr

class Int(Variable):
    """ An integer variable. """

    def canonicalize(self):
        """Variable must be an integer.
        """
        obj, constr = super(Int, self).canonicalize()
        return (obj, constr + [IntConstr(obj)])

    def __repr__(self):
        """String to recreate the object.
        """
        return "Int(%d, %d)" % self.size
