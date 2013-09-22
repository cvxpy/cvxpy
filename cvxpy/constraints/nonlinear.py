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

from cvxpy.expressions.variables import Variable
import cvxpy.interface.matrix_utilities as intf
from affine import AffEqConstraint, AffLeqConstraint

class NonlinearConstraint(object):
    """ 
    A nonlinear inequality constraint:
        f(x) <= 0
    where f is twice-differentiable.
    
    TODO: this may not be the best way to handle these constraints, but it is
    one of many (of course).
    """
    # f - a nonlinear function
    # x - the variables involved in the function
    def __init__(self, f, x):
        # assert(isinstance(f, NonlinearFunc))
        self.f = f
        self.vars_involved = x # TODO unify syntax with affine
        super(NonlinearConstraint, self).__init__()

    @property
    def size(self):
        return (self.f()[0],1)