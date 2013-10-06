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
from . variable import Variable
from .. affine import AffObjective
from ... constraints.semi_definite import SDP

class SemidefVar(Variable):
    """ A semidefinite variable. """
    def __init__(self, n=1, name=None):
        super(SemidefVar, self).__init__(n,n,name)
    
    # A semidefinite variable is no different from a normal variable except
    # that it adds an SDP constraint on the variable.
    def _constraints(self):
        # ECHU: sad face, when used in expressions this fails
        return [SDP(self._objective())]