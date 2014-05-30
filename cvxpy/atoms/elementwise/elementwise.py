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
from cvxpy.atoms.atom import Atom
import operator as op

class Elementwise(Atom):
    """ Abstract base class for elementwise atoms. """
    __metaclass__ = abc.ABCMeta

    def shape_from_args(self):
        """Shape is the same as the sum of the arguments.
        """
        return reduce(op.add, [arg.shape for arg in self.args])

    def validate_arguments(self):
        """
        Verify that all the shapes are the same
        or can be promoted.
        """
        shape = self.args[0].shape
        for arg in self.args[1:]:
            shape = shape + arg.shape
