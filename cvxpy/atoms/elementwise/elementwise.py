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
from ..atom import Atom

class Elementwise(Atom):
    """ Abstract base class for elementwise atoms. """
    __metaclass__ = abc.ABCMeta
    # Saves original arguments for indexing and transpose.
    def __init__(self, *args):
        super(Elementwise, self).__init__(*args)
        self.original_args = self.args

    # Return the given index into the atomic expression.
    def index_object(self, key):
        args = []
        for arg in self.original_args:
            if arg.size == (1,1):
                args.append(arg)
            else:
                args.append(arg[key])
        return self.__class__(*args)

    # Return the transpose of the atomic expression.
    def transpose(self):
        return self.__class__(*[arg.T for arg in self.original_args])