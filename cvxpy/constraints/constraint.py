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

import cvxpy.lin_ops.lin_utils as lu
import abc


class Constraint(object):
    """Abstract super class for constraints.

    TODO rationalize constraint classes. Make lin_op versions
    of SOC, SDP, etc.

    Attributes
    ----------
    args : list
        A list of expression trees.
    constr_id : int
        A unique id for the constraint.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, args, constr_id=None):
        self.args = args
        if constr_id is None:
            self.constr_id = lu.get_id()
        else:
            self.constr_id = constr_id
        super(Constraint, self).__init__()

    @property
    def id(self):
        """Wrapper for compatibility with variables.
        """
        return self.constr_id
