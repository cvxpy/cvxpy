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

import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
from cvxpy.expressions import cvxtypes
import abc
import numpy as np


class Constraint(u.Canonical):
    """The base class for constraints.

    A constraint is an equality, inequality, or more generally a generalized
    inequality that is imposed upon a mathematical expression or a list of
    thereof.

    Parameters
    ----------
    args : list
        A list of expression trees.
    constr_id : int
        A unique id for the constraint.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, args, constr_id=None):
        # TODO cast constants.
        # self.args = [cvxtypes.expression().cast_to_const(arg) for arg in args]
        self.args = args
        if constr_id is None:
            self.constr_id = lu.get_id()
        else:
            self.constr_id = constr_id
        self.dual_variables = [cvxtypes.variable()(arg.shape) for arg in args]
        super(Constraint, self).__init__()

    @abc.abstractmethod
    def is_dcp(self):
        """Checks whether the constraint is DCP.

        Returns
        -------
        bool
            True if the constraint is DCP, False otherwise.
        """
        return NotImplemented

    @abc.abstractproperty
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        NumPy.ndarray
            The residual, or None if the constrained expression does not have
            a value.
        """
        return NotImplemented

    def violation(self):
        """The numeric residual of the constraint.

        The violation is defined as the distance between the constrained
        expression's value and its projection onto the domain of the
        constraint:

        .. math::

            ||\Pi(v) - v||_2^2

        where :math:`v` is the value of the constrained expression and
        :math:`\\Pi` is the projection operator onto the constraint's domain .

        Returns
        -------
        NumPy.ndarray
            The residual value.

        Raises
        ------
        ValueError
            If the constrained expression does not have a value associated
            with it.
        """
        residual = self.residual
        if residual is None:
            raise ValueError("Cannot compute the violation of an constraint "
                             "whose expression is None-valued.")
        return residual

    def value(self, tolerance=1e-8):
        """Checks whether the constraint violation is less than a tolerance.

        Parameters
        ----------
            tolerance : float
                The absolute tolerance to impose on the violation.

        Returns
        -------
            bool
                True if the violation is less than ``tolerance``, False
                otherwise.

        Raises
        ------
            ValueError
                If the constrained expression does not have a value associated
                with it.
        """
        residual = self.residual
        if residual is None:
            raise ValueError("Cannot compute the value of an constraint "
                             "whose expression is None-valued.")
        return np.all(residual <= tolerance)

    @property
    def id(self):
        """Wrapper for compatibility with variables.
        """
        return self.constr_id

    def save_value(self, value):
        """Save the value of the dual variable for the constraint's parent.

        Args:
            value: The value of the dual variable.
        """
        pass

    def get_data(self):
        """Data needed to copy.
        """
        return [self.id]
