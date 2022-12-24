"""
Copyright 2013 Steven Diamond

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

import abc

import numpy as np

import cvxpy.lin_ops.lin_utils as lu
import cvxpy.utilities as u
from cvxpy.expressions import cvxtypes


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

    def __init__(self, args, constr_id=None) -> None:
        # TODO cast constants.
        # self.args = [cvxtypes.expression().cast_to_const(arg) for arg in args]
        self.args = args
        if constr_id is None:
            self.constr_id = lu.get_id()
        else:
            self.constr_id = constr_id
        self._construct_dual_variables(args)
        super(Constraint, self).__init__()

    def __str__(self):
        """Returns a string showing the mathematical constraint.
        """
        return self.name()

    def __repr__(self) -> str:
        """Returns a string with information about the constraint.
        """
        return "%s(%s)" % (self.__class__.__name__,
                           repr(self.args[0]))

    def _construct_dual_variables(self, args) -> None:
        self.dual_variables = [cvxtypes.variable()(arg.shape) for arg in args]

    @property
    def shape(self):
        """int : The shape of the constrained expression."""
        return self.args[0].shape

    @property
    def size(self):
        """int : The size of the constrained expression."""
        return self.args[0].size

    def is_real(self) -> bool:
        """Is the Leaf real valued?
        """
        return not self.is_complex()

    def is_imag(self) -> bool:
        """Is the Leaf imaginary?
        """
        return all(arg.is_imag() for arg in self.args)

    def is_complex(self) -> bool:
        """Is the Leaf complex valued?
        """
        return any(arg.is_complex() for arg in self.args)

    @abc.abstractmethod
    def is_dcp(self, dpp: bool = False) -> bool:
        """Checks whether the constraint is DCP.

        Returns
        -------
        bool
            True if the constraint is DCP, False otherwise.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def is_dgp(self, dpp: bool = False) -> bool:
        """Checks whether the constraint is DGP.

        Returns
        -------
        bool
            True if the constraint is DGP, False otherwise.
        """
        raise NotImplementedError()

    def is_dpp(self, context='dcp') -> bool:
        if context.lower() == 'dcp':
            return self.is_dcp(dpp=True)
        elif context.lower() == 'dgp':
            return self.is_dgp(dpp=True)
        else:
            raise ValueError("Unsupported context ", context)

    @abc.abstractproperty
    def residual(self):
        """The residual of the constraint.

        Returns
        -------
        NumPy.ndarray
            The residual, or None if the constrained expression does not have
            a value.
        """
        raise NotImplementedError()

    def violation(self):
        """The numeric residual of the constraint.

        The violation is defined as the distance between the constrained
        expression's value and its projection onto the domain of the
        constraint:

        .. math::

            ||\\Pi(v) - v||_2^2

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

    def value(self, tolerance: float = 1e-8):
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

    def get_data(self):
        """Data needed to copy.
        """
        return [self.id]

    def __nonzero__(self):
        """Raises an exception when called.

        Python 2 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        return self._chain_constraints()

    def _chain_constraints(self):
        """Raises an error due to chained constraints.
        """
        raise Exception(
            ("Cannot evaluate the truth value of a constraint or "
             "chain constraints, e.g., 1 >= x >= 0.")
        )

    def __bool__(self):
        """Raises an exception when called.

        Python 3 version.

        Called when evaluating the truth value of the constraint.
        Raising an error here prevents writing chained constraints.
        """
        return self._chain_constraints()

    # TODO(rileyjmurray): add a function to compute dual-variable violation.

    @property
    def dual_value(self):
        """NumPy.ndarray : The value of the dual variable.
        """
        dual_vals = [dv.value for dv in self.dual_variables]
        if len(dual_vals) == 1:
            return dual_vals[0]
        else:
            return dual_vals

    def save_dual_value(self, value) -> None:
        """Save the value of the dual variable for the constraint's parent.
        Args:
            value: The value of the dual variable.
        """
        self.dual_variables[0].save_value(value)
