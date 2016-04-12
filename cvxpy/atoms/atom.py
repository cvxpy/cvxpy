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


from .. import settings as s
from .. import utilities as u
from .. import interface as intf
from ..expressions.constants import Constant, CallbackParam
from ..expressions.variables import Variable
from ..expressions.expression import Expression
import abc
import sys
from toolz.functoolz import memoize
if sys.version_info >= (3, 0):
    from functools import reduce


class Atom(Expression):
    """ Abstract base class for atoms. """
    __metaclass__ = abc.ABCMeta
    # args are the expressions passed into the Atom constructor.
    def __init__(self, *args):
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError(
                "No arguments given to %s." % self.__class__.__name__
            )
        # Convert raw values to Constants.
        self.args = [Atom.cast_to_const(arg) for arg in args]
        self.validate_arguments()
        self._size = self.size_from_args()

    def name(self):
        """Returns the string representation of the function call.
        """
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join([arg.name() for arg in self.args]))

    def validate_arguments(self):
        """Raises an error if the arguments are invalid.
        """
        pass

    @abc.abstractmethod
    def size_from_args(self):
        """Returns the (row, col) size of the expression.
        """
        return NotImplemented

    @property
    def size(self):
        return self._size

    @abc.abstractmethod
    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        return NotImplemented

    @memoize
    def is_positive(self):
        """Is the expression positive?
        """
        return self.sign_from_args()[0]

    @memoize
    def is_negative(self):
        """Is the expression negative?
        """
        return self.sign_from_args()[1]

    @abc.abstractmethod
    def is_atom_convex(self):
        """Is the atom convex?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_atom_concave(self):
        """Is the atom concave?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return NotImplemented

    @abc.abstractmethod
    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return NotImplemented

    @memoize
    def is_convex(self):
        """Is the expression convex?
        """
        # Applies DCP composition rule.
        if not self.is_atom_convex():
            return False
        for idx, arg in enumerate(self.args):
            if not (arg.is_affine() or \
                    (arg.is_convex() and self.is_incr(idx)) or \
                    (arg.is_concave() and self.is_decr(idx))):
                return False
        return True

    @memoize
    def is_concave(self):
        """Is the expression concave?
        """
        # Applies DCP composition rule.
        if not self.is_atom_concave():
            return False
        for idx, arg in enumerate(self.args):
            if not (arg.is_affine() or \
                    (arg.is_concave() and self.is_incr(idx)) or \
                    (arg.is_convex() and self.is_decr(idx))):
                return False
        return True

    def canonicalize(self):
        """Represent the atom as an affine objective and conic constraints.
        """
        # Constant atoms are treated as a leaf.
        if self.is_constant():
            # Parameterized expressions are evaluated later.
            if self.parameters():
                rows, cols = self.size
                param = CallbackParam(lambda: self.value, rows, cols)
                return param.canonical_form
            # Non-parameterized expressions are evaluated immediately.
            else:
                return Constant(self.value).canonical_form
        else:
            arg_objs = []
            constraints = []
            for arg in self.args:
                obj, constr = arg.canonical_form
                arg_objs.append(obj)
                constraints += constr
            # Special info required by the graph implementation.
            data = self.get_data()
            graph_obj, graph_constr = self.graph_implementation(arg_objs,
                                                                self.size,
                                                                data)
            return (graph_obj, constraints + graph_constr)

    @abc.abstractmethod
    def graph_implementation(self, arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        return NotImplemented

    def variables(self):
        """Returns all the variables present in the arguments.
        """
        var_list = []
        for arg in self.args:
            var_list += arg.variables()
        # Remove duplicates.
        return list(set(var_list))

    def parameters(self):
        """Returns all the parameters present in the arguments.
        """
        param_list = []
        for arg in self.args:
            param_list += arg.parameters()
        # Remove duplicates.
        return list(set(param_list))

    @property
    def value(self):
        # Catch the case when the expression is known to be
        # zero through DCP analysis.
        if self.is_zero():
            result = intf.DEFAULT_INTF.zeros(*self.size)
        else:
            arg_values = []
            for arg in self.args:
                # A argument without a value makes all higher level
                # values None.
                # But if the atom is constant with non-constant
                # arguments it doesn't depend on its arguments,
                # so it isn't None.
                arg_val = arg.value
                if arg_val is None and not self.is_constant():
                    return None
                else:
                    arg_values.append(arg_val)
            result = self.numeric(arg_values)

        # Reduce to a scalar if possible.
        if intf.size(result) == (1, 1):
            return intf.scalar_value(result)
        else:
            return result

    @staticmethod
    def numpy_numeric(numeric_func):
        """Wraps an atom's numeric function that requires numpy ndarrays as input.
           Ensures both inputs and outputs are the correct matrix types.
        """
        def new_numeric(self, values):
            interface = intf.DEFAULT_INTF
            values = [interface.const_to_matrix(v, convert_scalars=True)
                      for v in values]
            result = numeric_func(self, values)
            return intf.DEFAULT_INTF.const_to_matrix(result)
        return new_numeric
