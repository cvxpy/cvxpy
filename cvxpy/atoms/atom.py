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
from ..expressions.constants import Constant
from ..expressions.variables import Variable
from ..expressions.expression import Expression
import abc

class Atom(Expression):
    """ Abstract base class for atoms. """
    __metaclass__ = abc.ABCMeta
    # args are the expressions passed into the Atom constructor.
    def __init__(self, *args):
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError(
                "No arguments given to '%s'." % self.__class__.__name__
            )
        # Convert raw values to Constants.
        self.args = map(Expression.cast_to_const, args)
        self.validate_arguments()
        self.init_dcp_attr()
        self.subexpressions = self.args

    # Returns the string representation of the function call.
    def name(self):
        return "%s(%s)" % (self.__class__.__name__,
                           ", ".join([arg.name() for arg in self.args]))

    # Determines the curvature, sign, and shape from the arguments.
    def init_dcp_attr(self):
        # Initialize _shape. Raises an error for invalid argument sizes.
        shape = self.shape_from_args()
        sign = self.sign_from_args()
        curvature = Atom.dcp_curvature(self.func_curvature(),
                                       self.args,
                                       self.monotonicity())
        self._dcp_attr = u.DCPAttr(sign, curvature, shape)

    # Returns argument curvatures as a list.
    def argument_curvatures(self):
        return [arg.curvature for arg in self.args]

    # Raises an error if the arguments are invalid.
    def validate_arguments(self):
        pass

    # The curvature of the atom if all arguments conformed to DCP.
    # Alternatively, the curvature of the atom's function.
    @abc.abstractmethod
    def func_curvature(self):
        return NotImplemented

    # Returns a list with the monotonicity in each argument.
    # monotonicity can depend on the sign of the argument.
    @abc.abstractmethod
    def monotonicity(self):
        return NotImplemented

    # Applies DCP composition rules to determine curvature in each argument.
    # The overall curvature is the sum of the argument curvatures.
    @staticmethod
    def dcp_curvature(curvature, args, monotonicities):
        if len(args) != len(monotonicities):
            raise Exception('The number of args be'
                            ' equal to the number of monotonicities.')
        arg_curvatures = []
        for arg,monotonicity in zip(args,monotonicities):
            arg_curv = u.monotonicity.dcp_curvature(monotonicity, curvature,
                                                    arg.sign, arg.curvature)
            arg_curvatures.append(arg_curv)
        return reduce(lambda x,y: x+y, arg_curvatures)

    # Represent the atom as an affine objective and affine/basic SOC constraints.
    def canonicalize(self):
        # Constant atoms are treated as a leaf.
        if self.curvature.is_constant():
            return (self, [])
        else:
            arg_objs = []
            constraints = []
            for arg in self.args:
                obj,constr = arg.canonicalize()
                arg_objs.append(obj)
                constraints += constr
            graph_obj,graph_constr = self.graph_implementation(arg_objs)
            return (graph_obj, constraints + graph_constr)

    def coefficients(self):
        """Coefficients for a constant expression with non-affine atoms.
        """
        if self.curvature.is_constant():
            return Constant(self.value).coefficients()
        else:
            return self.func_coefficients()

    def func_coefficients(self):
        """Only affine atoms can return coefficients if non-constant.
        """
        raise Exception("Cannot canonicalize a non-affine expression.")

    # Returns an affine expression and list of
    # constraints equivalent to the atom.
    # arg_objs - the canonical objectives of the arguments.
    @abc.abstractmethod
    def graph_implementation(self, arg_objs):
        return NotImplemented

    # Wraps an atom's numeric function that requires numpy ndarrays as input.
    # Ensures both inputs and outputs are the correct matrix types.
    @staticmethod
    def numpy_numeric(numeric_func):
        def new_numeric(self, values):
            values = map(intf.DEFAULT_NP_INTERFACE.const_to_matrix, values)
            result = numeric_func(self, values)
            return intf.DEFAULT_SPARSE_INTERFACE.const_to_matrix(result)
        return new_numeric
