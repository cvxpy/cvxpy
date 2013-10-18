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
from ..expressions.variables import Variable
from ..expressions.expression import Expression
from ..expressions.affine import AffObjective
from .. import utilities as u
from ..constraints.affine import AffEqConstraint, AffLeqConstraint

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
        self.subexpressions = self.args
        # Initialize _shape. Raises an error for invalid argument sizes.
        self.set_shape()
        # Initialize sign and curvature.
        self.set_sign_curv()
        super(Atom, self).__init__()

    # Returns the string representation of the function call.
    def name(self):
        return "%s(%s)" % (self.__class__.__name__, 
                           ", ".join([arg.name() for arg in self.args]))

    @abc.abstractmethod
    def set_shape(self):
        return NotImplemented

    # Sets signed curvature based on the arguments' signed curvatures.
    def set_sign_curv(self):
        sign = self.sign_from_args()
        curvature = Atom.dcp_curvature(self.base_curvature(), 
                                       self.args, 
                                       self.monotonicity())
        self._context = u.Context(sign, curvature, self._shape)

    # Returns argument curvatures as a list.
    def argument_curvatures(self):
        return [arg.curvature for arg in self.args]

    # The curvature of the atom if all arguments conformed to DCP.
    # @abc.abstractmethod
    def base_curvature(self):
        return NotImplemented

    # Returns a list with the monotonicity in each argument.
    # Monotonicity can depend on the sign of the argument.
    # @abc.abstractmethod
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
            arg_curv = monotonicity.dcp_curvature(curvature, arg.sign, arg.curvature)
            arg_curvatures.append(arg_curv)
        return reduce(lambda x,y: x+y, arg_curvatures)

    # Represent the atom as an affine objective and affine/basic SOC constraints.
    def canonicalize(self):
        # Constant atoms are treated as a leaf.
        if self.curvature.is_constant():
            obj = AffObjective([], ConstantAtom(self), self.shape)
            return (obj, final_constraints)
        # Non-constant atoms are expanded into an affine objective and constraints.
        else:
            var_args = []
            final_constraints = []
            for arg in self.args:
                # canonicalize arguments.
                obj,constraints = arg.canonical_form()
                var_args.append(obj)
                final_constraints += constraints
            graph_var,graph_constr = self.graph_implementation(var_args, self.size)
            obj = u.Affine.cast_as_affine(graph_var)
            return (obj,final_constraints + graph_constr)

    # Returns a variable and set of affine/SOC 
    # constraints equivalent to the atom.
    # var_args - a list of single variable arguments.
    # size - the dimensions of the variable to return.
    @abc.abstractmethod
    def graph_implementation(var_args, size):
        return NotImplemented