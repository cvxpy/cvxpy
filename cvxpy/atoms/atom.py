import abc
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.expression import Expression
import cvxpy.utilities as u
from cvxpy.constraints.affine import AffEqConstraint, AffLeqConstraint

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
        # Convert raw values to Constants
        self.args = map(Expression.cast_to_const, list(args))
        # Initialize _shape. Raises an error for invalid argument sizes.
        self.set_shape()
        super(Atom, self).__init__()

    # Returns the string representation of the function call.
    def name(self):
        return "%s(%s)" % (self.__class__.__name__, 
                           ", ".join([arg.name() for arg in self.args]))

    # Determines curvature from args and sign.
    @property
    def curvature(self):
        curvature = self.base_curvature()
        return Atom.dcp_curvature(curvature, self.args, self.monotonicity())

    @abc.abstractmethod
    def set_shape(self):
        return NotImplemented

    @property
    def size(self):
        return self._shape.size

    # Returns argument curvatures as a list.
    def argument_curvatures(self):
        return [arg.curvature for arg in self.args]

    # The curvature of the atom if all arguments conformed to DCP.
    @abc.abstractmethod
    def base_curvature(self):
        return NotImplemented

    # Returns a list with the monotonicity in each argument.
    # Monotonicity can depend on the sign of the argument.
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
        arg_curvatures = [monotonicity.dcp_curvature(curvature,arg.curvature)
                          for arg,monotonicity in zip(args,monotonicities)]
        return u.Curvature.sum(arg_curvatures)

    # Represent the atom as an affine objective and affine/basic SOC constraints.
    def canonicalize(self):
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