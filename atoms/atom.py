import abc
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity

class Atom(Expression):
    """ Abstract base class for atoms. """
    __metaclass__ = abc.ABCMeta
    # args are the expressions passed into the Atom constructor.
    def __init__(self, *args):
        # Throws error if args is empty.
        if len(args) == 0:
            raise TypeError("No arguments given to '%s'." % self.name())
        # Convert raw values to Parameters
        self.args = map(Expression.cast_to_const, list(args))

    # Returns the 
    def name(self):
        return "%s(%s)" % (self.__class__.__name__, 
                           ", ".join([arg.name() for arg in self.args]))

    # Determines curvature from args and sign.
    def curvature(self):
        curvature = self.base_curvature()
        return Atom.dcp_curvature(curvature, self.args, self.monotonicity())

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
        arg_curvatures = [monotonicity.dcp_curvature(curvature,arg.curvature())
                          for arg,monotonicity in zip(args,monotonicities)]
        return Curvature.sum(arg_curvatures)

    # Represent the atom as a linear objective and linear/basic SOC constraints.
    def canonicalize(self):
        obj,constraints = self.base_canonicalize()
        final_constraints = []
        for constr in constraints:
            final_constraints += constr.canonicalize()[1]
        return (obj,final_constraints)

    # The atom's canonicalize method.
    @abc.abstractmethod
    def base_canonicalize(self):
        return NotImplemented