import abc
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.expression import Expression
from cvxpy.expressions.curvature import Curvature
from monotonicity import Monotonicity
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
        # Validate arguments
        self.validate_arguments()
        # Initialize _shape
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
        return Curvature.sum(arg_curvatures)

    # Raises an error if the arguments passed to the atom are invalid.
    @abc.abstractmethod
    def validate_arguments(self):
        return NotImplemented

    # Represent the atom as a linear objective and linear/basic SOC constraints.
    def canonicalize(self):
        # canonicalize arguments.
        var_args = []
        final_constraints = []
        for arg in self.args:
            obj,constraints = arg.canonical_form()
            # Replace affine objective with a single variable.
            # TODO why does Grant do this?
            u = Variable(*arg.size)
            var_args.append(u)
            constraints.append( AffEqConstraint(u, obj) )
            final_constraints += constraints
        graph_obj,graph_constr = self.graph_implementation(var_args)
        # Replace the atom with a variable subject to a constraint
        # with graph_obj
        v = Variable(*self.size)
        v,dummy = v.canonical_form()
        graph_constr.append(self.graph_constraint(v, graph_obj))
        return (v,final_constraints + graph_constr)

    # Return the top level constraint for the graph implementation.
    # Of the form atom_var ==/>=/<= graph_obj
    def graph_constraint(self, atom_var, graph_obj):
        if self.base_curvature().is_affine():
            return AffEqConstraint(graph_obj, atom_var)
        elif self.base_curvature().is_convex():
            return AffLeqConstraint(graph_obj, atom_var)
        elif self.base_curvature().is_concave():
            return AffLeqConstraint(atom_var, graph_obj)

    # Returns an affine objective and set of affine/SOC 
    # constraints equivalent to the atom.
    # var_args - a list of single variable arguments.
    @abc.abstractmethod
    def graph_implementation(var_args):
        return NotImplemented