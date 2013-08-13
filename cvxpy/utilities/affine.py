import abc
import cvxpy.expressions.types as types

class Affine(object):
    """ Interface for affine objects. """
    __metaclass__ = abc.ABCMeta

    # Returns a dict of variable id to coefficient.
    @abc.abstractmethod
    def coefficients(self, interface):
        return NotImplemented

    # Returns a list of variables in the expression.
    @abc.abstractmethod
    def variables(self):
        return NotImplemented

    # Casts expression as an AffObjective.
    @staticmethod
    def cast_as_affine(expr):
        if isinstance(expr, types.aff_obj()):
            return expr
        elif isinstance(expr, types.expression()):
            obj,constr = expr.canonical_form()
            if len(constr) > 0:
                raise Exception("Non-affine argument '%s'." % expr.name())
            return obj
        else:
            return Affine.cast_as_affine(types.constant()(expr))