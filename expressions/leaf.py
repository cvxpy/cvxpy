import abc
from collections import deque
import types

class Leaf(object):
    """
    A leaf node, i.e. a Variable, Constant, or Parameter.
    """
    __metaclass__ = abc.ABCMeta
    # Every multiplication queue begins with the leaf itself.
    def terms(self):
        return [( self, deque([self]) )]

    def canonicalize(self):
        return (self,[])

    # The id of the leaf for coefficients.
    @abc.abstractproperty
    def id(self):
        return NotImplemented

    # The coefficient value.
    @abc.abstractmethod
    def coefficient(self, interface):
        return NotImplemented

    # Dequeue all the multiplications on the leaf,
    # returning a {leaf id: coefficient} dict and parameter equality constraints.
    # interface - the matrix interface to convert constants
    #             into a matrix of the target class.
    @staticmethod
    def dequeue_mults(mult_queue, interface):
        first_term = True
        constraints = []
        while len(mult_queue) > 0:
            term = mult_queue.popleft()
            # The beginning of a multiplication chain.
            if isinstance(term, types.variable()) or first_term:
                leaf = term
                coeff = leaf.coefficient(interface)
                coeff_expr = leaf
                lone_variable = isinstance(term, types.variable())
            # If a constant, update the coefficient.
            elif isinstance(term, types.constant()):
                lone_variable = False
                coeff = term.coefficient(interface) * coeff
                coeff_expr = types.constant()(coeff) * leaf
            # Create a parameter equality if it's a parameter.
            if isinstance(term, types.parameter()):
                # Create a constraint t = P*x
                if lone_variable:
                    coeff = term
                    size = (coeff * leaf).size
                    var = types.variable()(*size)
                    constr = (coeff * leaf == var)
                # Reduce the head of the queue to a lone variable.
                else:
                    var = types.variable()(*coeff_expr.size)
                    constr = (coeff_expr == var)
                    # Special case if the parameter is the first term.
                    if not first_term:
                        mult_queue.appendleft(term)
                constr.coefficients = {var.id: -var.coefficient(interface),
                                       leaf.id: coeff,
                                      }
                constraints.append(constr)
                mult_queue.appendleft(var)
            first_term = False
        return ({leaf.id: coeff}, constraints)