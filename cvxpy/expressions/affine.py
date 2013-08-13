import types
import cvxpy.settings as s
import cvxpy.utilities as u
from collections import deque

class AffObjective(u.Affine):
    """ An affine objective. The result of canonicalization. """
    # variables - a list of variables.
    # terms - a list of multiplication queues.
    # shape - an object representing the dimensions.
    def __init__(self, variables, terms, shape):
        self._vars = variables
        self._terms = terms
        self._shape = shape
        super(AffObjective, self).__init__()

    def name(self):
        return str(self._terms)

    def __str__(self):
        return self.name()

    def __repr__(self):
        return self.name()

    # The dimensions of the objective.
    @property
    def size(self):
        return self._shape.size

    # Returns a dict of term id to coefficient.
    # interface - the matrix interface to convert constants
    #             into a matrix of the target class.
    def coefficients(self, interface):
        final_coeffs = {}
        for mults in self._terms:
            root = mults[0]
            root_coeffs = root.coefficients(interface)
            for id,coeff in root_coeffs.items():
                coefficient = self.dequeue_mults(coeff, mults, interface)
                if id in final_coeffs:
                    final_coeffs[id] = final_coeffs[id] + coefficient
                else:
                    final_coeffs[id] = coefficient
        return final_coeffs

    # Resolves a multiplication stack into a single coefficient.
    @staticmethod
    def dequeue_mults(coefficient, mults, interface):
        for i in range(len(mults)-1):
            lh = mults[i+1]
            lh_coeff = lh.coefficients(interface)[s.CONSTANT]
            coefficient = lh_coeff * coefficient
        return coefficient

    # Returns a list of variables.
    def variables(self):
        return self._vars

    # Concatenates the terms.
    def __add__(self, other):
        return AffObjective(self.variables() + other.variables(),
                               self._terms + other._terms,
                               self._shape + other._shape)

    def __sub__(self, other):
        return self + -other

    # Distributes multiplications by left hand terms
    # across right hand terms.
    def __mul__(self, other):
        terms = AffObjective.mul_terms(self._terms, other._terms)
        return AffObjective(other.variables(), terms, 
                               self._shape * other._shape)

    # Multiplies every term by -1.
    def __neg__(self):
        lh_mult = deque([types.constant()(-1)])
        terms = AffObjective.mul_terms([lh_mult], self._terms)
        return AffObjective(self.variables(), terms, self._shape)

    # Utility function for multiplying lists of terms.
    @staticmethod
    def mul_terms(lh_terms, rh_terms):
        terms = []
        for lh_mult in lh_terms:
            for rh_mult in rh_terms:
                mult = deque(rh_mult)
                mult.extend(lh_mult)
                terms.append(mult)
        return terms