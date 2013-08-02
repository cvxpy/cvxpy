import types
from collections import deque

class AffineObjective(object):
    """ An affine objective. The result of canonicalization. """
    # terms - a list of (leaf, multiplication queue) tuples.
    # shape - an object representing the dimensions.
    def __init__(self, terms, shape):
        self._terms = terms
        self._shape = shape
        super(AffineObjective, self).__init__()

    # The dimensions of the objective.
    @property
    def size(self):
        return self._shape.size

    # Returns a dict of term id to coefficient.
    # interface - the matrix interface to convert constants
    #             into a matrix of the target class.
    def coefficients(self, interface):
        coeffs = {}
        for term,mults in self._terms:
            mult_coeffs = self.dequeue_mults(mults, interface)
            # got this nice piece of code off stackoverflow http://stackoverflow.com/questions/1031199/adding-dictionaries-in-python
            coeffs = dict( (n, mult_coeffs.get(n, 0) + coeffs.get(n, 0))
                            for n in set(mult_coeffs)|set(coeffs) )
        return coeffs

    # Resolves a multiplication stack into a dictionary of
    # id: coefficient.
    @staticmethod
    def dequeue_mults(mults, interface):
        mults = deque(mults) # Shallow copy for repeated leaves.
        root = mults.popleft()
        coeffs = root.coefficients(interface)
        while(len(mults) > 0):
            lh = mults.popleft()
            lh_coeffs = lh.coefficients(interface)
            # TODO only use constant?
            for k,lh_val in lh_coeffs.items():
                coeffs = dict((k,lh_val*v) for k,v in coeffs.items())
        return coeffs

    # Returns a dictionary of name to variable.
    def variables(self):
        vars = {}
        for term,mult in self._terms:
            if isinstance(term, types.variable()):
                vars[term.id] = term
        return vars

    # Concatenates the terms.
    def __add__(self, other):
        return AffineObjective(self._terms + other._terms,
                               self._shape + other._shape)

    def __sub__(self, other):
        return self + -other

    # Distributes multiplications by left hand terms
    # across right hand terms.
    def __mul__(self, other):
        terms = AffineObjective.mul_terms(self._terms, other._terms)
        return AffineObjective(terms, self._shape * other._shape)

    # Multiplies every term by -1.
    def __neg__(self):
        lh_mult = deque([types.constant()(-1)])
        lh_terms = [(None, lh_mult)]
        terms = AffineObjective.mul_terms(lh_terms, self._terms)
        return AffineObjective(terms, self._shape)

    # Utility function for multiplying lists of terms.
    @staticmethod
    def mul_terms(lh_terms, rh_terms):
        terms = []
        for lh_term,lh_mult in lh_terms:
            for rh_term,rh_mult in rh_terms:
                mult = deque(rh_mult)
                mult.extend(lh_mult)
                terms.append( (rh_term,mult) )
        return terms