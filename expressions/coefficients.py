import settings

class Coeff(object):
    """
    From Scoop/QCML
    A class / container for storing coefficients. These are stored in the 
    form of c + a1*p1 + a2*p2 + ... , where pi are expression names.
    
    This assumes that expressions are uniquely identified by their names. That
    is, we can't have a variable named 'x' that is a positive scalar and
    another variable named 'x' that is suddenly a negative matrix.
    """
    def __init__(self, coeff_dict):
        self.coeff_dict = coeff_dict

    def __repr__(self):
        return "Coeff(%s)" % (self.coeff_dict)

    # Is the only non-zero coefficient the constant term?
    def is_constant(self):
        return (settings.CONSTANT in self.coeff_dict and len(self.coeff_dict) == 1)

    # Get the constant term.
    def constant(self):
        return self.coeff_dict.get(settings.CONSTANT, 0)
    
    def __add__(self, other):
        # add the constants
        a = self.coeff_dict
        b = other.coeff_dict
        # got this nice piece of code off stackoverflow http://stackoverflow.com/questions/1031199/adding-dictionaries-in-python
        d = dict( (n, a.get(n, 0) + b.get(n, 0)) for n in set(a)|set(b) )
        
        return Coeff(d)
    
    def __neg__(self):
        d = {}
        for k in self.coeff_dict:
            d[k] = -self.coeff_dict[k]
        return Coeff(d)
    
    def __sub__(self,other):
        return self + (-other)
    
    # Multiply a parameter by an expression.
    def __mul__(self,other):
        if not self.is_constant():
            raise Exception("Cannot multiply by a non-constant on the left.") # TODO get name?
        if self.is_constant():
            const_term,coeffs = (self.constant(),other.coeff_dict)
        else:
            const_term,coeffs = (other.constant(),self.coeff_dict)
        d = dict( ( k,Coeff.constant_mul(const_term,v) ) for k,v in coeffs.iteritems() )
        return Coeff(d)

    # Handles matrix and scalar multiplication.
    @staticmethod
    def constant_mul(constant, value):
        try:
            return constant.dot(value)
        except Exception, e:
            return constant * value