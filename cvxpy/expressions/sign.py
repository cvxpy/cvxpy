class Sign(object):
    """ Sign of convex optimization expressions. """
    POSITIVE_KEY = 'POSITIVE'
    NEGATIVE_KEY = 'NEGATIVE'
    UNKNOWN_KEY = 'UNKNOWN'
    ZERO_KEY = 'ZERO'
    
    # SIGN_MAP for resolving sign addition using bitwise OR
    SIGN_MAP = {ZERO_KEY: 0, POSITIVE_KEY: 1, NEGATIVE_KEY: 2, UNKNOWN_KEY: 3}
    
    def __init__(self,sign_str):
        sign_str = sign_str.upper()
        if sign_str in Sign.SIGN_MAP.keys():
            self.sign_str = sign_str
        else:
            raise Exception("No such sign %s exists." % str(sign_str))

    # Returns whether the sign string is a valid sign type.
    @staticmethod
    def is_sign(sign_str):
        return sign_str in Sign.SIGN_MAP.keys()

    # Converts a number to a sign.
    @staticmethod
    def val_to_sign(val):
        if val > 0:
            return Sign.POSITIVE
        elif val == 0:
            return Sign.ZERO
        else:
            return Sign.NEGATIVE
    
    # Arithmetic operators  
    def __add__(self, other):
        sign_val = Sign.SIGN_MAP[self.sign_str] | Sign.SIGN_MAP[other.sign_str]
        for key,val in Sign.SIGN_MAP.items():
            if val == sign_val:
                return Sign(key)
    
    def __sub__(self, other):
        return self + -other
       
    def __mul__(self, other):
        if self == Sign.ZERO or other == Sign.ZERO:
            return Sign.ZERO
        elif self == Sign.UNKNOWN or other == Sign.UNKNOWN:
            return Sign.UNKNOWN
        elif self != other:
            return Sign.NEGATIVE
        else:
            return Sign.POSITIVE
        
    def __neg__(self):
        return self * Sign.NEGATIVE
    
    # Boolean operators
    def __eq__(self,other):
        return self.sign_str == other.sign_str

    # To string methods.
    def __repr__(self):
        return "Sign('%s')" % self.sign_str
    
    def __str__(self):
        return self.sign_str

# Class constants for all sign types.
Sign.POSITIVE = Sign(Sign.POSITIVE_KEY)
Sign.NEGATIVE = Sign(Sign.NEGATIVE_KEY)
Sign.ZERO = Sign(Sign.ZERO_KEY)
Sign.UNKNOWN = Sign(Sign.UNKNOWN_KEY)