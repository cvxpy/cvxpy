import settings as s
import interface.matrices as intf
from interface.shapeset import ShapeSet

class Coeff(object):
    """
    Utility class for building dictionaries of coefficients in an expression.
    """
    # Is the given coeff_dict that of a constant expression?
    @staticmethod 
    def is_constant(coeff_dict):
        return s.CONSTANT in coeff_dict and len(coeff_dict) == 1

    # Returns a new dict with all coefficients conformed to the given shapes.
    @staticmethod
    def conform_coeffs_to_shapes(coeff_dict, shapes):
        return dict( 
            (k,intf.conform_to_shapes(v, shapes)) for (k,v) in coeff_dict.items() 
        )

    # Evaluates the left hand and right hand expressions,
    # finds the possible shapes of the sum, converts all coefficients
    # to that shape and sums the dicts.
    # Returns (coefficient dict, set(possible shapes)).
    @staticmethod
    def add(lh_exp, rh_exp):
        (lh_coeff, lh_shapes) = lh_exp.coefficients()
        (rh_coeff, rh_shapes) = rh_exp.coefficients()
        shapes = lh_shapes + rh_shapes
        lh_coeff = Coeff.conform_coeffs_to_shapes(lh_coeff, shapes)
        rh_coeff = Coeff.conform_coeffs_to_shapes(rh_coeff, shapes)
        # got this nice piece of code off stackoverflow http://stackoverflow.com/questions/1031199/adding-dictionaries-in-python
        coeffs = dict( 
            (n, lh_coeff.get(n, 0) + rh_coeff.get(n, 0)) for n in set(lh_coeff)|set(rh_coeff) 
        )
        return (coeffs, shapes)

    # Evaluates the left hand and right hand expressions,
    # checks the left hand expression is constant,
    # finds the possible shapes of the left hand expression and converts it,
    # then multiplies all coefficients by the left hand value.
    # Returns (coefficient dict, set(possible shapes)).
    @staticmethod
    def mul(lh_exp, rh_exp):
        (lh_coeff, lh_shapes) = lh_exp.coefficients()
        if not Coeff.is_constant(lh_coeff): 
            raise Exception("Cannot multiply on the left by a non-constant.")
        (rh_coeff,rh_shapes) = rh_exp.coefficients()
        lh_shapes,result_shapes,rh_shapes = lh_shapes * rh_shapes
        lh_coeff = Coeff.conform_coeffs_to_shapes(lh_coeff, shapes)
        rh_coeff = Coeff.conform_coeffs_to_shapes(rh_coeff, shapes)
        coeffs = dict((k,lh_coeff[s.CONSTANT]*v) for v in rh_coeff.items())
        return (coeffs, result_shapes)