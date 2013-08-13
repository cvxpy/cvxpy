import cvxpy.utilities as u
import cvxpy.interface.matrix_utilities as intf

class AffVstack(u.Affine):
    """ Vertical concatenation of Affine Objectives. """
    def __init__(self, *args):
        self.args = [self.cast_as_affine(arg) for arg in args]
        cols = self.args[0].size[1]
        rows = sum(arg.size[0] for arg in self.args)
        self._shape = u.Shape(rows, cols)
        self._vars = []
        map(self._vars.extend, (arg.variables() for arg in self.args))
        super(AffVstack, self).__init__()

    def variables(self):
        return self._vars

    # Places the coefficients of all the blocks
    # as blocks in zero matrices.
    def coefficients(self, interface):
        coeffs = {}
        offset = 0
        for arg in self.args:
            arg_coeffs = arg.coefficients(interface)
            for k,v in arg_coeffs.items():
                zeros = interface.zeros(*self._shape.size)
                rows,cols = intf.size(v)
                interface.block_copy(zeros, v, offset, 0, rows, cols)
                if k in coeffs:
                    coeffs[k] = coeffs[k] + zeros
                else:
                    coeffs[k] = zeros
            offset += arg.size[0]
        return coeffs