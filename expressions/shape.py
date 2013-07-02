class Shape(object):
    """
    The dimensions of a term in an expression.
    """
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

    def __repr__(self):
        return "Shape(%d,%d)" % (self.rows,self.cols)

    # Is the object a scalar?
    def is_scalar(self):
        return self.rows == 1 and self.cols == 1

    # Is the object a vector?
    def is_vector(self):
        return self.rows == 1 or self.cols == 1

    def __add__(self, other, name):
        if self.rows == other.rows and self.cols == other.cols:
            return self
        else:
            raise Exception("Invalid dimensions for " + name)

    def __sub__(self, other, name):
        return self.__add__(other, name)

    def __mul__(self, other, name):
        # Scalar multiplication
        if self.is_scalar(): return other
        if other.is_scalar(): return self
        # Matrix/vector multiplication
        # Vectors are columns by default, but can be treated as rows if
        # necessary for matrix multiplication.
        if self.cols == other.rows:
            return Shape(self.rows, other.cols)
        elif self.is_vector() and self.rows == other.rows:
            return Shape(1, other.cols)
        else:
            raise Exception("Invalid dimensions for " + name)

    def __neg__(self):
        return self