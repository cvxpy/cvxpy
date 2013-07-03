class ShapeSet(object):
    """
    A set of possible shapes associated with an expression.
    Allows conversions such as 5 + [1,1] to [6,6].
    """
    # Matches any height/width
    ANY_DIM = -1

    # shapes - set of possible shapes for the expression.
    # lh_mul_shapes - possible shapes that could multiply the expression on the left.
    # rh_mul_shapes - possible shapes that could multiply the expression on the right.
    def __init__(self, shapes, lh_mul_shapes, rh_mul_shapes):
        self.shapes = shapes
        self.lh_mul_shapes = lh_mul_shapes
        self.rh_mul_shapes = rh_mul_shapes

    # If the shapes are equivalent, return the more specific one.
    @staticmethod
    def match(lh_shape, rh_shape):
        heights = [lh_shape[0], rh_shape[0]]
        heights.sort()
        widths = [lh_shape[1], rh_shape[1]]
        widths.sort()
        if (heights[0] == ShapeSet.ANY_DIM or heights[0] == heights[1]) and \
           (widths[0] == ShapeSet.ANY_DIM or widths[0] == widths[1]):
           return (heights[1], widths[1])

    # Returns the set of shapes conformant to both of the two given sets of shapes.
    @staticmethod
    def intersection(lh_shapes, rh_shapes):
        intersection = set()
        for lh_shape in lh_shapes:
            for rh_shape in rh_shapes:
                match = ShapeSet.match(lh_shape, rh_shape)
                if match: intersection.add(match)
        if len(intersection) == 0: raise Exception("Incompatible dimensions.")
        return intersection

    # Returns the possible shapes of the sum.
    def __add__(self, other):
        return ShapeSet.intersection(self.shapes, other.shapes)

    # Returns the possible shapes of the left hand multiplier,
    # the product, and the left hand multiplier.
    def __mul__(self, other):
        return (ShapeSet.intersection(self.shapes, other.lh_mul_shapes),
                ShapeSet.intersection(self.rh_mul_shapes, other.shapes),
               )