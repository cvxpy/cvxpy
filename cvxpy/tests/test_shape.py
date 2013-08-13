from cvxpy.expressions.variable import Variable
from cvxpy.utilities import Shape
import unittest

class TestShape(unittest.TestCase):
    """ Unit tests for the expressions/shape module. """
    def setUp(self):
        pass

    # Test the size method.
    def test_size(self):
        self.assertEqual(Shape(1,3).size, (1,3))
        self.assertEqual(Shape(2,1).size, (2,1))

    # Test adding two shapes.
    def test_add(self):
        self.assertEqual((Shape(3,4) + Shape(3,4)).size, (3,4))
        
        with self.assertRaises(Exception) as cm:
            (Shape(1,3) + Shape(4,3))
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

        # Promotion
        self.assertEqual((Shape(3,4) + Shape(1,1)).size, (3,4))
        self.assertEqual((Shape(1,1) + Shape(3,4)).size, (3,4))

    # Test multiplying two shapes.
    def test_mul(self):
        self.assertEqual((Shape(5,9) * Shape(9,2)).size, (5,2))
        
        with self.assertRaises(Exception) as cm:
            (Shape(5,3) * Shape(9,2))
        self.assertEqual(str(cm.exception), "Incompatible dimensions.")

        # Promotion
        self.assertEqual((Shape(3,4) * Shape(1,1)).size, (3,4))
        self.assertEqual((Shape(1,1) * Shape(3,4)).size, (3,4))