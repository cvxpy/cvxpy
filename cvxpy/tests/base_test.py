# Base class for unit tests.
import unittest

class BaseTest(unittest.TestCase):
    # AssertAlmostEqual for lists.
    def assertItemsAlmostEqual(self, a, b):
        a = list(a)
        b = list(b)
        for i in range(len(a)):
            self.assertAlmostEqual(a[i], b[i])

    # Overriden method to assume lower accuracy.
    def assertAlmostEqual(self, a, b, places=3):
        super(BaseTest, self).assertAlmostEqual(a,b,places=places)