# from cvxpy.expressions.variable import Variable
# from cvxpy.problems.objective import *
# import unittest

# class TestObjectives(unittest.TestCase):
#     """ Unit tests for the expression/expression module. """
#     def setUp(self):
#         self.x = Variable(2, name='x')
#         self.y = Variable(3, name='y')
#         self.z = Variable(2, name='z')

#     # Test the Minimize class.
#     def test_minimize(self):
#         exp = self.x + self.z
#         obj = Minimize(exp)
#         self.assertEqual(obj.name(), "minimize %s" % exp.name())
#         self.assertEqual(obj.coefficients().keys(), exp.coefficients().keys())
#         self.assertEqual(obj.variables(), exp.variables())

#         with self.assertRaises(Exception) as cm:
#             obj.size()
#         self.assertEqual(str(cm.exception), 
#             "The objective 'minimize x + z' must resolve to a scalar.")


#     # Test the Maximize class.
#     def test_maximize(self):
#         exp = self.x + self.z
#         obj = Maximize(exp)
#         self.assertEqual(obj.name(), "maximize %s" % exp.name())
#         self.assertEqual(obj.coefficients().keys(), exp.coefficients().keys())
#         self.assertEqual(obj.variables(), exp.variables())

#         with self.assertRaises(Exception) as cm:
#             obj.size()
#         self.assertEqual(str(cm.exception), 
#             "The objective 'maximize x + z' must resolve to a scalar.")