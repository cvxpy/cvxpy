# from cvxpy.expressions.leaf import Leaf
# from cvxpy.expressions.variable import Variable
# from cvxpy.expressions.constant import Constant
# from cvxpy.expressions.parameter import Parameter
# import cvxpy.interface.matrix_utilities as intf
# import cvxpy.settings as s
# from collections import deque
# import unittest

# class TestLeaf(unittest.TestCase):
#     """ Unit tests for the expressions.leaf module. """
#     def setUp(self):
#         self.x = Variable(2, name='x')
#         self.y = Variable(3, name='y')

#         self.A = Parameter(2,2,'A')
#         self.B = Parameter(2,2,'B')
#         self.C = Parameter(3,2,'C')
#         self.intf = intf.DEFAULT_INTERFACE()

#     # Overriden method to handle lists and lower accuracy.
#     def assertAlmostEqual(self, a, b, interface=intf.DEFAULT_INTERFACE):
#         try:
#             a = list(a)
#             b = list(b)
#             for i in range(len(a)):
#                 self.assertAlmostEqual(a[i], b[i])
#         except Exception:
#             super(TestLeaf, self).assertAlmostEqual(a,b,places=4)

#     # Test the dequeue_mults function with simple arguments.
#     def test_basic_dequeue_mults(self):
#         # Lone Variable
#         v,mult = self.x.terms()[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         self.assertEqual(constr, [])
#         self.assertEqual(coeffs.keys(), [v.id])
#         self.assertAlmostEqual(coeffs[v.id], v.coefficient(self.intf))
#         # Lone Constant
#         v,mult = Constant(2).terms()[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         self.assertEqual(constr, [])
#         self.assertEqual(coeffs.keys(), [v.id])
#         self.assertAlmostEqual(coeffs[v.id], v.coefficient(self.intf))
#         # Lone Parameter
#         v,mult = self.A.terms()[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         self.assertEqual(len(constr), 1)

#         var_id = coeffs.keys()[0]
#         self.assertItemsEqual(constr[0].coefficients.keys(), 
#                              [self.A.id, var_id])
#         self.assertAlmostEqual(coeffs[var_id], 
#                                Variable(2,2).coefficient(self.intf))

#     # Test the dequeue_mults function with complex arguments.
#     def test_complex_dequeue_mults(self):
#         # Only constants and variables.
#         exp = Constant(4)*Constant(3)*self.x
#         terms = exp.terms()
#         self.assertEqual(len(terms), 1)
#         v,mult = terms[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         self.assertItemsEqual(coeffs.keys(), [self.x.id])
#         self.assertAlmostEqual(coeffs[self.x.id], 12*self.x.coefficient(self.intf))

#         # Parameter * Constant
#         c = Constant([4,2])
#         exp = self.A * c
#         terms = exp.terms()
#         v,mult = terms[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         var_id = coeffs.keys()[0]
#         assert var_id in constr[1].coefficients.keys()
#         var2_id = [id for id in constr[1].coefficients.keys() if id != var_id][0]
#         self.assertAlmostEqual(coeffs[var_id], 
#                                Variable(2,2).coefficient(self.intf))
#         self.assertAlmostEqual(constr[1].coefficients[var_id], 
#                                -Variable(2,2).coefficient(self.intf))
#         self.assertEqual(constr[1].coefficients[var2_id], 
#                          self.A.coefficient(self.intf))
#         self.assertItemsEqual(constr[0].coefficients.keys(), [var2_id, c.id])
#         self.assertAlmostEqual(constr[0].coefficients[var2_id], 
#                                -Variable(2).coefficient(self.intf))
#         self.assertAlmostEqual(constr[0].coefficients[c.id], 
#                                c.coefficient(self.intf))

#         # Constant * Parameter
#         T = Constant([[4,2],[7,5]])
#         exp = T * self.A
#         terms = exp.terms()
#         v,mult = terms[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         var_id = coeffs.keys()[0]
#         self.assertAlmostEqual(coeffs[var_id], 
#                                T.coefficient(self.intf))
#         self.assertItemsEqual(constr[0].coefficients.keys(), [self.A.id, var_id])
#         self.assertAlmostEqual(constr[0].coefficients[var_id], 
#                                -Variable(2,2).coefficient(self.intf))
#         self.assertEqual(constr[0].coefficients[self.A.id], self.A)

#         # Parameter * Variable
#         exp = self.A * self.x
#         terms = exp.terms()
#         v,mult = terms[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         var_id = coeffs.keys()[0]
#         self.assertAlmostEqual(coeffs[var_id], 
#                                Variable(2).coefficient(self.intf))
#         self.assertItemsEqual(constr[0].coefficients.keys(), [self.x.id, var_id])
#         self.assertAlmostEqual(constr[0].coefficients[var_id], 
#                                -Variable(2).coefficient(self.intf))
#         self.assertEqual(constr[0].coefficients[self.x.id], self.A)

#         # Parameter * Constant * Variable
#         T = Constant([[4,2],[7,5]])
#         exp = self.A * T * self.x
#         terms = exp.terms()
#         v,mult = terms[0]
#         coeffs,constr = Leaf.dequeue_mults(mult, self.intf)
#         self.assertEqual(len(constr), 2)
#         var_id = coeffs.keys()[0]
#         self.assertAlmostEqual(coeffs[var_id], 
#                                Variable(2).coefficient(self.intf))
#         assert var_id in constr[1].coefficients.keys()
#         var2_id = [id for id in constr[1].coefficients.keys() if id != var_id][0]
#         self.assertAlmostEqual(constr[1].coefficients[var_id], 
#                                -Variable(2,2).coefficient(self.intf))
#         self.assertEqual(constr[1].coefficients[var2_id], 
#                          self.A)
#         self.assertItemsEqual(constr[0].coefficients.keys(), [var2_id, self.x.id])
#         self.assertAlmostEqual(constr[0].coefficients[var2_id], 
#                                -Variable(2).coefficient(self.intf))
#         self.assertAlmostEqual(constr[0].coefficients[self.x.id], 
#                                T.coefficient(self.intf))