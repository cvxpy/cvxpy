"""
Copyright 2017 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from cvxpy.expressions.variables import Variable
from cvxpy.constraints.second_order import SOC
from cvxpy.tests.base_test import BaseTest
import numpy as np
import sys
PY2 = sys.version_info < (3, 0)

class TestConstraints(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable(2, 2, name='A')
        self.B = Variable(2, 2, name='B')
        self.C = Variable(3, 2, name='C')

    def test_constr_str(self):
        """Test string representations of the constraints.
        """
        constr = self.x <= self.x
        self.assertEqual(repr(constr), "LeqConstraint(%s, %s)" % (repr(self.x), repr(self.x)))
        constr = self.x <= 2*self.x
        self.assertEqual(repr(constr), "LeqConstraint(%s, %s)" % (repr(self.x), repr(2*self.x)))
        constr = 2*self.x >= self.x
        self.assertEqual(repr(constr), "LeqConstraint(%s, %s)" % (repr(self.x), repr(2*self.x)))

    def test_eq_constraint(self):
        """Test the EqConstraint class.
        """
        constr = self.x == self.z
        self.assertEqual(constr.name(), "x == z")
        self.assertEqual(constr.size, (2, 1))
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.x.save_value([2,2])
        self.z.save_value([2,2])
        assert constr.value
        self.x.save_value([3,3])
        assert not constr.value

        self.x.value = [2, 1]
        self.z.value = [2, 2]
        assert not constr.value
        self.assertItemsAlmostEqual(constr.violation, [0, 1])
        self.assertItemsAlmostEqual(constr.residual.value, [0, 1])

        self.z.value = [2, 1]
        assert constr.value
        self.assertItemsAlmostEqual(constr.violation, [0, 0])
        self.assertItemsAlmostEqual(constr.residual.value, [0, 0])

        with self.assertRaises(Exception) as cm:
            (self.x == self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A, self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)
        self.assertTrue(copy.args[1] is self.B)

    def test_leq_constraint(self):
        """Test the LeqConstraint class.
        """
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2, 1))
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.x.save_value([1,1])
        self.z.save_value([2,2])
        assert constr.value
        self.x.save_value([3,3])
        assert not constr.value
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])

        self.x.value = [2, 1]
        self.z.value = [2, 0]
        assert not constr.value
        self.assertItemsAlmostEqual(constr.violation, [0, 1])
        self.assertItemsAlmostEqual(constr.residual.value, [0, 1])

        self.z.value = [2, 2]
        assert constr.value
        self.assertItemsAlmostEqual(constr.violation, [0, 0])
        self.assertItemsAlmostEqual(constr.residual.value, [0, 0])

        with self.assertRaises(Exception) as cm:
            (self.x <= self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A, self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)
        self.assertTrue(copy.args[1] is self.B)

    def test_psd_constraint(self):
        """Test the PSD constraint <<.
        """
        constr = self.A >> self.B
        self.assertEqual(constr.name(), "A >> B")
        self.assertEqual(constr.size, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.A.save_value(np.matrix("2 -1; 1 2"))
        self.B.save_value(np.matrix("1 0; 0 1"))
        assert constr.value
        self.assertAlmostEqual(constr.violation, 0)
        self.assertAlmostEqual(constr.residual.value, 0)

        self.B.save_value(np.matrix("3 0; 0 3"))
        assert not constr.value
        self.assertAlmostEqual(constr.violation, 1)
        self.assertAlmostEqual(constr.residual.value, 1)

        with self.assertRaises(Exception) as cm:
            (self.x >> self.y)
        self.assertEqual(str(cm.exception), "Non-square matrix in positive definite constraint.")

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.B, self.A])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.B)
        self.assertTrue(copy.args[1] is self.A)

    def test_nsd_constraint(self):
        """Test the PSD constraint <<.
        """
        constr = self.A << self.B
        self.assertEqual(constr.name(), "B >> A")
        self.assertEqual(constr.size, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        assert constr.value is None
        self.B.save_value(np.matrix("2 -1; 1 2"))
        self.A.save_value(np.matrix("1 0; 0 1"))
        assert constr.value
        self.A.save_value(np.matrix("3 0; 0 3"))
        assert not constr.value

        with self.assertRaises(Exception) as cm:
            (self.x << self.y)
        self.assertEqual(str(cm.exception), "Non-square matrix in positive definite constraint.")

    def test_lt(self):
        """Test the < operator.
        """
        constr = self.x < self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2, 1))

        with self.assertRaises(Exception) as cm:
            (self.x < self.y)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

    def test_geq(self):
        """Test the >= operator.
        """
        constr = self.z >= self.x
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2, 1))

        with self.assertRaises(Exception) as cm:
            (self.y >= self.x)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

    def test_gt(self):
        """Test the > operator.
        """
        constr = self.z > self.x
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.size, (2, 1))

        with self.assertRaises(Exception) as cm:
            (self.y > self.x)
        self.assertEqual(str(cm.exception), "Incompatible dimensions (2, 1) (3, 1)")

    # Test the SOC class.
    def test_soc_constraint(self):
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(scalar_exp, [exp])
        self.assertEqual(constr.size, (3, 1))

    def test_chained_constraints(self):
        """Tests that chaining constraints raises an error.
        """
        error_str = ("Cannot evaluate the truth value of a constraint or "
                     "chain constraints, e.g., 1 >= x >= 0.")
        with self.assertRaises(Exception) as cm:
            (self.z <= self.x <= 1)
        self.assertEqual(str(cm.exception), error_str)

        with self.assertRaises(Exception) as cm:
            (self.x == self.z == 1)
        self.assertEqual(str(cm.exception), error_str)

        if PY2:
            with self.assertRaises(Exception) as cm:
                (self.z <= self.x).__nonzero__()
            self.assertEqual(str(cm.exception), error_str)
        else:
            with self.assertRaises(Exception) as cm:
                (self.z <= self.x).__bool__()
            self.assertEqual(str(cm.exception), error_str)
