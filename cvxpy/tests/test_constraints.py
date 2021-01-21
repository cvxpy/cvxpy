"""
Copyright 2013 Steven Diamond

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

from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.reshape import reshape as reshape_atom
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.power import PowCone3D, PowConeND
from cvxpy.tests.base_test import BaseTest
import numpy as np


class TestConstraints(BaseTest):
    """ Unit tests for the expression/expression module. """

    def setUp(self):
        self.a = Variable(name='a')
        self.b = Variable(name='b')

        self.x = Variable(2, name='x')
        self.y = Variable(3, name='y')
        self.z = Variable(2, name='z')

        self.A = Variable((2, 2), name='A')
        self.B = Variable((2, 2), name='B')
        self.C = Variable((3, 2), name='C')

    def test_equality(self):
        """Test the Equality class.
        """
        constr = self.x == self.z
        self.assertEqual(constr.name(), "x == z")
        self.assertEqual(constr.shape, (2,))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.x.save_value(2)
        self.z.save_value(2)
        assert constr.value()
        self.x.save_value(3)
        assert not constr.value()

        self.x.value = np.array([2, 1])
        self.z.value = np.array([2, 2])
        assert not constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 1])
        self.assertItemsAlmostEqual(constr.residual, [0, 1])

        self.z.value = np.array([2, 1])
        assert constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 0])
        self.assertItemsAlmostEqual(constr.residual, [0, 0])

        # Incompatible dimensions
        with self.assertRaises(ValueError):
            (self.x == self.y)

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)

    def test_inequality(self):
        """Test the Inequality class.
        """
        constr = self.x <= self.z
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.shape, (2,))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.x.save_value(1)
        self.z.save_value(2)
        assert constr.value()
        self.x.save_value(3)
        assert not constr.value()
        # self.assertItemsEqual(constr.variables().keys(), [self.x.id, self.z.id])

        self.x.value = np.array([2, 1])
        self.z.value = np.array([2, 0])
        assert not constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 1])
        self.assertItemsAlmostEqual(constr.residual, [0, 1])

        self.z.value = np.array([2, 2])
        assert constr.value()
        self.assertItemsAlmostEqual(constr.violation(), [0, 0])
        self.assertItemsAlmostEqual(constr.residual, [0, 0])

        # Incompatible dimensions
        with self.assertRaises(Exception):
            (self.x <= self.y)

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.A])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.A)

    def test_psd_constraint(self):
        """Test the PSD constraint <<.
        """
        constr = self.A >> self.B
        self.assertEqual(constr.name(), "A + -B >> 0")
        self.assertEqual(constr.shape, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.A.save_value(np.array([[2, -1], [1, 2]]))
        self.B.save_value(np.array([[1, 0], [0, 1]]))
        assert constr.value()
        self.assertAlmostEqual(constr.violation(), 0)
        self.assertAlmostEqual(constr.residual, 0)

        self.B.save_value(np.array([[3, 0], [0, 3]]))
        assert not constr.value()
        self.assertAlmostEqual(constr.violation(), 1)
        self.assertAlmostEqual(constr.residual, 1)

        with self.assertRaises(Exception) as cm:
            (self.x >> 0)
        self.assertEqual(str(cm.exception), "Non-square matrix in positive definite constraint.")

        # Test copy with args=None
        copy = constr.copy()
        self.assertTrue(type(copy) is type(constr))
        # A new object is constructed, so copy.args == constr.args but copy.args
        # is not constr.args.
        self.assertEqual(copy.args, constr.args)
        self.assertFalse(copy.args is constr.args)
        # Test copy with new args
        copy = constr.copy(args=[self.B])
        self.assertTrue(type(copy) is type(constr))
        self.assertTrue(copy.args[0] is self.B)

    def test_nsd_constraint(self):
        """Test the PSD constraint <<.
        """
        constr = self.A << self.B
        self.assertEqual(constr.name(), "B + -A >> 0")
        self.assertEqual(constr.shape, (2, 2))
        # Test value and dual_value.
        assert constr.dual_value is None
        with self.assertRaises(ValueError):
            constr.value()
        self.B.save_value(np.array([[2, -1], [1, 2]]))
        self.A.save_value(np.array([[1, 0], [0, 1]]))
        assert constr.value()
        self.A.save_value(np.array([[3, 0], [0, 3]]))
        assert not constr.value()

        with self.assertRaises(Exception) as cm:
            self.x << 0
        self.assertEqual(str(cm.exception),
                         "Non-square matrix in positive definite constraint.")

    def test_geq(self):
        """Test the >= operator.
        """
        constr = self.z >= self.x
        self.assertEqual(constr.name(), "x <= z")
        self.assertEqual(constr.shape, (2,))

        # Incompatible dimensions
        with self.assertRaises(ValueError):
            (self.y >= self.x)

    # Test the SOC class.
    def test_soc_constraint(self):
        exp = self.x + self.z
        scalar_exp = self.a + self.b
        constr = SOC(scalar_exp, exp)
        self.assertEqual(constr.size, 3)

        # Test invalid dimensions.
        error_str = ("Argument dimensions (1,) and (1, 4), with axis=0, "
                     "are incompatible.")
        with self.assertRaises(Exception) as cm:
            SOC(Variable(1), Variable((1, 4)))
        self.assertEqual(str(cm.exception), error_str)

    def test_pow3d_constraint(self):
        n = 3
        np.random.seed(0)
        alpha = 0.275
        x, y, z = Variable(n), Variable(n), Variable(n)
        con = PowCone3D(x, y, z, alpha)
        # check violation against feasible values
        x0, y0 = 0.1 + np.random.rand(n), 0.1 + np.random.rand(n)
        z0 = x0**alpha * y0**(1-alpha)
        z0[1] *= -1
        x.value, y.value, z.value = x0, y0, z0
        viol = con.residual()
        self.assertLessEqual(viol, 1e-7)
        # check violation against infeasible values
        x1 = x0.copy()
        x1[0] *= -0.9
        x.value = x1
        viol = con.residual()
        self.assertGreaterEqual(viol, 0.99*abs(x1[0]))
        # check invalid constraint data
        with self.assertRaises(ValueError):
            con = PowCone3D(x, y, z, 1.001)
        with self.assertRaises(ValueError):
            con = PowCone3D(x, y, z, -0.00001)

    def test_pownd_constraint(self):
        n = 4
        W, z = Variable(n), Variable()
        np.random.seed(0)
        alpha = 0.5 + np.random.rand(n)
        alpha /= np.sum(alpha)
        with self.assertRaises(ValueError):
            # entries don't sum to one
            con = PowConeND(W, z, alpha+0.01)
        with self.assertRaises(ValueError):
            # shapes don't match exactly
            con = PowConeND(W, z, alpha.reshape((n, 1)))
        with self.assertRaises(ValueError):
            # wrong axis
            con = PowConeND(reshape_atom(W, (n, 1)), z,
                            alpha.reshape((n, 1)),
                            axis=1)
        # Compute a violation
        con = PowConeND(W, z, alpha)
        W0 = 0.1 + np.random.rand(n)
        z0 = np.prod(np.power(W0, alpha))+0.05
        W.value, z.value = W0, z0
        viol = con.violation()
        self.assertGreaterEqual(viol, 0.01)
        self.assertLessEqual(viol, 0.06)

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

        with self.assertRaises(Exception) as cm:
            (self.z <= self.x).__bool__()
        self.assertEqual(str(cm.exception), error_str)
