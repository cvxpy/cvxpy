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

import cvxpy.lin_ops.lin_utils as lu
from cvxpy.atoms.elementwise.elementwise import Elementwise
import numpy as np
from cvxpy.utilities.power_tools import (is_power2, gm_constrs, pow_mid,
                                         pow_high, pow_neg)
import scipy.sparse as sp


class power(Elementwise):
    r""" Elementwise power function :math:`f(x) = x^p`.

    If ``expr`` is a CVXPY expression, then ``expr**p``
    is equivalent to ``power(expr, p)``.

    Specifically, the atom is given by the cases

    .. math::

        \begin{array}{ccl}
        p = 0 & f(x) = 1 & \text{constant, positive} \\
        p = 1 & f(x) = x & \text{affine, increasing, same sign as $x$} \\
        p = 2,4,8,\ldots &f(x) = |x|^p  & \text{convex, signed monotonicity, positive} \\
        p < 0 & f(x) = \begin{cases} x^p & x > 0 \\ +\infty & x \leq 0 \end{cases}
          & \text{convex, decreasing, positive} \\
        0 < p < 1 & f(x) = \begin{cases} x^p & x \geq 0 \\ -\infty & x < 0 \end{cases}
          & \text{concave, increasing, positive} \\
        p > 1,\ p \neq 2,4,8,\ldots & f(x) = \begin{cases} x^p & x \geq 0 \\
          +\infty & x < 0 \end{cases} & \text{convex, increasing, positive}.
        \end{array}

    .. note::

        Generally, ``p`` cannot be represented exactly, so a rational,
        i.e., fractional, **approximation** must be made.

        Internally, ``power`` computes a rational approximation
        to ``p`` with a denominator up to ``max_denom``. The resulting
        approximation can be found through the attribute ``power.p``.
        The approximation error is given by the attribute ``power.approx_error``.
        Increasing ``max_denom`` can give better approximations.

        When ``p`` is an ``int`` or ``Fraction`` object, the approximation
        is usually **exact**.

    .. note::

        The final domain, sign, monotonicity, and curvature of the ``power`` atom
        are determined by the rational approximation to ``p``, **not** the input parameter ``p``.

        For example,

        >>> from cvxpy import Variable, power
        >>> x = Variable()
        >>> g = power(x, 1.001)
        >>> g.p
        Fraction(1001, 1000)
        >>> g
        Expression(CONVEX, POSITIVE, (1, 1))

        results in a convex atom with implicit constraint :math:`x \geq 0`, while

        >>> g = power(x, 1.0001)
        >>> g.p
        1
        >>> g
        Expression(AFFINE, UNKNOWN, (1, 1))

        results in an affine atom with no constraint on ``x``.


    - When :math:`p > 1` and ``p`` is not a power of two, the monotonically increasing version
      of the function with full domain,

      .. math::

          f(x) = \begin{cases} x^p & x \geq 0 \\ 0 & x < 0 \end{cases}

      can be formed with the composition ``power(pos(x), p)``.

    - The symmetric version with full domain,

      .. math::

          f(x) = |x|^p

      can be formed with the composition ``power(abs(x), p)``.


    Parameters
    ----------

    x : cvx.Variable

    p : int, float, or Fraction
        Scalar power.

    max_denom : int
        The maximum denominator considered in forming a rational approximation of ``p``.



    """

    def __init__(self, x, p, max_denom=1024):
        p_old = p

        # how we convert p to a rational depends on the branch of the function
        if p > 1:
            p, w = pow_high(p, max_denom)
        elif 0 < p < 1:
            p, w = pow_mid(p, max_denom)
        elif p < 0:
            p, w = pow_neg(p, max_denom)

        # note: if, after making the rational approximation, p ends up being 0 or 1,
        # we default to using the 0 or 1 behavior of the atom,
        # which affects the curvature, domain, etc...
        # maybe unexpected behavior to the user if they put in 1.00001?

        if p == 1:
            # in case p is a fraction equivalent to 1
            p = 1
            w = None
        if p == 0:
            p = 0
            w = None

        self.p, self.w = p, w

        self.approx_error = float(abs(self.p - p_old))

        super(power, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        # Throw error if negative and power doesn't handle that.
        if self.p < 0 and values[0].min() <= 0:
            raise ValueError(
                "power(x, %.1f) cannot be applied to negative or zero values." % float(self.p)
            )
        elif not is_power2(self.p) and self.p != 0 and values[0].min() < 0:
            raise ValueError(
                "power(x, %.1f) cannot be applied to negative values." % float(self.p)
            )
        else:
            return np.power(values[0], float(self.p))

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.p == 1:
            # Same as input.
            return (self.args[0].is_positive(), self.args[0].is_negative())
        else:
            # Always positive.
            return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        # p == 0 is affine here.
        return self.p <= 0 or self.p >= 1

    def is_atom_concave(self):
        """Is the atom concave?
        """
        # p == 0 is affine here.
        return 0 <= self.p <= 1

    def is_constant(self):
        """Is the expression constant?
        """
        return self.p == 0 or super(power, self).is_constant()

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        if 0 <= self.p <= 1:
            return True
        elif self.p > 1:
            if is_power2(self.p):
                return self.args[idx].is_positive()
            else:
                return True
        else:
            return False

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        if self.p <= 0:
            return True
        elif self.p > 1:
            if is_power2(self.p):
                return self.args[idx].is_negative()
            else:
                return False
        else:
            return False

    def is_quadratic(self):
        if self.p == 0:
            return True
        elif self.p == 1:
            return self.args[0].is_quadratic()
        elif self.p == 2:
            return self.args[0].is_affine()
        else:
            return self.args[0].is_constant()

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size[0]*self.args[0].size[1]
        cols = self.size[0]*self.size[1]
        if self.p == 0:
            # All zeros.
            return [sp.csc_matrix((rows, cols), dtype='float64')]
        # Outside domain or on boundary.
        if not is_power2(self.p) and np.min(values[0]) <= 0:
            if self.p < 1:
                # Non-differentiable.
                return [None]
            else:
                # Round up to zero.
                values[0] = np.maximum(values[0], 0)

        grad_vals = float(self.p)*np.power(values[0], float(self.p)-1)
        return [power.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        if (self.p < 1 and not self.p == 0) or \
           (self.p > 1 and not is_power2(self.p)):
            return [self.args[0] >= 0]
        else:
            return []

    def validate_arguments(self):
        pass

    def get_data(self):
        return [self.p, self.w]

    def copy(self, args=None):
        """Returns a shallow copy of the power atom.

        Parameters
        ----------
        args : list, optional
            The arguments to reconstruct the atom. If args=None, use the
            current args of the atom.

        Returns
        -------
        power atom
        """
        if args is None:
            args = self.args
        # Avoid calling __init__() directly as we do not have p and max_denom.
        copy = type(self).__new__(type(self))
        # Emulate __init__()
        copy.p, copy.w = self.get_data()
        copy.approx_error = self.approx_error
        super(type(self), copy).__init__(*args)
        return copy

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        """Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        size : tuple
            The size of the resulting expression.
        data :
            Additional data required by the atom.

        Returns
        -------
        tuple
            (LinOp for objective, list of constraints)
        """
        x = arg_objs[0]
        p, w = data

        if p == 1:
            return x, []
        else:
            one = lu.create_const(np.mat(np.ones(size)), size)
            if p == 0:
                return one, []
            else:
                t = lu.create_var(size)

                if 0 < p < 1:
                    return t, gm_constrs(t, [x, one], w)
                elif p > 1:
                    return t, gm_constrs(x, [t, one], w)
                elif p < 0:
                    return t, gm_constrs(one, [x, t], w)
                else:
                    raise NotImplementedError('this power is not yet supported.')

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p)
