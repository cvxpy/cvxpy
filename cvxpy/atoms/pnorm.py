"""
Copyright 2013 Steven Diamond

This file is part of CVXPY.

CVXPY is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CVXPY is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CVXPY.  If not, see <http://www.gnu.org/licenses/>.
"""

from cvxpy.atoms.atom import Atom
import cvxpy.utilities as u
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
from ..utilities.power_tools import sanitize_scalar, pow_high, gm_constrs
from cvxpy.constraints.second_order import SOC

# todo: OK, we've got a couple of (vector and matrix) norms here. maybe we dispatch to a pnorm (vector) atom

# todo: make sure (along with power and geo_mean) that we don't make extra variables and constraints if we don't need
# to, in the case where we have trivial powers like (0, 1, 0)


class pnorm(Atom):
    r"""The p-norm given by

    .. math::

        \left(\sum_i |x_i|^p \right)^{1/p} \leq t,

    where :math:`p \geq 1`. (Including :math:`p = +\infty`.)

    .. note::

        Generally, ``p`` cannot be represented exactly, so a rational,
        i.e., fractional, **approximation** must be made.

        Internally, ``pnorm`` computes a rational approximation
        to the reciprocal :math:`1/p` with a denominator up to ``max_denom``.
        The resulting
        approximation can be found through the attribute ``pnorm.p``.
        The approximation error is given by the attribute ``pnorm.approx_error``.
        Increasing ``max_denom`` can give better approximations.

        When ``p`` is an ``int`` or ``Fraction`` object, the approximation
        is usually **exact**.


    Notes
    -----

    For general ``p``, the p-norm is equivalent to the following convex inequalities:

    .. math::

        |x_i| &\leq s_i^{1/p} t^{1 - 1/p}\\
        \sum_i s_i &\leq t,

    where :math:`p \geq 1`.

    These inequalities are also correct for :math:`p = +\infty` if we interpret :math:`1/\infty` as :math:`0`.


    Although the inequalities above are correct, for a few special cases, we can represent the p-norm
    more efficiently and with fewer variables and inequalities.

    - For :math:`p = 1`, we use the representation

        .. math::

            |x_i| &\leq s_i\\
            \sum_i s_i &\leq t

    - For :math:`p = \infty`, we use the representation

        .. math::

            |x_i| &\leq t

      Note that we don't need the :math:`s` variables or the sum inequality.

    - For :math:`p = 2`, we use the natural second-order cone representation

        .. math::

            \|x\|_2 \leq t

      Note that we could have used the set of inequalities given above if we wanted an alternate decomposition
      of a large second-order cone into into several smaller inequalities.


    Parameters
    ----------
    x : cvxpy.Variable
        The value to take the norm of.

    p : int, float, Fraction, or string
        If ``p`` is an ``int``, ``float``, or ``Fraction`` then we must have :math:`p \geq 1`.

        The only other valid inputs are ``numpy.inf``, ``float('inf')``, ``float('Inf')``, or
        the strings ``"inf"`` or ``"inf"``, all of which are equivalent and give the infinity norm.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    def __init__(self, x, p=2, max_denom=1024):
        p_old = p
        if p in ('inf', 'Inf', np.inf):
            p, w = np.inf, None
        elif p < 1:
            raise ValueError("Norm must have p >= 1. {} is an invalid input.".format(p))
        elif p > 1:
            p = sanitize_scalar(p)
            p, w = pow_high(p, max_denom)
        elif p == 1:
            p, w = 1, None

        if p == 1:
            self.p, self.w = 1, None
        if p == 2:
            self.p, self.w = 2, None
        else:
            self.p, self.w = p, w

        super(pnorm, self).__init__(x)

        if self.p == np.inf:
            self.approx_error = 0
        else:
            self.approx_error = float(abs(self.p - p_old))


    @Atom.numpy_numeric
    def numeric(self, values):
        """Returns the p-norm of x.
        """
        values = np.array(values[0]).flatten()
        return np.linalg.norm(values, self.p)


    def shape_from_args(self):
        """Resolves to a scalar.
        """
        return u.Shape(1, 1)

    def sign_from_args(self):
        """Always positive.
        """
        return u.Sign.POSITIVE

    def func_curvature(self):
        """Default curvature is convex.
        """
        return u.Curvature.CONVEX

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        return [u.monotonicity.SIGNED]

    def get_data(self):
        return self.p, self.w

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p)

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
        p, w = data
        x = arg_objs[0]
        t = lu.create_var((1, 1))

        # todo: clean up this mess of conditionals
        if p == 2:
            return t, [SOC(t, [x])]
        elif p == np.inf:
            promoted_t = lu.promote(t, x.size)
            constraints = [lu.create_geq(lu.sum_expr([x, promoted_t])),
                           lu.create_leq(x, promoted_t)]
            return t, constraints
        else:
            r = lu.create_var(x.size)
            constraints = [lu.create_geq(lu.sum_expr([x, r])),
                           lu.create_leq(x, r)]
            if p == 1:
                # todo: can gm_constr handle this elegantly?
                return lu.sum_entries(r), constraints
            else:
                promoted_t = lu.promote(t, x.size)
                s = lu.create_var(x.size)
                # todo: no need to run gm_constr to form the tree each time. we only need to form the tree once
                constraints += gm_constrs(r, [s, promoted_t], w)
                constraints += [lu.create_leq(lu.sum_entries(s), t)]
                return t, constraints
