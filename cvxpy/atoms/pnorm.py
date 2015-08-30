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
from ..utilities.power_tools import pow_high, pow_mid, pow_neg, gm_constrs
from cvxpy.constraints.second_order import SOC
from fractions import Fraction


class pnorm(Atom):
    r"""The vector p-norm.

    If given a matrix variable, ``pnorm`` will treat it as a vector, and compute the p-norm
    of the concatenated columns.

    For :math:`p \geq 1`, the p-norm is given by

    .. math::

        \|x\|_p = \left(\sum_i |x_i|^p \right)^{1/p},

    with domain :math:`x \in \mathbf{R}^n`.

    For :math:`p < 1,\ p \neq 0`, the p-norm is given by

    .. math::

        \|x\|_p = \left(\sum_i x_i^p \right)^{1/p},

    with domain :math:`x \in \mathbf{R}^n_+`.

    - Note that the "p-norm" is actually a **norm** only when
      :math:`p \geq 1` or :math:`p = +\infty`. For these cases,
      it is convex.
    - The expression is not defined when :math:`p = 0`.
    - Otherwise, when :math:`p < 1`, the expression is
      concave, but it is not a true norm.

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


    Parameters
    ----------
    x : cvxpy.Variable
        The value to take the norm of.

    p : int, float, Fraction, or string
        If ``p`` is an ``int``, ``float``, or ``Fraction`` then we must have :math:`p \geq 1`.

        The only other valid inputs are ``numpy.inf``, ``float('inf')``, ``float('Inf')``, or
        the strings ``"inf"`` or ``"inf"``, all of which are equivalent and give the infinity norm.

    max_denom : int
        The maximum denominator considered in forming a rational approximation for ``p``.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """
    def __init__(self, x, p=2, max_denom=1024):
        p_old = p
        if p in ('inf', 'Inf', np.inf):
            self.p = np.inf
        elif p < 0:
            self.p, _ = pow_neg(p, max_denom)
        elif 0 < p < 1:
            self.p, _ = pow_mid(p, max_denom)
        elif p > 1:
            self.p, _ = pow_high(p, max_denom)
        elif p == 1:
            self.p = 1
        else:
            raise ValueError('Invalid p: {}'.format(p))

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

        if self.p < 1 and np.any(values < 0):
            return -np.inf

        if self.p < 0 and np.any(values == 0):
            return 0.0

        return np.linalg.norm(values, float(self.p))


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
        if self.p >= 1:
            return u.Curvature.CONVEX
        else:
            return u.Curvature.CONCAVE

    def monotonicity(self):
        """Increasing for positive arguments and decreasing for negative.
        """
        if self.p >= 1:
            return [u.monotonicity.SIGNED]
        else:
            return [u.monotonicity.INCREASING]


    def get_data(self):
        return [self.p]

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p)

    @staticmethod
    def graph_implementation(arg_objs, size, data=None):
        r"""Reduces the atom to an affine expression and list of constraints.

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

        Notes
        -----

        Implementation notes.

        - For general :math:`p \geq 1`, the inequality :math:`\|x\|_p \leq t`
          is equivalent to the following convex inequalities:

          .. math::

              |x_i| &\leq r_i^{1/p} t^{1 - 1/p}\\
              \sum_i r_i &= t.

          These inequalities happen to also be correct for :math:`p = +\infty`,
          if we interpret :math:`1/\infty` as :math:`0`.

        - For general :math:`0 < p < 1`, the inequality :math:`\|x\|_p \geq t`
          is equivalent to the following convex inequalities:

          .. math::

              r_i &\leq x_i^{p} t^{1 - p}\\
              \sum_i r_i &= t.

        - For general :math:`p < 0`, the inequality :math:`\|x\|_p \geq t`
          is equivalent to the following convex inequalities:

          .. math::

              t &\leq x_i^{-p/(1-p)} r_i^{1/(1 - p)}\\
              \sum_i r_i &= t.




        Although the inequalities above are correct, for a few special cases, we can represent the p-norm
        more efficiently and with fewer variables and inequalities.

        - For :math:`p = 1`, we use the representation

            .. math::

                x_i &\leq r_i\\
                -x_i &\leq r_i\\
                \sum_i r_i &= t

        - For :math:`p = \infty`, we use the representation

            .. math::

                x_i &\leq t\\
                -x_i &\leq t

          Note that we don't need the :math:`r` variable or the sum inequality.

        - For :math:`p = 2`, we use the natural second-order cone representation

            .. math::

                \|x\|_2 \leq t

          Note that we could have used the set of inequalities given above if we wanted an alternate decomposition
          of a large second-order cone into into several smaller inequalities.

        """
        p = data[0]
        x = arg_objs[0]
        t = lu.create_var((1, 1))
        constraints = []

        # first, take care of the special cases of p = 2, inf, and 1
        if p == 2:
            return t, [SOC(t, [x])]

        if p == np.inf:
            t_ = lu.promote(t, x.size)
            return t, [lu.create_leq(x, t_), lu.create_geq(lu.sum_expr([x, t_]))]

        # we need an absolute value constraint for the symmetric convex branches (p >= 1)
        # we alias |x| as x from this point forward to make the code pretty :)
        if p >= 1:
            absx = lu.create_var(x.size)
            constraints += [lu.create_leq(x, absx), lu.create_geq(lu.sum_expr([x, absx]))]
            x = absx

        if p == 1:
            return lu.sum_entries(x), constraints

        # now, we take care of the remaining convex and concave branches
        # to create the rational powers, we need a new variable, r, and
        # the constraint sum(r) == t
        r = lu.create_var(x.size)
        t_ = lu.promote(t, x.size)
        constraints += [lu.create_eq(lu.sum_entries(r), t)]

        # make p a fraction so that the input weight to gm_constrs
        # is a nice tuple of fractions.
        p = Fraction(p)
        if p < 0:
            constraints += gm_constrs(t_, [x, r],  (-p/(1-p), 1/(1-p)))
        if 0 < p < 1:
            constraints += gm_constrs(r,  [x, t_], (p, 1-p))
        if p > 1:
            constraints += gm_constrs(x,  [r, t_], (1/p, 1-1/p))

        return t, constraints

        # todo: no need to run gm_constr to form the tree each time. we only need to form the tree once

