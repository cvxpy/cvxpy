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

from cvxpy.atoms.axis_atom import AxisAtom
from cvxpy.atoms.norm1 import norm1
from cvxpy.atoms.norm_inf import norm_inf
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import scipy.sparse as sp
from cvxpy.utilities.power_tools import pow_high, pow_mid, pow_neg, gm_constrs
from cvxpy.constraints.second_order import SOC
from fractions import Fraction


def pnorm(x, p=2, axis=None, keepdims=False, max_denom=1024):
    """Factory function for a mathematical p-norm.

    TODO(akshayka): Documentation.
    """
    if p == 1:
        return norm1(x, axis=axis, keepdims=keepdims)
    elif p in [np.inf, 'inf', 'Inf']:
        return norm_inf(x, axis=axis, keepdims=keepdims)
    else:
        return Pnorm(x, p=p, axis=axis, keepdims=keepdims, max_denom=max_denom)


class Pnorm(AxisAtom):
    r"""The vector p-norm, for p not equal to 1 or infinity.

    If given a matrix variable, ``pnorm`` will treat it as a vector, and
    compute the p-norm of the concatenated columns. Only accepts p values
    that are not equal to 1 or infinity; the norm1 and norm_inf classes
    handle those norms.

    For :math:`p > 1`, the p-norm is given by

    .. math::

        \|x\|_p = \left(\sum_i |x_i|^p \right)^{1/p},

    with domain :math:`x \in \mathbf{R}^n`.

    For :math:`p < 1,\ p \neq 0`, the p-norm is given by

    .. math::

        \|x\|_p = \left(\sum_i x_i^p \right)^{1/p},

    with domain :math:`x \in \mathbf{R}^n_+`.

    - Note that the "p-norm" is actually a **norm** only when
      :math:`p > 1`. For these cases, it is convex.
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

    p : int, float, or Fraction
        We require that :math:`p > 1`, but :math:`p \neq \infty`. See the
        norm1 and norm_inf classes for these norms, or use the pnorm
        function wrapper to instantiate them.


    max_denom : int
        The maximum denominator considered in forming a rational approximation
        for ``p``.

    axis : 0 or 1
           The axis to apply the norm to.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """

    def __init__(self, x, p=2, axis=None, keepdims=False, max_denom=1024):
        if p < 0:
            # TODO(akshayka): Why do we accept p < 0?
            self.p, _ = pow_neg(p, max_denom)
        elif 0 < p < 1:
            self.p, _ = pow_mid(p, max_denom)
        elif p > 1:
            self.p, _ = pow_high(p, max_denom)
        elif p == 1:
            raise ValueError('Use the norm1 class to instantiate a one norm.')
        elif p == 'inf' or p == 'Inf' or p == np.inf:
            raise ValueError('Use the norm_inf class to instantiate an '
                             'infinity norm.')
        else:
            raise ValueError('Invalid p: {}'.format(p))
        self.approx_error = float(abs(self.p - p))
        super(Pnorm, self).__init__(x, axis=axis, keepdims=keepdims)

    def numeric(self, values):
        """Returns the p-norm of x.
        """

        if self.axis is None:
            values = np.array(values[0]).flatten()
        else:
            values = np.array(values[0])

        if self.p < 1 and np.any(values < 0):
            return -np.inf
        if self.p < 0 and np.any(values == 0):
            return 0.0

        return np.linalg.norm(values, float(self.p), axis=self.axis,
                              keepdims=self.keepdims)

    def validate_arguments(self):
        super(Pnorm, self).validate_arguments()
        # TODO(akshayka): Why is axis not supported for other norms?
        if self.axis is not None and self.p != 2:
            raise ValueError(
                "The axis parameter is only supported for p=2.")

    def sign_from_args(self):
        """Returns sign (is positive, is negative) of the expression.
        """
        # Always positive.
        return (True, False)

    def is_atom_convex(self):
        """Is the atom convex?
        """
        return self.p > 1

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return self.p < 1

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.p < 1 or (self.p > 1 and self.args[0].is_nonneg())

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.p > 1 and self.args[0].is_nonpos()

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return False

    def get_data(self):
        return [self.p, self.axis]

    def name(self):
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p)

    def _domain(self):
        """Returns constraints describing the domain of the node.
        """
        if self.p < 1 and self.p != 0:
            return [self.args[0] >= 0]
        else:
            return []

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        return self._axis_grad(values)

    def _column_grad(self, value):
        """Gives the (sub/super)gradient of the atom w.r.t. a column argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            value: A numeric value for a column.

        Returns:
            A NumPy ndarray matrix or None.
        """
        rows = self.args[0].size
        value = np.matrix(value)
        # Outside domain.
        if self.p < 1 and np.any(value <= 0):
            return None
        D_null = sp.csc_matrix((rows, 1), dtype='float64')
        denominator = np.linalg.norm(value, float(self.p))
        denominator = np.power(denominator, self.p - 1)
        # Subgrad is 0 when denom is 0 (or undefined).
        if denominator == 0:
            if self.p > 1:
                return D_null
            else:
                return None
        else:
            nominator = np.power(value, self.p - 1)
            frac = np.divide(nominator, denominator)
            return np.reshape(frac.A, (frac.size, 1))

    @staticmethod
    def graph_implementation(arg_objs, shape, data=None):
        r"""Reduces the atom to an affine expression and list of constraints.

        Parameters
        ----------
        arg_objs : list
            LinExpr for each argument.
        shape : tuple
            The shape of the resulting expression.
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




        Although the inequalities above are correct, for a few special cases,
        we can represent the p-norm more efficiently and with fewer variables and inequalities.

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

          Note that we could have used the set of inequalities given above if we wanted
          an alternate decomposition of a large second-order cone into into several
          smaller inequalities.

        """
        p = data[0]
        axis = data[1]
        x = arg_objs[0]
        t = lu.create_var((1, 1))
        constraints = []

        # first, take care of the special cases of p = 2, inf, and 1
        if p == 2:
            if axis is None:
                return t, [SOC(t, x)]

            else:
                t = lu.create_var(shape)
                return t, [SOC(lu.reshape(t, (t.shape[0]*t.shape[1], 1)),
                               x, axis)]

        if p == np.inf:
            t_ = lu.promote(t, x.shape)
            return t, [lu.create_leq(x, t_), lu.create_geq(lu.sum_expr([x, t_]))]

        # we need an absolute value constraint for the symmetric convex branches (p >= 1)
        # we alias |x| as x from this point forward to make the code pretty :)
        if p >= 1:
            absx = lu.create_var(x.shape)
            constraints += [lu.create_leq(x, absx), lu.create_geq(lu.sum_expr([x, absx]))]
            x = absx

        if p == 1:
            return lu.sum(x), constraints

        # now, we take care of the remaining convex and concave branches
        # to create the rational powers, we need a new variable, r, and
        # the constraint sum(r) == t
        r = lu.create_var(x.shape)
        t_ = lu.promote(t, x.shape)
        constraints += [lu.create_eq(lu.sum(r), t)]

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

        # todo: no need to run gm_constr to form the tree each time.
        # we only need to form the tree once
