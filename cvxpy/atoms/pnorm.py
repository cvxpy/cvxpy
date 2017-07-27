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

from cvxpy.atoms.atom import Atom
from cvxpy.atoms.axis_atom import AxisAtom
import cvxpy.lin_ops.lin_utils as lu
import numpy as np
import scipy.sparse as sp
from cvxpy.utilities.power_tools import pow_high, pow_mid, pow_neg, gm_constrs
from cvxpy.constraints.second_order import SOC
from cvxpy.constraints.soc_axis import SOC_Axis
from fractions import Fraction


class pnorm(AxisAtom):
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

    axis : 0 or 1
           The axis to apply the norm to.

    Returns
    -------
    Expression
        An Expression representing the norm.
    """

    def __init__(self, x, p=2, axis=None, max_denom=1024):
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

        super(pnorm, self).__init__(x, axis=axis)

        if self.p == np.inf:
            self.approx_error = 0
        else:
            self.approx_error = float(abs(self.p - p_old))

    @Atom.numpy_numeric
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

        retval = np.linalg.norm(values, float(self.p), axis=self.axis)

        # NOTE: workaround for NumPy <=1.9 and no keepdims for norm()
        if self.axis is not None:
            if self.axis == 0:
                retval = np.reshape(retval, (1, self.args[0].size[1]))
            else:  # self.axis == 1:
                retval = np.reshape(retval, (self.args[0].size[0], 1))

        return retval

    def validate_arguments(self):
        super(pnorm, self).validate_arguments()
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
        return self.p >= 1

    def is_atom_concave(self):
        """Is the atom concave?
        """
        return self.p < 1

    def is_incr(self, idx):
        """Is the composition non-decreasing in argument idx?
        """
        return self.p < 1 or (self.p >= 1 and self.args[0].is_positive())

    def is_decr(self, idx):
        """Is the composition non-increasing in argument idx?
        """
        return self.p >= 1 and self.args[0].is_negative()

    def is_pwl(self):
        """Is the atom piecewise linear?
        """
        return (self.p == 1 or self.p == np.inf) and self.args[0].is_pwl()

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
        rows = self.args[0].size[0]*self.args[0].size[1]
        value = np.matrix(value)
        # Outside domain.
        if self.p < 1 and np.any(value <= 0):
            return None
        D_null = sp.csc_matrix((rows, 1), dtype='float64')
        if self.p == 1:
            D_null += (value > 0)
            D_null -= (value < 0)
            return sp.csc_matrix(D_null.A.ravel(order='F')).T
        denominator = np.linalg.norm(value, float(self.p))
        denominator = np.power(denominator, self.p - 1)
        # Subgrad is 0 when denom is 0 (or undefined).
        if denominator == 0:
            if self.p >= 1:
                return D_null
            else:
                return None
        else:
            nominator = np.power(value, self.p - 1)
            frac = np.divide(nominator, denominator)
            return np.reshape(frac.A, (frac.size, 1))

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
                return t, [SOC(t, [x])]

            else:
                t = lu.create_var(size)
                return t, [SOC_Axis(lu.reshape(t, (t.size[0]*t.size[1], 1)),
                                    x, axis)]

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

        # todo: no need to run gm_constr to form the tree each time.
        # we only need to form the tree once
