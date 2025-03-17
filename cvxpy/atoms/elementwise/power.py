"""
Copyright 2013 Steven Diamond

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at


Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import List, Tuple

import numpy as np
import scipy.sparse as sp

import cvxpy.utilities as u
from cvxpy.atoms.elementwise.elementwise import Elementwise
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities.power_tools import is_power2, pow_high, pow_mid, pow_neg


def _is_const(p) -> bool:
    return isinstance(p, cvxtypes.constant())


class power(Elementwise):
    r""" Elementwise power function :math:`f(x) = x^p`.

    If ``expr`` is a CVXPY expression, then ``expr**p``
    is equivalent to ``power(expr, p)``.

    For DCP problems, the exponent `p` must be a numeric constant. For DGP
    problems, `p` can also be a scalar Parameter.


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

        For DCP problems, ``power`` assumes ``p`` has a rational representation
        with a small denominator. **Approximations** are employed when this is not
        the case. Specifically, ``power`` computes a rational approximation
        to ``p`` with a denominator up to ``max_denom``.
        Increasing ``max_denom`` can give better approximations.
        When ``p`` is an ``int`` or ``Fraction`` object, the approximation
        is usually *exact*. No such approximation
        is used for DGP problems.

        CVXPY supports exponential cone and power cone constraints.
        Such constraints could be used to handle the ``power`` atom in DCP problems
        without relying on approximations. Such an approach would also result in
        fewer variables than the current method, even when the current method is
        an exact reformulation. If you're interested in helping enhance CVXPY with
        this ability, please get in touch with us and check out
        `GitHub Issue 1222 <https://github.com/cvxpy/cvxpy/issues/1222>`_!

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

    x : cvxpy.Variable

    p : int, float, Fraction, or Parameter.
        Scalar power. ``p`` may be a Parameter in DGP programs, but not
        in DCP programs.

    max_denom : int
        The maximum denominator considered in forming a rational approximation
        of ``p``; only relevant when solving as a DCP program.
    """

    def __init__(self, x, p, max_denom: int = 1024) -> None:
        self._p_orig = p
        # NB: It is important that the exponent is an attribute, not
        # an argument. This prevents parametrized exponents from being replaced
        # with their logs in Dgp2Dcp.
        self.p = cvxtypes.expression().cast_to_const(p)
        if not (isinstance(self.p, cvxtypes.constant()) or
                isinstance(self.p, cvxtypes.parameter())):
            raise ValueError("The exponent `p` must be either a Constant or "
                             "a Parameter; received ", type(p))
        self.max_denom = max_denom

        self.p_rational = None
        if isinstance(self.p, cvxtypes.constant()):
            # Compute a rational approximation to p, for DCP (DGP doesn't need
            # an approximation).

            if not isinstance(self._p_orig, cvxtypes.expression()):
                # converting to a CVXPY Constant loses the dtype (eg, int),
                # so fetch the original exponent when possible
                p = self._p_orig
            else:
                p = self.p.value
            # how we convert p to a rational depends on the branch of the function
            if p > 1:
                p, w = pow_high(p, max_denom)
            elif 0 < p < 1:
                p, w = pow_mid(p, max_denom)
            elif p < 0:
                p, w = pow_neg(p, max_denom)

            # note: if, after making the rational approximation, p ends up
            # being 0 or 1, we default to using the 0 or 1 behavior of the
            # atom, which affects the curvature, domain, etc... maybe
            # unexpected behavior to the user if they put in 1.00001?
            if p == 1:
                # in case p is a fraction equivalent to 1
                p = 1
                w = None
            if p == 0:
                p = 0
                w = None

            self.p_rational, self.w = p, w
            self.approx_error = float(abs(self.p_rational - p))
        super(power, self).__init__(x)

    @Elementwise.numpy_numeric
    def numeric(self, values):
        return np.power(values[0], float(self.p.value))

    def sign_from_args(self) -> Tuple[bool, bool]:
        """Returns sign (is positive, is negative) of the expression.
        """
        if self.p.value == 1:
            # Same as input.
            return (self.args[0].is_nonneg(), self.args[0].is_nonpos())
        else:
            # Always positive.
            return (True, False)

    def is_atom_convex(self) -> bool:
        """Is the atom convex?
        """
        # Parametrized powers are not allowed for DCP (curvature analysis
        # depends on the value of the power, not just the sign).
        #
        # p == 0 is affine here.
        return _is_const(self.p) and (self.p.value <= 0 or self.p.value >= 1)

    def is_atom_concave(self) -> bool:
        """Is the atom concave?
        """
        # Parametrized powers are not allowed for DCP.
        #
        # p == 0 is affine here.
        return _is_const(self.p) and 0 <= self.p.value <= 1

    def parameters(self):
        # This is somewhat of a hack. When checking DPP for DGP,
        # we need to know whether the exponent p is a parameter, because
        # expressions like power(power(x, parameter), parameter) are
        # unallowed.
        #
        # It seems natural that p should be an argument, not
        # a member of the atom. However, this doesn't work because power
        # is a special case: while in general parameters in a DGP program
        # must be positive, they can have any sign when appearing as an
        # exponent, since in this case we don't need to take the log
        # (eg, in a monomial x_1^a_1x_2a^2, a_1 and a_2 don't need to be
        # positive). If the parameter p were an arg and was negative,
        # then x^p would get falsely flagged as unknown curvature under DGP.
        #
        # So, as a workaround, we overload the parameters method.
        return self.args[0].parameters() + self.p.parameters()

    def is_atom_log_log_convex(self) -> bool:
        """Is the atom log-log convex?
        """
        if u.scopes.dpp_scope_active():
            # This branch applies curvature rules for DPP.
            #
            # Because a DPP scope is active, parameters will be
            # treated as affine (like variables, not constants) by curvature
            # analysis methods.
            #
            # A power x^p is log-log convex (actually, affine) as long as
            # at least one of x and p do not contain parameters.
            #
            # Note by construction (see __init__, p is either a Constant or
            # a Parameter, ie, either isinstance(p, Constant) or isinstance(p,
            # Parameter)).
            x = self.args[0]
            p = self.p
            return not (x.parameters() and p.parameters())
        else:
            return True

    def is_atom_log_log_concave(self) -> bool:
        """Is the atom log-log concave?
        """
        return self.is_atom_log_log_convex()

    def is_constant(self) -> bool:
        """Is the expression constant?
        """
        return (_is_const(self.p) and self.p.value == 0) or super(power, self).is_constant()

    def is_incr(self, idx) -> bool:
        """Is the composition non-decreasing in argument idx?
        """
        if not _is_const(self.p):
            return self.p.is_nonneg() and self.args[idx].is_nonneg()

        p = self.p_rational
        if 0 <= p <= 1:
            return True
        elif p > 1:
            if is_power2(p):
                return self.args[idx].is_nonneg()
            else:
                return True
        else:
            return False

    def is_decr(self, idx) -> bool:
        """Is the composition non-increasing in argument idx?
        """
        if not _is_const(self.p):
            return self.p.is_nonpos() and self.args[idx].is_nonneg()

        p = self.p_rational
        if p <= 0:
            return True
        elif p > 1:
            if is_power2(p):
                return self.args[idx].is_nonpos()
            else:
                return False
        else:
            return False

    def is_quadratic(self) -> bool:
        if not _is_const(self.p):
            return False

        p = self.p_rational
        if p == 0:
            return True
        elif p == 1:
            return self.args[0].is_quadratic()
        elif p == 2:
            return self.args[0].is_affine()
        else:
            return self.args[0].is_constant()

    def has_quadratic_term(self) -> bool:
        """Does the affine head of the expression contain a quadratic term?

        The affine head is all nodes with a path to the root node
        that does not pass through any non-affine atom. If the root node
        is non-affine, then the affine head is the root alone.
        """
        if not _is_const(self.p):
            return False

        p = self.p_rational
        if p == 1:
            return self.args[0].has_quadratic_term()
        elif p == 2:
            return True
        else:
            return False

    def is_qpwa(self) -> bool:
        if not _is_const(self.p):
            # disallow parameters
            return False

        p = self.p_rational
        if p == 0:
            return True
        elif p == 1:
            return self.args[0].is_qpwa()
        elif p == 2:
            return self.args[0].is_pwl()
        else:
            return self.args[0].is_constant()

    def _quadratic_power(self) -> bool:
        """Utility function to check if power is 0, 1 or 2."""
        p = self.p_rational
        return p in [0, 1, 2]

    def _grad(self, values):
        """Gives the (sub/super)gradient of the atom w.r.t. each argument.

        Matrix expressions are vectorized, so the gradient is a matrix.

        Args:
            values: A list of numeric values for the arguments.

        Returns:
            A list of SciPy CSC sparse matrices or None.
        """
        rows = self.args[0].size
        cols = self.size

        if self.p_rational is not None:
            p = self.p_rational
        elif self.p.value is not None:
            p = self.p.value
        else:
            raise ValueError("Cannot compute grad of parametrized power when "
                             "parameter value is unspecified.")

        if p == 0:
            # All zeros.
            return [sp.csc_array((rows, cols), dtype='float64')]
        # Outside domain or on boundary.
        if not is_power2(p) and np.min(values[0]) <= 0:
            if p < 1:
                # Non-differentiable.
                return [None]
            else:
                # Round up to zero.
                values[0] = np.maximum(values[0], 0)

        grad_vals = float(p)*np.power(values[0], float(p)-1)
        return [power.elemwise_grad_to_diag(grad_vals, rows, cols)]

    def _domain(self) -> List[Constraint]:
        """Returns constraints describing the domain of the node.
        """
        if not isinstance(self._p_orig, cvxtypes.expression()):
            p = self._p_orig
        else:
            p = self.p.value

        if p is None:
            raise ValueError("Cannot compute domain of parametrized power when "
                             "parameter value is unspecified.")
        elif (p < 1 and not p == 0) or (p > 1 and not is_power2(p)):
            return [self.args[0] >= 0]
        else:
            return []

    def get_data(self):
        return [self._p_orig, self.max_denom]

    def copy(self, args=None, id_objects=None) -> "power":
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
        return power(args[0], self._p_orig, self.max_denom)

    def name(self) -> str:
        return "%s(%s, %s)" % (self.__class__.__name__,
                               self.args[0].name(),
                               self.p.value)
