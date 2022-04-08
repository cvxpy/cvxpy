#!/usr/bin/env python
"""
Copyright, the CVXPY authors

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
import numpy as np

from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions import cvxtypes
from cvxpy.utilities import scopes


class FiniteSet(Constraint):
    """A class for constraining given expressions to a set of finite size composed of real numbers.

    Parameters
    ----------
    expre : Expression
        The given expression to be constrained. Note that, ``expre`` can have multiple features, and the
        constraint is applied element-wise to each feature of the ``Expression`` i.e.:

        .. code-block:: python

            for i in range(expre.size):
                print(expre[i] in vec) # => True

    vec : NumPy.ndarray/set
        The finite set of values to which the given (affine) expression is to be constrained.
    """
    def __init__(self, expre, vec, ineq_form: bool = False, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        if isinstance(vec, set):
            vec = list(vec)
        vec = Expression.cast_to_const(vec)
        self.expre = expre
        self.vec = vec
        self._ineq_form = ineq_form
        super(FiniteSet, self).__init__([expre, vec], constr_id)

    def name(self) -> str:
        return "%s FS 0" % self.args[0]

    def get_data(self):
        return [self._ineq_form]

    def is_dcp(self, dpp: bool = False) -> bool:
        """
        A ``FiniteSet`` constraint is DCP, if the constrained expression is affine
        """
        if dpp:
            with scopes.dpp_scope():
                return self.args[0].is_affine()
        return self.args[0].is_affine()

    def is_dgp(self, dpp: bool = False) -> bool:
        return False

    def is_dqcp(self) -> bool:
        return self.is_dcp()

    @property
    def size(self):
        return self.expre.size

    @property
    def ineq_form(self) -> bool:
        """
        Choose between two constraining methodologies, use ``ineq_form=False`` while working with
        ``Parameter`` types
        """
        return self._ineq_form

    @property
    def shape(self):
        return self.expre.shape

    @property
    def residual(self):
        """
        The residual of the constraint.

        Returns
        -------
        float
        """
        val = self.expre.value
        res = np.min(np.abs(val - self.vec.value))
        return res
