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
        The given expression to be constrained. If ``expre`` has multiple elements, then
        the constraint is applied separtely to each element. I.e., after solving a problem
        with this constraint, we should have:

        .. code-block:: python

            for e in expre.flatten():
                print(e.value in vec) # => True

    vec : Union[Expression, np.ndarray, set]
        The finite set of values to which the given (affine) expression is to be constrained.
        
    ineq_form : bool
        Controls how this contraint is canonicalized into mixed integer linear constraints.
        
        If True, then we use a formulation with ``vec.size - 1`` inequality constraints, one 
        equality constraint, and ``vec.size - 1`` binary variables for each element of ``expre``.
        
        If False, then we use a formuation with ``vec.size`` binary variables and two equality
        constraints for each element of ``expre``.

        Defaults to False. The case ``ineq_form=True`` is provided in the hopes that it may 
        speed up some mixed-integer solvers that use simple branch and bound methods.
    """

    def __init__(self, expre, vec, ineq_form: bool = False, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        if isinstance(vec, set):
            vec = list(vec)
        vec = Expression.cast_to_const(vec).flatten()
        self.expre = expre
        self.vec = vec
        self._ineq_form = ineq_form
        super(FiniteSet, self).__init__([expre, vec], constr_id)

    def name(self) -> str:
        return "FiniteSet(%s, %s)" % (self.args[0], self.args[1])

    def get_data(self):
        return [self._ineq_form]

    def is_dcp(self, dpp: bool = False) -> bool:
        """
        A ``FiniteSet`` constraint is DCP if the constrained expression is affine
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
        expre_val = np.array(self.expre.value).flatten()
        vec_val = self.vec.value
        resids = [np.min(np.abs(val - vec_val)) for val in expre_val]
        res = max(resids)
        return res
