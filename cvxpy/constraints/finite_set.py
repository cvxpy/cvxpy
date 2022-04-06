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
    def __init__(self, expre, vec, flag: bool = False, constr_id=None) -> None:
        Expression = cvxtypes.expression()
        vec = Expression.cast_to_const(vec)
        flag = Expression.cast_to_const(flag)
        self.expre = expre
        self.vec = vec
        self.flag = flag
        super(FiniteSet, self).__init__([expre, vec, flag], constr_id)

    def name(self) -> str:
        return "%s FS 0" % self.args[0]

    def is_dcp(self, dpp: bool = False) -> bool:
        """A FiniteSet constraint imposed by exprval_in_vec makes the MICP problem DCP"""
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
