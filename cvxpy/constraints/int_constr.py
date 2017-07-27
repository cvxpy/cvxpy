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

import cvxpy.settings as s
from cvxpy.constraints.bool_constr import BoolConstr


class IntConstr(BoolConstr):
    """
    An integer constraint:
        X_{ij} in Z for all i,j.

    Attributes:
        noncvx_var: A variable constrained to be elementwise integral.
        lin_op: The linear operator equal to the noncvx_var.
    """
    CONSTR_TYPE = s.INT_IDS

    def __str__(self):
        return "IntConstr(%s)" % self.lin_op
