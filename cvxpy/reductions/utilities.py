"""
Copyright 2018 Akshay Agrawal

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


from cvxpy.constraints.zero import Zero
from cvxpy.constraints.nonpos import NonPos


def lower_inequality(inequality):
    lhs = inequality.args[0]
    rhs = inequality.args[1]
    return NonPos(lhs - rhs, constr_id=inequality.constr_id)


def lower_equality(equality):
    lhs = equality.args[0]
    rhs = equality.args[1]
    return Zero(lhs - rhs, constr_id=equality.constr_id)


def are_args_affine(constraints):
    return all(arg.is_affine() for constr in constraints
               for arg in constr.args)
