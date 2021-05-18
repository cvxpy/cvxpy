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

from cvxpy.expressions.variable import Variable

DCP = 'DCP'
DGP = 'DGP'


def build_non_disciplined_error_msg(problem, discipline_type) -> str:
    prop_name = None
    prefix_conv = ""
    if discipline_type == DCP:
        prop_name = "is_dcp"
    elif discipline_type == DGP:
        prop_name = "is_dgp"
        prefix_conv = "log_log_"
    else:
        raise ValueError("Unknown discipline type")

    def find_non_prop_leaves(expr, res=None):
        if res is None:
            res = []
        if (len(expr.args) == 0 and getattr(expr, prop_name)()):
            return res
        if ((not getattr(expr, prop_name)()) and
                all(getattr(child, prop_name)() for child in expr.args)):
            str_expr = str(expr)
            if discipline_type == DGP and isinstance(expr, Variable):
                str_expr += " <-- needs to be declared positive"
            res.append(str_expr)
        for child in expr.args:
            res = find_non_prop_leaves(child, res)
        return res

    if not getattr(problem.objective, prop_name)():
        non_disciplined_leaves = find_non_prop_leaves(problem.objective.expr)
        if len(non_disciplined_leaves) > 0:
            msg = "The objective is not {}. Its following subexpressions are not:".format(
                discipline_type
            )
        else:
            convex_str = "{}{}".format(prefix_conv, "convex")
            concave_str = "{}{}".format(prefix_conv, "concave")
            fun_attr_check = getattr(problem.objective.args[0], "is_{}".format(convex_str))()
            msg = ("The objective is not {}, even though each sub-expression is.\n"
                   "You are trying to {} a function that is {}.").format(
                        discipline_type,
                        problem.objective.NAME,
                        convex_str if fun_attr_check else concave_str
                    )
        for expr in non_disciplined_leaves:
            msg += '\n%s' % (str(expr,))
        return msg
    not_disciplined_constraints = [expr for expr in problem.constraints if not expr.is_dcp()]
    msg = "The following constraints are not {}:".format(discipline_type)
    for expr in not_disciplined_constraints:
        msg += '\n%s , because the following subexpressions are not:' % (expr,)
        non_disciplined_leaves = find_non_prop_leaves(expr)
        for subexpr in non_disciplined_leaves:
            msg += '\n|--  %s' % (str(subexpr,))
    return msg
