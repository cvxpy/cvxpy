"""
Copyright 2025 CVXPY developers

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

from cvxpy.atoms.elementwise.rel_entr import rel_entr
from cvxpy.reductions.expr2smooth.canonicalizers.rel_entr_canon import rel_entr_canon


def kl_div_canon(expr, args):
    _rel_entr = rel_entr(args[0], args[1])
    rel_entr_expr, constr = rel_entr_canon(_rel_entr, _rel_entr.args)
    return rel_entr_expr - args[0] + args[1] , constr

