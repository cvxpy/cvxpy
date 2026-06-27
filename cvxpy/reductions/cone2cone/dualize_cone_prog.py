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

import cvxpy.settings as s
from cvxpy.reductions.cone2cone.affine2direct import Dualize as A2DDualize
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.reduction import Reduction


class DualizeConeProg(Reduction):
    """Flags a ParamConeProg for dualized interpretation by the solver.

    This reduction sits between ConeMatrixStuffing and the solver.  It
    does **not** rewrite any matrices — it simply sets the ``dualized``
    flag on the ``ParamConeProg`` so that downstream solvers interpret
    the existing data ``(A, b, c, d, P, cone_dims)`` as the Lagrangian
    dual:

        (D)  max  -b'y  [- ½w'Pw]  + d
             s.t. A'y   [+ Pw]      = c
                  y in K*

    where ``K*`` is the dual of the cone ``K`` described by
    ``cone_dims``.  The slack variable ``w`` (size ``x.size``) is
    introduced by the solver only when ``P`` is not ``None``.

    Primal recovery
    ---------------
    ``x*`` is obtained from the dual of the equality constraint
    ``A'y = c`` (equivalently, ``A'y + Pw = c`` when ``P`` is
    present).  Original constraint duals are the optimal ``y``
    blocks.

    The solver is responsible for:
    1. Interpreting the flag (building a max problem, placing cones
       on variables, etc.).
    2. Returning a ``Solution`` whose ``primal_vars`` and
       ``dual_vars`` follow the ``affine2direct.Dualize`` convention
       so that ``invert`` can recover the original solution.

    Position in the chain
    ---------------------
    ::

        ... -> ConeMatrixStuffing -> DualizeConeProg -> Solver
    """

    def accepts(self, problem) -> bool:
        return (isinstance(problem, ParamConeProg)
                and not problem.is_mixed_integer()
                and not problem.dualized)

    def apply(self, problem):
        problem.dualized = True
        inv_data = {
            'x_id': problem.x.id,
            'constr_map': problem.constr_map,
        }
        return problem, inv_data

    def invert(self, solution, inverse_data):
        """Delegates to the existing affine2direct.Dualize.invert logic."""
        # Build the inv_data dict that affine2direct.Dualize.invert expects.
        inv = {
            s.OBJ_OFFSET: 0,  # offset already baked into solver's opt_val
            'constr_map': inverse_data['constr_map'],
            'x_id': inverse_data['x_id'],
        }
        return A2DDualize.invert(solution, inv)
