"""
Copyright 2017 Robin Verschueren

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
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.error import SolverError
from cvxpy.reductions.canonical_maps import tensor_matrix_block, tensor_vector_block
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.solvers.solver import Solver


def _has_unsupported_cones(cone_dims) -> bool:
    """Check if cone dimensions contain cones unsupported by QP solvers."""
    return (cone_dims.exp > 0 or cone_dims.soc or cone_dims.psd
            or cone_dims.p3d or cone_dims.pnd)


def _get_unsupported_cone_message(cone_dims) -> str:
    """Get a descriptive message about unsupported cones."""
    unsupported = []
    if cone_dims.exp > 0:
        unsupported.append(f"{cone_dims.exp} exponential cones")
    if cone_dims.soc:
        unsupported.append(f"{len(cone_dims.soc)} second-order cones")
    if cone_dims.psd:
        unsupported.append(f"{len(cone_dims.psd)} PSD cones")
    if cone_dims.p3d:
        unsupported.append(f"{len(cone_dims.p3d)} 3D power cones")
    if cone_dims.pnd:
        unsupported.append(f"{len(cone_dims.pnd)} ND power cones")
    return ', '.join(unsupported)


class QpSolver(Solver):
    """
    A QP solver interface.

    QP solvers accept ParamConeProg with only Zero and NonNeg constraints
    (i.e., equality and inequality constraints) and convert them to the
    standard QP form:

        minimize      1/2 x' P x + q' x
        subject to    A x =  b
                      F x <= g
    """
    # Every QP solver supports Zero and NonNeg constraints.
    SUPPORTED_CONSTRAINTS = [Zero, NonNeg]

    # Some solvers cannot solve problems that do not have constraints.
    # For such solvers, REQUIRES_CONSTR should be set to True.
    REQUIRES_CONSTR = False

    IS_MIP = "IS_MIP"

    def supports_quad_obj(self) -> bool:
        """QP solvers support quadratic objectives."""
        return True

    def accepts(self, problem):
        return (isinstance(problem, ParamConeProg)
                and (self.MIP_CAPABLE or not problem.is_mixed_integer())
                and not convex_attributes([problem.x])
                and (len(problem.constraints) > 0 or not self.REQUIRES_CONSTR)
                and all(type(c) in self.SUPPORTED_CONSTRAINTS for c in problem.constraints)
                and not _has_unsupported_cones(problem.cone_dims))

    def apply(self, problem):
        """
        Construct QP problem data stored in a dictionary.

        Converts a ParamConeProg (with only Zero and NonNeg constraints) to QP form:

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        The rows arrive in stuffing order -- Zero then NonNeg, established by
        ``order_cone_constraints`` -- and unformatted: ``_build_solving_chain``
        appends ``ConeFormat`` only for conic solvers. With just those two
        cones the layout would be the identity anyway, so all it would do is
        negate the Zero rows, and the split below applies its own signs.
        """
        # Negating twice would silently flip the equality block.
        assert not problem.formatted, "QP solvers consume unformatted rows"
        if not self.accepts(problem):
            if _has_unsupported_cones(problem.cone_dims):
                raise SolverError(
                    f"QP solver {self.name()} cannot handle: "
                    f"{_get_unsupported_cone_message(problem.cone_dims)}. "
                    "This may indicate a bug in solver selection. Please report this issue."
                )

        maps = self.cached_canonical_maps(problem)
        if maps is not NotImplemented:
            return self._apply_canonical_maps(problem, maps)

        data, inv_data = self._prepare_data_and_inv_data(problem)

        # Apply parameters with quadratic objective
        P, q, d, AF, bg = problem.apply_parameters(quad_obj=True)
        inv_data[s.OFFSET] = d

        # Get number of variables
        n = problem.x.size
        len_eq = problem.cone_dims.zero
        len_leq = problem.cone_dims.nonneg

        # Split into equality and inequality constraints
        if len_eq > 0:
            A = AF[:len_eq, :]
            b = -bg[:len_eq]
        else:
            A, b = sp.csr_array((0, n)), -np.array([])

        if len_leq > 0:
            F = -AF[len_eq:, :]
            g = bg[len_eq:]
        else:
            F, g = sp.csr_array((0, n)), -np.array([])

        # Create dictionary with problem data
        data[s.P] = sp.csc_array(P)
        data[s.Q] = q
        data[s.A] = sp.csc_array(A)
        data[s.B] = b
        data[s.F] = sp.csc_array(F)
        data[s.G] = g
        data['n_eq'] = A.shape[0]
        data['n_ineq'] = F.shape[0]
        # apply_parameters() has evaluated any parametric bounds.
        data[s.LOWER_BOUNDS] = problem.lower_bounds
        data[s.UPPER_BOUNDS] = problem.upper_bounds

        return data, inv_data

    def _prepare_data_and_inv_data(self, problem):
        """Problem data and inverse data that do not depend on parameters."""
        data = {}
        inv_data = {self.VAR_ID: problem.x.id}

        data[QpSolver.DIMS] = problem.cone_dims
        inv_data[QpSolver.DIMS] = data[QpSolver.DIMS]
        inv_data[QpSolver.IS_MIP] = problem.is_mixed_integer()
        data[s.PARAM_PROB] = problem

        # Store constraint lists for dual variable mapping
        # Constraints are ordered: Zero (equality) first, then NonNeg (inequality)
        inv_data[self.EQ_CONSTR] = [c for c in problem.constraints if type(c) == Zero]
        inv_data[self.NEQ_CONSTR] = [c for c in problem.constraints if type(c) == NonNeg]

        data[s.BOOL_IDX] = [t[0] for t in problem.x.boolean_idx]
        data[s.INT_IDX] = [t[0] for t in problem.x.integer_idx]
        data['n_var'] = problem.x.size
        # Variable bounds are not set here: parametric bounds are only
        # evaluated by apply_parameters(), so read them after it.
        return data, inv_data

    def _qp_canonical_maps(self, problem):
        """The maps of ``apply``'s split form, as ``(P, q, d, AF, bg)``.

        ``P`` is the full symmetric matrix. ``AF x <= bg`` stacks the
        equality rows (which hold with equality) above the inequality rows:
        ``apply``'s ``A, b`` and ``F, g`` are its row blocks.
        """
        n_params, q, AF, P = problem.input_tensors()
        n = problem.x.size
        len_eq = problem.cone_dims.zero
        m = len_eq + problem.cone_dims.nonneg
        # The stuffed rows read ``A x + b`` in Zero and NonNeg cones.
        sign = np.concatenate([np.ones(len_eq), -np.ones(m - len_eq)])
        return (tensor_matrix_block(P, n, n, n_params),
                tensor_vector_block(q, 0, n, n_params),
                tensor_vector_block(q, n, n + 1, n_params),
                tensor_matrix_block(AF, m, n, n_params).scale_rows(sign),
                tensor_vector_block(AF, m * n, m * (n + 1), n_params).scale(-sign))

    def _apply_canonical_maps(self, problem, maps):
        """``apply`` for an interface whose canonical maps are implemented.

        Interfaces override this to turn their evaluated inputs into the data
        their ``solve_via_data`` consumes.
        """
        raise NotImplementedError()
