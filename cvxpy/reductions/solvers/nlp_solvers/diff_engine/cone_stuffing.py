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

Cone matrix stuffing via the C diff engine. ``ConeMatrixStuffing.apply``
dispatches here for canon_backend="DIFFENGINE"; everything
backend-specific lives in this package so the dispatch seam stays fixed
as the backend evolves.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cvxpy.settings as s
from cvxpy.constraints import (
    PSD,
    SOC,
    ExpCone,
    NonNeg,
    PowCone3D,
    PowConeND,
    SvecPSD,
    Zero,
)
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.matrix_stuffing import (
    _has_parametric_bounds,
    extract_lower_bounds,
    extract_mip_idx,
    extract_upper_bounds,
)
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.extractor import DiffEngineExtractor
from cvxpy.reductions.utilities import group_constraints


def stuff_cone_program(problem, cons, inverse_data, quad_obj):
    """Stuff a parameter-free problem by evaluating the expression trees
    with the C diff engine instead of building parameter tensors.

    Produces the same artifact as the tensor path: a ParamConeProg whose
    single-column tensors hold the constant slice, so every downstream
    consumer (formatting, solvers, inversion) runs the stock code path.

    ``cons`` are the lowered (but not yet ordered) constraints from
    ``ConeMatrixStuffing.apply``. Returns ``(new_prob, inverse_data)``.
    """
    variables = problem.variables()
    if _has_parametric_bounds(variables):
        raise NotImplementedError(
            f"The {s.DIFFENGINE_CANON_BACKEND} canonicalization backend "
            "does not support parametric variable bounds.")
    if problem.parameters():
        raise ValueError(
            f"The {s.DIFFENGINE_CANON_BACKEND} canonicalization backend "
            "does not yet support parametric problems. Solve with "
            "ignore_dpp=True to evaluate parameters before "
            "canonicalization. If you already passed ignore_dpp=True, the "
            "remaining parameters most likely come from parametric "
            "variable bounds, which the DIFFENGINE backend does not "
            "support.")

    # Reorder constraints to Zero, NonNeg, SOC, PSD, EXP, PowCone3D, PowConeND
    constr_map = group_constraints(cons)
    ordered_cons = constr_map[Zero] + constr_map[NonNeg] + \
        constr_map[SOC] + constr_map.get(PSD, []) + \
        constr_map.get(SvecPSD, []) + constr_map[ExpCone] + \
        constr_map[PowCone3D] + constr_map[PowConeND]
    inverse_data.cons_id_map = {con.id: con.id for con in ordered_cons}
    inverse_data.constraints = ordered_cons
    inverse_data.minimize = type(problem.objective) == Minimize

    # One-shot extraction of the concrete cone matrices at x = 0.
    expr_list = [arg for c in ordered_cons for arg in c.args]
    extractor = DiffEngineExtractor(inverse_data).build(
        problem.objective.expr, expr_list, quad_obj)
    q, d, A, b, P = extractor.extract(quad_obj)

    n = inverse_data.x_length
    # Encode as the single-column (constant-slice) tensors ParamConeProg
    # decodes: [q; d] for the objective, [A | b] flattened column-major
    # for the constraints, P flattened column-major for the quadratic
    # term. Everything downstream then matches the tensor path exactly.
    q_tensor = sp.csr_array(np.concatenate([q, [d]])[:, None])
    A_tensor = sp.csc_array(
        sp.hstack([A, sp.csc_array(b[:, None])]).reshape((-1, 1), order='F'))
    P_tensor = (sp.csc_array(P.reshape((n * n, 1), order='F'))
                if P is not None else None)

    boolean, integer = extract_mip_idx(variables)
    x = Variable(n, boolean=boolean, integer=integer)
    new_prob = ParamConeProg(
        q_tensor,
        x,
        A_tensor,
        variables,
        inverse_data.var_offsets,
        ordered_cons,
        [],
        inverse_data.param_id_map,
        P=P_tensor,
        lower_bounds=extract_lower_bounds(variables, n),
        upper_bounds=extract_upper_bounds(variables, n),
    )
    return new_prob, inverse_data
