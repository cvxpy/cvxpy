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

Chain step that pre-applies the solver's cone-restructuring matrix to the
symbolic parametric DIFFENGINE program, so it reaches the solver already
``formatted=True`` and is never replaced by a stock ParamConeProg inside
``ConicSolver.format_constraints``.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.reduction import Reduction


def build_restruct_mat(solver, constraints, m):
    """Materialize the solver's cone-restructuring matrix R as CSC.

    Implemented as an identity probe of the frozen
    ``ConicSolver.format_constraints``: a stub parameter-free ParamConeProg
    whose constraint matrix is ``[I_m | 0]`` is formatted, and R is read off
    the restructured result. This cannot drift from the solver's actual
    restructuring conventions because it *is* the solver's restructuring.
    """
    # Encode [I_m | 0] with x.size == m: flat row of entry (i, i) is i*(m+1).
    flat_rows = np.arange(m, dtype=np.int64) * (m + 1)
    A_probe = sp.csc_array(
        (np.ones(m), (flat_rows, np.zeros(m, dtype=np.int64))),
        shape=(m * (m + 1), 1))
    q_probe = sp.csr_array((m + 1, 1))
    stub = ParamConeProg(
        q_probe, Variable(m), A_probe,
        variables=[], var_id_to_col={},
        constraints=constraints, parameters=[], param_id_to_col={})
    formatted = type(solver).format_constraints(stub, solver.EXP_CONE_ORDER)
    m_out = formatted.A.shape[0] // (m + 1)
    M = formatted.A.reshape((m_out, m + 1), order='F').tocsc()
    return M[:, :m]


class DiffengineConeFormat(Reduction):
    """Pre-format the symbolic DIFFENGINE program for a conic solver."""

    def __init__(self, solver) -> None:
        super().__init__()
        self.solver = solver

    def accepts(self, problem) -> bool:
        return True

    def apply(self, problem):
        from cvxpy.reductions.solvers.nlp_solvers.diff_engine.parametric_program import (
            DiffengineParamConeProg,
        )
        if not isinstance(problem, DiffengineParamConeProg):
            # Canonicalization consumed every parameter (e.g. a constraint
            # quad_form factorized at its current value), so stuffing
            # produced a stock program; the solver formats it normally.
            return problem, None
        R = build_restruct_mat(
            self.solver, problem.constraints, problem.A.shape[0] // (problem.x.size + 1))
        return problem.with_restruct(R), None

    def invert(self, solution, inverse_data):
        return solution
