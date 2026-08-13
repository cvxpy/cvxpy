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
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

import cvxpy as cp
from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.reductions.solvers.nlp_solvers.diff_engine.converters import (
    convert_expr,
    convert_symbolic_quad_form,
)
from cvxpy.tests.base_test import BaseTest

SOLVER = cp.CLARABEL


class TestDiffengineConverter(BaseTest):
    """Converter correctness and fail-loud behavior: unsupported constructs
    must raise immediately instead of silently miscompiling."""

    def test_unsupported_atom_raises(self) -> None:
        """convert_expr names the offending atom."""
        x = cp.Variable(4)
        with self.assertRaisesRegex(NotImplementedError, "cumsum"):
            convert_expr(cp.cumsum(x), {x.id: None}, 4, {})

    def test_symbolic_quad_form_block_indices_raises(self) -> None:
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.sum_squares(x),
                               block_indices=[np.array([0, 1]), np.array([2, 3])])
        with self.assertRaisesRegex(NotImplementedError, "block_indices"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_symbolic_quad_form_unsupported_orig_raises(self) -> None:
        x = cp.Variable(4)
        sqf = SymbolicQuadForm(x, sp.eye_array(4), cp.norm1(x))
        with self.assertRaisesRegex(NotImplementedError, "norm1"):
            convert_symbolic_quad_form(sqf, {}, 4, {})

    def test_symbolic_quad_form_parametric_P_raises(self) -> None:
        """A parametric P is out of scope for the diff engine; the converter
        must fail loud rather than freeze the current value."""
        n = 4
        x = cp.Variable(n)
        P = cp.Parameter((n, n), PSD=True)
        P.value = np.eye(n)
        sqf = SymbolicQuadForm(x, cp.psd_wrap(P), cp.quad_form(x, cp.psd_wrap(P)))
        with self.assertRaisesRegex(NotImplementedError, "parametric P"):
            convert_symbolic_quad_form(sqf, {x.id: None}, n, {P.id: None})
