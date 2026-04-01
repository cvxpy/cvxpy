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

Chordal decomposition of PSD constraints.

Replaces large PSD constraints with smaller overlapping PSD constraints
based on the sparsity pattern of the data matrix, when the aggregate
sparsity pattern is chordal (or can be made chordal via a fill-reducing
chordal extension).

This reduction operates on a ParamConeProg (the output of ConeMatrixStuffing)
and produces a new ParamConeProg with decomposed PSD cones.

References
----------
.. [1] Vandenberghe, L. and Andersen, M.S., 2015.
       Chordal graphs and semidefinite optimization.
       Foundations and Trends in Optimization, 1(4), pp.241-433.

.. [2] Sun, Y., Vandenberghe, L. et al., 2014.
       Decomposition methods for sparse matrix nearness problems.
       SIAM Journal on Matrix Analysis and Applications, 35(3), pp.1028-1047.
"""
from __future__ import annotations

from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ParamConeProg
from cvxpy.reductions.reduction import Reduction


class ChordalDecomp(Reduction):
    """Replace PSD constraints with smaller PSD constraints via chordal decomposition.

    This reduction exploits sparsity in semidefinite constraints.  When the
    aggregate sparsity pattern of a PSD constraint is chordal (or can be
    extended to one cheaply), the single large PSD constraint can be
    equivalently replaced by several smaller PSD constraints on the
    principal submatrices corresponding to the maximal cliques of the
    chordal graph.

    This reduction sits between ``ConeMatrixStuffing`` and the solver in
    the solving chain.  It receives and produces a ``ParamConeProg``.

    Parameters
    ----------
    use_decomposition : bool, optional
        If *False* the reduction is a no-op (pass-through).
        Default *True*.
    """

    def __init__(self, use_decomposition: bool = True) -> None:
        super().__init__()
        self.use_decomposition = use_decomposition

    def accepts(self, problem) -> bool:
        """Accept any ParamConeProg (always applicable in the chain)."""
        return isinstance(problem, ParamConeProg)

    def apply(self, problem):
        """Decompose PSD constraints in *problem*.

        Parameters
        ----------
        problem : ParamConeProg
            The stuffed cone program from ``ConeMatrixStuffing``.

        Returns
        -------
        ParamConeProg
            A (possibly modified) cone program with decomposed PSD cones.
        inverse_data : dict
            Data needed by :meth:`invert` to reassemble the solution.
        """
        # TODO: implement chordal decomposition here.
        inverse_data = {"no_decomposition": True}
        return problem, inverse_data

    def invert(self, solution, inverse_data):
        """Map the solver solution back through the decomposition.

        Parameters
        ----------
        solution : dict
            The raw solution from the solver (or the next reduction).
        inverse_data : dict
            Data from :meth:`apply`.

        Returns
        -------
        dict
            The solution with dual variables reassembled for the
            original (pre-decomposition) PSD constraints.
        """
        # TODO: reassemble PSD dual variables from clique duals.
        return solution
