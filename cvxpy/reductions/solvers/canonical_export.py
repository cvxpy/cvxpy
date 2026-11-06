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

Assembles ``Problem.get_canonical_maps()`` from a compiled solving chain.
"""
from __future__ import annotations

import numpy as np

from cvxpy.constraints import PSD, RSOC, SOC, ExpCone, PowCone3D, PowConeND
from cvxpy.lin_ops.lin_op import CONSTANT_ID
from cvxpy.reductions.canonical_maps import (
    CanonicalMaps,
    DualInfo,
    SolverInputMaps,
    UnsupportedRecovery,
    VarInfo,
)
from cvxpy.reductions.cvx_attr2constr import SYMMETRIC_ATTRIBUTES, attributes_present
from cvxpy.reductions.eval_params import EvalParams


def dual_variable_indices(constraint, size: int) -> list[np.ndarray] | None:
    """Where each dual variable's entries sit in the constraint's dual.

    The constraint's dual is the flat vector that recovery produces for it
    (see :class:`~cvxpy.reductions.canonical_maps.RecoveryMap`). The returned
    index arrays list each dual variable's entries in Fortran order. They
    replay ``save_dual_value`` on an index array. ``size`` is the length of
    the constraint's dual. ``None`` if the constraint type does not store
    duals.
    """
    idx = np.arange(size)
    if isinstance(constraint, SOC):
        cone_size = constraint._cone_size()
        value = np.reshape(idx, (-1, cone_size))
        t, X = value[:, 0], value[:, 1:]
        if len(constraint.args[1].shape) == 0:
            X = X[0, 0]
        elif constraint.axis == 0:
            X = X.T
        parts = [t, X]
    elif isinstance(constraint, RSOC):
        n_cones = constraint.args[0].size
        dt, du = idx[:n_cones], idx[n_cones:2 * n_cones]
        dx = np.reshape(idx[2 * n_cones:], (n_cones, -1), order='F')
        if n_cones == 1:
            parts = [dt[0], du[0], dx[0]]
        else:
            parts = [dt, du, dx if constraint.axis == 1 else dx.T]
    elif isinstance(constraint, ExpCone):
        value = np.reshape(idx, (-1, 3))
        parts = [value[:, 0], value[:, 1], value[:, 2]]
    elif isinstance(constraint, PowCone3D):
        value = np.reshape(idx, constraint.shape, order='F')
        parts = [value[0], value[1], value[2]]
    elif isinstance(constraint, PowConeND):
        value = np.reshape(idx, constraint.shape, order='F')
        dW = value[:-1, :]
        parts = [dW.T if constraint.axis == 1 else dW, value[-1, :]]
    elif len(constraint.dual_variables) == 1:
        parts = [idx]
    else:
        return None
    return [np.ravel(part, order='F') for part in parts]


def _resolve(rec, what: str):
    if isinstance(rec, UnsupportedRecovery):
        raise NotImplementedError(
            f"Canonical maps cannot recover {what}: {rec.reason}. "
            "Pass `variables` to request only supported outputs.")
    return rec


def build_canonical_maps(problem, chain, param_prog, inverse_data,
                         variables=None, solver_inputs: bool = True) -> CanonicalMaps:
    """Export the canonical maps of a compiled problem.

    Parameters
    ----------
    problem : Problem
        The original problem.
    chain : SolvingChain
        The chain ``problem.get_problem_data()`` compiled with.
    param_prog : ParamConeProg
        The program the chain's solver interface receives.
    inverse_data : list
        The inverse data of every reduction in ``chain``, solver last.
    variables : list of Variable, optional
        Primal variables of ``problem`` and dual variables of its
        constraints. ``None`` selects all primal variables and all duals of
        ``problem.constraints``.
    solver_inputs : bool
        If False, export only parameter preparation and recovery, leaving
        ``maps`` and ``structures`` empty, for interfaces that do not export
        input maps yet. Recovery then assumes the default solver vector
        layout of :meth:`~cvxpy.reductions.solvers.solver.Solver.canonical_recovery`.
        This is a transitional escape hatch, not public API.
    """
    solver = chain.solver
    if any(isinstance(r, EvalParams) for r in chain.reductions):
        raise ValueError("Canonical maps require a DPP compilation; the chain "
                         "evaluates parameters to constants.")
    input_maps = solver.cached_canonical_maps(param_prog) if solver_inputs \
        else SolverInputMaps()
    if input_maps is NotImplemented:
        raise NotImplementedError(
            f"The {solver.name()} interface does not support canonical maps yet.")

    recovery = solver.canonical_recovery(param_prog)
    for r, inv in reversed(list(zip(chain.reductions[:-1], inverse_data[:-1]))):
        recovery = r.compose_recovery(recovery, inv)

    primal_vars = problem.variables()
    dual_owner = {dv.id: (c, i) for c in problem.constraints
                  for i, dv in enumerate(c.dual_variables)}
    if variables is None:
        requested_primal = primal_vars
        requested_dual = [dv for c in problem.constraints for dv in c.dual_variables]
    else:
        primal_ids = {v.id for v in primal_vars}
        requested_primal, requested_dual = [], []
        for v in variables:
            if v.id in primal_ids:
                requested_primal.append(v)
            elif v.id in dual_owner:
                requested_dual.append(v)
            else:
                raise ValueError(
                    f"{v} is neither a variable of the problem nor a dual "
                    "variable of one of its constraints.")

    primal_recovery, primal_info = {}, []
    for var in requested_primal:
        rec = _resolve(recovery.get_primal(var.id), f"variable {var}")
        assert rec.source == 'x'
        if var.attributes['sparsity']:
            sparse_idx = var.sparse_idx
        elif var.attributes['diag']:
            sparse_idx = (np.arange(var.shape[0]),) * 2
        else:
            sparse_idx = None
        primal_recovery[var.id] = rec.finalize()
        primal_info.append(VarInfo(
            var.id, var.shape, rec.size, sparse_idx,
            bool(attributes_present([var], SYMMETRIC_ATTRIBUTES))))

    dual_recovery, dual_info = {}, []
    for dv in requested_dual:
        constraint, i = dual_owner[dv.id]
        what = f"the dual of {type(constraint).__name__} constraint {constraint.id}"
        rec = _resolve(recovery.get_dual(constraint.id), what)
        assert rec.source == 'y'
        parts = dual_variable_indices(constraint, rec.size)
        if parts is None:
            _resolve(UnsupportedRecovery("its constraint type stores no duals"), what)
        sub = rec.take(parts[i])
        if sub.size != dv.size:
            raise NotImplementedError(f"Inconsistent dual size for {what}.")
        dual_recovery[dv.id] = sub.finalize()
        dual_info.append(DualInfo(dv.id, constraint.id, dv.shape, dv.size,
                                  type(constraint).__name__, isinstance(constraint, PSD)))

    param_id_to_col = {pid: col for pid, col in param_prog.param_id_to_col.items()
                       if pid != CONSTANT_ID}
    return CanonicalMaps(
        maps=input_maps.maps,
        structures=input_maps.structures,
        primal_recovery=primal_recovery,
        dual_recovery=dual_recovery,
        primal_vars=primal_info,
        dual_vars=dual_info,
        cone_dims=param_prog.cone_dims,
        constraints=param_prog.constraints,
        parameters=param_prog.parameters,
        param_transforms=chain.param_transforms,
        param_id_to_col=param_id_to_col,
        param_id_to_size={pid: param_prog.param_id_to_size[pid] for pid in param_id_to_col},
        total_param_size=param_prog.total_param_size,
    )
