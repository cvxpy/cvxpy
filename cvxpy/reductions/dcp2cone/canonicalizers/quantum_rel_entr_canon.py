import numpy as np
import cvxpy as cp
from cvxpy import Variable
from cvxpy.atoms.affine.kron import kron
from cvxpy.constraints import OpRelEntrConeQuad
from cvxpy.utilities.solver_context import SolverInfo


def quantum_rel_entr_canon(expr, args, solver_context: SolverInfo | None = None):
    X, Y = args
    n = X.shape[0]
    Imat = np.eye(n)

    # ── FAST PATH: one argument is constant ──────────────────────────────────
    # Block size drops from 2n² × 2n² → 2n × 2n  (see Fawzi & Fawzi 2018, Table 1 footnote b)

    if X.is_constant():
        # D(X‖Y) with X constant.
        # quantum_rel_entr = Tr[X(log X - log Y)]
        #                  = Tr[op_rel_entr(X, Y)]   <-- scalar trace
        # We introduce a n×n epi variable T such that X ≼_OpRelEntr T (w.r.t. Y)
        # and minimize Tr[T].
        epi = Variable(shape=(n, n), symmetric=True)
        orec_con = OpRelEntrConeQuad(
            X, Y, epi,
            expr.quad_approx[0], expr.quad_approx[1]
        )
        main_con, aux_cons = cp.reductions.cone2cone.approx.OpRelEntrConeQuad_canon(
            orec_con, None
        )
        obj = cp.trace(epi)           # Tr[T]  →  scalar objective
        return obj, [main_con] + aux_cons

    if Y.is_constant():
        # D(X‖Y) with Y constant.
        # Same idea, swap roles.
        epi = Variable(shape=(n, n), symmetric=True)
        orec_con = OpRelEntrConeQuad(
            X, Y, epi,
            expr.quad_approx[0], expr.quad_approx[1]
        )
        main_con, aux_cons = cp.reductions.cone2cone.approx.OpRelEntrConeQuad_canon(
            orec_con, None
        )
        obj = cp.trace(epi)
        return obj, [main_con] + aux_cons

    # ── GENERAL PATH: both X and Y are variables ─────────────────────────────
    # Must lift to n²×n² space via Kronecker products.
    e = Imat.ravel().reshape(n**2, 1)

    first_arg  = cp.atoms.affine.wraps.symmetric_wrap(kron(X, Imat))
    second_arg = cp.atoms.affine.wraps.symmetric_wrap(kron(Imat, Y))
    epi = Variable(shape=first_arg.shape, symmetric=True)

    orec_con = OpRelEntrConeQuad(
        first_arg, second_arg, epi,
        expr.quad_approx[0], expr.quad_approx[1]
    )
    main_con, aux_cons = cp.reductions.cone2cone.approx.OpRelEntrConeQuad_canon(
        orec_con, None
    )
    return e.T @ epi @ e, [main_con] + aux_cons