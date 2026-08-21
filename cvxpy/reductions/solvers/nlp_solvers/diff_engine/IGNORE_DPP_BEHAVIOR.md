# The symbolic ignore_dpp / non-DPP path

## What changes

On the `ignore_dpp` / non-DPP branch, parameters are no longer baked to
constants by `EvalParams`. The chain becomes

```
[CallbackParamFold, ..., Dcp2Cone, ..., ConeMatrixStuffing(DIFFENGINE),
 (DiffengineConeFormat,) solver]
```

and canonicalization keeps `Parameter` leaves symbolic. `ConeMatrixStuffing`
dispatches to `diff_engine/cone_stuffing.py::stuff_cone_program`, which
compiles the canonicalized expression trees into a C diff-engine program and
returns a `DiffengineParamConeProg` (`diff_engine/parametric_program.py`) — a
`ParamConeProg` subclass that re-evaluates `(q, d, A, b, P)` at the current
parameter values on every `apply_parameters()` call, instead of multiplying
parameter tensors.

Parameter-free problems produce a stock parameter-free `ParamConeProg`
(single constant-column tensors) and run the unmodified tensor-path code
downstream.

## CallbackParamFold

`Dcp2Cone` epigraph-relaxes nonlinear atoms. For a *variable-free parametric*
subtree in a concave position (e.g. `x <= power(t, 2)` with parameter `t`)
that relaxation is unsound — the epigraph variable is unconstrained by data.
`CallbackParamFold` (`cvxpy/reductions/fold_callback_params.py`) walks the
tree first and replaces every maximal variable-free parametric subtree that is
NOT parameter-affine (checked under `dpp_scope`) with a `CallbackParam` whose
callback re-evaluates the subtree. Canonicalization then sees an opaque
parameter leaf; its value refreshes on every solve because `CallbackParam.value`
re-runs the closure.

Parameter-affine subtrees stay symbolic (the engine handles them natively),
and anything containing a variable is untouched.

## Formatting (conic solvers)

`ConicSolver.format_constraints` constructs a stock `ParamConeProg` — it would
silently replace the re-extractable program at format time. The chain therefore
inserts `DiffengineConeFormat` (`diff_engine/formatting.py`) between stuffing
and a conic solver for parametric problems: it materializes the solver's
cone-restructuring matrix `R` once — by an identity probe of the real
`format_constraints`, so it cannot drift from upstream conventions — and hands
the solver the program with `R` pre-applied and `formatted=True`.
`apply_parameters()` re-applies the stored `R` to freshly extracted `(A, b)`.
QP solvers consume the program directly and need no formatting step.

## Selection and fallbacks

- Default (no `canon_backend`): the branch selects DIFFENGINE when the problem
  is ≤2-D, not DGP, and has no parametric variable bounds. Otherwise it falls
  back silently to `EvalParams` + the tensor backends (N-D SCIPY fallback,
  bounds tensors, DGP's post-EvalParams parameters).
- An explicit tensor `canon_backend` on the symbolic parametric branch raises:
  tensor backends cannot keep parameters symbolic here. Parameter-free
  problems honor any explicit backend.
- An explicit `canon_backend="DIFFENGINE"` on a DPP problem also takes the
  symbolic path.
- The perspective canonicalizer pins its internal aux chains to CPP and
  pre-bakes their parameters, preserving its bake-at-canon semantics.

## Error semantics

- Cone-emitting parametric composites (e.g. `log_det(P)`) now count their
  cones in problem analysis even under `ignore_dpp`, so a QP-only solver
  raises `SolverError` instead of silently solving a baked approximation.
  Evaluate the term numerically (e.g. `np.linalg.slogdet`) if that is wanted.
- The parameter differentiation contract (`requires_grad`,
  `problem.derivative`/`backward`, diffcp's `keep_zeros`) is not supported:
  `DiffengineParamConeProg` raises `NotImplementedError` — the engine
  re-evaluates a nonlinear map rather than applying a stored linear one.

## Caching status

The compiled symbolic program is NOT yet cached across solves
(`SolvingChain.uncached_param_prog`): every solve rebuilds the chain and the
engine program, and the first solve extracts twice (once at stuffing, once in
the solver's `apply_parameters`). Caching with a theta short-circuit and a
record-at-site guard for the one value-consuming canonicalizer (the cone
quad_form canon calls `decomp_quad` on the current `P.value`) is the follow-up
PR.
