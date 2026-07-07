# `ignore_dpp=True` → DIFFENGINE backend: behavior notes

Developer-facing notes on a subtle semantic change introduced by the diffengine backend
work. The change is easy to miss because it does not touch any test file, yet it alters
what `Problem.solve(..., ignore_dpp=True)` does. Read this before changing the
`ignore_dpp` branch of the solving chain or the DIFFENGINE backend.

## Summary: the old vs. new `ignore_dpp` contract

`ignore_dpp=True` used to prepend the `EvalParams` reduction (baking every `Parameter`
into a `Constant`). It now instead selects the **DIFFENGINE** canon backend and leaves
the parameters symbolic — the C diff engine re-evaluates the parametric expression tree
on every solve, preserving the problem's conic structure.

| | Old (master) | New (this branch) |
|---|---|---|
| What `ignore_dpp=True` does | prepend `EvalParams` (params → constants) | set `canon_backend = DIFFENGINE` (re-evaluate params each solve, keep conic structure) |
| `EvalParams` in the chain? | yes | no |
| `ConeMatrixStuffing.canon_backend` | `None` (→ CPP) | `'DIFFENGINE'` |
| Parameter-dependent atoms | folded to constants before canonicalization | canonicalized symbolically (cones preserved) |

### Where this lives in the code

- **The switch** — `cvxpy/reductions/solvers/solving_chain.py:209-212`:

  ```python
  if ignore_dpp:
      canon_backend = DIFFENGINE_CANON_BACKEND   # line 210 (new)
  else:
      reductions = [EvalParams()] + reductions    # line 212 (old path, now non-DPP only)
  ```

- **The dispatch** — `cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py:467-471`: when
  `canon_backend == DIFFENGINE`, stuffing builds a `DiffengineConeProgram` instead of the
  normal `ParamConeProg`.
- **The constant** — `cvxpy/settings.py`:
  `DIFFENGINE_CANON_BACKEND = "DIFFENGINE"`. Auto-activated for `ignore_dpp` / non-DPP
  solves; since 2026-07 it is also documented as explicitly user-selectable
  (`canon_backend="DIFFENGINE"`, experimental) on the DPP path, where the chain and the
  `DiffengineConeProgram` are cached across solves (no `EvalParams` in the chain).

## Deep dive: why `log_det(P)` with a parameter `P` is conic, not a constant

This is the most surprising consequence, surfaced by
`test_dpp.py::test_log_det_with_parameter_ignore_dpp`. Consider:

```python
P = cp.Parameter((2, 2)); P.value = np.eye(2)
obj = cp.sum_squares(x + y) - 2 * cp.log_det(P)
```

Intuitively `P` has a value, so `log_det(P)` "should" just be the constant `0.0`. It is
not, and here is why.

### A `Parameter` is a symbolic placeholder, not a constant

A `cp.Parameter` deliberately models *a value that varies between solves*. For curvature
and canonicalization it is treated as **affine, not constant**, so the canonicalization can
be cached and reused when only `P.value` changes. Setting `P.value` does **not** make the
canonicalizer fold it away. The only thing that substitutes the value into the tree is the
`EvalParams` reduction:

- `cvxpy/reductions/eval_params.py:8-16` — `replace_params_with_consts` walks the tree and
  replaces each `Parameter` leaf with `Constant(expr.value)` (line 16).

### How `log_det(P)` canonicalizes (LDL → PSD + exponential cones)

`cvxpy/reductions/dcp2cone/canonicalizers/log_det_canon.py` reformulates `log_det(A)` via an
LDL-style factorization:

- introduce an upper-triangular `Z` and its diagonal `D` (lines 69-71),
- impose the LMI `[[D, Z], [Zᵀ, A]] ⪰ 0` — **one PSD cone** (lines 72-73:
  `X = bmat([[D, Z], [Z.T, A]])`, `constraints = [PSD(X)]`),
- bound `tᵢ ≤ log(Dᵢᵢ)` via `log_canon`, which emits **one exponential cone per diagonal
  entry** (line 75).

For a 2×2 `P` this is exactly **1 PSD cone + 2 exponential cones** — the cones named in the
`SolverError` that OSQP raises. The parameter `P` enters as the **bottom-right block** of the
PSD matrix, i.e. `P`'s entries become parameter coefficients of that constraint's affine map.
Because the constraint is conic, a QP-only solver (OSQP) cannot accept it.

### Why `EvalParams` made it solvable by a QP solver

Under the old path, `EvalParams` runs first and rewrites `log_det(P)` → `log_det(Constant(eye))`.
Then `Dcp2Cone` short-circuits constant subtrees:

- `cvxpy/reductions/dcp2cone/dcp2cone.py:122-125`:
  `if expr.is_constant() and not expr.parameters(): return expr, []` — **no cones emitted**.

The atom's `numeric` then folds `slogdet(eye) = 0.0`, the objective collapses to
`sum_squares(x + y)`, a pure QP that OSQP solves.

### What the DIFFENGINE path does instead

With `ignore_dpp=True` and the DIFFENGINE backend, `P` stays symbolic, so `log_det(P)`
canonicalizes into the full conic program above. `DiffengineConeProgram.apply_parameters`
re-extracts the `A`/`b` matrices from the current `P.value` on each solve (that is the whole
point of the backend — fast parametric re-solves without re-canonicalizing). The conic
structure is preserved, so a conic solver (CLARABEL/SCS) is required.

## Consistency note: the new behavior matches the *default* DPP path

There is a sibling test, `test_dpp.py::test_log_det_with_parameter` (no `ignore_dpp`), whose
docstring states that the **default DPP solve must use a conic solver (SCS)** and that *"QP
solvers cannot handle"* the problem because canonicalization preserves the parameter.

So keeping `log_det(P)` conic under `ignore_dpp` is in fact **more consistent** with the
default DPP path. The old `EvalParams` behavior was the special case — it only worked by
folding the parameter away, which contradicts what `ignore_dpp` is otherwise for (fast
re-solves with changing parameters).

## Resolution: no folding; EvalParams only for N-D (2026-07)

`FoldVariableFreeParams` (which folded variable-free parametric composites like
`log_det(P)`, `norm(p)` to constants before canonicalization) is **deleted**, and
`EvalParams` survives in exactly one niche. The non-DPP / `ignore_dpp` branch of
`solving_chain.py` now:

- **≤2-D + parameters** → `canon_backend = DIFFENGINE`, no reduction prepended. Parameters
  stay symbolic end to end.
- **>2-D + parameters** → `EvalParams` + tensor-backend selection (the diff engine is 2-D
  only; this preserves the pre-DIFFENGINE semantics for backwards compatibility — see
  `test_einsum.py::test_einsum_ignore_dpp_avoids_diffengine`).
- **explicit non-DIFFENGINE `canon_backend` + parameters (≤2-D)** → `ValueError` (the
  opt-out that restored baking semantics is gone). Parameter-free problems honor the
  requested backend.

### Soundness: parametric constants must not be canonicalized on this path

Removing the fold exposed a correctness constraint, not just a performance one. `Dcp2Cone`
applies an atom's graph implementation (epigraph substitution) whose direction is only valid
where DCP-with-params-affine curvature says it is. A parametric-constant composite can sit in
a position that is DCP-valid *only because parameters count as constants* — e.g.
`x <= power(t, 2)` with parameter `t`. Epigraph-canonicalizing that (`x <= s, s >= t**2`) is
**vacuous** — the problem becomes unbounded. DQCP bisection generates exactly these
subproblems (`ceil`/`floor` lowering), so this is not a corner case.

The fix lives at two altitudes:

1. `Dcp2Cone(canon_param_constants=False)` on the non-DPP / `ignore_dpp` branch
   (`dcp2cone.py::canonicalize_expr`): parametric-constant subtrees keep their original atom
   instead of receiving a graph implementation. On the DPP path (`canon_param_constants=True`,
   the default) behavior is unchanged — DPP compliance guarantees the graph implementation is
   sound there, and the tensor backends require it (e.g. DPP `log_det(P)` in an objective is
   *made* affine-in-parameter by its cone reformulation).
2. The diff-engine converter (`converters.py::convert_expr`) turns a variable-free
   subtree whose atom has no symbolic converter (e.g. `floor(t)`) into a lazy synthetic
   parameter (`helpers.SyntheticParams`) re-evaluated in Python on every parameter
   update; supported atoms (e.g. `power`) stay symbolic and are re-evaluated by the
   engine.

### Caching on this path (2026-07)

The compiled `DiffengineConeProgram` is now **cached across solves** for parametric
non-DPP / `ignore_dpp` (≤2-D) problems; `apply_parameters` refreshes values through the
cached program. `safe_to_cache` in `cvxpy/problems/problem.py` disables caching in exactly
two recorded cases:

1. `SolvingChain.uncached_param_prog` — the N-D `EvalParams` fallback (baked values must
   refresh via re-canonicalization);
2. `InverseData.param_values_consumed` — set by `Dcp2Cone` when the cone quad_form canon
   fires with parametric `P` (`decomp_quad(P.value)`, the only value-consuming
   canonicalizer per the exhaustive audit).

What made everything else cacheable (see `TODO.md` §3 for the full account):
the quad-path `quad_over_lin` denominator was already symbolic (`eye/y`); a SparseDiffEngine
fix propagates `needs_parameter_refresh` into composite `param_source` subtrees (they
previously served stale values on re-use); the engine now owns registered parameter nodes
(a node referenced only by the registration list used to dangle); and the converter's
variable-free fallback became **lazy synthetic parameters** (`helpers.SyntheticParams`) —
engine parameter nodes re-evaluated from current `Parameter` values on every
`C_problem.update_params`, instead of baked constants. DQCP bisection subproblems
(`floor(t)`) now cache across iterations.

### QP solvers and the numpy remedy

Because nothing folds `log_det(P)` / `norm(p)` anymore, they keep their cones (consistent
with the default DPP path), so QP solvers raise `SolverError` for such problems under
`ignore_dpp`. Users who want a QP compute the value themselves:

```python
log_det_val = np.linalg.slogdet(P.value)[1]   # instead of cp.log_det(P)
```

See `test_dpp.py::test_log_det_with_parameter_ignore_dpp_qp_solver_raises`.

### No fallback (fail loud)

There is **no fallback** to full parameter baking. If the diff engine cannot convert an atom
with variables that survives symbolically (e.g. a parametric `kron` was one), the solve
raises (`NotImplementedError` from `diff_engine/converters.py` / `registry.py`). Variable-free
subtrees never raise — they become lazy synthetic parameters re-evaluated per solve. The new
`ValueError` (explicit-backend parametric) is the loud edge of the removed opt-out; N-D
parametric problems keep the `EvalParams` fallback for backwards compatibility.

### Test updates

- `test_diffengine_backend.py` was renamed to **`test_ignore_dpp.py`**; it pins the
  explicit-backend `ValueError`, the N-D EvalParams fallback, the epigraph-soundness case
  (`x <= power(t, 2)`), and the numeric fallback (`floor(p)`).
- `test_dpp.py::test_log_det_with_parameter_ignore_dpp` — CLARABEL solves and refreshes;
  the OSQP variant asserts `SolverError` + demonstrates the `np.linalg.slogdet` remedy.
- `test_complex_dpp.py`, `test_parametric_bounds.py`, `test_quad_dpp.py` — assertions on the
  deleted classes became name-string regression guards.
- `test_atoms.py::test_convolve` — dropped its explicit CPP backend (now a `ValueError` for
  non-DPP parametric problems); solves via DIFFENGINE.

## TODOs / follow-ups

Tracked work left after the variable-free-composite-folding change. Grouped by kind. The two
hard failures (`test_mip_vars::test_miqp_warning`, `test_cone2cone::test_mi_socp_2`) are
**pre-existing and unrelated** (no mixed-integer solver installed → weak `ECOS_BB` fallback).

### A. Diff-engine converter gaps (the remaining `xfail`ed tests)

Keeping bare parameters symbolic means parametric atoms now reach the diff-engine converter
instead of being baked away. The remaining features are unimplemented; each `xfail` test is the
acceptance test — implementing the feature should make it pass (remove the marker).

1. **Parametric divisor** — division by a parameter. ✅ **Implemented.** `convert_div`
   (`diff_engine/registry.py`) builds a `make_power(d, -1)` reciprocal node, re-evaluated each
   solve, and routes it through the same scalar/vector `param_mult` dispatch as the constant
   path. Surfaced by complex division by a `Parameter` (`Complex2Real` rewrites `z / c` into
   `z·conj(c)/|c|²`, a variable-free composite divisor that reaches the engine because
   `EvalParams` runs before `Complex2Real`). Tests now passing:
   `test_complex.py::test_div_complex_divisor`, `test_dpp.py::TestDcp::test_quad_over_lin`.

2. **`kron`** ✅ **Implemented** as a native engine atom (`new_kron` /`make_kron` in
   SparseDiffEngine, `convert_kron` in `registry.py`). Every kron output entry depends on a single
   child entry, so the output Jacobian is the child Jacobian's rows gathered (with repetition) and
   scaled by the variable-free operand — no coefficient matrix, `O(nnz(result))`. Handles
   `kron(param/const, var)` and `kron(var, param/const)`. Tests now passing:
   `test_kron_canon.py::TestKronRightVar::test_gen_kronr_param`,
   `TestKronLeftVar::{test_gen_kronl_param, test_scalar_kronl_param, test_symvar_kronl_param}`.

3. **>2-D expressions** — `einsum` builds a 3-D constant; the converter's `Constant` base case
   assumes ≤2-D.
   - Raises: `ValueError: too many values to unpack (expected 2, got 3)`
     (`diff_engine/converters.py:178`, via `normalize_shape(expr.shape)`).
   - Needs: >2-D shape handling in the converter (the engine is 2-D today), or an explicit,
     clearer "diff engine: >2-D not supported" error.
   - Test: `test_einsum.py::TestEinsum::test_einsum_solve` (xfail).

### B. Code cleanups in this change

4. ✅ **Resolved by deletion (2026-07).** The fold (`FoldVariableFreeParams`) was removed
   outright; `EvalParams` survives only for the N-D fallback. The uncached-chain coupling
   is now an explicit `SolvingChain.uncached_param_prog` flag consumed by `safe_to_cache`
   (see "Resolution: no folding; EvalParams only for N-D" above).

### C. Pre-existing / edge-case correctness

6. ✅ **Resolved (2026-07).** The `problem_form.eval_params` exclusion machinery was deleted
   along with the fold: cone analysis now always counts parametric subtrees' cones
   (conservative for atoms the diff engine evaluates numerically, but never under-counting).

### D. Deferred optimization

7. **Reuse Jacobian/Hessian sparsity across per-solve capsule rebuilds.** Because the
   non-DPP / `ignore_dpp` chain is uncached (`uncached_param_prog`), the DIFFENGINE
   `C_problem` is rebuilt each solve. Changing parameter values never changes the `A`/`P`
   sparsity, so the sparsity pattern is invariant and could be reused. This needs a new **sparsediffpy C
   binding** (`problem_init_jacobian_coo_from` / Hessian equivalent) to initialize a fresh
   capsule's derivative structures from a cached COO pattern — `_sparsediffengine` currently
   exposes `problem_init_jacobian_coo` (computes the pattern) and `get_*_sparsity_coo` (reads
   it) but nothing to *set* a known pattern. Store the pattern on the persistent
   `ConeMatrixStuffing` instance. Best tackled alongside the gaps in section A.
