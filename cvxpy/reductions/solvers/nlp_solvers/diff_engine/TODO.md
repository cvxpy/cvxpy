# DIFFENGINE (`ignore_dpp` / non-DPP path) — follow-up work

Action list for the diff-engine backend (see `IGNORE_DPP_BEHAVIOR.md` for the design and
rationale, §7 below for the cvxcore-tracker replication status). Status as of 2026-07: the
kron/perf work (§1a, §2) is done and pushed; the fold is removed
(`FoldVariableFreeParams` deleted; `EvalParams` survives only for the N-D fallback — see
`IGNORE_DPP_BEHAVIOR.md`, "Resolution: no folding; EvalParams only for N-D"); re-solve
caching is parked pending a sound value-consumption detector (§3); what remains is small
cleanups (§4) and the release/merge checklist (§5). §6 collects possible-but-not-planned
performance follow-ups. The `einsum` gap (§1c) is closed: >2-D problems fall back to
EvalParams + tensor backends, pinned by
`test_einsum.py::test_einsum_ignore_dpp_avoids_diffengine`.

## 1. Missing atom implementations (converter gaps)

These atoms now reach the diff-engine converter (because parameters/structure stay symbolic
on the `ignore_dpp` / non-DPP path) and raise. There is **no fallback** — they fail loud by
design, so each is tracked here with its gating test.

### 1a. `kron` ✅ DONE (native engine atom, sparse-only)
- Implemented as native compiled constructors `new_left_kron`/`new_right_kron`
  (`SparseDiffEngine/src/atoms/affine/kron.c`, bindings `make_left_kron`/`make_right_kron`),
  wired by `convert_kron` in `registry.py`. cvxpy's `kron(A, B)` always has one variable-free
  operand; every output entry depends on a single child entry, so the output Jacobian is the
  child Jacobian's rows **gathered (with repetition) and scaled** by the variable-free operand —
  no coefficient matrix, no matmul, `O(nnz(result))`. Built **sparse-only**: cvxpy passes the
  constant operand's active (nonzero) block indices, so structurally-zero blocks (e.g. the
  off-diagonal blocks of `kron(I, ·)`) are never materialized; a `Parameter` operand gets all
  blocks since its zeros aren't permanent. Re-evaluates the variable-free operand each solve.
- Was: `NotImplementedError: Atom 'kron' not supported`.
- Now passing: `test_kron_canon.py::TestKronRightVar::test_gen_kronr_param`,
  `TestKronLeftVar::{test_gen_kronl_param, test_scalar_kronl_param, test_symvar_kronl_param}`
  (xfail markers removed). The SDP benchmark `sdp_segfault_1132` (param-free but variable-coupled
  `kron(e, reshape(diag(V@G@V.T), ...))`) now canonicalizes instead of erroring — see §2.

### 1b. Parametric divisor — division by a parameter ✅ DONE
- `convert_div` (`diff_engine/registry.py`) now handles a parametric divisor `d` via a
  `make_power(d, -1)` reciprocal node that the engine re-evaluates from the current parameter
  value each solve, then routes it through the same scalar/vector `param_mult` dispatch as the
  constant path. The divisor is variable-free, so it is the "parameter" side of the product,
  exactly like the constant operand in `convert_multiply`.
- Surfaced by complex division by a `Parameter`: `Complex2Real` rewrites `z / c` into
  `z·conj(c)/|c|²`, a variable-free composite divisor that `EvalParams` does not fold (it runs
  before `Complex2Real`), so it reaches the diff engine.
- Gating tests (now passing): `test_complex.py::TestComplex::test_div_complex_divisor`,
  `test_dpp.py::TestDcp::test_quad_over_lin` (xfail marker removed).

### 1c. >2-D expressions (e.g. `einsum`) ✅ DONE (fallback, not converter support)
- >2-D problems never reach the converter: the ignore_dpp/non-DPP path detects
  `problem._max_ndim() > 2` and falls back to EvalParams + tensor-backend selection
  (mirroring the CPP N-D treatment, kept for backwards compatibility); explicit
  `canon_backend="DIFFENGINE"` raises a clear ValueError. `normalize_shape` fails loud
  on >2-D as a backstop.
- Tests: `test_einsum.py::test_einsum_ignore_dpp_avoids_diffengine`,
  `test_ignore_dpp.py::{test_nd_problem_falls_back_like_cpp,
  test_nd_problem_explicit_diffengine_raises}`. No xfail markers anywhere.
- Native >2-D converter support would only matter if the engine itself goes N-D; not
  planned.

## 2. Still-slow problems ✅ RESOLVED (2026-07)

Every previously-slow benchmark is now at or faster than CPP. Cold first-compile (s),
`solver=CLARABEL`, `d/CPP` = DIFFENGINE / CPP (>1 = diffengine slower). Suite run:
`benchmarks/results_kron_sparse_v3.txt` (single cold compile, noisy ±10-15%; the
OptimalAdvertising suite row read 1.58× under machine contention — a clean re-run gives 0.86×).

| benchmark                         | vars    | CPP   | DIFFENG | d/CPP | was    |
|-----------------------------------|---------|-------|---------|-------|--------|
| finance.CVaRBenchmark             | 769     | 37.4  | 35.8    | 0.96x | 0.85x  |
| gini_portfolio.Murray             | 245,400 | 18.8  | 7.3     | 0.39x | 2.26x  |
| optimal_advertising.OptimalAdv.   | 250,000 | 6.0   | 5.2     | 0.86x | 0.81x  |
| sdp_segfault_1132                 | 9,801   | 114.6 | 15.7    | 0.14x | 0.47x  |
| quantum_hilbert_matrix            | 4,096   | 3.8   | 1.6     | 0.43x | 17.19x |
| simple_QP.UnconstrainedQP         | 18      | 9.3   | 0.08    | 0.01x | 3.79x  |

(quantum/UnconstrainedQP DIFFENG numbers from clean isolated runs; issue #2205's original
script now runs end to end in ~1.2s where it reported ~12s, and the N_t=2 case that took
~200s runs in ~6s — remaining time is cvxpy *expression construction*, not canonicalization.)

Four fixes, by layer (see also `test_cvxcore_issues.py` and §7):

1. **Sparse-only kron atom** (engine + `convert_kron`): `new_left_kron`/`new_right_kron` take
   the constant operand's active block indices; `kron(I_8, ·)` materializes 32k rows, not 262k.
   See §1a.
2. **Output-driven `block_left_multiply_fill_sparsity`**
   (`SparseDiffEngine/src/utils/linalg_sparse_matmuls.c`): was O(n_vars × n_blocks × A_rows)
   via per-row `has_overlap` rescans; now gathers result rows from a one-time CSC view of A
   with generation-stamp dedup, byte-identical output. Same treatment for
   `csr_csc_matmul_alloc` (bivariate matmul / convolve init).
3. **Lazy constant matmul operands** (`converters.py::convert_expr`): plain-constant matmul
   operands are no longer densified into `make_parameter` nodes that `convert_matmul` then
   discards (it reads the scipy value directly). This was ~25s of quantum's 30s.
4. **Matmul chain normalization** (`converters.py::_normalize_matmul`): reassociates
   `A @ E @ B @ x` so plain constants fold together, pushes through `NegExpression`, and
   distributes a constant *vector* over `AddExpression` — collapsing matrix-valued
   intermediate Jacobians to vectors. This was UnconstrainedQP's entire 35s (one
   `problem_eval_jacobian_vals` call over dense 63k-row intermediates).

To reproduce: `BENCH_ONLY="QuantumHilbertMatrix,UnconstrainedQP,SDPSegfault1132Benchmark,Murray"
BENCH_ITERS=1 .venv/bin/python benchmarks/run_upstream_benchmarks.py` (BENCH_OUT is resolved
relative to `benchmarks/`).

## 3. Re-solve caching — REVISED 2026-07: no-fold detection was unsound; needs
##    value-consumption detection instead

Status: the duplicate per-solve extraction is FIXED (each `DiffengineConeProgram` records
the parameter vector its matrices were extracted at; `apply_parameters` reuses on exact
match — was ~30% of large-problem re-solve time). The fold itself is now GONE
(`FoldVariableFreeParams` deleted; `EvalParams` survives only for the N-D fallback — see
`IGNORE_DPP_BEHAVIOR.md`, "Resolution: no folding; EvalParams only for N-D"); the chain is
kept uncached via the explicit `SolvingChain.uncached_param_prog` flag.

The previously decided plan — **no-fold detection** (cache whenever the problem contains
no variable-free parametric composite) — was **discovered to be unsound** while removing
the fold: canonicalization of non-DPP problems can consume parameter values even with no
foldable composite present. Concretely, `quad_over_lin_canon.py` (quad path) divides by
`y.value` for a bare-parameter denominator, and constraint `quad_form`s bake numeric
Cholesky factors of `P.value`. Caching such a program serves stale data
(`test_dpp.py::test_quad_over_lin` catches the quad_over_lin case).

A future caching scheme must detect **any** parameter-value consumption during
canonicalization/conversion — candidate design: a scope/flag flipped wherever a
reduction or the converter reads `Parameter.value`, with `safe_to_cache` honoring it.
Until then, parametric non-DPP / ignore_dpp solves rebuild the capsule per solve.

### The three sites where parameters do NOT reach the engine symbolically

The exhaustive list (as of 2026-07) of where a parameter's *value* is consumed before the
diff engine sees the tree. These are simultaneously (a) what a value-consumption detector
must catch, and (b) the work-list for widening cacheability — each one made symbolic
removes a caching blocker.

1. **Constraint `quad_form(x, P)` — numeric factorization.** FUNDAMENTAL.
   `dcp2cone/canonicalizers/quad_form_canon.py:28` calls `decomp_quad(args[1].value)`
   (marked `# TODO this doesn't work with parameters!`) and embeds the factor as
   `Constant(M.T)` rows of the SOC data. Cannot go symbolic in this architecture: the
   conic form needs literal factor rows, and even the factor's shape/branch (PSD vs NSD,
   rank) depends on the values. Per-solve re-canonicalization is the correct behavior.
2. **Quad-objective-path `quad_over_lin(x, p)` — denominator evaluated.** FIXABLE.
   `dcp2cone/canonicalizers/quad/quad_over_lin_canon.py:33` builds the SymbolicQuadForm
   matrix as `eye/y.value`. Not fundamental: the engine already represents parametric
   reciprocals (`make_power(d, -1)`, see §1b) and the sparsity of `I/p` is
   value-invariant. Fix = keep a parametric denominator symbolic when stuffing targets
   DIFFENGINE. Gating test: `test_dpp.py::test_quad_over_lin` (stale-cache canary).
3. **Converter numeric fallback for variable-free atoms.** FIXABLE PER-ATOM.
   `diff_engine/converters.py::convert_expr` (the `not expr.variables()` branch)
   evaluates `expr.value` for variable-free subtrees whose atom has no symbolic
   converter — `floor(t)` from DQCP bisection, and composites left intact by
   `Dcp2Cone(canon_param_constants=False)` (raw `log_det(P)`, `norm(p)`, ...). Each new
   engine atom converter shrinks this set. Gating tests:
   `test_ignore_dpp.py::{test_unsupported_variable_free_atom_evaluates,
   test_param_constant_in_concave_position_sound}`.

Measured ceiling (re-run 2026-07-06 with the fixed runner — fresh benchmark instance per
strategy; earlier `eval_params`/Table-2 numbers were invalid because strategies shared a
Problem whose chain cache key excludes canon_backend/is_dpp. `de_cached` =
`canon_backend="DIFFENGINE"` on the DPP path = what ignore_dpp would become if a sound
caching scheme lands; results file since deleted from `benchmarks/`):

| benchmark                    | dpp    | rebuild (today) | de_cached |
|------------------------------|--------|-----------------|-----------|
| FactorCovarianceModel        | 0.163  | 0.913           | 0.298     |
| ConvexPlasticity             | 2.04   | 2.11            | 2.05      |
| ParamConeMatrixStuffing      | 0.0028 | 0.133           | 0.0009    |
| ParamSmallMatrixStuffing     | 0.0047 | 0.195           | 0.0016    |
| SimpleScalarParametrizedLP   | 0.315  | 1.218           | 0.243     |

Rejected alternatives: pseudo-parameter placeholders (sign/attribute propagation
complexity); the pattern-reuse engine API — prototyped end to end as
SparseDifferentiation/SparseDiffEngine#103 (`jacobian-pattern-reuse`:
`problem_init_jacobian_coo_from` + Hessian analog, tested, 399 engine tests pass), then
PARKED: profiling a 1M-var LP rebuild showed it recovers only ~45ms of ~350ms (~13%) —
per-node `jacobian_init` and expression conversion, the bulk of the rebuild, cannot be
skipped by a final-pattern API since evaluation needs the per-node structures, and once
the capsule is cached there is no rebuild to accelerate. SparseDiffPy bindings were
deliberately not written. Revisit only if a rebuild-heavy workload (variable-free
composites + tight re-solve loop) shows up after no-fold detection lands.

## 4. Smaller code cleanups (from `IGNORE_DPP_BEHAVIOR.md`)

- ~~Rename `EvalParams`~~ / ~~comment the `safe_to_cache` coupling~~ /
  ~~`problem_form.eval_params` exclusion~~ — all RESOLVED 2026-07 by deleting the fold
  (`EvalParams` survives only for the N-D fallback; explicit
  `SolvingChain.uncached_param_prog` flag; `problem_form` eval_params machinery removed;
  see `IGNORE_DPP_BEHAVIOR.md`).
- **Drop the unused `sparse_dot_offset`** in `SparseDiffEngine/src/utils/
  linalg_sparse_matmuls.c` (pre-existing `-Wunused-function` warning; nothing calls it).

## 5. Release / merge checklist

Ordered — the cvxpy branch does not work against the released sparsediffpy.

1. **Merge the engine kron PR** (SparseDifferentiation/SparseDiffEngine#101, branch
   `kron-native-atom`: sparse-only left/right kron + output-driven matmul sparsity fills,
   411 C tests) and the **bindings branch** (SparseDiffPy `kron-binding`:
   `make_left_kron`/`make_right_kron`, submodule bump).
2. **Release sparsediffpy 0.6.0** and **bump the cvxpy pin** — `pyproject.toml` still says
   `sparsediffpy >= 0.5.0, < 0.6.0`, but `convert_kron` needs the 0.6.0 bindings. On the
   PyPI 0.5.0 wheel, `test_kron_canon.py -k param` fails with
   `AttributeError: no attribute 'make_left_kron'`. Until the release, install the dev build:
   `uv pip install --python .venv/bin/python --force-reinstall --no-deps ~/Documents/SparseDiffPy`.
3. **Merge the cvxpy branch** (`diffengine-backend-ignoredpp`). Call out the user-facing
   behavior changes in the PR description: (a) `ignore_dpp=True` + explicit CPP/SCIPY
   backend raises for parametric ≤2-D problems (the baking opt-out is gone; N-D problems
   keep the EvalParams fallback for backwards compatibility); (b) QP solvers raise `SolverError`
   for `log_det(P)`/`norm(p)`-style composites under ignore_dpp — remedy: evaluate with
   numpy (`np.linalg.slogdet(P.value)`). Then close out the tracker items per §7: comment/close cvxpy#2205 (fixed, benchmark + regression test),
   reference the diffengine numbers on #1132 and #1611, and post the drafted closing
   comment on #579 (§7 appendix).
4. Decide whether any of `benchmarks/` (the upstream-suite runner, `results_*.txt`,
   `issue_1611_pickled_problem.py`) should be committed somewhere; it is all untracked today.

## 6. Possible future performance work (not planned)

- **Upstream matmul-chain reassociation.** `_normalize_matmul` fixes the
  `A @ E @ B @ x` association problem for DIFFENGINE only, but CPP pays for the same thing
  (9.3s on `UnconstrainedQP`, same root cause). A backend-agnostic reduction in cvxpy proper
  would benefit every backend — bigger blast radius, needs its own discussion/PR.
- **Row-vector-head mirror** for `_normalize_matmul` (`c @ E @ D` chains). Deliberately
  omitted — no measured workload exercises it; mechanical to re-add (see the note in
  `converters.py`) if one shows up.
- **Frontend expression construction on huge krons.** Issue #2205's largest case
  (`N_r=16, N_t=8`, dim 1280) is no longer canonicalization-bound: `cp.kron` of a dense
  matrix builds millions of scalar subexpressions in the Python frontend before the diff
  engine is ever reached. Any fix lives in cvxpy's expression layer, not this backend.
- **DIFFENGINE on very large mixed problems.** Issue #1611's 270k-var MIQP compiles at CPP
  parity (24.7s vs 22.1s); if that class of problem matters, profile where the remaining
  time goes (likely candidates: per-expression conversion overhead, Jacobian init).

## 7. Discussion #3221 tracker status (was `ISSUES_3221_STATUS.md`, removed from the PR)

Replicated + regression-tested in `cvxpy/tests/test_cvxcore_issues.py`:
- **#1132** SDP kron segfault -> solves, matches CPP (`test_issue_1132_sdp_kron_segfault`);
  full size ~2x faster than CPP.
- **#2205** kron QP minutes-long compile -> fixed (36.0s -> 0.08s;
  `test_issue_2205_kron_unconstrained_qp`).
- **#1043** dictionary-learning segfault -> synthesized structure solves
  (`test_issue_1043_param_scaled_frobenius`).
- **#1611** pickled 270k-var MIQP -> compiles at CPP parity; loader
  `benchmarks/issue_1611_pickled_problem.py` (needs 42 MB attachment, not a checked-in test).

cvxcore-internal, moot once cvxcore is deleted: #2927/#2519 (int-through-double COO indices),
#2537, #1876 (broken `_cvxcore` extension builds), #1690 (cvxcore build asserts).
Not replicable: #1332 (repro repo 404), #579 (2018 Cython proposal -- superseded).

Draft closing comment for #579:

> Closing this as superseded. When this was filed (2018, cvxpy 1.0), compilation was a
> pure-Python bottleneck and Cython was a plausible medium-term accelerator. Since then the
> compile-time story has been addressed structurally instead: the C++ cvxcore backend became
> the default canonicalization engine, the vectorized SciPy and COO backends cover
> large/parametric problems in pure Python, and current work is consolidating
> canonicalization on a compiled sparse differentiation engine (SparseDiffEngine) with the
> goal of retiring cvxcore entirely.
>
> Adding Cython now would introduce a second compiled toolchain (build matrix, wheels,
> debugging) to accelerate exactly the code paths that are being replaced. Nothing here
> rules out revisiting compiled acceleration for a specific hot loop if profiling ever
> points at one, but as a standing initiative this is no longer the direction.
>
> If you hit slow compiles today, please open a fresh issue with a repro -- those are
> actionable individually (see e.g. the recent kron/#2205 work).
