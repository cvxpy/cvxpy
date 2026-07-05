# Discussion #3221 tracker: status on the DIFFENGINE path

Status of every issue referenced by the cvxcore tracker discussion
([cvxpy/cvxpy#3221](https://github.com/cvxpy/cvxpy/discussions/3221)), replayed against the
DIFFENGINE canon backend (`ignore_dpp=True` on the `diffengine-backend-ignoredpp` branch).
Scaled-down regression tests live in `cvxpy/tests/test_diffengine_issues.py`; full-size
timings in `benchmarks/run_upstream_benchmarks.py`.

## Replicated and verified on the diffengine path

| issue | what it was | status on DIFFENGINE |
|---|---|---|
| [#1132](https://github.com/cvxpy/cvxpy/issues/1132) | segfault canonicalizing an SDP (PSD var inside `kron`/`reshape`/`diag` under a fro-norm) at n=300 | Solves. Test: `test_issue_1132_sdp_kron_segfault` (n=25, value matches CPP). Full-size: `sdp_segfault_1132` benchmark — DIFFENGINE was already ~2× faster than CPP. |
| [#2205](https://github.com/cvxpy/cvxpy/issues/2205) | minutes-long compile of an unconstrained QP with `kron(diag(ones), diag(complex_var))` between dense DFT matrices | Fixed (sparse kron + matmul chain normalization — see `TODO.md` §2). `UnconstrainedQP` benchmark: **36.0s → 0.08s** (CPP: 9.3s). The issue's original script: ~12s reported → 1.2s end to end; its ~200s case → ~6s. Test: `test_issue_2205_kron_unconstrained_qp` (recovers the planted error, both `sum_squares` and `norm2` forms). |
| [#1043](https://github.com/cvxpy/cvxpy/issues/1043) | segfault on a dictionary-learning problem (param-scaled squared fro-norms) with real data | Original `.npy` data unavailable; same structure synthesized at reduced scale solves and matches a constant-baked baseline across two parameter combinations. Test: `test_issue_1043_param_scaled_frobenius`. |
| [#1611](https://github.com/cvxpy/cvxpy/issues/1611) | QpMatrixStuffing regression 1.1.7→1.1.18 (compile 8s → 128s) on a shipped pickled problem (270,936 vars, 3,528 booleans, 37 constraints) | The `Problem.prob.zip` shelve unpickles on modern cvxpy with two shims (stub modules for a stale cached solving chain; newer Leaf attribute defaults) — the expression tree itself loads as real cvxpy objects. Rebuilt fresh and compiled (HIGHS target): **CPP 22.1s, DIFFENGINE 24.7s** — the 1.1.18 regression (128s) is long gone and the diffengine is at CPP parity. Loader: `benchmarks/issue_1611_pickled_problem.py` (needs the 42 MB attachment, so not a checked-in test). |

## Nothing to replicate — cvxcore-internal, gone when cvxcore is deleted

| issue | why it evaporates |
|---|---|
| [#2927](https://github.com/cvxpy/cvxpy/issues/2927) / [#2519](https://github.com/cvxpy/cvxpy/issues/2519) | `getI()`/`getJ()` in `ProblemData.hpp` return int indices through `double*` buffers. The diffengine extractor gets COO indices from `_sparsediffengine` as int32 numpy arrays end to end (`diff_engine/extractor.py`); no float round-trip exists on this path. |
| [#2537](https://github.com/cvxpy/cvxpy/issues/2537) | segfault at `import cvxpy` from a broken `_cvxcore` extension build (macOS/Py3.12). sparsediffpy is a separate, scikit-build-core-built extension; the failure mode is specific to the cvxcore/SWIG build. |
| [#1876](https://github.com/cvxpy/cvxpy/issues/1876) | `_cvxcore` DLL load failure in offline installs — same category as above. |
| [#1690](https://github.com/cvxpy/cvxpy/issues/1690) | compiling cvxcore with assertions enabled breaks — cvxcore build system issue. |

## Not replicable

| issue | reason |
|---|---|
| [#1332](https://github.com/cvxpy/cvxpy/issues/1332) | repro repo `kmonson/cvxpy-performance` returns 404; no inline code in the issue. The reported symptom (DPP much slower than non-DPP first solve) is a DPP-tensor-backend property, orthogonal to this path. |
| [#579](https://github.com/cvxpy/cvxpy/issues/579) | not a bug — a 2018 proposal to Cython-accelerate compilation. Recommended for closing as superseded; draft closing comment below. |

## Appendix: draft closing comment for #579

> Closing this as superseded. When this was filed (2018, cvxpy 1.0), compilation was a pure-Python
> bottleneck and Cython was a plausible medium-term accelerator. Since then the compile-time story
> has been addressed structurally instead: the C++ cvxcore backend became the default
> canonicalization engine, the vectorized SciPy and COO backends cover large/parametric problems in
> pure Python, and current work is consolidating canonicalization on a compiled sparse
> differentiation engine (SparseDiffEngine) with the goal of retiring cvxcore entirely.
>
> Adding Cython now would introduce a second compiled toolchain (build matrix, wheels, debugging)
> to accelerate exactly the code paths that are being replaced — the maintenance cost lands where
> the project is trying to shed complexity. Nothing here rules out revisiting compiled acceleration
> for a specific hot loop if profiling ever points at one, but as a standing initiative this is no
> longer the direction. No activity or champion in 7+ years also suggests nobody is blocked on it.
>
> If you hit slow compiles today, please open a fresh issue with a repro — those are actionable
> individually (see e.g. the recent kron/#2205 work).
