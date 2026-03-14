# CVXPY Canonicalization Optimization — Agent Protocol

## Goal
Make CVXPY canonicalization (Problem → solver-ready matrices) as fast as possible.
Metric: geometric mean of `canon_benchmark.py` timings (lower = better).

## Setup (run once per session)
```bash
source .venv/bin/activate
export ENSUE_API_KEY=$(cat .autoresearch-key)
```

## Agent Loop

### 1. RECALL
```python
from coordinator import Coordinator
coord = Coordinator()
coord.analyze()                       # Full summary
coord.ask("recent insights")          # Semantic search
best = coord.pull_best()              # Current best config + diff
hypotheses = coord.list_hypotheses(status="open")  # What to try next
```

### 2. THINK
- Pick the highest-priority open hypothesis, OR
- Identify a new optimization target from profiling/code reading
- Focus on one change at a time — keep diffs minimal

### 3. IMPLEMENT
- Edit files in `cvxpy/lin_ops/backends/` or related hot paths
- Keep changes focused and reversible
- Run `pytest cvxpy/tests/ -x -q` to verify correctness

### 4. BENCHMARK
```bash
# Quick check
python canon_benchmark.py --quick --json

# Full benchmark (before publishing)
python canon_benchmark.py --all-backends --json > /tmp/bench_result.json
```

Verify:
- All problems pass (no errors)
- Check geomean_ms vs baseline

### 5. PUBLISH
```python
import json, subprocess

# Get the diff
diff = subprocess.check_output(["git", "diff"]).decode()

# Load results
with open("/tmp/bench_result.json") as f:
    bench = json.load(f)

# Determine status
status = "keep" if bench["geomean_ms"] < best["geomean_ms"] else "discard"

coord.publish_result(
    "Description of what was changed",
    bench,
    diff,
    status
)

# Record what you learned
coord.post_insight("What we learned from this experiment")

# Propose next steps
coord.publish_hypothesis(
    "Next thing to try",
    "Detailed description of the hypothesis",
    priority=2  # 1=high, 2=medium, 3=low
)

# Update hypothesis status if you tested one
coord.publish_hypothesis("Title of tested hypothesis", "...", status="tested")
```

### 6. REPEAT
Go back to step 1.

## Key Files

| File | Role |
|------|------|
| `cvxpy/lin_ops/backends/base.py` | `build_matrix()`, `process_constraint()`, `flatten_tensor()` |
| `cvxpy/lin_ops/backends/coo_backend.py` | COO backend — best for DPP problems |
| `cvxpy/lin_ops/backends/scipy_backend.py` | SciPy backend — reference implementation |
| `cvxpy/cvxcore/python/canonInterface.py` | CPP backend entry point |
| `cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py` | Canonicalization orchestrator |
| `coordinator.py` | Ensue research coordinator |
| `canon_benchmark.py` | Standardized benchmark harness |

## Optimization Targets (by priority)

### P1 — Hot Path
1. **Parallelize `build_matrix()` constraint loop** — constraints are independent
2. **Optimize `mul` in COO backend** — DPP hot path
3. **Speed up `flatten_tensor()` COO→CSC** — called on every problem

### P2 — Medium Impact
4. **`CooTensor.select_rows()` skip argsort** for pre-sorted inputs
5. **Pre-allocate arrays in `TensorRepresentation.combine()`**
6. **int32 indices** when dimensions fit (saves memory bandwidth)

### P3 — Exploratory
7. **LinOp fusion**: `neg(mul(...))` → single op
8. **Smart backend auto-selection** based on problem structure
9. **Rust backend completion** for tree traversal

## Rules
- One optimization per experiment — isolate effects
- Always verify correctness before publishing
- Record negative results too (status="discard") — they prevent re-trying
- Keep git working tree clean between experiments (stash or revert failed attempts)
