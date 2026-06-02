# Post-solver feasibility checks design note

## Goal

Explore a small feasibility-check layer for CVXPY that reports whether the original problem constraints are satisfied after solution values are unpacked.

The current local spike uses the existing `constraint.violation()` path after `Problem.unpack()` and computes the maximum violation over the original constraints.

## Main design question

The main question is the API surface:

- Should users call an explicit method after solve?
- Should users enable it through a solve flag?
- Should the report be stored somewhere after solve?
- Should the check be lazy or eager?

## Option A: explicit lazy method

```python
prob.solve()
report = prob.feasibility_report(tol=1e-6)
```

Pros:

- No overhead unless requested.
- Clear user intent.
- Avoids changing default `solve()` behavior.
- Avoids warning users unless they explicitly ask.

Cons:

- Users need to know the method exists.
- Report may become stale if variable values are manually changed after solve.

## Option B: optional solve flag

```python
prob.solve(check_feasibility=True, feasibility_tol=1e-6)
```

Pros:

- Convenient.
- Runs immediately after values are unpacked.
- Could warn when violation exceeds tolerance.

Cons:

- Adds solve-time overhead.
- Needs careful behavior for inaccurate/infeasible statuses.
- Warning behavior needs design.

## Option C: stored report

```python
prob.solve()
prob.feasibility_report
```

Pros:

- Easy to inspect after solve.
- Keeps the report attached to the problem.

Cons:

- If eager, it adds overhead to every solve.
- If lazy, it needs cache/staleness rules.
- Could be confused with `solver_stats`, even though this is CVXPY-side validation rather than solver-returned metadata.

## Current preference

Start with a lazy explicit method:

```python
prob.solve()
report = prob.feasibility_report(tol=1e-6)
```

Reason:

- Lowest risk.
- No default overhead.
- No new default warnings.
- Easier to benchmark.
- Easier to extend later into a solve flag.

## Minimal report

```python
@dataclass
class FeasibilityReport:
    is_feasible: bool
    max_violation: float
```

For the first version, I would avoid adding solver metadata because `SolverStats` already covers solver-returned information. This report should focus only on CVXPY-side validation of the original constraints.

## Definition of violation

Initial definition:

```text
max_violation is the maximum value returned by constraint.violation()
across all original problem constraints after solution values are unpacked.
```

Initial audit: `Constraint.violation()` is documented as the distance from the current expression value to the constraint domain. Existing tests cover violation behavior for equality, PSD, power cone, boolean/inequality-style, and linear cone cases. One open question is aggregation: some constraints return vector residuals, while others return scalar norms, so the feasibility report needs a clear rule for reducing all constraint violations to one `max_violation` value.

Before proposing a public API, the next step is to check the existing coverage report and tests for `constraint.violation()` across the main constraint classes, especially equality, inequality, PSD, SOC, exponential, and power constraints.

Constraint types to check:

- Equality constraints.
- Inequality constraints.
- PSD constraints.
- SOC constraints.
- Exponential/power constraints.
- How vector-valued residuals and scalar norm violations should be aggregated into one report value.

## Benchmark plan and initial result

Benchmark feasibility-check overhead in two cases.

### Case 1: vectorized constraint

```python
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])
prob.solve()
report = prob.feasibility_report()
```

### Case 2: poorly vectorized many-scalar-constraint problem

```python
x = cp.Variable(n)
constraints = [x[i] >= 0 for i in range(n)]
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
prob.solve()
report = prob.feasibility_report()
```

Measure:

- Solve time.
- Feasibility check time.
- Scaling with `n`.
- Whether eager checking would be too expensive for poorly vectorized problems.

### Initial local result

A small local benchmark showed the many-scalar-constraint case is much slower than the vectorized case.

For `n = 100`:

- Vectorized constraint check time: `0.000033s`
- Many scalar constraints check time: `0.000574s`
- Scalar/vectorized ratio: `17.5x`

For `n = 1000`:

- Vectorized constraint check time: `0.000027s`
- Many scalar constraints check time: `0.005542s`
- Scalar/vectorized ratio: `203.7x`

The absolute times are small in this toy benchmark, but the scaling difference supports making the first public API lazy or explicitly opt-in rather than eager by default.

## Proposed first PR

A first PR could stay small:

- Internal helper to compute max original-constraint violation.
- Minimal `FeasibilityReport`.
- Tests for a solved feasible problem.
- Tests for manually corrupted variable values.
- No public API until the design direction is agreed on.

After that, expose either:

```python
prob.feasibility_report(tol=...)
```

or:

```python
prob.solve(check_feasibility=True, feasibility_tol=...)
```
