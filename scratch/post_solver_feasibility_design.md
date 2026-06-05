# Post-solver feasibility checks design note

## Goal

Explore a small post-solver validation layer for CVXPY. The first API direction considered here is a raw scalar violation value computed from checks on the original problem after solution values are unpacked. The first pass should cover primal constraint violation, and audit whether dual feasibility and complementarity can be included cleanly.

## Main design question

The main question is the API surface:

* Should users call an explicit method after solve?
* Should users enable it through a solve flag?
* Should the violation value be stored somewhere after solve?
* Should the check be lazy or eager?

## Option A: explicit lazy method

```python
prob.solve()
violation = prob.violation()
```

Pros:

* No overhead unless requested.
* Clear user intent.
* Avoids changing default `solve()` behavior.
* Keeps the first API focused on reporting a raw violation value.

Cons:

* Users need to know the method exists.
* The value may become stale if variable values are manually changed after solve.

## Option B: optional solve flag

```python
prob.solve(check_feasibility=True, feasibility_tol=1e-6)
```

Pros:

* Convenient.
* Runs immediately after values are unpacked.
* Could warn when violation exceeds tolerance.

Cons:

* Adds solve-time overhead.
* Needs careful behavior for inaccurate/infeasible statuses.
* Warning behavior needs design.

## Option C: stored violation value

```python
prob.solve()
prob.violation_value  # stored value, name TBD
```

Pros:

* Easy to inspect after solve.
* Keeps the value attached to the problem.

Cons:

* If eager, it adds overhead to every solve.
* If lazy, it needs cache/staleness rules.
* Could be confused with `solver_stats`, even though this is CVXPY-side validation rather than solver-returned metadata.

## Current preference

Start with a lazy explicit method returning a raw scalar violation value:

```python
prob.solve()
violation = prob.violation()
```

Reason:

* Lowest risk.
* No default overhead.
* No new default warnings.
* Separates raw post-solver validation values from tolerance-based feasibility decisions.
* Easier to extend later into warning behavior or a solve flag.

## Minimal first API

For the first public API, prefer exposing a scalar violation value rather than a full feasibility report:

```python
violation = prob.violation()
```

This should be interpreted as a CVXPY-side post-solver validation value computed from the original problem after solution values are unpacked. The initial implementation should define exactly which checks are included. It should not be framed as an infeasibility certificate.

Tolerance-based behavior, warnings, or an `is_feasible` boolean can be considered later once the raw violation definition is clear.

## Definition of violation

Working definition:

```text
Assuming each constraint violation is first normalized to a scalar,
prob.violation() would return the maximum scalar violation across
all original problem constraints after solution values are unpacked.
```

Initial audit: `Constraint.violation()` is documented as the distance from the current expression value to the constraint domain. Before building a problem-level API, the main open question is whether each individual constraint violation should first be normalized to a scalar with a clearly documented norm/distance convention. Once each constraint reports a scalar consistently, `prob.violation()` can aggregate those scalar values directly.

This API should report post-solver validation against the original problem data. It should not be presented as an infeasibility certificate or as a substitute for solver certificates such as a Farkas ray.

Constraint types to check:

* Equality constraints.
* Inequality constraints.
* PSD constraints.
* SOC constraints.
* Exponential cone constraints.
* Power cone constraints.
* Quantum constraints using quadrature approximation.
* Whether each `constraint.violation()` implementation can return a scalar consistently.
* What norm/distance convention each scalar violation uses.

## Benchmark plan and initial result

Benchmark feasibility-check overhead in two cases.

### Case 1: vectorized constraint

```python
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x >= 0])
prob.solve()
violation = prob.violation()
```

### Case 2: poorly vectorized many-scalar-constraint problem

```python
x = cp.Variable(n)
constraints = [x[i] >= 0 for i in range(n)]
prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), constraints)
prob.solve()
violation = prob.violation()
```

Measure:

* Solve time.
* Feasibility check time.
* Scaling with `n`.
* Whether eager checking would be too expensive for poorly vectorized problems.

### Initial local result

A small local benchmark showed the many-scalar-constraint case is much slower than the vectorized case.

For `n = 100`:

* Vectorized constraint check time: `0.000033s`
* Many scalar constraints check time: `0.000574s`
* Scalar/vectorized ratio: `17.5x`

For `n = 1000`:

* Vectorized constraint check time: `0.000027s`
* Many scalar constraints check time: `0.005542s`
* Scalar/vectorized ratio: `203.7x`

The absolute times are small in this toy benchmark, but the scaling difference supports making the first public API lazy or explicitly opt-in rather than eager by default.

## Proposed first PR

A first PR could stay small:

* Internal helper to normalize each primal constraint violation to a scalar and compute the max original-constraint violation.
* Audit whether dual feasibility can be checked from unpacked `constraint.dual_value` values.
* Audit whether complementarity can be evaluated cleanly for the main constraint classes.
* Tests for a solved feasible problem.
* Tests for manually corrupted variable values.
* Tests covering scalar normalization behavior across core constraint classes.
* No public API until the violation definition and API direction are agreed on.

Once the violation definition and API direction are agreed on, expose something like:

```python
prob.violation()
```

Possible follow-up work could add warning behavior through a solve flag:

```python
prob.solve(check_feasibility=True, feasibility_tol=...)
```

## First-pass scope

Based on the design discussion, the first pass should audit and, where clean, include the checks that can be evaluated directly from unpacked primal and dual values in the original problem space:

* **Primal constraint violation:** Evaluate the original high-level `Constraint` objects using the unpacked primal values.
* **Dual feasibility:** Check whether the unpacked `constraint.dual_value` satisfies the expected dual-domain conditions for each constraint type.
* **Complementarity:** Check primal-dual complementarity where it can be evaluated cleanly for a given constraint type.

The next step is to audit the main constraint classes, including equality, inequality, PSD, SOC, exponential cone, power cone, and quantum/quadrature constraints, to see which of these checks can consistently return scalar values.

## Deferred scope: duality gap and stationarity

Duality gap and stationarity should stay out of the first pass. They require more care in the original problem space, especially for reductions, functional convex problems, subgradients, and solver-specific dual information.
