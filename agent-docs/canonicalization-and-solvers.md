<!--
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
-->

# Canonicalization and solvers

[`Problem.get_problem_data()`](../cvxpy/problems/problem.py) applies the
solving chain and returns the data for the selected solver, the chain, and
the inverse data. Inspect this data to test compilation without running
the numerical solver. See [Testing](testing.md#test-at-the-layer-that-owns-the-behavior)
for an example.

## The solving chain

For DCP problems, `Dcp2Cone` lowers expressions using atom canonicalizers.
`CvxAttr2Constr` lowers variable attributes to equivalent variables and,
where needed, explicit constraints. `ConeMatrixStuffing` extracts
the coefficients from the resulting affine expressions. The chain
can then format cone data for the solver, and the solver interface
prepares the data for its numerical API.

The exact chain depends on the problem and the solver.
For example, it can convert complex expressions to real ones before
`Dcp2Cone` or retain a quadratic objective for a solver that accepts one.
A reduction's `apply` returns the transformed problem and the inverse
data needed by `invert` to map a solution back.

[Chain construction](../cvxpy/reductions/solvers/solving_chain.py) ·
[Reduction interface](../cvxpy/reductions/reduction.py) ·
[Matrix stuffing](../cvxpy/reductions/dcp2cone/cone_matrix_stuffing.py) ·
[Solver interfaces](../cvxpy/reductions/solvers/)

## Other problem classes

- For DGP, `solve(gp=True)` uses `Dgp2Dcp` to convert the problem to DCP using
  a log-log transformation. [DGP reduction](../cvxpy/reductions/dgp2dcp/)
- For a DQCP problem outside DCP, `solve(qcp=True)` uses `Dqcp2Dcp` to build
  a family of DCP feasibility problems and uses bisection to find the
  optimal value. [DQCP reduction](../cvxpy/reductions/dqcp2dcp/) ·
  [Bisection](../cvxpy/reductions/solvers/bisection.py)
- For DNLP, `solve(nlp=True)` checks `problem.is_dnlp()` and lowers the
  problem through `dnlp2smooth` for an NLP solver. Nonconvex atoms such
  as `cp.nlp.sin` live in the `cp.nlp` namespace.
  The NLP interfaces include IPOPT, UNO, KNITRO, and COPT.
  [NLP solving path](../cvxpy/reductions/solvers/nlp_solving_chain.py) ·
  [NLP interfaces](../cvxpy/reductions/solvers/nlp_solvers/)

## Disciplined parametrized programming

DPP lets CVXPY reuse a compiled parameterized program when only parameter
values change. In a DCP check under DPP rules, parameters are treated as
affine, and the product rule distinguishes parameter-affine expressions
from expressions that are free of parameters. For example, the product
of a parameter and a variable is DPP, as is a constraint on the norm of
a parameter. The product of two independent parameters is DCP and fails
the DPP check.

```python
import cvxpy as cp

p = cp.Parameter(2)
q = cp.Parameter(2)
x = cp.Variable(2)

assert cp.multiply(p, x).is_dcp(dpp=True)
assert (cp.norm(p) <= 1).is_dcp(dpp=True)
assert cp.multiply(p, q).is_dcp()
assert not cp.multiply(p, q).is_dcp(dpp=True)
```

Use `problem.is_dpp()` to check a problem. Use `enforce_dpp=True` when a
compilation test needs to exercise DPP. For a cache regression, change a
parameter to a value that requires different compiled coefficients and
check the new data.

[DPP tutorial](../doc/source/tutorial/dpp/index.rst) ·
[DPP scope](../cvxpy/utilities/scopes.py)

## Matrix stuffing backends

`CPP` uses the [C++ core](../cvxpy/cvxcore/).
`SCIPY` uses SciPy sparse matrices, `COO` uses a sparse coordinate tensor
suited to large DPP problems, and `RUST` is experimental.
[Python backend entry points](../cvxpy/lin_ops/backends/)

`DIFFENGINE` uses the nonlinear expression engine for conic matrix
stuffing. It requires a parameter-free lowered problem and expressions
of at most two dimensions.
[DiffEngine stuffing](../cvxpy/reductions/solvers/nlp_solvers/diff_engine/cone_stuffing.py)

Pass `canon_backend` to `solve()` or `get_problem_data()` to target a
particular backend. With no explicit backend, the solving chain can select
COO for a DPP problem whose total parameter size reaches
`DPP_PARAM_THRESHOLD` (1,000). During stuffing, problems with expressions
that exceed two dimensions or lack CPP support can fall back to SciPy.
If stuffing still has no backend choice, it reads
`CVXPY_DEFAULT_CANON_BACKEND`. The built-in default is CPP, or SciPy on
Emscripten.

[Chain selection](../cvxpy/reductions/solvers/solving_chain.py) ·
[Stuffing compatibility checks](../cvxpy/reductions/solvers/solving_chain_utils.py) ·
[Backend dispatch](../cvxpy/cvxcore/python/canonInterface.py) ·
[Defaults](../cvxpy/settings.py)
