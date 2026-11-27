# CVXPY agent guide

CVXPY requires Python 3.11 or newer. See the [contributor guide](CONTRIBUTING.md)
for the project's code style, license header, and contribution checklist.

## Writing tests

- Start with one small, deterministic case. Parameterize when cases exercise
  distinct behavior, code paths, or boundaries.
- Check expression behavior directly. For canonicalization, inspect the
  reduction or the data from `get_problem_data()`. Use a solve when the
  assertion requires a solver result.
- Give new and substantially revised tests a natural-language docstring.
  Describe what the test sets up, what it exercises, and how it decides
  whether the result is correct.

## Where to start

| Task | Guide |
| --- | --- |
| Set up the checkout, rebuild the C++ core, or open a PR | [Development](agent-docs/development.md) |
| Change expression behavior or add an atom | [Expressions and atoms](agent-docs/expressions-and-atoms.md) |
| Change a reduction, matrix stuffing backend, or solver interface | [Canonicalization and solvers](agent-docs/canonicalization-and-solvers.md) |
| Design or run a regression test | [Testing](agent-docs/testing.md) |
