# Agent conventions for CVXPY

This file records project conventions for AI agents and contributors. Prefer
these over inventing parallel patterns.

## Imports

- Put imports at the **top** of the file.
- **Inline / deferred imports are prohibited** unless required to break a
  circular import. If deferred, add a one-line comment naming the cycle
  (e.g. `# Deferred: debug_tools imports Expression via Power/Constraint`).
- Do not use deferred imports "just in case."

## Typing and defensiveness

- Prefer **`isinstance`** over comparing `type(x).__name__` strings.
- Do not use **`getattr` / `hasattr`** to paper over uncertain types. Pass
  properly typed objects (`Expression`, `Constraint`, …) and use their APIs.
- Add type annotations on new public and internal helpers.
- Do not access private attributes (e.g. `_label`) when a public API exists
  (`format_labeled()`, `name()`, properties).

## Expression / constraint display

- Use existing **`name()`** / **`format_labeled()`** for stringifying
  expressions and constraints. Do not reimplement pretty-printers in
  utilities.
- Atom-specific display choices belong on the **atom class** (e.g. `Power`
  owns `atom_name()` / `name()` for `sqrt` / `square`), not in call-site
  special cases.

## Public API surface

- Prefer **one** way to do a thing. If a capability exists as a method
  parallel to an existing API (e.g. `is_dcp()` → `explain_dcp()`), do not
  also export a duplicate top-level function unless there is a clear need.
- Match existing patterns on `Expression`, `Objective`, `Constraint`, and
  `Problem` rather than inventing a parallel dispatcher.

## Duplication

- Before adding helpers, search for existing logic (`format_labeled`,
  curvature checks on `Atom`, constraint `name()`, etc.).
- Mirroring a rule for *explanation text* (e.g. restating DCP composition
  failures) is acceptable; copying formatting or type-dispatch that already
  lives on the objects is not.

## Other style (see also `CLAUDE.md`)

- Line length 100; ruff via pre-commit.
- Apache 2.0 license header on new files.
- Tests subclass `cvxpy.tests.base_test.BaseTest`; use `solver=cp.CLARABEL`
  when calling `solve()`.
