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

# Development

## Set up the checkout

Use Python 3.11 or newer in a virtual environment. From the repository root,
install CVXPY in editable mode with its test dependencies.

```bash
python -m pip install -e ".[testing]"
```

After changing the C++ core or its SWIG interface, install SWIG and run
`bash rebuild_cvxcore.sh` from the repository root. The script regenerates
the SWIG bindings and reinstalls the editable package.

[Python and dependency requirements](../pyproject.toml) ·
[C++ rebuild script](../rebuild_cvxcore.sh)

## Code style and license

Keep Python lines within 100 characters and imports at the top of the file.
A circular dependency can require a deferred import. Use the CVXPY Apache 2.0
license header in new files. Ruff checks Python style, and the repository
provides pre-commit hooks for running Ruff and checking configuration files.

[License header](../CONTRIBUTING.md#license) ·
[Code style](../CONTRIBUTING.md#code-style) ·
[Pre-commit configuration](../.pre-commit-config.yaml)

## Verify and open a PR

Start with the tests for the changed behavior. See [Testing](testing.md) for
commands and guidance on solver-dependent tests. Run the relevant benchmarks
for changes that can affect performance.

Use the [PR template](../.github/pull_request_template.md). Describe the
behavior that changed and report the checks you ran. Mark only completed
items in the contribution checklist.

[Contributor guide](../doc/source/contributing/index.rst) ·
[Benchmarks](../CONTRIBUTING.md#benchmarks)
