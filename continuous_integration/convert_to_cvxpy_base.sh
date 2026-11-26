#!/usr/bin/env bash
# Rewrite pyproject.toml and setup.py in place so the build produces the
# cvxpy-base distribution: renamed, and without the default solver dependencies.
# `sed -i.bak` (rather than `sed -i`) keeps this portable to macOS sed.
set -euo pipefail

sed -i.bak -e 's/name = "cvxpy"/name = "cvxpy-base"/g' pyproject.toml
sed -i.bak '/clarabel >= /d' pyproject.toml
sed -i.bak '/osqp >= /d' pyproject.toml
sed -i.bak '/scs >= /d' pyproject.toml
sed -i.bak '/highspy >= /d' pyproject.toml
sed -i.bak -e 's/name="cvxpy",/name="cvxpy-base",/g' setup.py
rm -f pyproject.toml.bak setup.py.bak
