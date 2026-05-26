"""
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
"""

import sys
import warnings

import pytest

from cvxpy.reductions.solvers.openmp_conflict import warn_if_omp_conflict


@pytest.fixture
def darwin_isolated(monkeypatch):
    """Pretend to be macOS with no OMP-bundling packages loaded.

    The conflict warning is gated on ``sys.platform == "darwin"``, so to
    exercise it on Linux/Windows CI runners we monkeypatch the platform.
    """
    monkeypatch.setattr(sys, "platform", "darwin")
    for pkg in ("knitro", "cyipopt", "cvxopt"):
        monkeypatch.delitem(sys.modules, pkg, raising=False)


def test_no_warning_when_alone(darwin_isolated) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # turn any warning into an exception
        warn_if_omp_conflict("knitro")


def test_warns_when_cvxopt_already_loaded(darwin_isolated, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "cvxopt", object())
    with pytest.warns(RuntimeWarning, match=r"knitro.*cvxopt"):
        warn_if_omp_conflict("knitro")


def test_warns_when_knitro_already_loaded(darwin_isolated, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "knitro", object())
    with pytest.warns(RuntimeWarning, match=r"cyipopt.*knitro"):
        warn_if_omp_conflict("cyipopt")


def test_lists_all_loaded_siblings(darwin_isolated, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "knitro", object())
    monkeypatch.setitem(sys.modules, "cyipopt", object())
    with pytest.warns(RuntimeWarning) as record:
        warn_if_omp_conflict("cvxopt")
    msg = str(record[0].message)
    assert "knitro" in msg and "cyipopt" in msg


def test_silent_for_unrelated_package(darwin_isolated, monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "knitro", object())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_omp_conflict("numpy")  # not an OMP-bundling package


def test_silent_on_non_darwin(monkeypatch) -> None:
    """On Linux/Windows the gate short-circuits even when a conflict exists."""
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setitem(sys.modules, "cvxopt", object())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warn_if_omp_conflict("knitro")
