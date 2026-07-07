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

import importlib.abc
import importlib.machinery
import sys

from cvxpy.reductions.solvers.solver import Solver, module_spec_available


class _BasePackageFinder(importlib.abc.MetaPathFinder):
    """Expose a top-level package through sys.meta_path without filesystem paths."""

    def __init__(self, package_name):
        self.package_name = package_name

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.package_name:
            return None
        spec = importlib.machinery.ModuleSpec(fullname, loader=None)
        spec.submodule_search_locations = []
        return spec


def test_module_spec_available_does_not_import_dotted_module(tmp_path, monkeypatch):
    package = tmp_path / "probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('package imported')\n")
    (package / "child.py").write_text("raise RuntimeError('child imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert module_spec_available("probe_pkg.child") is True
    assert "probe_pkg" not in sys.modules
    assert "probe_pkg.child" not in sys.modules


def test_module_spec_available_uses_meta_path_for_base_package(monkeypatch):
    package_name = "meta_path_solver_backend"
    finder = _BasePackageFinder(package_name)
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    assert module_spec_available(package_name) is True
    assert package_name not in sys.modules


def test_solver_is_installed_falls_back_when_dotted_module_is_ambiguous(monkeypatch):
    package_name = "ambiguous_solver_backend"
    finder = _BasePackageFinder(package_name)
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    class AmbiguousSolver(Solver):
        REQUIRED_MODULES = (f"{package_name}.child",)
        import_called = False

        def name(self):
            return "AMBIGUOUS_SOLVER"

        def import_solver(self):
            self.import_called = True

        def apply(self, problem):
            raise NotImplementedError

        def invert(self, solution, inverse_data):
            raise NotImplementedError

        def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
            raise NotImplementedError

        def cite(self, data):
            return ""

    solver = AmbiguousSolver()

    assert module_spec_available(f"{package_name}.child") is None
    assert solver.is_installed()
    assert solver.import_called
    assert package_name not in sys.modules


def test_solver_is_not_installed_when_ambiguous_fallback_import_fails(monkeypatch):
    package_name = "missing_ambiguous_solver_backend"
    finder = _BasePackageFinder(package_name)
    monkeypatch.setattr(sys, "meta_path", [finder] + sys.meta_path)

    class MissingAmbiguousSolver(Solver):
        REQUIRED_MODULES = (f"{package_name}.child",)

        def name(self):
            return "MISSING_AMBIGUOUS_SOLVER"

        def import_solver(self):
            raise ModuleNotFoundError("missing")

        def apply(self, problem):
            raise NotImplementedError

        def invert(self, solution, inverse_data):
            raise NotImplementedError

        def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
            raise NotImplementedError

        def cite(self, data):
            return ""

    assert not MissingAmbiguousSolver().is_installed()


def test_solver_is_installed_uses_required_modules(tmp_path, monkeypatch):
    package = tmp_path / "lazy_solver_backend"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('backend imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    class LazySolver(Solver):
        REQUIRED_MODULES = ("lazy_solver_backend",)

        def name(self):
            return "LAZY_SOLVER"

        def import_solver(self):
            raise AssertionError("availability check imported the solver")

        def apply(self, problem):
            raise NotImplementedError

        def invert(self, solution, inverse_data):
            raise NotImplementedError

        def solve_via_data(self, data, warm_start, verbose, solver_opts, solver_cache=None):
            raise NotImplementedError

        def cite(self, data):
            return ""

    assert LazySolver().is_installed()
    assert "lazy_solver_backend" not in sys.modules
