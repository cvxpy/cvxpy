import sys

from cvxpy.reductions.solvers.solver import Solver, module_spec_available


def test_module_spec_available_does_not_import_dotted_module(tmp_path, monkeypatch):
    package = tmp_path / "probe_pkg"
    package.mkdir()
    (package / "__init__.py").write_text("raise RuntimeError('package imported')\n")
    (package / "child.py").write_text("raise RuntimeError('child imported')\n")
    monkeypatch.syspath_prepend(str(tmp_path))

    assert module_spec_available("probe_pkg.child")
    assert "probe_pkg" not in sys.modules
    assert "probe_pkg.child" not in sys.modules


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
