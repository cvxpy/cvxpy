"""
Lightweight runner for cvxpy/tests/test_labels.py without pytest.

Discovers functions named `test_*` in the module and invokes them.
Reports first failure and exits non-zero, or prints OK if all pass.
"""
import importlib
import inspect
import sys
import traceback


def main() -> int:
    mod = importlib.import_module("cvxpy.tests.test_labels")
    tests = [name for name, obj in inspect.getmembers(mod, inspect.isfunction) if name.startswith("test_")]
    failures = []
    for name in tests:
        func = getattr(mod, name)
        try:
            func()
        except Exception as e:  # noqa: BLE001
            failures.append((name, e, traceback.format_exc()))
            break
    if failures:
        name, exc, tb = failures[0]
        print(f"FAILED: {name}: {exc}")
        print(tb)
        return 1
    print(f"OK: {len(tests)} tests ran")
    return 0


if __name__ == "__main__":
    sys.exit(main())

