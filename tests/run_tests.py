"""Run the test suite without third-party test tooling.

Usage::

    python tests/run_tests.py

pytest works too (``python -m pytest tests -q``) because the tests are plain
``test_*`` functions; this runner exists so the suite can be executed in an
environment where pytest is not installed.
"""

from __future__ import annotations

import importlib
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")

for path in (HERE, SRC):
    if path not in sys.path:
        sys.path.insert(0, path)

MODULES = ("test_features", "test_training")


def main() -> int:
    total = 0
    failures: list[tuple[str, str]] = []

    for module_name in MODULES:
        module = importlib.import_module(module_name)
        for name in sorted(vars(module)):
            if not name.startswith("test_"):
                continue
            function = getattr(module, name)
            if not callable(function):
                continue
            total += 1
            label = f"{module_name}::{name}"
            try:
                function()
                print(f"PASS  {label}")
            except Exception:  # noqa: BLE001 - this is a test runner
                failures.append((label, traceback.format_exc()))
                print(f"FAIL  {label}")

    for label, trace in failures:
        print(f"\n--- {label} ---\n{trace}")

    passed = total - len(failures)
    print(f"\n{passed}/{total} tests passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
