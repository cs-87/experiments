"""
Minimal test runner: `python -m src.spread_spectrum.tests.run [substring ...]`

pytest is not installed in this environment and the repository has no test
infrastructure, so the suite is written as plain `test_*` functions that pytest would
also collect unchanged if it ever is.
"""

import importlib
import pkgutil
import sys
import time
import traceback

PACKAGE = "src.spread_spectrum.tests"


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pkg = importlib.import_module(PACKAGE)
    names = sorted(m.name for m in pkgutil.iter_modules(pkg.__path__)
                   if m.name.startswith("test_"))
    passed = failed = skipped = 0
    t0 = time.time()
    for mod_name in names:
        mod = importlib.import_module(f"{PACKAGE}.{mod_name}")
        tests = sorted(n for n in dir(mod) if n.startswith("test_"))
        shown = False
        for name in tests:
            if argv and not any(a in f"{mod_name}.{name}" for a in argv):
                skipped += 1
                continue
            if not shown:
                print(f"\n{mod_name}")
                shown = True
            t = time.time()
            try:
                getattr(mod, name)()
            except Exception:
                failed += 1
                print(f"  FAIL {name}")
                print("".join("       " + l for l in
                              traceback.format_exc(limit=3).splitlines(True)))
            else:
                passed += 1
                print(f"  ok   {name}  ({time.time() - t:.2f}s)")
    print(f"\n{passed} passed, {failed} failed"
          + (f", {skipped} not selected" if skipped else "")
          + f" in {time.time() - t0:.1f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
