"""Check that the simulated layers never reach for time or randomness directly.

A deterministic simulation is only as good as its weakest module. One direct
`time.time()` inside the sync engine and an entire run stops being a function
of its seed — and the failure is silent: the suite still passes, the
simulation still runs, and a bug found at seed 8,271,443 simply refuses to
reappear when somebody tries to reproduce it.

That is worth a lint rather than a convention, so this is one. It fails the
build if any module that the simulator drives reads the clock or rolls a die
without going through `aegis.core.determinism`.
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

# Everything the deterministic simulator drives.
# `ids.py` is in this list because an operation id is not cosmetic: it is what
# the IBLT hashes and what a fetch is sorted by. While ids came from the wall
# clock, two sweeps over the same 500 seeds returned 454 and 456 failures.
SIMULATED = ("aegis/sync/", "aegis/core/clock.py", "aegis/core/determinism.py",
             "aegis/core/ids.py")
# Reading a duration for a metric is not a correctness decision, and forcing it
# through the environment would make the real node's telemetry lie about how
# long things took. Each exemption names a reason.
EXEMPT = {
    "aegis/core/determinism.py": "it *is* the environment",
    "aegis/sync/transport.py": "network clients own their own timeouts",
    "aegis/sync/meshlink.py": "measures real RTT to a real peer over HTTP",
    "aegis/sync/compression.py": "no clock or randomness in a codec",
}
BANNED = {
    ("time", "time"): "determinism.now()",
    ("time", "perf_counter"): "determinism.monotonic()",
    ("time", "monotonic"): "determinism.monotonic()",
    ("random", "random"): "determinism.rng().random()",
    ("random", "randint"): "determinism.rng().randint()",
    ("random", "choice"): "determinism.rng().choice()",
    ("random", "sample"): "determinism.rng().sample()",
    ("random", "shuffle"): "determinism.rng().shuffle()",
    ("random", "uniform"): "determinism.rng().uniform()",
    ("random", "randrange"): "determinism.rng().randrange()",
    ("random", "getrandbits"): "determinism.rng().getrandbits()",
    ("os", "urandom"): "determinism.rng().getrandbits()",
}


def offences(path: Path) -> list[tuple[int, str, str]]:
    found: list[tuple[int, str, str]] = []
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError as exc:
        return [(exc.lineno or 0, "syntax error", str(exc))]

    # A module that reaches for `determinism` without importing it raises
    # NameError the first time that line is reached, which — because these are
    # clock calls on paths that only run under load or during recovery — can be
    # a long way from the change that caused it. Two modules were converted
    # without their import and the suite found out, not this script. It should
    # have been this script.
    uses_determinism = any(
        isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
        and node.value.id == "determinism" for node in ast.walk(tree))
    imports_determinism = any(
        (isinstance(node, ast.Import)
         and any(a.name.endswith("determinism") for a in node.names))
        or (isinstance(node, ast.ImportFrom)
            and any(a.name == "determinism" for a in node.names))
        or (isinstance(node, ast.ImportFrom) and (node.module or "").endswith("determinism"))
        for node in ast.walk(tree))
    if uses_determinism and not imports_determinism:
        found.append((1, "determinism.* with no import",
                      "from ..core import determinism"))

    # A bare reference, not a call: `field(default_factory=time.time)` never
    # appears as an `ast.Call` on `time.time`, so the check below walked
    # straight past two of them. They sat in `causal.py` and `conflict.py`
    # through three thousand simulated executions, comparing virtual time
    # against the real wall clock, which quietly made one expiry path dead in
    # every seed.
    called = {id(node.func) for node in ast.walk(tree)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or not isinstance(node.value, ast.Name):
            continue
        if id(node) in called:
            continue                      # an ordinary call; the loop below has it
        key = (node.value.id, node.attr)
        if key in BANNED:
            found.append((node.lineno, f"{node.value.id}.{node.attr} passed as a value",
                          BANNED[key]))

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        value = node.func.value
        if not isinstance(value, ast.Name):
            continue
        key = (value.id, node.func.attr)
        if key in BANNED:
            found.append((node.lineno, f"{value.id}.{node.func.attr}()", BANNED[key]))
    return found


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    failures = 0
    checked = 0
    for pattern in SIMULATED:
        target = root / pattern
        files = sorted(target.rglob("*.py")) if target.is_dir() else [target]
        for path in files:
            relative = str(path.relative_to(root))
            if relative in EXEMPT:
                print(f"  skip  {relative:<34} ({EXEMPT[relative]})")
                continue
            checked += 1
            for line, used, instead in offences(path):
                failures += 1
                print(f"  FAIL  {relative}:{line}  {used} — use {instead}")
    if failures:
        print(f"\n{failures} determinism fault(s) in simulated code.")
        print("A single one makes a seed stop reproducing, silently.")
        return 1
    print(f"\n{checked} simulated modules draw time and randomness from the environment.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
