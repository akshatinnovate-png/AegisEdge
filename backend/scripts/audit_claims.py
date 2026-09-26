"""Do the numbers in the documentation still match the artefacts?

    python3 scripts/audit_claims.py

Every measured claim in this repository is supposed to come from a committed
result. Nothing enforced that. A number can be right when it is written and
wrong three commits later, and prose does not fail a build — which is the
mechanism by which almost every README in the world ends up lying slightly.

This checks a declared set of claims against the files that produced them, and
exits non-zero on drift. It deliberately does not parse arbitrary prose: that
is fragile, and a checker that cries wolf gets removed. Each claim names the
document, the exact text that must appear, and where the number comes from.

Adding a measured number to the docs without adding it here is possible. The
point is not that evasion is impossible; it is that drift is no longer silent,
and that the cost of keeping the documentation honest is one line.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[2]
O, R, B, D, G = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m", "\033[32m"


def load(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


@dataclass
class Claim:
    """One number in the documentation, and the artefact it came from."""

    document: str
    text: Callable[[Any], str]        # the exact string the document must contain
    artefact: str
    describe: str

    def check(self) -> str | None:
        source = ROOT / self.artefact
        if not source.exists():
            return f"artefact missing: {self.artefact}"
        page = ROOT / self.document
        if not page.exists():
            return f"document missing: {self.document}"
        expected = self.text(load(self.artefact))
        if expected not in page.read_text(encoding="utf-8"):
            return f"{self.document} does not say {expected!r}"
        return None


CLAIMS = [
    # The README and the instructions quote the *quick* control, because that
    # is the command they tell a reader to run. The engineering log quotes the
    # long one. Checking both against the same artefact was the first thing
    # this script caught, on its first run: two true statements about two
    # different runs, one of which would have drifted silently.
    Claim("README.md", lambda d: f"{len(d['failures'])} of {d['executions']}",
          "testlogs/simulation_unsigned_quick.json",
          "the control result the README quotes, from the command it prints"),
    Claim("docs/RUNNING.md", lambda d: f"{len(d['failures'])} of {d['executions']}",
          "testlogs/simulation_unsigned_quick.json",
          "the same, where a reader is told what to expect"),
    Claim("docs/ENGINEERING.md", lambda d: f"**{len(d['failures'])} of {d['executions']}**",
          "testlogs/simulation_unsigned.json",
          "the long control in the engineering log"),
    Claim("README.md", lambda d: f"{d['executions']:,}-execution",
          "testlogs/simulation.json",
          "the size of the signed sweep"),
    # Derived from `simulated_seconds`, the way the script that wrote the file
    # derives it — not from the stored `fleet_days`, which is already rounded
    # to two places and rounds again to a different first place. The audit
    # caught that on itself: 8.15 stored, "8.1" printed by the script, "8.2"
    # expected by the checker, and neither of them wrong about the number.
    Claim("README.md", lambda d: f"{d['simulated_seconds'] / 86400.0:,.1f} fleet-days",
          "testlogs/simulation.json",
          "how much simulated time that was"),
    Claim("docs/ENGINEERING.md", lambda d: f"invariant failures   {len(d['failures'])} of "
                                           f"{d['executions']:,}",
          "testlogs/simulation.json",
          "the signed sweep's result, quoted as the script printed it"),
]


def main() -> int:
    print(f"{O}{B}documentation claims{R}  {len(CLAIMS)} checked against committed artefacts\n")
    failures = 0
    for claim in CLAIMS:
        problem = claim.check()
        if problem is None:
            print(f"  {G}ok{R}    {claim.describe}")
        else:
            failures += 1
            print(f"  {O}DRIFT{R} {claim.describe}\n        {problem}")
    print()
    if failures:
        print(f"{O}{B}{failures} claim(s) no longer match the results they came from.{R}")
        print(f"{D}Either the documentation is stale or the artefact was regenerated "
              f"without it.{R}")
        return 1
    print(f"{D}Every checked number in the documentation is the number in the file that "
          f"produced it.{R}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
