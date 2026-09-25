"""Deterministic simulation: thousands of fleet-days, and a seed for every bug.

    python3 scripts/simulate.py --seeds 2000 --steps 300 --peers 5
    python3 scripts/simulate.py --replay 8271443          # see it again

Each seed is one complete distributed execution — writes, gossip, partitions,
heals, clock steps in both directions, crashes that lose what was never
shared, and duplicate delivery from a peer that ignores its own egress filter.
Time is virtual, so a simulated hour costs nothing and a run covers far more
of the ordering space than any suite of hand-written tests.

After every step the invariants are checked. When one breaks, the run stops
and the shrinker takes over: remove one action, replay, does it still fail?
Keep the removal if so. What survives is the shortest sequence that still
reproduces the failure, which is usually short enough to read.

A clean sweep is not a proof. It is "no counterexample in N executions", and N
is reported so a reader can judge it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.sim.world import Violation, run_seed          # noqa: E402

O, R, B, D = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m"


def shrink(violation: Violation, peers: int, budget: int = 400,
           signed: bool = True) -> tuple[list, int]:
    """Delta-debug the failing history down to what actually matters."""
    history = list(violation.history)
    attempts = 0
    changed = True
    while changed and attempts < budget:
        changed = False
        index = 0
        while index < len(history) and attempts < budget:
            candidate = history[:index] + history[index + 1:]
            attempts += 1
            again, _ = run_seed(violation.seed, steps=len(candidate),
                                peers=peers, replay=candidate, signed=signed)
            if again is not None and again.invariant == violation.invariant:
                history = candidate
                changed = True
            else:
                index += 1
    return history, attempts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=500)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--peers", type=int, default=5)
    parser.add_argument("--replay", type=int, default=None,
                        help="re-run one seed and print its history")
    parser.add_argument("--out", default=None)
    parser.add_argument("--no-shrink", action="store_true",
                        help="count failures without delta-debugging each one — for a "
                             "control run where the rate is the result, not the trace")
    parser.add_argument("--unsigned", action="store_true",
                        help="run the mesh without operation signatures — the control "
                             "that shows the Byzantine invariants can still fail")
    parser.add_argument("--stop-after", type=int, default=3,
                        help="stop the sweep after this many distinct failures")
    args = parser.parse_args()

    if args.replay is not None:
        violation, stats = run_seed(args.replay, steps=args.steps, peers=args.peers,
                                    signed=not args.unsigned)
        print(f"{O}{B}seed {args.replay}{R} · {json.dumps(stats)}")
        if violation is None:
            print("  clean — no invariant broke in this execution")
            return 0
        print(f"  {O}{violation}{R}")
        for i, (action, params) in enumerate(violation.history[-25:]):
            print(f"  {D}{i:>4}  {action:<10} {params}{R}")
        return 1

    print(f"{O}{B}deterministic simulation{R}  {args.seeds:,} executions · "
          f"{args.steps} steps · {args.peers} peers · "
          f"{'unsigned (control)' if args.unsigned else 'signed'}")
    print(f"{D}every execution is a pure function of its seed; a failure here is a "
          f"failure forever{R}\n")

    started = time.perf_counter()
    failures: list[dict] = []
    simulated_seconds = 0.0
    operations = 0
    for offset in range(args.seeds):
        seed = args.start + offset
        violation, stats = run_seed(seed, steps=args.steps, peers=args.peers,
                                    signed=not args.unsigned)
        simulated_seconds += stats.get("simulated_seconds", 0.0)
        operations += stats.get("operations", 0)
        if violation is None:
            continue
        print(f"  {O}FAIL{R} seed {B}{seed}{R} · {violation.invariant} · "
              f"{violation.detail[:90]}")
        if args.no_shrink:
            minimal, attempts = list(violation.history), 0
        else:
            minimal, attempts = shrink(violation, args.peers, signed=not args.unsigned)
        if not args.no_shrink:
            print(f"       shrunk {len(violation.history)} steps → {B}{len(minimal)}{R} "
                  f"in {attempts} replays")
            for i, (action, params) in enumerate(minimal[:12]):
                print(f"       {D}{i:>3}  {action:<10} {params}{R}")
        failures.append({**violation.as_dict(), "minimal": [
            {"action": a, "args": list(b)} for a, b in minimal]})
        if len(failures) >= args.stop_after:
            print(f"\n  stopping after {len(failures)} distinct failures")
            break

    elapsed = time.perf_counter() - started
    days = simulated_seconds / 86400.0
    print(f"\n{O}{B}{args.seeds:,} executions{R} in {elapsed:.1f}s real time")
    print(f"  simulated            {B}{simulated_seconds:,.0f} seconds{R} "
          f"({days:,.1f} fleet-days)")
    print(f"  operations exchanged {B}{operations:,}{R}")
    print(f"  invariant failures   {B}{len(failures)}{R}"
          f" of {B}{offset + 1:,}{R} executions run")
    if not failures:
        print(f"{D}  no counterexample found. That is not a proof — it is "
              f"{args.seeds:,} executions without one.{R}")

    if args.out:
        out = Path(args.out).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({
            "executions": args.seeds, "steps": args.steps, "peers": args.peers,
            "signed": not args.unsigned,
            "real_seconds": round(elapsed, 2),
            "simulated_seconds": round(simulated_seconds, 1),
            "fleet_days": round(days, 2), "operations": operations,
            "failures": failures,
            "claim": ("no counterexample in these executions; this samples the "
                      "space of orderings rather than covering it"),
        }, indent=2), encoding="utf-8")
        print(f"  wrote {out}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
