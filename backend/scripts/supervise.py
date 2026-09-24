"""Run a node that can be killed, and bring it back.

A durability claim that cannot be tested is a slogan. "Nothing becomes
searchable before it is recoverable" is either true — in which case killing
the process mid-write costs milliseconds and no memories — or it is not, and
the only way to tell from outside is to kill it and look.

A process cannot, obviously, restart itself. So this runs the node as a child,
and when the child dies for any reason it starts another one and records what
happened: which signal, how long the gap was, and how many times. That is the
whole supervisor. It deliberately has no API of its own, because anything it
exposed would be a second thing to trust.

    python3 scripts/supervise.py --port 8000

The node then answers `POST /api/v1/chaos/kill`, which is how a person with a
browser and no terminal can pull the rug out. `SIGKILL` is the default and the
only interesting one: `SIGTERM` lets the node shut down tidily, which proves
nothing about what happens when a battery is removed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

DEATHS_FILE = "supervisor-deaths.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--node-id", default=None)
    parser.add_argument("--mesh-transport", default=None)
    parser.add_argument("--max-restarts", type=int, default=0,
                        help="0 means restart forever")
    parser.add_argument("--log-level", default="warning")
    args = parser.parse_args()

    backend = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    if args.data_dir:
        env["AEGIS_DATA_DIR"] = args.data_dir
    if args.node_id:
        env["AEGIS_NODE_ID"] = args.node_id
    if args.mesh_transport:
        env["AEGIS_MESH_TRANSPORT"] = args.mesh_transport
    # The node reads this to know it has somewhere to fall: without a
    # supervisor, offering a kill button would just end the demo.
    env["AEGIS_SUPERVISED"] = "1"

    deaths_path = Path(env.get("AEGIS_DATA_DIR", backend / ".aegis")) / DEATHS_FILE
    deaths: list[dict] = []
    if deaths_path.exists():
        try:
            deaths = json.loads(deaths_path.read_text())
        except Exception:
            deaths = []

    command = [sys.executable, "-m", "uvicorn", "aegis.main:app",
               "--host", args.host, "--port", str(args.port),
               "--log-level", args.log_level]

    generation = 0
    child: subprocess.Popen | None = None

    def forward(signum, _frame):
        """Ctrl-C should stop the whole thing, not orphan a node."""
        if child and child.poll() is None:
            child.send_signal(signum)
        sys.exit(0)

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)

    while True:
        generation += 1
        env["AEGIS_GENERATION"] = str(generation)
        started = time.time()
        print(f"[supervisor] generation {generation} starting on "
              f"{args.host}:{args.port}", flush=True)
        child = subprocess.Popen(command, cwd=backend, env=env)
        code = child.wait()
        lived = time.time() - started

        # A negative return code is the signal that killed it. -9 is the one
        # that matters: no handler ran, no buffers flushed, nothing tidied.
        killed_by = -code if code is not None and code < 0 else None
        record = {
            "generation": generation,
            "exit_code": code,
            "killed_by_signal": killed_by,
            "signal_name": signal.Signals(killed_by).name if killed_by else None,
            "lived_s": round(lived, 2),
            "died_at": time.time(),
            "hard_kill": killed_by == signal.SIGKILL,
        }
        deaths.append(record)
        try:
            deaths_path.parent.mkdir(parents=True, exist_ok=True)
            deaths_path.write_text(json.dumps(deaths[-50:], indent=2))
        except Exception:
            pass
        print(f"[supervisor] generation {generation} died: {json.dumps(record)}", flush=True)

        if args.max_restarts and generation >= args.max_restarts:
            print("[supervisor] restart limit reached", flush=True)
            return
        # A node that dies instantly and repeatedly is broken, not being
        # tested; backing off keeps a crash loop from looking like a demo.
        time.sleep(0.4 if lived > 3 else 2.0)


if __name__ == "__main__":
    main()
