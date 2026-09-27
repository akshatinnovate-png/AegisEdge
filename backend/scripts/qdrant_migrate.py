"""Move existing collections onto the Qdrant hybrid schema.

    python3 scripts/qdrant_migrate.py --dry-run
    python3 scripts/qdrant_migrate.py --apply

A node created before the hybrid schema holds one unnamed dense vector per
collection, which is why `search_native()` declines on it: there is no `lex`
sparse vector to prefetch and no named `dense` to query. Nothing is silently
rewritten at boot — a recreate deletes a collection, and doing that to an
operator's data because a new build prefers a different layout is not a
migration, it is data loss with a changelog entry.

This does it deliberately, and it can only be honest about it because Qdrant is
not the only copy: the node's write-ahead log and local index hold every memory,
so the points come back from there. If Qdrant holds more points for a collection
than the node can supply, that difference is data only Qdrant has, and the
migration refuses to touch that collection unless `--force` says otherwise.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aegis.config import Settings                                # noqa: E402
from aegis.node import EdgeNode                                  # noqa: E402

O, R, B, D, G, Y = ("\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m",
                    "\033[38;5;42m", "\033[38;5;220m")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the migration")
    parser.add_argument("--force", action="store_true",
                        help="recreate even when Qdrant holds points the node cannot supply")
    args = parser.parse_args()

    node = EdgeNode(Settings())
    await node.start()
    store = node.store.store
    if not hasattr(store, "hybrid"):
        print(f"{Y}this node is not on Qdrant — nothing to migrate{R}")
        await node.stop()
        return 1

    schema = store.hybrid.schema
    legacy = [name for name, kind in schema.items() if not kind.startswith("hybrid")]
    print(f"{O}{B}qdrant schema{R}  backend {B}{store.backend}{R}")
    for name, kind in schema.items():
        try:
            held = getattr(store.client.get_collection(name), "points_count", "?")
        except Exception:
            held = "?"
        mine = sum(1 for p in node.store.points.values() if p.collection == name)
        colour = G if kind.startswith("hybrid") else Y
        print(f"  {name:<12} {colour}{kind:<12}{R}{D}qdrant holds {held}, "
              f"the node can supply {mine}{R}")

    if not legacy:
        print(f"\n{G}every collection is already on the hybrid schema{R}")
        await node.stop()
        return 0
    if not args.apply:
        print(f"\n{Y}{len(legacy)} collection(s) on the old schema. "
              f"Re-run with --apply to migrate them.{R}")
        print(f"{D}until then those collections keep the local-index query path, which is "
              f"correct — just not the engine-side one{R}")
        await node.stop()
        return 0

    report = store.migrate_schema(list(node.store.points.values()), force=args.force)
    print(f"\n{json.dumps(report, indent=2)}")
    if report["refused"]:
        print(f"\n{Y}refused {len(report['refused'])} collection(s) to avoid deleting points "
              f"only Qdrant holds. Re-run with --force if that is what you want.{R}")
    else:
        print(f"\n{G}migrated {report['points']} point(s){R}")
    await node.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
