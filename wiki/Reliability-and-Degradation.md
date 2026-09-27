# Reliability & Degradation

## Durability

Every mutation goes to a **write-ahead log**, fsynced before the write is
acknowledged. On boot the node replays it, reports what it recovered and what it
could not, and exposes that at `/api/v1/integrity/recovery` rather than starting
quietly.

Archived history lives in **immutable segments** with a manifest and a
checkpoint LSN. A corrupted segment is detected by `fsck`, repaired from the
rest where possible, and named where not. Point-in-time restore works from the
manifest.

`/api/v1/chaos/kill` sends the node `SIGKILL` — refused without a supervisor —
so "it survives a hard kill" is a thing you can do rather than a thing you read.

## The SLO ladder

The node defends an error budget rather than reacting to an instantaneous spike.
Burn rate against the query objective drives a five-rung ladder, with hysteresis
so it cannot flap:

| Level | What it sheds |
|---|---|
| 0 · FULL | Nothing |
| 1 · ECONOMISE | Wide fetch narrows |
| 2 · TRIM | Graph boost, diversity |
| 3 · ESSENTIAL | Cross-encoder rerank — dense+sparse fusion only |
| 4 · SURVIVAL | Everything but recall |

Each rung is announced with the reason (`burn rate 2.3x over budget`), and the
degradation level travels back with the query so an answer can say what it was
allowed to use. Degradation also changes egress: a degraded node takes the
faster query path, because undercutting the ladder while it sheds features
defeats what it is defending.

## Live invariants

The deterministic simulator checks invariants after every step. Those same
invariants run **in production**, on a rolling cursor with a budget, publishing
violations rather than raising — because a node that crashes on a self-check is
worse than one that reports the problem.

`/api/v1/integrity/invariants` reports what the node has asserted about itself
while running, per-invariant counters, and — importantly — **the scope each
local check can actually establish**. A single node cannot verify fleet-wide
convergence, and saying so is more useful than a green tick that means less than
it looks.

### The dead assertion

Two of the seven invariants read `agent.causal.clock.last`. `VectorClock` has no
attribute `last`. The lookup returned `None`, the loop hit `continue` on every
agent, and both checks returned clean for **three thousand executions**.

So "seven invariants checked after every step" was five. It was found by reading
the code, not by a failure — which is how a dead assertion is always found: it
cannot report itself.

Behind it was a real, undefended attack. The simulator mounts `forge_clock`, and
`HLC.dominates` compares `wall_ms` with no opinion about whether that number is
plausible. With the bound removed: **40 of 40 seeds falsified**. With it:
**0 of 60**.

## Thermal and power

A governor reads platform thermal and battery telemetry via `psutil` and sheds
inference work as ceilings approach. Where a platform exposes no reading, it is
reported **unavailable** rather than synthesised — a fabricated temperature is
worse than a missing one.

`/api/v1/energy` reports joules per operation and answers per 1% of battery,
billed against **CPU time** rather than wall time, because wall time charges a
query for every millisecond it spent queued behind something else.

## Chaos

`/api/v1/chaos/{fault}` injects one of: disk full, thermal spike, corrupt WAL,
partition, corrupt segment, memory pressure, query storm, peer churn, packet
loss, process kill. Each clears itself after a duration.

These are how the twelve stress-found defects were found, and they stay in the
build so they can be found again.
