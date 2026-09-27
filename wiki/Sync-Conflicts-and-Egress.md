# Sync, Conflicts & Egress

The hard half of "offline-first" is not working offline. It is what happens when
the link comes back.

## The op log

Every local mutation becomes a CRDT operation: an append-only causal log with
last-writer-wins per point, resolved by **hybrid logical clock**, and OR-Set
deletes. Resolution is by HLC rather than arrival order, which is what makes the
log order-independent — a property the egress scheduler depends on and a test
verifies by replaying the same operations forward, backward and shuffled.

Operations are **Ed25519-signed** by the authoring device. See
[Security Model](Security-Model).

## Finding what diverged

| Mechanism | Job |
|---|---|
| **Merkle range digests** | Compare two devices' state in one round trip and learn *which ranges* differ, not which points |
| **IBLT set reconciliation** | Recover the actual symmetric difference of two id sets in space proportional to the difference, not the sets |
| **Vector clocks + causal delivery** | Buffer an operation that arrives before its causal predecessor instead of applying it out of order |
| **Gossip anti-entropy** | Device-to-device convergence with no coordinator |

## Conflicts

Two devices writing the same point while partitioned is not an error, it is the
normal case. A four-rung arbiter handles it:

1. **Clock** — one clearly happens-after the other; converge, no conflict.
2. **Concurrent within the skew window** — a genuine conflict, recorded.
3. **Policy** — the stricter sync class and sensitivity always win. A memory
   marked for redaction at its origin may never arrive freely shareable.
4. **A person decides** — a review queue, surfaced at
   `/api/v1/sync/conflicts`, because some conflicts are not the machine's to
   resolve.

## Connectivity

Polling every 30 s means discovering the link 30 s late. The oracle keeps EWMA
estimates of RTT, jitter and loss, classifies the link continuously
(`offline` / `degraded` / `metered` / `healthy`), and fires callbacks on the
*transition* — so reconnection work starts in milliseconds.

The smoothing is deliberately asymmetric. A completed round trip is unambiguous
evidence the link works, so recovery is fast; a single timeout is not proof of
an outage, so degradation is slow.

## Egress: what a dying link should carry first

Egress used to be four lines — while the queue has anything, send it, in write
order. Correct for a link that stays up. This node's premise is a link that does
not.

The queue is durable, so nothing is lost either way. The question is **what goes
first**, because the prefix that lands before the drop is the part that was
worth sending.

`aegis/sync/egress.py` makes three decisions:

**Redundant operations are never sent.** Two queued operations on the same point
mean the older one is dead on arrival — LWW resolves by HLC, so applying it
changes nothing — and FIFO pays full price to transmit a no-op. They are *held*,
not dropped: an operation leaves the durable queue only once the operation that
supersedes it is acknowledged. A tombstone is never suppressed this way.

**The rest go by value per byte.** Cost is the real encoded size from the wire
codec. Value comes from what the node already knows:

| Signal | Weight | Reasoning |
|---|---|---|
| Delete | 40 | The cloud still holding what the edge deleted is a *wrong* state, not a stale one |
| Redaction / metadata-only | 24 | Same class of obligation |
| Inside a divergent Merkle range | 12 | Closes a gap the last cycle actually found |
| Local retrievals | 6 × log1p(reads) | A memory being read is one another device will want |
| Pinned | 8 | |
| Low confidence | ×(0.6 + c) | Worth less to the fleet |

**Nothing starves.** Value ordering alone would never send a dull operation
while interesting ones keep arriving, so anything waiting longer than
`STARVATION_S` (120 s) is promoted ahead of the ordering entirely. The wait is
bounded by a number rather than by hope.

### What it is worth

`scripts/egress_bakeoff.py` builds a real queue and cuts the link after a fixed
number of bytes. **321 operations, 289 KB if all of them went, a link that
carries 16 KB and then drops:**

| | FIFO | value-first |
|---|---|---|
| operations landed | 18 | 18 |
| value landed | 49.5 | **387.7** |
| deletes landed | **0 of 7** | **7 of 7** |
| share of the queue's total worth | 4.0% | **31.4%** |

Same bytes, same starvation rule, same queue. FIFO left every deletion
un-propagated.

Separately, **14 operations were superseded before they left, saving 12.7 KB**
that FIFO would have spent transmitting no-ops.

## The defect this uncovered

Building that measurement found something worse than the thing it was aimed at.

`record_local` stamped every operation with `point.hlc` — a field written once,
at ingest, and never again. So a tombstone carried **the timestamp of the write
it was deleting**, and resolution is `dominates()`, strictly happens-after. A
delete that ties with its own upsert loses.

```
delete op hlc: 1790509144046.00000.edge-07
point  hlc:    1790509144046.00000.edge-07     # identical
oplog tombstones: 0   rejected_stale: 1
sync: {'state': 'CONVERGED', 'pushed': 1}
```

The node rejected its own delete and reported success. On a peer it was worse —
the tombstone landed inside the 1000 ms concurrency window and was filed as a
**genuine conflict for a human to arbitrate**:

| | author tombstones | peer tombstones | peer verdict |
|---|---|---|---|
| tombstone reuses the point's HLC | 0 | 0 | conflicted |
| tombstone stamped at delete time | 1 | 1 | accepted |

A memory someone asked to be forgotten stayed on every other device in the
fleet, silently. For a system whose whole egress story is about what may leave a
device, that is the worst defect in it.

An operation is now stamped when it happens. Pinned by a test that asserts the
broken behaviour *and* the fixed one, because a regression test that only checks
the fix cannot tell you the bug was real.

## Egress policy

Not everything is allowed to leave. The policy engine assigns each memory a sync
class at ingest:

| Class | Behaviour |
|---|---|
| `full` | Body and vectors sync |
| `redacted` | Sensitive spans replaced via the redaction vault before egress |
| `metadata_only` | Collection, timestamp and sensitivity leave; the content never does |
| Sensitive | Never leaves the device |

Egress runs **through** the policy engine, not around it, and a withheld memory
is announced on the event bus rather than dropped quietly.
