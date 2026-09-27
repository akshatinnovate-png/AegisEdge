# Defect Log

**35 defects found and fixed, each with the regression test that would have
caught it.** The tally lives in one ledger (`testlogs/defects.json`) and CI
checks the documentation against it — because this count had already drifted
once, with `STATISTICS.md` saying thirty-one while the README said thirty-five.

| Found by | Count |
|---|---:|
| Load and stress testing | 12 |
| Deterministic simulation | 4 |
| A security review of that work | 5 |
| A correctness review of it | 10 |
| Racing Qdrant's engine against this node's own index | 3 |
| Asking what a dying link should carry first | 1 |
| **Total** | **35** |

Full narratives are in [docs/ENGINEERING.md](https://github.com/akshatinnovate-png/AegisEdge/blob/main/docs/ENGINEERING.md).
The ones worth knowing about:

---

## The tombstone that never deleted anything

`record_local` stamped every operation with `point.hlc` — written once at ingest
and never again — so a delete carried the timestamp of the write it was
deleting. Resolution is `dominates()`, strictly happens-after, so a tie loses.

The node rejected its own tombstone (`tombstones: 0, rejected_stale: 1`) and
reported `CONVERGED`. A peer filed the delete as a **concurrent write for a
human to arbitrate**, so the memory stayed on every device in the fleet.

For a system whose entire egress story is about what may leave a device, a
delete that does not propagate is the worst defect in it. **Fixed:** an
operation is stamped when it happens.

## Two invariants that asserted nothing

`_clock_monotonic` and `_clock_not_poisoned` both read
`agent.causal.clock.last`. `VectorClock` has no attribute `last`. Both returned
clean for **three thousand executions**.

Behind them was a real, undefended clock-poisoning attack: with the bound
removed, **40 of 40 seeds falsified**; with it, **0 of 60**. Found by reading
the code, because a dead assertion cannot report itself.

## The codec that edited what had been signed

The wire codec quantized vectors on the way out — *after* the operation was
signed. Every real upsert carrying a vector therefore failed verification at the
far end as a forgery. **Fixed:** quantize at creation; the codec compresses, it
does not edit.

## The delta dictionary that survived a frame

The encoder's delta context persisted across frames while the decoder started
empty on each one. The first operation carrying `sensitivity: restricted` sent
the label; every later one sent a hole the receiver could not fill. An operation
the policy would have refused arrived looking ordinary — and was accepted.

Worse than a compression bug: the receiving node decides whether it may hold an
operation by reading exactly that label. Surfaced by the simulator as a
signature mismatch, which is the whole argument for signing.

## Operation ids from the wall clock

Two identical sweeps gave 454 and 456 results. `ids.py` was reading the real
clock, so seeds stopped reproducing. **Fixed**, and `ids.py` brought into the
determinism lint's scope — the lint had not been looking at it.

## The double length-normalisation

The sparse encoder's BM25 weights already normalise for length, and
`SparseIndex.search` divided by the document's L2 norm as well. Long memories
sank for being long, twice. Found because Qdrant's sparse ranking disagreed with
ours — **0.717 agreement, now 1.000**.

## The cache that dropped its own confidence

Both cache layers returned `confidence: {}`, and the exact layer returned
`trace: {}`. Asking the same question twice produced an answer with no
calibrated guarantee and no account of itself — on the repeat, which is the one
most likely to be the one an operator is staring at.

## Resource bounds a peer controlled

From the security review: a decompression bomb (51 KB → 50 MB), an IBLT cell
count taken from the wire (69 MB from one integer), an unauthenticated
`/mesh/attack`, probe identities that could squat on real device ids, and a
drill that polluted real incident counters.

The lesson, now written into the code: **anything a peer controls is an
allocation a peer controls.**

## The cost model that timed the wrong thing

It timed a graph hop as one vectorised numpy call — which captures the
arithmetic and none of the per-node interpreter bookkeeping that dominates a
real traversal. Overhead came out ≈1, the crossover collapsed onto its floor,
and the node chose HNSW from 5,000 points while flat was 4.9× faster and exact.
Re-timed with the bookkeeping (overhead 12.1×), crossover **110,000 points**.

## Numbers in prose that stopped being true

The determinism lint's module count said twelve while the lint covered fourteen.
The defect tally said thirty-one in one file and thirty-five in another. Both
were written truthfully, at different times.

Both are now generated from a single source and checked by `audit_claims.py`,
which is the same mechanism that caught a rounding bug in itself.
