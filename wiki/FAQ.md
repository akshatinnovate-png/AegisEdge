# FAQ

### Is any of this simulated?

No. Real ONNX weights compiled at first boot, a real embedded Qdrant, a real
index, real Ed25519 signatures, a real WAL that survives `SIGKILL`. The node
**refuses to start** without real weights or a real vector store — there is no
fallback encoder and no internal store masquerading as Qdrant.

Three things are honestly bounded rather than claimed: a remote Qdrant Server
run, a multi-device federated learning deployment, and Triton escalation against
a live GPU. In each case the code path is real and exercised; the deployment is
not claimed. See [Open Problems](Open-Problems).

### Does it actually work without a network?

The query path has no network call in it. Weights ship in the wheel and
provisioning never touches the network. Pull the link with
`POST /api/v1/chaos/partition` and keep asking questions.

### Why is Qdrant slower than your own index, then?

Because the embedded client's local mode is a pure-Python implementation
intended for development, and this node's index is a calibrated HNSW with a
quantized cold tier. Against a Qdrant Server the engine is compiled and the four
round trips are real, and the trade flips.

The important part is that this is **measured, not assumed** — and that the two
paths are identical stage for stage, so the node can route between them safely.
See [Qdrant at the Centre](Qdrant-at-the-Centre).

### How do I know the numbers in the docs are true?

`python3 backend/scripts/audit_claims.py` checks every measured claim against
the committed artefact that produced it and exits non-zero on drift. It runs in
CI. It has caught drift three times, including on itself.

### What is the single most convincing thing to run?

```bash
cd backend
python3 scripts/simulate.py --seeds 200 --unsigned
```

The unsigned control. A clean signed sweep proves nothing on its own; it proves
something next to a run, on the same seeds with the same attacks, that comes
back dirty. CI **fails the build if that control comes back clean**.

### What is the most serious bug you found?

A delete that never deleted anything. Tombstones carried the HLC of the write
they were deleting, so they never dominated it — the node rejected its own
delete, reported `CONVERGED`, and peers filed the tombstone as a conflict for a
human to arbitrate. A memory someone asked to be forgotten stayed on every
device in the fleet, silently. See [Defect Log](Defect-Log).

### Why are there so many "controls" everywhere?

Because a measurement without one is a number, not evidence. Signed sweep and
unsigned control. Engine path and interpreter path. Value-ordered egress and
FIFO. Flat, HNSW and IVF-PQ on one corpus. Every bake-off prints a verdict
computed from its own table rather than typed into it.

### Can it run against a real Qdrant Server?

Yes — `AEGIS_QDRANT_URL=http://host:6333`. The same adapter, the same code path.
On a server the router always picks the engine, because one call beats four.

### What happens when two devices edit the same memory offline?

Hybrid logical clocks resolve by last-writer-wins where one clearly
happens-after. Genuinely concurrent writes inside the skew window go to a
four-rung arbiter, where the stricter sensitivity and sync class always win, and
what the machine should not decide lands in a human review queue at
`/api/v1/sync/conflicts`.

### Is the fancy console hiding a thin backend?

The console reads `/api/v1/*` and nothing else. Every panel has an endpoint
behind it that you can `curl`, and `/api/v1/qdrant`, `/api/v1/sync/egress` and
`/api/v1/qdrant/map` are all plain JSON. Turn the node off and the console goes
blank rather than degrading to a demo.

### How big is it?

**19,874** lines of node code across **109** modules, **5,217** lines of tests,
**4,348** lines of harnesses, and a **2,887**-line frontend with no build step.
Test code is about 26% the size of the code it tests — not a target, just what
happened when every defect had to arrive with the test that would have caught
it.

### What would you fix next?

The ~2.8 KB/query resident growth. It has a committed one-variable reproducer,
it has been shown to be a leak rather than a transient, and it has been shown
not to belong to any pipeline stage. That is three things ruled out and the
cause still unknown, which is exactly where the next session should start.
