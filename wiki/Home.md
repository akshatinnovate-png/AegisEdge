# AegisEdge

**An offline-first semantic memory node for the edge, built on Qdrant.**
Code Cubicle 6.0 — Problem Statement 03, *AI-Powered Edge Memory & Intelligence
Platform*.

A device that remembers what it has seen, answers questions about it with no
network, decides for itself what is worth sending to the cloud, and can show you
exactly how it reached every one of those conclusions.

---

## The short version

| | |
|---|---|
| **Runs with no network** | Real ONNX weights, a real embedded Qdrant, a real index. Nothing is stubbed and nothing phones home. |
| **Hybrid search in one engine call** | Dense recall, sparse recall, RRF fusion and a late-interaction rerank, executed by Qdrant as a single `query_points`. |
| **Decides where to run each query** | The engine or the local index, chosen from measured p95 against the query objective — and re-measured, not frozen at boot. |
| **Decides what to sync** | On a link that will not last, value per byte: obligations first, no-ops never, nothing starving. |
| **Can be inspected** | A console that draws the plan the engine ran, the corpus it holds, and the queue it is about to send. |
| **Proves its own claims** | 385 tests, a deterministic simulator, three bake-offs, and a CI job that fails when a documented number stops being true. |

---

## Evidence, not adjectives

| | |
|---:|---|
| **385** | tests |
| **3,000** | deterministic executions with signing on |
| **0** | invariant failures in those |
| **435 / 500** | executions falsified with signing **off** — the control that makes the 0 mean something |
| **8.1** | simulated fleet-days |
| **1,733,206** | operations exchanged in simulation |
| **35** | defects found and fixed, each with the regression test that would have caught it |

The control is the whole argument. A clean sweep proves nothing on its own; it
proves something next to a run, on the same seeds with the same attacks, that
comes back dirty. CI **fails the build if the control comes back clean**.

---

## Where to start

- Want to run it? → **[Quick Start](Quick-Start)**
- Want the design? → **[Architecture](Architecture)**
- Want to know how Qdrant is actually used? → **[Qdrant at the Centre](Qdrant-at-the-Centre)**
- Want to know whether any of this is true? → **[How This Was Tested](How-This-Was-Tested)**
- Want the things that are still wrong? → **[Open Problems](Open-Problems)**

---

## How it maps to the problem statement

| PS requirement | Where it lives |
|---|---|
| Offline-first semantic memory | [Architecture](Architecture) · real weights, embedded Qdrant, no network path in the query loop |
| Low-latency hybrid search without network | [Hybrid Retrieval](Hybrid-Retrieval) · dense + sparse + RRF + MaxSim, measured p50 **2.5 ms** at 2,000 points |
| Local vs cloud decisions | [Sync, Conflicts & Egress](Sync-Conflicts-and-Egress) · value-per-byte egress, policy-governed egress classes |
| Intermittent connectivity | [Sync, Conflicts & Egress](Sync-Conflicts-and-Egress) · connectivity oracle, resumable cursor, durable queue, starvation ceiling |
| Edge ↔ cloud sync | [Sync, Conflicts & Egress](Sync-Conflicts-and-Egress) · CRDT op log, Merkle range digests, IBLT set reconciliation |
| Conflict handling | [Sync, Conflicts & Egress](Sync-Conflicts-and-Egress) · HLC last-writer-wins, a four-rung arbiter, a human review queue |
| User-facing inspection | [Console Guide](Console-Guide) · query plans, the corpus plotted, the egress queue priced |
