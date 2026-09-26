# Measurements

*Part of [AegisEdge](../README.md). Every number here was produced by a harness in this repository, and every harness is committed.*

---

## 2. Measured, not claimed

Every number on this page comes from `backend/scripts/stress.py` on the
hardware it names. The full log is **[Appendix 1](#appendix-1--stress-test-log)**
and the raw results are committed under [`testlogs/`](../testlogs/).

| | Measured |
|---|---|
| Embedding throughput | **29,421 docs/s** at batch 128, four cores, no GPU |
| Search p95 @ 20,000 points | **1.80 ms**, recall@10 **1.000** (exhaustive) |
| Ingest, full pipeline | 12,000 documents, **zero failures**, p99 30.8 ms |
| Adversarial payloads | 22 hostile inputs, **0 timeouts**, fsck clean, audit chain intact |
| Fault storm | 12 simultaneous faults, **190 queries answered, 0 failed**, p99 138.7 ms, 11/11 subsystems alive |
| Query throughput | **6,176 q/s** on repeated questions, **1,599 q/s** on questions never asked before, at 256 concurrent |
| Energy | **40,191 answers per 1% of a 50 Wh pack**, 44.8 mJ each |
| Mesh | 50 peers, 2,000 divergent ops, **50/50 converged in 7 waves**, 8,187 ops/s across the mesh |
| Durability | 27 memories, SIGKILL, **back in 2.9 s with 27 memories and nothing lost** |
| Soak | 150 s mixed read/write — and an **open memory finding**, below |

One line in that table is an admission rather than a result. A steady-state
soak — corpus held still, nothing written, queries only — shows the resident
set growing about **2.6 KB per query under concurrent load**, linearly over
64,000 queries with no plateau. Python objects, the ONNX memory arena, input
shapes, the encoder, the micro-batcher, tracing, metrics, the event bus,
background writes and allocator retention have each been measured and ruled
out, and two attempted fixes measured as doing nothing and were therefore not
kept. It is on the status list as open, with the reproduction. A README that
said "no unbounded growth" — as this one did until the soak was taught to hold
the corpus still — would have been wrong rather than reassuring.

Two figures are deliberately *not* on that list. AegisEdge does not do "billions
of inputs per millisecond" — that would be ~10¹² ops/s, several orders of
magnitude past this machine's memory bandwidth. The honest ceiling is **29
embeddings per millisecond** on this box, and the report says where it flattens
and why. And the shipped default tenant quota is **600 ingests/minute**, which
is the first ceiling anyone will hit — about 3,100× below what the inference
layer can sustain. The harness lifts it explicitly and records that it did,
because a stress test run with the production guardrail on is measuring the
guardrail.

---

## 4. On-device representation learning

The node can fit a better embedding space for its own corpus than the one it
shipped with — and it is not allowed to believe that without proving it.

**The shipped space is badly conditioned.** Measured on the real encoder over
8,000 documents: unrelated documents sit at cosine **0.285**, and the
256-dimensional space has an **effective rank of 29.5**. Most of what a cosine
score reports is a common component every vector shares.

**So the device measures its own covariance** and fits a whitening transform,
with Ledoit–Wolf shrinkage so it never inverts an estimate it cannot support.
A packing line and a clinic converge on different transforms, because they see
different data, and neither needed a network to learn it.

**Rank selection is not optional, and finding that out cost a bug.** Whitening
at full dimension amplifies ~220 near-zero directions by four orders of
magnitude; it took cold-tier recall@10 from 0.77 to **0.11**. The transform now
keeps a rank bounded by both retained energy and a conditioning floor.

**Nothing is armed without evidence.** `AdaptationGate` builds a paraphrase
task out of the node's own memories — delete half a stored document's words,
and the document it came from is the correct answer by construction, so no
labels are needed — and adopts a candidate only when a **paired bootstrap** puts
the 95% confidence interval clear of zero. That is not ceremony: whitening
measured **+0.0175 MRR on one probe sample of the same corpus and −0.0001 on
another**. The gate rejects at 250 probes, rejects at 400 (+0.0092, CI still
spans zero), and adopts at 600 (**+0.0221, CI [+0.0059, +0.0371]**).

**And the gate earns its keep by rejecting things.** Smooth Inverse Frequency
pooling is one of the most cited results in sentence embedding, and it needs no
graph change here — the pooling graph computes `sum(emb·mask)/sum(mask)`, so
real-valued weights in place of the 0/1 mask make it a weighted mean exactly.
It also makes retrieval monotonically **worse** on this model: MRR@10 0.7827 →
0.7457 at a=1e-3, → 0.6376 at a=1e-4. The token table is a distillation trained
*for* mean pooling, so it has already absorbed the correction SIF exists to
apply. Shipping it on the strength of the citation would have cost ~5% of
retrieval quality, silently, on every device. It ships off.

**Measuring never arms anything.** The winning transform changes the output
dimension 256 → 195, so switching it on without re-embedding the corpus would
leave stored vectors and queries in differently shaped spaces — not a
degradation but nonsense, which no retrieval metric would report. Arming is an
explicit call behind a renewal migration, and the API refuses it while any
stored vector is still in the old space.

### 4.1 RaBitQ: 1-bit codes that know how wrong they are

The cold tier stored `sign(v)` — 32× compression, silently lossy, and no way for
anything downstream to reason about the loss, so the only safe use was a fixed
6× over-fetch that nobody could justify. RaBitQ (Gao & Long, SIGMOD 2024)
replaces the proxy with an **unbiased estimator and a concentration bound**:
centre on the corpus centroid, apply a random orthogonal rotation (a fast
Hadamard construction, O(d log d), when the dimension is a power of two), and
keep one float for how well each code aligns with the vector it encodes.

| | sign bits | RaBitQ |
|---|---|---|
| recall@10, raw space, 6× shortlist | 0.938 | **0.997** |
| recall@10, whitened space | 0.675 | **0.893** |
| bias | — (not an estimator) | **< 1e-3** |
| bound holds | no bound | **99.1–99.4%** against a stated 98.5% |
| bytes/vector @ d=256 | 32 | 40 |

Eight bytes per vector to turn a guess into a bound — and the bound then
replaces the guess. The k-th largest *lower* bound is a score the true top-k
provably reaches, so any candidate whose upper bound falls short is never
fetched from disk. Easy queries collapse to a handful of reads, ambiguous ones
widen by themselves, and a latency budget caps it when the bound stops being
informative — recorded when it binds, because a guarantee that silently lapsed
is worse than one never claimed.

### 4.2 The index strategy was choosing wrong

`scripts/strategy_bakeoff.py` forces each strategy onto the same corpus, because
"the adaptive index never switched" is either a cost model working or a cost
model broken, and only a measurement can say which:

| 20,000 points | p50 | recall@10 | build |
|---|---|---|---|
| **flat** | **0.886 ms** | **1.000** | 0 s |
| hnsw | 4.340 ms | 0.773 | 252 s |
| ivf_pq | 550.512 ms | 0.997 | 123 s |

Exhaustive search was **4.9× faster than the graph**, exact where the graph lost
a quarter of its recall, and free to build — and the cost model was selecting
the graph from 5,000 points upward. It timed a hop as one vectorised numpy call,
capturing the arithmetic and none of the per-node interpreter bookkeeping that
actually dominates. The overhead ratio came out ≈1 and the crossover collapsed
onto its floor.

It only failed to make anything *worse* because index migration is deferred to
the maintenance lane, which the scale runs never reach — a latent
misconfiguration masked by an unrelated fix. The hop is now timed with its
bookkeeping (overhead **12.1×**), and the crossover is solved from the two
measurements: **110,000 points**, with the model predicting 4.82 ms for HNSW at
20k against a measured 4.34 ms.


---

## 4.3 PROVE IT — four claims, each with the button that would falsify it

Every team claims. Almost none invite verification, because offering to be
checked only works if you would survive it. A third mode, **PROVE IT**, is the
offer.

| | |
|---|---|
| ![Prove panel](../testlogs/images/11-prove-panel.png) | ![Under load](../testlogs/images/12b-prove-pinned.png) |
| **The panel.** Durability, degradation, energy, provenance and time travel — each with the control that would break it if the claim were false. | **Under load, at a pinned rung.** ESSENTIAL suspends eight stages *by name*; in-flight requests are capped and what the cap holds back is counted. |
| ![Killed](../testlogs/images/13-prove-killed.png) | ![Recovered](../testlogs/images/14-prove-recovered.png) |
| **SIGKILL, from the browser.** No handler runs, no buffer is flushed, nothing is tidied. | **Back in 2.9 s.** 36 operations replayed in 52.7 ms. *27 memories before the kill, 27 after — nothing was lost.* |

### Hand the judge the crowbar

`scripts/supervise.py` runs the node as a child and restarts it, recording each
death: which signal, how long it lived, whether it was a hard kill.
`POST /api/v1/chaos/kill` lets somebody with a browser and no terminal pull the
rug out. SIGKILL is the default because it is the only interesting one. The
endpoint **refuses unless a supervisor is present** — a kill button with
nothing behind it is not a demonstration, it is the end of the demo.

`GET /api/v1/integrity/recovery` then answers the only question that matters
after a crash, in the terms the claim was made in. A torn tail record is
reported and explained rather than glossed: the WAL is fsynced *before* a write
is acknowledged, so a half-written record is one whose caller never got an
answer, and discarding it is the only correct thing to do.

Beside it, every fault the chaos controller knows — corrupt the WAL, corrupt a
segment, fill the disk, skew the clock, memory pressure, query storm — and
`fsck` to check afterwards.

### Joules, not milliseconds

**40,191 answers per 1% of a 50 Wh pack, at 44.8 mJ each.** Latency is the
figure everything reports and the wrong one to optimise a battery-powered
device against: answering in 4 ms while holding four cores flat is worse, on a
robot, than 12 ms on one core.

Three sources, tried in order, and every reading says which produced it — the
CPU package's RAPL counter, battery discharge, or CPU-seconds times an assumed
per-core draw. The first two are measurements. The third is a **model**, and it
is labelled `modelled` everywhere it appears, because a model dressed as a
measurement survives exactly until somebody checks it. The coefficient is
configuration rather than a constant buried in a module: the right value for a
Xeon and an A78 differ by an order of magnitude and only the deployer knows
which they have.

### The receipt

A demo is an assertion, and nobody watching one can tell a node running the
committed code from a node running a branch with the hard parts stubbed. So the
node computes, at runtime, a **Merkle root** over its own source, the weights
it loaded and the graphs it compiled from them:

```bash
python3 -m aegis.core.provenance        # in a clean clone
curl localhost:8000/api/v1/provenance   # on the running node
```

If the roots match, the process answering you is the code that is public. A
tree rather than a flat hash, so a mismatch is *located* file by file. A dirty
working tree is reported, not hidden. And the receipt states its own limit:
this is integrity, not authenticity — it proves the running process matches a
tree with this root, not who produced the tree.

### Degradation you can drive

A dial that issues real queries, and the ladder lighting up as the node sheds.
Two honesty fixes were needed to make the reading mean anything. At 900 q/s the
browser had 1,333 fetches outstanding and the panel reported a p50 of 3.6
seconds while the node's own burn rate sat at zero — the node was fine and the
*browser* was the bottleneck; in-flight requests are now capped and what the
cap turns away is counted and shown. And one browser may simply be unable to
make a node breach its objective, which is an honest result rather than a
broken demo: the panel says so instead of manufacturing a stall, and the rung
pins beside the dial are labelled as pins, because a manual override is not the
same evidence as organic shedding.

### Time travel

![Time travel](../testlogs/images/15-prove-timetravel.png)

A slider over the last 72 hours: what this device believed then, and what it has
learned or retracted since. The bitemporal graph has been in the system all
along and nothing had ever surfaced it.

### A peer that lies

![A peer that lies](../testlogs/images/07-a-peer-that-lies.png)

Four buttons, four attacks, run against the node serving the page. Each one
builds a real `Operation`, encodes it with the real wire codec and hands it to
the same `GossipAgent.handle` a peer device reaches over `/mesh/exchange` —
nothing is staged, and the only thing the route knows that a peer does not is
the attacker's own key, which it uses exactly as an attacker would.

| Button | What it sends | What should happen |
|---|---|---|
| send an honest operation | signed by the device that wrote it | **accepted** |
| rewrite it in flight | somebody else's operation, body changed, original signature kept | refused |
| write in another device's name | the attacker's signature under the victim's `device_id` | refused |
| send it unsigned | no signature at all | refused |

The honest case is on the card for a reason: it is the control. A node that
refused everything would pass the other three for entirely the wrong reason,
and a panel that only ever shows refusals cannot tell the difference. The card
reports the node behaving *as claimed*, which for the first button means
accepting — and if an outcome is ever wrong, it says "wrong outcome" on screen
rather than in a log.

The counters beside it are the node's own: operations verified, refused as
forged, refused by policy on arrival. They are read from `/api/v1/mesh/status`,
the same endpoint anything else would use.

---

# Appendix 1 — stress test log

The complete run, all nine phases, is committed at
**[`testlogs/STRESS-REPORT.md`](../testlogs/STRESS-REPORT.md)**, rendered from the
raw results in [`testlogs/stress-raw.json`](../testlogs/stress-raw.json) by
`backend/scripts/report.py`. Also committed:

| File | What it holds |
|---|---|
| [`testlogs/STRESS-REPORT.md`](../testlogs/STRESS-REPORT.md) | The full nine-phase report, with every table |
| [`testlogs/stress-raw.json`](../testlogs/stress-raw.json) | Raw structured results — the report is generated from this, so the two cannot drift |
| [`testlogs/ab-before.json`](../testlogs/ab-before.json) · [`ab-after.json`](../testlogs/ab-after.json) | The quadratic-append regression, measured either side of the fix |
| [`testlogs/geometry-eval.json`](../testlogs/geometry-eval.json) | The embedding-space sweep: anisotropy, whitening configurations, quantization recall |
| [`testlogs/strategy-bakeoff.json`](../testlogs/strategy-bakeoff.json) | Flat vs HNSW vs IVF-PQ forced onto the same corpus |
| [`testlogs/images/`](../testlogs/images/) | The frames above, captured against a live node |
| [`testlogs/simulation.json`](../testlogs/simulation.json) · [`.txt`](../testlogs/simulation.txt) | The deterministic sweep, signed build — executions, fleet-days, and any counterexample with its shrunk history |
| [`testlogs/simulation_unsigned.json`](../testlogs/simulation_unsigned.json) · [`.txt`](../testlogs/simulation_unsigned.txt) | The control: the same simulator with signatures off, so "the attack no longer fires" can be read against a run where it does |

Reproduce any of it:

```bash
cd backend
python3 scripts/stress.py --phases all --out ../testlogs   # ~21 min, nine phases
python3 scripts/report.py ../testlogs                      # render the report
python3 scripts/strategy_bakeoff.py                        # index strategy bake-off
python3 scripts/geometry_eval.py                           # embedding space sweep
python3 scripts/capture.py --out ../testlogs/images        # the screenshots

python3 scripts/audit_determinism.py                       # the determinism lint
python3 scripts/simulate.py --seeds 3000 --steps 400 --peers 6 \
        --out ../testlogs/simulation.json                  # the deterministic sweep
python3 scripts/simulate.py --seeds 500 --steps 400 --peers 6 --unsigned \
        --no-shrink --out ../testlogs/simulation_unsigned.json    # the control
```

The harness records failures as results. Where a subsystem gave way, the number
that broke it is in the table rather than absent from it — including the
30-second query and the `FULL` degradation level that were still wrong when the
run was taken, and which the fixes above landed after.

---

