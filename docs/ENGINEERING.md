# Engineering log — what broke, and how it was found

*Part of [AegisEdge](../README.md). This is the long version: sixteen defects, the instruments that found them, and the ones still open.*

---

## 3. What the stress runs broke

The point of a stress test is the things it breaks. Sixteen of this project's
thirty-one defects are below, each found by pushing until something gave way
and then reading what actually happened rather than what was supposed to. All
are fixed, with a regression test each — and one finding that is still open,
because not finding the cause is also a result.

The other fifteen came from reviewing this work rather than running it: five in
a security pass (§3.7) and ten in a correctness pass (§3.8), including two
invariants that had never worked at all.

Twelve came from load. The last four came from a different instrument
entirely — a deterministic simulator that runs the whole fleet as a pure
function of one integer, described in §3.1. Load testing asks whether the
system is fast. The simulator asks whether it is *right*, across orderings no
human would think to write down.

| # | Defect | What it cost |
|---|---|---|
| 1 | Vector append was quadratic — `np.vstack` copied the whole matrix per insert | **403× slower** inserts at 32k points; a scale run that never returned |
| 2 | Index migration ran inline on the write path | one write blocked **30.8 seconds** while a graph was built under it |
| 3 | Catastrophic regex backtracking in the PII classifier | one 1 MB ingest wedged the node for **79 minutes** at 100% CPU — a denial-of-service vector on an unauthenticated path |
| 4 | `slo.observe()` was called only in the HTTP layer | the degradation ladder was blind to every query from the WebSocket, agent, mesh or internal retrieval: **p99 1,850 ms** against a 150 ms target, burn rate **0.0**, nothing shed |
| 5 | The semantic cache was keyed on tenant alone | a `collection="*"` answer was served verbatim for a scoped query — **five hits from a collection holding nothing** |
| 6 | Inferred query narrowing applied as a hard filter | ordinary queries returned **zero** hits: `conveyor` → 5, `conveyor vibration night shift` → 0, on a corpus where every document contains all four words |
| 7 | The tokenizer encoded the whole input to keep 128 tokens | 1 MB query **853 ms → 35.5 ms**; the same text stored as a document paid it again on every rerank |
| 8 | In-flight sync outlived the storage handle | shutdown raised bare `RuntimeError` into un-awaited background tasks |
| 9 | **The semantic cache had never answered a question** | 0 entries, 0 hits, 0 misses across 128 queries — a `if not filters` guard that query understanding made true on almost every query. Fixed and moved in front of the encoder: **11.16 ms → 0.139 ms, 80×** |
| 10 | The index cost model chose the slower, less accurate structure | it picked HNSW from 5,000 points where flat was **4.9× faster and exact** — it had timed a graph hop as one vectorised call, capturing the arithmetic and none of the interpreter cost |
| 11 | The index calibration was not reproducible | consecutive calibrations on one machine derived crossovers between 40,000 and 110,000 points, so a node's index strategy depended on what else was running when it booted |
| 12 | A handling class did not survive the trip between devices | a memory marked for redaction at its origin arrived on the next device freely shareable, and that device's onward decisions rested on a class it never had |
| 13 | Egress policy was enforced on the sender only | a device that skips its own filter — buggy or compromised — pushed a never-leaves-the-device memory to every peer, and each stored it without a murmur. Found deterministically at **seed 5**, and found there every time |
| 14 | **Operations had no integrity protection at all** | a relay could rewrite the body of somebody else's operation in flight and every downstream node accepted the altered version. **435 of 500** simulated executions found it |
| 15 | Operation ids were drawn from the wall clock | the ids the IBLT hashes and a fetch is sorted by were not part of the seeded execution, so two sweeps over the same 500 seeds returned **454 failures, then 456** — close enough to read as noise, and a flat contradiction of the claim that a run is a pure function of its seed |
| 16 | The wire codec silently dropped body fields | it elided a repeated field against encoder state that outlived the frame, while the decoder started empty on every frame. The second operation carrying `sensitivity: restricted` arrived with **no sensitivity label at all** — and that label is exactly what the receiving node reads to decide whether it may hold the memory |

Three of these deserve more than a row.

**The classifier hang (3)** is the one that matters most, because it is
reachable by anyone who can write a memory. The pattern was
`[\w.+-]+@[\w-]+\.[\w.]+`. Against a megabyte of word characters the local
part matches greedily to the end, fails to find the `@`, backtracks the whole
way, and the engine restarts from the next offset — measured at exactly **4× the
time for 2× the input**, extrapolating to 79 minutes for a single 1 MB write.
The classifier sits on the ingest path *by design*, so it cannot be bypassed,
which is also why one write could take the device off the air. Anchoring the
pattern fixed the bug; the scan now also runs in overlapping windows so no
future pattern can be handed an unbounded string. Truncating was rejected — this
is a security control, and a secret at offset two megabytes must not escape
classification because scanning it was inconvenient. **79 minutes → 199 ms**,
flat at 199 µs/KB from 32 KB to 4 MB.

**The blind ladder (4)** was advertised as the thing that protects p99 under
load, and it could not see the load. A burn rate of exactly 0.0 across 2,688
queries is not a controller choosing to hold; it is a controller with an empty
input. The observation now happens in the pipeline, where every query passes
regardless of transport. The HTTP layer keeps the failures — which never reach
the pipeline — and gives up the successes, which would otherwise be counted
twice and halve the apparent breach rate.

**The dead cache (9)** is the one that cost the most performance and hid the
longest. The guard read as conservative — a filtered result is not cacheable by
vector alone — but query understanding infers a collection filter on most
queries, so it was almost always true. Filters belong *in the key*, not in a
condition that skips the cache. Fixing the key raised the hit rate and bought
almost nothing, because the lookup happened *after* the encoder ran: a hit was
still paying for the most expensive part of the query it existed to avoid. So
there is now an exact layer in front of the embed, and a hit in the vector
layer teaches it — a cache that only learns from full work never learns from
itself.

**A note on the bake-off guard.** The cost model's crossover is derived from
what the device measures, which is the design — and it means the bake-off's
conclusion ("flat wins to 20,000 points") is binding only on a machine like the
one it was measured on. On a container whose numpy takes **7 ms** for a
2048×256 matvec — about 75 MFLOP/s, two orders of magnitude off a healthy CPU —
the linear scan really is slower than a graph walk at that scale, and a model
preferring the graph there is the model working.

The guard used to assert the bake-off's *conclusion* directly, so it failed on
such a machine and read as a regression in code that had not changed. It is now
three assertions that hold on any device — an honestly timed hop must cost
measurably more than a vectorised one (which is the actual defect, timed both
ways on the same data); the calibration must be stable across consecutive runs;
and `choose` must have the right shape around the model's own crossover — plus
the bake-off assertion, which now measures the machine first and **skips with
the measurement in the skip reason** rather than failing quietly or passing
vacuously.

**The empty results (5 and 6)** are the pair that would have ruined a demo.
Query understanding read "conveyor vibration night shift" as sensor intent and
applied `collection="sensor"` as a hard filter, though the caller asked for
`*`; the cache bug had been hiding it by serving the unscoped answer to every
scoped query. Inferred narrowing is now advisory: when it matches nothing the
query is retried at the scope the caller asked for, and the result records what
was dropped. Only what *this layer* added is ever backed out — an explicit
filter is obeyed even when it matches nothing, and tenant isolation is
explicitly excluded, because an empty visible set is a boundary, not a bad
guess, and retrying wider there would turn a correct empty answer into an
isolation breach.

---

## 3.1 Deterministic simulation — a seed for every bug

Load testing answers "is it fast?". It cannot answer "is it correct under an
ordering I did not think of?", because the orderings that break distributed
systems are the rare ones, and a load test explores whichever orderings the
operating system happens to hand it.

So the whole fleet was made reproducible. `aegis/core/determinism.py` holds a
swappable ambient environment; inside `simulated(seed)`, `determinism.now()`,
`determinism.monotonic()`, `determinism.rng()` and `determinism.sleep()` are
served by a virtual clock and a seeded generator. Time moves only when the
simulation moves it, and `await sleep(3600)` advances an hour and returns
immediately.

The consequence is the point: **an entire distributed execution becomes a pure
function of one integer.** Twelve peers, four hundred steps, partitions,
crashes, clock skew and three Byzantine attacks — all of it replays byte for
byte from the seed. A failure found once is a failure that can be found again,
on any machine, forever.

### The lint that keeps it true

One `time.time()` left in a simulated module and a seed silently stops
reproducing — the run still passes, it just no longer means anything. So it is
not a convention, it is a build step:

```bash
python3 scripts/audit_determinism.py
#   skip  aegis/sync/compression.py   (no clock or randomness in a codec)
#   skip  aegis/sync/meshlink.py      (measures real RTT to a real peer over HTTP)
#   skip  aegis/sync/transport.py     (network clients own their own timeouts)
#   skip  aegis/core/determinism.py   (it *is* the environment)
#
#   12 simulated modules draw time and randomness from the environment.
```

It is an AST pass, not a grep, and each exemption carries the reason it is
exempt. It also checks that a module reaching for `determinism` actually
imports it — a rule added after two modules were converted without their
import and the *test suite* found out rather than the linter. The failure mode
there is a `NameError` on a clock call that only runs under load or during
recovery, a long way from the change that caused it.

The list of modules it covers was also learned the hard way. Two sweeps over
the same 500 seeds returned **454** failures and then **456**. Two out of five
hundred reads as noise; it was not noise. Operation ids came from
`time.time()` and `os.urandom`, and an operation id is not cosmetic — it is
what the IBLT hashes and what a fetch is sorted by, so reconciliation was
drifting while every other part of the execution replayed perfectly. Ids now
come from the environment, the per-millisecond counter is reset when a
simulation begins, and `ids.py` is inside the lint's scope. The two sweeps now
agree exactly: same failing seeds, same details, the same 80,618 operations
exchanged.

That is the mechanism working on itself. A simulator whose own instrument
drifts produces numbers that look like measurements.

### What the world can do

`aegis/sim/world.py` draws each step from a weighted population:

| Action | Weight | What it does |
|---|---|---|
| `write` | 34 | a device records a memory; 15% are marked never-leaves-the-device |
| `gossip` | 30 | one anti-entropy round between two peers |
| `partition` / `heal` | 8 / 8 | cut and restore a link |
| `skew` | 6 | step a clock, in either direction |
| `crash` | 5 | lose everything a device had not yet shared |
| `duplicate` | 5 | a peer redelivers what it already sent |
| `forge_clock` | 4 | a peer stamps an operation a century ahead so it wins every conflict — **signed honestly**, so only a bound on the clock can stop it |
| `impersonate` | 4 | a peer writes an operation in another device's name |
| `tamper` | 4 | a relay forwards somebody else's operation with the body rewritten |
| `idle` | 4 | time passes and nothing happens, which is most of a real device's life |

After **every** step, seven invariants are checked:

| Invariant | The promise |
|---|---|
| `no-fabrication` | no device holds an operation no device ever wrote |
| `policy` | a never-leaves-the-device memory is on exactly one device |
| `origin-retains` | the device that wrote a memory still has it |
| `no-duplicates` | redelivery never double-applies |
| `clock-monotonic` | a device's own clock never goes backwards in its log |
| `clock-not-poisoned` | one peer claiming the future cannot drag the fleet there |
| `bodies-intact` | what a device reads is what the author wrote |

When one breaks, the run stops and a delta-debugging shrinker takes over:
remove one action, replay, does it still fail? What survives is the shortest
history that still reproduces it — usually short enough to read in full.

### What it found

**Seed 5, immediately: egress policy was enforced on the sender only.** A
restricted memory written on `edge-01` was sitting on `edge-03`. The sending
side filtered correctly; the receiving side simply believed what it was handed.
That is right for an honest peer and worth nothing against a compromised one,
and a mesh of field devices is a population where "one handset is lost" is a
Tuesday. The receiver now enforces its own policy on arrival, counts what it
refuses, and says so on the event bus.

**Then the one that mattered: operations had no integrity protection at all.**
The `tamper` action — a relay forwarding an operation with the body rewritten —
was accepted by every downstream node in **435 of 500 executions** — every one
of them the same invariant, `bodies-intact`. The signed build runs the same
seeds, with the same attacks still firing, and the invariant holds.

The full signed sweep, with the results committed:

```
deterministic simulation  3,000 executions · 400 steps · 6 peers · signed

  3,000 executions in 5201.9s real time
  simulated            704,174 seconds (8.1 fleet-days)
  operations exchanged 1,733,206
  invariant failures   0 of 3,000 executions run
  no counterexample found. That is not a proof — it is 3,000 executions
  without one.
```

**8.1 fleet-days of a six-device mesh, just under two million operations
exchanged, seven invariants checked after every one of 1.2 million steps, and
nothing broke.** The last sentence of that output is the script's own, and it
is there because the alternative — printing "verified" — would be a lie about
what a sweep is. It samples the space of orderings. It does not cover it.

```
deterministic simulation  500 executions · 400 steps · 6 peers · unsigned (control)

  FAIL seed 497 · bodies-intact · edge-02 holds 01HF7YRWDT4K5PZCZ5XH04JDGT
                  with content that differs from what edge-04 wrote
  ...
  500 executions in 63.5s real time
  operations exchanged 80,618
  invariant failures   435 of 500 executions run
```

The control is kept runnable — `--unsigned` is a flag, not a deleted commit.
A defence you cannot switch off is a defence you can no longer demonstrate
works, and "the attack no longer fires" is only meaningful next to a run where
it does.

Worth being precise about why the Byzantine invariants initially found
*nothing*: the first sweep came back clean, because the invariants were not
watching what the attacks targeted. `clock-not-poisoned` and `bodies-intact`
were added *because* a clean result from an adversarial run is a claim about
the invariants, not about the system. A simulator that cannot fail is a
simulator that cannot tell you anything.

### The fix: every operation says who wrote it

`aegis/sync/identity.py`. Each device holds an Ed25519 key pair, persisted
beside its data — an identity regenerated at boot is not an identity, because
every peer would see the key change and, correctly, refuse everything the
device had ever said.

The signature covers `op_id`, `kind`, `point_id`, `hlc`, `device_id` and
`body`, over canonical sorted-key JSON, so two encodings of the same operation
produce the same signature. `ts` is deliberately *outside* the signature — a
relay may touch routing, it may not touch content — and there is a test that
fails if anyone widens that set without meaning to.

Measured on this hardware: **62 µs to sign, 129 µs to verify, 88 bytes on the
wire.** Against an 18 ms ingest that is 0.7% — and verification happens on
receipt, not on the query path.

Key distribution is trust-on-first-use, and the limit is written down rather
than glossed:

- The first key a device presents for an identity is believed.
- Any *later* change to that identity's key is refused, loudly, as an
  `identity_conflict` event.
- Keys travel with operations, so `edge-02` can verify an `edge-00` operation
  relayed by `edge-01` without ever having met `edge-00`. Without that,
  signing would break the partition tolerance it exists to protect.
- **This does not defeat an attacker present at the very first contact.**
  Closing that needs an enrolment authority, which is a deployment decision
  rather than a library one — and is now built: see §3.3.

What it does defeat, with a regression test each: rewriting an operation in
flight, writing in another device's name, and taking over an identity that is
already in use.

### The bug underneath the bug

Turning signing on broke honest convergence — 6 operations out of 28 reaching
their peers. The signatures were valid at rest and invalid on arrival, which
meant something between the two was changing the operation.

It was the wire codec. It elided a repeated low-cardinality field — one of
which is `sensitivity` — against a dictionary that lived on the **encoder** and
survived across frames, while the decoder started from an empty context on
**every** frame. The first operation carrying `sensitivity: restricted` sent
the label. Every later one sent a hole. Every hole was dropped.

That is worse than a compression bug, because the receiving node decides
whether it may hold an operation by reading exactly that field. An operation
the policy would have refused arrived looking ordinary — and the receive-side
check added at seed 5 was reading a label the compressor underneath it had
already removed.

Cross-frame state was never sound here anyway: this transport drops, reorders
and duplicates frames, so "the value from the previous frame" is not something
a receiver can know. The context is now rebuilt per frame and never survives
one; a hole with nothing to fill it from raises rather than passing silently.
Compression is barely affected, because the redundancy that pays for it is
*within* a batch.

**This is the argument for signing, stated as cleanly as it can be stated.**
Signing did not prevent this bug. Signing *revealed* it — a silent data-loss
path, sitting on a security control, that four hundred passing tests and nine
phases of load testing had not touched.

### Running it

```bash
cd backend
python3 scripts/audit_determinism.py                          # the lint
python3 scripts/simulate.py --seeds 3000 --steps 400 --peers 6 \
        --out ../testlogs/simulation.json                     # the sweep
python3 scripts/simulate.py --seeds 500 --unsigned --no-shrink # the control
python3 scripts/simulate.py --replay 5                        # see one seed again
```

A clean sweep is **not a proof**, and the script says so in its own output: it
is "no counterexample in N executions", with N printed so a reader can judge
it. It samples the space of orderings; it does not cover it.

---

## 3.2 The invariants run in production too

Three thousand simulated executions finding no counterexample is a real result
and a bounded one. It says the code holds under the orderings the simulator
sampled, with the faults it knows how to inject, on the machine that ran it. It
says nothing about the device in somebody's hand.

So `aegis/core/invariants.py` checks the same seven properties against the
running node, every two seconds, for as long as it is up.

```
GET /api/v1/integrity/invariants

  ticks        1,284
  assertions   8,988
  violations   0
```

The headline is on `/api/v1/health` as well, and the breakdown — per invariant,
with its promise, its scope and its own counters — is on the route above.
`POST /api/v1/integrity/invariants/check` runs a tick immediately.

**Two things make it affordable.** Verifying every signature in the log on
every tick is O(n) per tick and O(n²) over a run, which would make the check
the most expensive thing the node does. Each invariant instead advances a
rolling cursor through its own subject and spends a fixed budget, so a long log
is covered across many ticks rather than all at once — the cost is bounded by
the budget, not by how much the device has remembered. There is a test that
fails if a 4,000-operation log costs materially more per tick than a 64-operation
one, and another that fails if the cursor never moves, because a budget that
only ever examined the first 64 entries would be theatre.

**The local checks are weaker than the simulated ones, and say so.** The
simulator is omniscient: it can assert things about a fleet that no single
device can see. A node sees only itself. Each invariant carries the scope of
what it can actually establish, so the API reports `this device's own egress
set; it cannot see what peers hold` rather than implying a guarantee about the
mesh.

| Invariant | What a node can establish about itself |
|---|---|
| `bodies-intact` | every signed operation it holds still verifies against its author's key |
| `policy` | nothing its own policy refuses is in the set it would hand a peer |
| `no-fabrication` | every operation names a device, and a signed one names a device it has a key for |
| `no-duplicates` | no operation id appears twice in its log |
| `clock-monotonic` | successive reads of its own clock never go backwards |
| `clock-not-poisoned` | its clock is not days ahead of its own wall clock |
| `origin-retains` | operations it authored are still in its log |

**A violation is not an exception.** It is counted, published on the bus at
`error`, and kept in a ring of recent findings. The node keeps serving, because
a node that halts on a detected inconsistency converts a partial fault into a
total one, and an operator needs the evidence more than they need the process
dead. A check that *throws* is itself recorded as a finding — otherwise the
assertion counter would keep climbing while nothing was being checked, which
looks exactly like health.

Every invariant has a test that breaks the property on purpose and asserts the
monitor fires. A monitor that never fires is indistinguishable from one that
cannot.

---

## 3.3 Enrolment — closing the gap the signing work named

§3.1 ends by admitting what trust-on-first-use cannot do: an attacker present
at a device's *very first* contact is believed, because first contact is the
one moment with nothing to compare against. That is now closed, for fleets that
want it closed.

```bash
python3 scripts/enrol.py init --out fleet/                      # once per fleet
python3 scripts/enrol.py device --fleet fleet/ --id edge-01 --data /var/aegis/edge-01
```

The fleet has one root key pair. Every device is issued a certificate — its id
**bound to its public key**, signed by the root — before it ships. A node
holding the root *public* key accepts a peer only on a valid certificate, so a
stranger at first contact is refused like any other: it cannot produce the
root's signature over a name it was never issued.

Both halves are covered by the signature on purpose. Signing the key alone
would let a device present somebody else's certificate under its own name;
signing the id alone would let it present any key it liked under a name it was
issued. There is a test for each.

The root private key is written once and never leaves the directory it was
made in. It is not on any device and is not needed to run one — only to enrol
the next. Losing it means you cannot add devices; leaking it means somebody
else can, which is the whole of its threat model.

**It is a switch, not a spectrum.** A node either requires certificates or it
does not:

```bash
AEGIS_FLEET_ROOT=<root public key>  AEGIS_DEVICE_CERT=<this device's certificate>
```

With both set the node requires every peer to be enrolled. With either missing
it falls back to trust-on-first-use and **says so** in
`/api/v1/mesh/status`, under `mode`. Requiring enrolment with no root key to
check against is refused at construction rather than quietly downgraded — a
node that claimed enrolment and believed strangers anyway would be worse than
one that never claimed it, because the claim is what an operator plans around.

Enrolment does not replace the change-refusal from §3.1; it runs in front of
it. A certificate says whose a key is. It does not say an identity may have two
keys at once, so a reissued device is still refused by peers that knew the old
one, loudly, rather than swapped in quietly.

---

## 3.4 The open defect, narrowed to one variable

The README has carried an open item for a while: resident-set growth under
concurrent load, roughly 2.6 KB per query, cause unknown. It is still open.
It is no longer vague.

```bash
cd backend && python3 scripts/repro_growth.py

  without a sequential warm-up   + 0.06 MB over 8,000 queries =     7.2 B/query
     with a sequential warm-up   +22.17 MB over 8,000 queries =  2,771.5 B/query

  385x more resident growth per query, from one difference before the
  measurement window opened.
```

Two runs, identical in every respect but one: whether sixty-four **sequential**
searches happen before the load. Both then warm with 8,000 concurrent queries
and measure the next 8,000, so neither is measuring start-up.

That warm-up length is not adjustable, and the reason is worth stating: run at
2,400 queries instead, the same script reports **12,029 B/query for the clean
arm against 372 for the dirty one** — the opposite conclusion, stated just as
confidently. The arm that has *not* done a sequential warm-up pays more of its
start-up inside a short measured window. A reproducer that inverts under a
smaller budget is a trap, so the budget is fixed and the script says why.

### What this round established

**It is a leak, not a transient.** Thirty-two thousand queries, sampled every
two thousand: the first ~8,000 carry a large one-time cost, and everything
after is flat at 2,740–2,850 B/query with no sign of levelling.

```
  8,000 queries   total + 64.07 MB   window  8,175 B/query   ← start-up
 16,000 queries   total + 86.22 MB   window  2,766 B/query
 24,000 queries   total +108.42 MB   window  2,785 B/query
 32,000 queries   total +131.63 MB   window  2,746 B/query
```

**It is not any pipeline stage.** The SLO ladder is a ready-made ablation
instrument — each rung switches off another stage — so every rung was run with
the load held identical. With a concurrent warm-up, *every* rung sits at
6–70 B/query, top to bottom. There is nothing to attribute to graph boosting,
late interaction, the adapter, query understanding, the cross-encoder or
diversity, because none of them leaks.

A first attempt at that ablation ran all five rungs in one process and produced
a clean staircase — 41.9, 21.9, 6.1, 2.6, 2.6 MB — which looked like a decisive
attribution. Running the rungs in the *reverse* order produced 3.9, 2.6, 2.6,
41.4, 21.9: the numbers tracked position, not rung. The staircase was the
one-time start-up cost being absorbed by whichever rung ran first. Each rung now
runs in its own process and measures only the flat region.

**It is not the ONNX runtime's shape planning.** `enable_mem_pattern = False`
was the one knob in that area never tried, and it produces the same numbers to
within a tenth of a megabyte at every checkpoint — alongside
`enable_cpu_mem_arena = False`, ruled out earlier the same way.

**The obvious fix does not work.** If the trigger is the node's first inference
running at batch size one, then making the node warm itself concurrently at
boot should disarm it. Measured: 2,770.9 B/query, unchanged. The sequential
searches arm it wherever they happen, not only when they come first. That is
why it is not in the shipped code — a fix that does not fix it, shipped on the
strength of a plausible story, would be worse than an open item.

### What is left

A reproducer this sharp is most of the way to a cause. What is known: the
encoder at batch one is clean in isolation (0.0 MB over 12,000 texts), so it is
not simply "small batches allocate". Something about running the *whole
pipeline* at concurrency one puts the process into a state that then leaks
under concurrency eight, and stays in it.

Stated plainly because the alternative is worse: a device that serves a few
queries a minute before a burst — which is most of them — will drift. On this
hardware that is roughly 2.8 KB per query; a node answering ten thousand
queries a day grows about 28 MB a day. It is not a reason to avoid deploying
this; it is a reason to restart nodes on a schedule until it is found, and
that is a sentence an operator can act on.

---

## 3.5 Scale: where it broke, and what it costs now

Every number in [Measurements](MEASUREMENTS.md) was taken on a small corpus. "Edge" does not mean small —
a body-cam fleet or a factory line generates millions of records — so the
question is not how fast the node is at two thousand memories but which term
in its cost grows with the corpus, and how far that carries.

Two terms did, and both were the same mistake in two places.

### A vector stored as a list of Python floats

A 256-dimension vector costs **8,344 bytes** as a `list[float]` and **1,136**
as a float32 array, because every element is a separate 24-byte object with a
pointer to it. The node stored one on every `MemoryPoint` — and another,
uncompressed, inside every `Operation` in a log that is never trimmed.

Measured in one process, on the same vector:

| | bytes | |
|---|---:|---|
| list of 256 Python floats | 8,344 | |
| float32 ndarray | 1,136 | 7.3× |
| int8 quantized, as an operation body carries it | 877 | 9.5× |
| **per memory, before** (point list + op list) | **16,688** | |
| **per memory, after** (point array + op codes) | **2,013** | **8.3×** |

`MemoryPoint.dense` is float32 now. The trap that introduces is worth naming:
`if point.dense` *raises* on an array of more than one element, so there is
`has_dense`, and `__post_init__` normalises whatever a caller passes — points
are built from the wire, from the WAL, from a peer's operation and from tests,
and any one of those handing over a list would leave a point costing eight
times what it should.

### What that turned up underneath

Operation bodies now carry the vector already quantized, rather than being
quantized by the wire codec on the way out. The codec used to do it — a lossy,
non-round-tripping transform, applied to content that is **signed**.

The vector that was signed was therefore never the vector that arrived. Every
real upsert carrying a vector failed verification at the receiver and was
refused as a forgery: a mesh that would have looked, from the inside, like it
was under attack by its own peers.

Nothing caught it. Not the 332 tests — none had signed a body with a vector in
it. Not the deterministic simulator — its operations carry text and a
sensitivity label and no vector at all. It surfaced from a test written to
check a *memory* saving, which is not where anybody would have gone looking.

The fix is a principle rather than a patch: **the codec compresses, it does
not edit.** Quantizing happens where the operation is built, which is the only
place it can happen without the signature and the wire disagreeing. The wire
is the same size it was; what changed is that what a peer checks is what the
author signed.

One number in the old codec test was flattering itself, and is now honest. It
asserted a compression ratio above 5×, reachable only because `raw_bytes` was
measured *before* the lossy transform and `wire_bytes` after it. Against the
same content the ratio is 2.2×, and an operation carrying a 256-dimension
vector costs about **325 bytes** on the wire.

### The curve

Identical methodology either side — ingest to 10,000 points, 2,500 at a time,
sampling ingest rate, query latency and resident set at each step. This
isolates the operation-body change; the float32 point change is in both arms.

| Points | | ingest | query p50 | query p95 | resident |
|---|---|---:|---:|---:|---:|
| 2,500 | before | 59.9 /s | 19.8 ms | 27.8 ms | 455 MB |
| | after | **67.5 /s** | 19.9 ms | **27.1 ms** | **385 MB** |
| 5,000 | before | 55.4 /s | 23.7 ms | 34.4 ms | 629 MB |
| | after | **58.4 /s** | **22.1 ms** | **33.1 ms** | **514 MB** |
| 7,500 | before | 42.4 /s | 30.5 ms | 55.8 ms | 817 MB |
| | after | **53.3 /s** | **28.9 ms** | **33.3 ms** | **602 MB** |
| 10,000 | before | 35.0 /s | 64.6 ms | 170.9 ms | 1,041 MB |
| | after | **48.7 /s** | **36.3 ms** | **64.9 ms** | **764 MB** |

**At ten thousand memories: 1.39× the ingest rate, 2.6× better p95, and 277 MB
less resident.** The marginal cost of a memory fell from 78.1 KB to 50.5 KB.

### What is still true

Ingest still decays with corpus size — 67.5 to 48.7 docs/s over ten thousand
points — and resident memory still grows at 50 KB per memory, which is far
more than the 2 KB the vectors now account for. Both are real and neither is
fixed. The remaining bulk is the operation log, which no process trims: it
retains every write forever so that a peer which has been offline for a month
can still reconcile. That is a deliberate property with an undeliberate
bound, and compaction — dropping bodies that the store can rebuild, keeping
ids for idempotency — is the obvious next step and is not done.

Reproduce the curve with `backend/scripts/scale_probe.py`.

---

## 3.6 Where a memory's resident cost actually goes

§3.5 ends with a number rather than an explanation: a memory costs about
50 KB of resident set, and the vectors now account for 2 KB of it. That is an
uncomfortable place to leave a scale claim, so the rest was attributed.

```bash
cd backend && python3 scripts/memory_breakdown.py --count 6000
```

Ingest six thousand memories, then walk the node's own structures. A shared
`seen` set runs across every subsystem, so an object two of them reference is
charged once — which makes the individual rows approximate and the total
sound, the right way round for this question.

| | | per memory |
|---|---:|---:|
| vector index | 32.2 MB | 5.2 KB |
| operation log | 19.6 MB | 3.2 KB |
| `store.points` | 15.7 MB | 2.6 KB |
| query understanding | 7.9 MB | 1.3 KB |
| embedder geometry | 5.0 MB | 0.8 KB |
| op log index, graph, mesh, merkle, sparse, cache | 3.4 MB | 0.5 KB |
| **attributed** | **83.9 MB** | **13.7 KB** |
| **unattributed — native, invisible to Python** | **203.7 MB** | **33.2 KB** |

Three candidates were then removed one at a time and measured:

| Removed | Resident change |
|---|---:|
| Embedded Qdrant's upsert (internal store write kept) | 2.9 KB/memory |
| glibc arenas capped at two (`MALLOC_ARENA_MAX=2`) | 4.1 KB/memory |
| `malloc_trim(0)` after the ingest | 1.6 KB/memory |

**None of them is the bulk.** Roughly 25 KB per memory is native, grows with
the corpus, and is not yet accounted for. That is written here because a scale
section that reported only the things it had explained would be describing a
different system than the one that runs.

### What changed as a result

`MALLOC_ARENA_MAX=2` is now set by `scripts/supervise.py` before the node
starts — it has to be in the environment first, because libc reads it once.
Measured on a 4,000-memory ingest: **43.8 KB per memory unbounded against 39.7
with two arenas**, for no change in throughput. It is a `setdefault`, so an
operator who has tuned it for their own hardware keeps their value.

### What did not change, and why

The operation log retains every write forever, so that a peer which has been
offline for a month can still reconcile. Compacting it — dropping bodies the
store can rebuild, keeping ids for idempotency — was the obvious next move and
was the plan until this table existed. At **3.2 KB of 46.8**, it would buy
about seven per cent, in exchange for a change to the part of the system that
signatures, gossip and the `bodies-intact` invariant all depend on.

Measuring first turned a confident plan into a bad trade. It stays on the open
list, honestly sized.

---

---

## 3.7 Reviewing my own security work

The signing, the enrolment and the live invariants were written by the same
person who then wrote the section saying they were sound. That is the
arrangement under which bugs survive, so this is a pass over that code with the
opposite intent: assume it is wrong, and find where.

It found five things. Two of them are remotely exploitable.

### A decompression bomb, on an unauthenticated endpoint

`/api/v1/mesh/exchange` has no credential, by design — a peer device has to be
able to reach it — and it handed the frame straight to `zlib.decompress` with
no bound on the output.

```
wire bytes an attacker sends: 50,976
peak RSS grew 50 MB from a 50,976-byte request
```

**About a thousandfold**, which makes a 5 MB request worth 5 GB. Anyone able to
open a socket to the node could exhaust it. The limits are now 8 MB on the wire
and 16 MB expanded, checked before decoding rather than after, using
`decompressobj(...).decompress(data, max_length)` so the expansion stops at the
ceiling instead of completing and then being measured. The same request now
allocates nothing.

A bound that also breaks anti-entropy would be the worse bug, so there is a
test that a real 400-operation frame — each carrying a 256-dimension vector —
still round-trips inside the limit.

### The same class again, one handler over

The digest handler took its table size from the peer:

```python
cells = int(payload.get("cells", 128))
table = IBLT(cells).insert_many(mine.keys())
```

`{"cells": 3000000}` allocated 69 MB, linearly in the number. The fix is not a
magic constant but the observation that **a difference table never needs more
cells than the set it is reconciling** — so the request is bounded by what this
node could actually fill, and by an absolute ceiling above that. A request for
100 million cells now builds 128.

Finding the first bug is worth less than finding its class. The fetch handler's
`op_ids` list was unbounded for the same reason and is capped the same way.

### The attack route was the one unauthenticated write in the API

§3.2's demonstration — mount the simulator's attacks against the live node from
the browser — shipped without a scope guard, while every other write in the API
had one. A route whose entire purpose is to hand a node something a peer should
not be able to hand it, and which teaches that node a key on the way, is a poor
choice for the exception. It requires `ADMIN` now, and a test asserts that a
principal holding `WRITE` is refused.

### The probe was squatting real device names

Its identities were `edge-victim` and `edge-attacker`, learned into the node's
**real** trust store. Trust-on-first-use is permanent by design — that is what
makes it useful — so any fleet that ever shipped a device by one of those names
would have found it refused forever, by a demo. They are namespaced now, with
this node's own id, which no real device will present.

### A drill was indistinguishable from an incident

Running the probe moved `refused_forged`: the counter an operator watches to
learn they are under attack. A drill that looks exactly like an attack in your
telemetry is worse than no drill, because it teaches people to ignore the
alarm. Probe effects are accounted separately, and a test fails if three
refused forgeries from a drill move the incident counter.

### And two smaller ones

Private keys were written and then `chmod`-ed, which leaves a window in which
they exist at whatever the umask allows — commonly world-readable. For the
fleet root — whoever holds it can enrol a device into the fleet, that window
is worth closing. They are created with the mode already set, atomically. The
test watches the file from the moment it appears rather than checking after the
call returns, because after the call returns is exactly when the old code
looked correct.

And one of these tests was itself flaky. The calibration-stability guard
compared two single timing samples: fine on an idle machine — eight consecutive
runs spread 1.13× — and unreliable on the loaded one that the test suite itself
creates. It compares medians of five now, which survives four deliberate CPU
burners and still catches the 2.75× swing the original defect produced.

### What this says about the rest

Five findings in code written carefully, by someone who had just finished
arguing it was correct. The honest conclusion is not that the code is now
clean; it is that a second pass with adversarial intent is worth more than a
first pass with careful intent, and that nothing here has had a *third*.

---

## 3.8 The correctness review, and the dead assertion behind it

§3.7 was a security pass. This is a second pass with a different lens —
correctness rather than exploitability — over the same code. It found ten
things. One of them was hiding a vulnerability, and one of them means a
sentence in an earlier version of this document was false.

### Two of the seven invariants asserted nothing

`_clock_monotonic` and `_clock_not_poisoned` both read
`agent.causal.clock.last`. `VectorClock` has no attribute `last`. The lookup
returned `None`, the loop hit `continue` on every agent, and both checks
returned clean for **three thousand executions**.

So the claim "seven invariants checked after every step" was five. It was found
by reading the code, not by a failure, which is how a dead assertion is always
found: it cannot report itself.

### Behind it was a real, undefended attack

The simulator mounts `forge_clock` — a peer stamping an operation a century
ahead. Last-writer-wins resolves by `HLC.dominates`, which compares `wall_ms`
and has no opinion about whether that number is plausible, so such an operation
beats every honest write to the same point forever.

With the invariant repaired, the very first sweep said so:

```
seed 0 -> clock-not-poisoned: edge-01 holds 01HF7YAWE3D3QTWT3919WBRWBN
          stamped 36,458 days beyond the world's clock
```

Forty seeds out of forty. The check had been written to catch exactly this,
and had been unable to fail since the day it was written.

The defence is a bound: `GossipAgent` refuses an operation whose clock is more
than an hour beyond its own, counts it, and says so on the bus. An hour is
generous against real drift, and the cost is stated rather than hidden — a
device whose clock is badly wrong has its writes refused, and is told, rather
than discovering it later as silent data loss.

| | signed sweep |
|---|---|
| bound removed (the defence off) | **40 of 40 executions falsified** |
| bound in place | **0 of 60** |

A guard that cannot fail proves nothing, so both directions are measured.

### A tautology, in the live invariants

`_policy_holds` asked whether anything in `_shareable()` failed `may_share`.
`_shareable()` is *defined* as the set filtered by `may_share`. It could not
fail against real code, and fired only against a test stub whose fake
`_shareable()` returned everything.

It now asks the property the simulator actually found at seed 5: a device may
keep its own restricted memories and must not hold anybody else's. There are
two tests — one that a foreign restricted memory is caught, one that this
device's own is not — because the second is what makes the first meaningful.

### Enrolment converged exactly one hop

Keys travel with operations so a node can verify an author it has never met.
In enrolment mode a key without its certificate is refused — and every node
offered its own certificate and nobody else's. Measured on a four-device chain:

```
hop 1 (direct):  edge-01 has it: True
hop 2 (relayed): edge-02 has it: False    refused_forged: 5
```

The two-node test passed throughout, because two nodes are always one hop.
Certificates are relayed now, which is safe for the reason they exist: a
certificate carries the root's signature over an id bound to a key, so the
recipient checks the root rather than the carrier. Four hops, zero forgeries,
with a test that walks the chain without touching the author again.

### The determinism lint had a blind spot

`field(default_factory=time.time)` never appears as an `ast.Call` on
`time.time`, and the audit only inspected calls. Two of them sat in
`causal.py` and `conflict.py` through every sweep, comparing virtual time
against the real wall clock — which made one buffer-expiry path dead in every
seed, and meant the "pure function of its seed" claim had a hole in it that the
tool built to guarantee it could not see.

Both converted; the lint now catches a banned name passed as a value, verified
by reintroducing the pattern and watching it fail.

### And four smaller ones

- **The honest probe left a real memory behind.** The accepted case is
  materialised by design — indexed, retrievable, gossipable — and only
  `mesh.known` was rolled back. A drill that leaves real state behind is not a
  drill. The point is deleted now, and the test asserts it is neither resident
  nor findable.
- **`set_dense` did not copy.** `np.ascontiguousarray` returns the *same
  object* for an already-contiguous float32 input, which is exactly what the
  micro-batcher produces — so every point held a view into the batch array it
  came from, pinning the buffer. The docstring claimed a copy. It is one now.
- **Two live invariants scanned everything** every two seconds, against a
  module whose docstring promises a bounded budget. Both windowed.
- **`set.pop()` evicted an arbitrary id** from the authored-operations cache,
  frequently the newest — so `origin-retains` had unpredictable gaps past the
  cap. Oldest-first now.

Plus an unbounded key store on the unauthenticated endpoint, which is the same
class as §3.7's three and was missed by that pass.

### What two reviews say that one does not

The security pass found five. This pass, over the same code, found ten more,
including two assertions that had never worked and one vulnerability they were
written to catch. Neither pass was lazy. The conclusion is not that the code
is now correct — it is that **the number of passes is the variable**, and this
code has had two.

