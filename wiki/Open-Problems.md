# Open Problems

Four things are wrong or unexplained. They are written down rather than closed
quietly, because a list of known unknowns is worth more than a clean sheet that
took effort to keep clean.

---

## 1. ~2.8 KB per query of resident growth

**Status:** open, narrowed to one variable, reproducer committed.

```bash
cd backend && python3 scripts/repro_growth.py

  without a sequential warm-up   + 0.06 MB over 8,000 queries =     7.2 B/query
     with a sequential warm-up   +22.17 MB over 8,000 queries =  2,771.5 B/query
```

Two runs, identical in every respect but one: whether sixty-four **sequential**
searches happen before the concurrent load. **385× more growth per query from
one difference before the measurement window opened.**

What has been established:

- **It is a leak, not a transient.** 32,000 queries sampled every 2,000: the
  first ~8,000 carry a one-time cost, everything after is flat at
  2,740–2,850 B/query with no sign of levelling.
- **It is not any pipeline stage.** The SLO ladder is a ready-made ablation
  instrument — each rung switches off another stage. With a concurrent warm-up,
  *every* rung sits at 6–70 B/query, top to bottom. Nothing to attribute to
  graph boosting, late interaction, the adapter, query understanding, the
  cross-encoder or diversity.

The measurement budget is fixed and the script explains why: at 2,400 queries
the same script reports the **opposite** conclusion just as confidently, because
the arm that has not done a sequential warm-up pays more of its start-up inside
a short window. A reproducer that inverts under a smaller budget is a trap.

---

## 2. ~25 KB per memory of unattributed resident cost

**Status:** open, measured, unexplained.

`scripts/memory_breakdown.py` accounts for what it can — vectors, index
structures, payload, graph — and a gap remains that none of the named
subsystems own. It is reported as unattributed rather than assigned to whatever
is nearest.

---

## 3. Ingest decay with corpus size

**Status:** open, measured.

Ingest falls from **67.5 to 48.7 docs/s** as the corpus grows. The shape is
known; the dominant term is not isolated. `scripts/scale_probe.py` samples
ingest rate, query latency and resident set at each step, because the question
is not whether the node is fast at two thousand memories but what happens to
each of those as it grows — "edge" does not mean small.

---

## 4. Op-log compaction: measured, priced, declined

**Status:** closed as a deliberate non-fix.

Compaction was measured at **3.2 KB of 46.8 KB — about 7%** — and judged a bad
trade against the complexity and the risk of compacting history a peer has not
yet seen. It is written down as a decision with a number attached rather than
left as a silent omission.

---

## Things that are deliberately not done

| | Why |
|---|---|
| A remote Qdrant Server run | The adapter, wire format and configuration are real and exercised against embedded Qdrant. The org egress policy blocks the network, so the run itself is not claimed. |
| Real federated learning across devices | The secure aggregation round is real and tested in-process; a multi-device deployment is not claimed. |
| Triton escalation against a live GPU server | The client is real, the escalation path is real and declined while offline. A GPU server was not available. |

Each of these is a case where the code is honest and the *claim* is bounded, not
a case where a stub is standing in for something real.
