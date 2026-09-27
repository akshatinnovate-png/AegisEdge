# How This Was Tested

The argument of this project is not "it works". It is "here is what would have
told us if it did not".

## Deterministic simulation

A seeded virtual clock and a seeded RNG make a whole distributed execution a
pure function of one integer. A fleet of peers, weighted fault and Byzantine
actions, invariants checked after every step — and when a seed falsifies an
invariant, a delta-debugging shrinker reduces it to the smallest schedule that
still fails.

A bug is no longer a story about what someone saw. It is an integer.

```bash
python3 scripts/simulate.py --seeds 200                # expect nothing
python3 scripts/simulate.py --seeds 200 --unsigned     # expect plenty
```

| | signed | unsigned control |
|---|---|---|
| executions | 3,000 | 500 |
| invariant failures | **0** | **435** |
| simulated fleet-time | 8.1 days | |
| operations exchanged | 1,733,206 | |

**CI fails the build if the control comes back clean.** A sweep that finds
nothing means nothing on its own; it means something beside a run, on the same
seeds with the same attacks, that finds something. If the control ever goes
quiet, the simulator has stopped mounting the attack — and the clean sweep above
it has stopped being evidence.

## The determinism lint

A single `time.time()` or `random.random()` in simulated code makes a seed stop
reproducing, silently. `scripts/audit_determinism.py` is an AST lint: no
simulated module may reach past the ambient environment. It covers **14
modules**, names the four it exempts and why, and catches banned names passed as
values — not just called.

It runs first in CI and on its own. If simulated code has started reading the
real clock, every result below it is about a different program than the one the
seeds describe.

## Bake-offs

A measurement without a control is a number. Three pairings:

| Bake-off | Control | Finding |
|---|---|---|
| `strategy_bakeoff.py` | flat vs HNSW vs IVF-PQ, one corpus | The cost model was choosing the graph from 5,000 points when flat was 4.9× faster and exact. Crossover recomputed to 110,000. |
| `qdrant_bakeoff.py` | the engine vs the local index | Identical stage for stage; the whole top-k difference is the fusion constant. Found a double length-normalisation in our sparse scorer. |
| `egress_bakeoff.py` | value-first vs write order | On a 16 KB link, FIFO landed 0 of 7 deletes; value-first landed 7 of 7. |

Each script prints a **verdict computed from its own table**, not typed into it.

## The claims audit

Prose does not fail a build, which is how almost every README in the world ends
up lying slightly: a number is right when it is written and wrong three commits
later.

`scripts/audit_claims.py` checks every measured claim against the committed
artefact that produced it, and exits non-zero on drift. It has caught drift
three times, including **on itself** — double-rounding `fleet_days` so the
checker expected "8.2" where the script printed "8.1".

It also verifies that the duplicated documents agree: Appendix 2 of the README
must match `backend/README.md` whole, and every `##` section of every `docs/`
module must appear in the README.

## Continuous integration

Every push runs:

1. The determinism lint (first, alone) — plus a staleness check on its own
   module count
2. The documentation claims audit
3. `STATISTICS.md` checked for staleness
4. The full test suite — **385 tests**
5. A signed simulation sweep, expected to find nothing
6. An unsigned control, **which fails the build if it comes back clean**
7. The Qdrant bake-off — fails if the two query paths stop computing the same
   pipeline
8. The egress bake-off — fails if a budgeted link stops carrying the obligations
   first

## The shape of the test suite

385 tests across 28 files, about 26% the size of the code they test. That is not
a target; it is what happened when every defect had to arrive with the test that
would have caught it.

Tests are named as sentences, because a failure should read as a statement about
the system:

- `test_a_tombstone_that_reuses_the_points_clock_never_deletes_anything`
- `test_the_merge_is_order_independent_which_is_what_lets_egress_reorder`
- `test_a_tenancy_allow_set_is_never_handed_to_the_engine`
- `test_sparse_ranking_does_not_penalise_a_long_memory_twice`
- `test_migration_refuses_to_delete_points_only_qdrant_holds`
- `test_an_inferred_narrowing_never_empties_the_results`

Several assert the **broken** behaviour alongside the fixed one, because a
regression test that only checks the fix cannot tell you the bug was real.
