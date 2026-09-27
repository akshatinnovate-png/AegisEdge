# Console Guide

Three modes, one page, no build step. When the node is unreachable the console
says so and shows nothing — a dashboard that invents plausible values is worse
than a blank one, because you cannot tell the difference until it matters.

| Mode | For |
|---|---|
| **USE IT** | Ingest and ask. What the thing does. |
| **PROVE IT** | Four claims, each with the button that would falsify it. |
| **INSPECT IT** | The node console — everything below. |

## Cold boot

The boot screen reads the node's own `/health` while it plays, so each line is
filled in from the real answer rather than asserted: execution provider,
embedding model, vector backend, resident memories, uplink, degradation level.

## QDRANT · QUERY ENGINE

Draws **the plan the engine actually executed** — the two prefetches, the
fusion, the MaxSim stage — inside an engine boundary that goes dim and releases
the stages when the query ran on the local index instead. The graphic is a
record of where the work happened, not an illustration of the architecture.

There is **no millisecond on any individual stage**. The engine runs all of them
in one call and returns one timing for it; splitting that total four ways would
put a number on the screen that nothing measured. The call's real timing sits on
the boundary.

`RACE BOTH PATHS` runs the bake-off for one query live and draws two bars, with
top-k overlap and rank agreement underneath.

The badge row says `48 PAYLOAD IDX INERT (LOCAL MODE)` when that is true. The
embedded client warns that payload indexes have no effect there, and reporting
them as live would be the one lie this console exists to avoid.

## THE CORPUS, AS QDRANT HOLDS IT

The stored vectors projected to two dimensions, scrolled back out of the engine
rather than read from the page's own copy — a picture of the corpus sourced from
the thing drawing it would agree with itself whatever Qdrant stored.

The query is marked, its retrieved neighbours numbered and joined to it, and the
caption carries the number that keeps the plot honest: **the two axes hold about
23% of the variance.** The numbered hits are the nearest in 256 dimensions,
which is exactly why they are visibly *not* the nearest on the page. A vector
plot without that sentence is the most common way this kind of picture misleads.

The query is projected through the corpus's own basis rather than refitted with
it, so one vector cannot move every document to accommodate itself.

## EGRESS — WHAT THE LINK WOULD CARRY

Prices the queue **without sending anything**: the link and its budget, the
operations that would go now, and the reason each earned its place, in the
planner's own words —

> *a delete the cloud has not applied is a wrong state, not a stale one*
> *retrieved 6× locally — the fleet will want it*
> *waited 143s — promoted ahead of the ordering*

Beside it, what the same bytes would have carried in write order. When the whole
queue fits the link the panel says so plainly: both orderings carry the same
value, because ordering only decides anything once the link is too small for the
queue.

## The rest of the console

| Card | Shows |
|---|---|
| **LOCAL MEMORY** | Points resident, hot/warm/cold tiers |
| **LOCAL INFERENCE** | Execution provider, embedder, token budget, embed p95, escalations |
| **SYNC** | State, queued ops, divergent ranges, conflicts, `FORCE RECONCILE` |
| **DATA RENEWAL** | Dual-space migration state, re-embedded count, stale points |
| **HYBRID RETRIEVAL** | Ask the node what it remembers; results carry their scores and which space matched |
| **EVENT STREAM** | Live telemetry, sync events and reasoning traces over the WebSocket |

## The rule the whole console follows

Every number comes from the node. Nothing is interpolated, nothing is smoothed,
and when a subsystem cannot answer, the card says that instead of showing a
plausible zero.
