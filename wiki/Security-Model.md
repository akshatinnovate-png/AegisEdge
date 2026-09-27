# Security Model

A mesh where any device can send any other device an operation is a mesh where
any device can rewrite the fleet's memory. That is the threat this addresses.

## Device identity

Every device holds an **Ed25519** keypair. The private key is written with mode
`0600` via `os.open`, through an atomic temp-then-replace, so it is never briefly
world-readable.

Every operation is signed over its immutable fields — `op_id`, `kind`,
`point_id`, `hlc`, `device_id`, `body` — canonicalised so two spellings of the
same operation produce the same bytes. The signature is not part of what the
signature covers, for the obvious reason.

## Admission

An operation arriving from a peer is checked before anything else looks at it:

1. **Policy** — may this node hold a memory of this sensitivity at all?
2. **Clock plausibility** — an operation stamped a century in the future is
   refused. There is a bound (`MAX_CLOCK_DRIFT_S`) and it is enforced.
3. **Signature** — verified against the key this node has for that device.

Counters for each outcome — `verified_ops`, `refused_forged`, `refused_future`,
`refused_inbound` — are reported at `/api/v1/mesh/status`, so "the mesh is
secure" is a number rather than a claim.

## Enrolment

Trust-on-first-use is the default: the first key seen for a device id is the key
for that device id, and any later contradiction is a forgery.

TOFU has a real gap — whoever speaks first wins — so a fleet can close it. A
root key signs device certificates binding an id to a public key. With
`AEGIS_ROOT_PUBLIC_KEY` set, a peer without a valid certificate is refused
outright. `scripts/enrol.py` issues them.

Certificates relay: a device that has verified a peer's certificate can pass it
on, so enrolment converges across the mesh rather than requiring every device to
meet the root.

## What the simulator does about it

The deterministic simulator mounts four Byzantine behaviours:

| Attack | What it tries |
|---|---|
| `forge_clock` | Stamp an operation a century ahead so LWW always picks it |
| `impersonate` | Claim another device's id |
| `tamper_body` | Alter a body after signing |
| `duplicate` | Replay an operation |

Over **3,000 executions** with signing on: **0 invariant failures**. Over 500
with signing off: **435**. That pairing is the evidence. One number without the
other is decoration.

## Live attack surface

`/api/v1/mesh/attack` mounts one of four operations — honest, tampered,
impersonated, unsigned — against this node through the same handler a peer
reaches, and reports what it did with it. The honest case is the control.

Three things that took a security review to get right:

- The endpoint is **not** unauthenticated. It was.
- The drill uses reserved probe identities so it cannot squat on a real device
  id, and cannot be used to poison the key table.
- The drill's outcomes are counted separately from real incidents, so running it
  does not inflate the node's forgery counters.

## Multi-tenancy

Tenants have scoped API keys and quotas. The visible id set is intersected with
the query planner's allow-set, so a tenant cannot see another tenant's memories
through *any* path — including the semantic cache, which is namespaced, and
including the Qdrant engine, which is never handed a query whose filter resolved
to an explicit id set.

An empty visible set is a tenancy boundary, not an inference that went wrong.
The pipeline deliberately does **not** relax the scope and retry, because that
would turn a correct empty answer into an isolation breach.

## Audit

A hash-chained audit log records ingest, egress decisions, policy verdicts and
redactions. `/api/v1/audit` returns the entries and verifies the chain.
`/api/v1/provenance` returns a Merkle receipt over the model graphs and
configuration, with a file-by-file comparison.

## Bounds that exist because they were missing

A security review of this work found five defects. Four were resource bounds a
hostile peer could have used:

- A decompression bomb: 51 KB expanding to 50 MB. The codec now decompresses
  incrementally with a ceiling and checks for a truncated stream.
- An IBLT `cells` count taken from the wire: one integer could allocate 69 MB.
  Now bounded by `MAX_DIGEST_CELLS`.
- Unbounded fetch id lists and learned-key tables. Both bounded.
- Frame size ceilings on both the wire and the frame.

The general lesson, written into the code: **anything a peer controls is an
allocation a peer controls.**
