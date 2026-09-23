"""Durability: segment integrity, crash-safe manifests, fsck, PITR, repair."""
from __future__ import annotations

import json
import os
import random

import pytest

from aegis.memory.segments import (FOOTER, Manifest, SegmentCorrupt, SegmentReader,
                                   SegmentStore, SegmentWriter)


def _store(tmp_path, segments: int = 3, per_segment: int = 20) -> SegmentStore:
    store = SegmentStore(tmp_path / "segments", "edge-test")
    for generation in range(segments):
        store.write_segment(
            [{"lsn": generation * per_segment + i, "op": "upsert",
              "body": {"id": f"p{generation}-{i}", "text": f"record {i}"}}
             for i in range(per_segment)],
            checkpoint_lsn=(generation + 1) * per_segment)
    return store


def test_sealed_segments_roundtrip(tmp_path):
    store = _store(tmp_path)
    assert len(list(store.read_all())) == 60
    assert store.fsck().clean


def test_segment_id_is_its_content_hash(tmp_path):
    store = SegmentStore(tmp_path / "s", "n")
    first = store.write_segment([{"lsn": 1, "body": {"id": "a"}}])
    second = SegmentStore(tmp_path / "s2", "n").write_segment([{"lsn": 1, "body": {"id": "a"}}])
    assert first.segment_id == second.segment_id      # same bytes, same name
    third = store.write_segment([{"lsn": 2, "body": {"id": "b"}}])
    assert third.segment_id != first.segment_id


def test_bit_rot_is_detected_not_returned(tmp_path):
    store = _store(tmp_path)
    victim = store.manifest.segments[1]
    data = bytearray(open(victim.path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(victim.path, "wb").write(bytes(data))

    report = store.fsck()
    assert not report.clean
    assert "sha256" in report.corrupt[0]["reason"]
    assert store.lost_records(report) == 20
    with pytest.raises(SegmentCorrupt):
        list(SegmentReader(victim.path).read(strict=True))


def test_truncated_segment_is_detected(tmp_path):
    store = _store(tmp_path, segments=1)
    victim = store.manifest.segments[0]
    with open(victim.path, "r+b") as handle:
        handle.truncate(os.path.getsize(victim.path) - FOOTER.size - 10)
    healthy, reason = SegmentReader(victim.path).verify()
    assert not healthy
    assert "truncated" in reason


def test_repair_quarantines_rather_than_deletes(tmp_path):
    store = _store(tmp_path)
    victim = store.manifest.segments[1]
    data = bytearray(open(victim.path, "rb").read())
    data[10] ^= 0xFF
    open(victim.path, "wb").write(bytes(data))

    report = store.fsck(repair=True)
    assert report.repaired == 1
    assert report.quarantined == 1
    assert list(store.quarantine.glob("*"))            # kept for forensics
    assert len(list(store.read_all())) == 40           # the rest still reads


def test_manifest_survives_a_crash_mid_write(tmp_path):
    store = _store(tmp_path)
    expected = len(store.manifest.segments)
    # simulate a crash: a half-written temp manifest left behind
    (store.directory / ".MANIFEST.999.tmp").write_text("{broken", encoding="utf-8")
    reopened = SegmentStore(tmp_path / "segments", "edge-test")
    assert len(reopened.manifest.segments) == expected


def test_corrupt_manifest_falls_back_to_the_previous_one(tmp_path):
    store = _store(tmp_path)
    expected = len(store.manifest.segments)
    store.manifest_path.write_text("not json at all", encoding="utf-8")
    reopened = SegmentStore(tmp_path / "segments", "edge-test")
    assert len(reopened.manifest.segments) in {expected, expected - 1}   # prev manifest


def test_corrupt_manifest_does_not_crash_the_node(tmp_path):
    """A damaged manifest falls back; it does not take the process down."""
    store = _store(tmp_path, segments=2, per_segment=5)
    store.manifest_path.write_text("\x00\x00 not json", encoding="utf-8")
    reopened = SegmentStore(tmp_path / "segments", "edge-test")
    assert reopened.manifest.segments                      # recovered from MANIFEST.prev


def test_future_format_version_is_refused_not_guessed(tmp_path):
    store = _store(tmp_path, segments=1)
    row = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    row["format_version"] = SegmentStore.FORMAT_VERSION + 5
    store.manifest_path.write_text(json.dumps(row), encoding="utf-8")
    (store.directory / "MANIFEST.prev").unlink(missing_ok=True)
    from aegis.memory.segments import IncompatibleFormat
    with pytest.raises(IncompatibleFormat, match="newer than this build"):
        SegmentStore(tmp_path / "segments", "edge-test")


def test_orphan_segments_are_reported(tmp_path):
    store = _store(tmp_path, segments=1)
    (store.directory / "deadbeef" .ljust(32, "0")).with_suffix(".seg").write_bytes(b"junk")
    assert store.fsck().orphans


def test_point_in_time_restore(tmp_path):
    store = _store(tmp_path, segments=4, per_segment=10)
    assert len(list(store.read_all())) == 40
    dropped = store.restore_to(2)
    assert dropped == 2
    assert len(list(store.read_all())) == 20
    reopened = SegmentStore(tmp_path / "segments", "edge-test")
    assert len(list(reopened.read_all())) == 20         # the restore is durable


def test_aborted_writer_leaves_nothing_behind(tmp_path):
    directory = tmp_path / "segments"
    writer = SegmentWriter(directory)
    writer.append({"lsn": 1})
    writer.abort()
    assert not list(directory.glob("*.seg"))
    assert not list(directory.glob(".building-*"))


@pytest.mark.parametrize("seed", [1, 7, 23])
def test_random_corruption_never_yields_bad_records(tmp_path, seed):
    """Property: whatever we damage, read(strict=False) returns only intact records."""
    rng = random.Random(seed)
    store = _store(tmp_path / str(seed), segments=1, per_segment=40)
    victim = store.manifest.segments[0]
    data = bytearray(open(victim.path, "rb").read())
    for _ in range(6):
        data[rng.randrange(len(data))] ^= 1 << rng.randrange(8)
    open(victim.path, "wb").write(bytes(data))

    records = list(SegmentReader(victim.path).read(strict=False))
    assert len(records) <= 40
    for record in records:
        assert "lsn" in record and "body" in record     # nothing malformed escaped


@pytest.mark.asyncio
async def test_node_archives_and_survives_restart(node, settings):
    for i in range(6):
        await node.remember(f"observation {i} that must outlive a restart")
    sealed = node.store.archive()
    assert sealed and sealed["records"] > 0
    assert node.segments.fsck().clean

    from aegis.node import EdgeNode

    # A restart is a restart: the running node releases its handles first.
    # Embedded Qdrant is single-writer, so overlapping nodes is not a scenario
    # that can occur in production either.
    node.close()
    reopened = EdgeNode(settings)
    try:
        assert reopened.segments.manifest.generation >= 1
        assert len(list(reopened.segments.read_all())) >= 6
        reopened.store.recover()
        assert len(reopened.store.points) >= 6
    finally:
        reopened.close()


@pytest.mark.asyncio
async def test_scrub_reports_clean_when_nothing_is_damaged(node):
    await node.remember("a memory worth scrubbing")
    node.store.archive()
    result = await node.repair.scrub()
    assert result["clean"] is True


@pytest.mark.asyncio
async def test_scrub_distinguishes_lost_data_from_a_lost_durable_copy(node):
    """A corrupt segment whose memories are still resident is not lost data."""
    for i in range(4):
        await node.remember(f"observation {i}")
    node.store.archive()
    victim = node.segments.manifest.segments[0]
    data = bytearray(open(victim.path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(victim.path, "wb").write(bytes(data))

    result = await node.repair.scrub(repair=True)
    assert result["clean"] is False
    assert result["repair"]["still_resident"] > 0        # memories survived in RAM
    assert result["repair"]["damaged"] == 0              # nothing actually lost
    assert result["resealed"] is not None                # a fresh durable copy was written
    assert node.segments.fsck().clean                    # and the store is healthy again


@pytest.mark.asyncio
async def test_scrub_reports_genuinely_lost_memories(node):
    """When the memory is gone from RAM too, the loss is named rather than hidden."""
    for i in range(4):
        await node.remember(f"observation {i}")
    node.store.archive()
    victim = node.segments.manifest.segments[0]
    lost_ids = [p.id for p in list(node.store.points.values())[:4]]
    for point_id in lost_ids:
        node.store.points.pop(point_id, None)            # simulate a lost in-memory state
    data = bytearray(open(victim.path, "rb").read())
    data[len(data) // 2] ^= 0xFF
    open(victim.path, "wb").write(bytes(data))

    result = await node.repair.scrub(repair=True)
    assert result["clean"] is False
    assert result["repair"]["damaged"] >= 1
    # no peers, no cloud in this fixture: the loss must be reported, not papered over
    assert result["repair"]["complete"] is False
    assert result["repair"]["unrecoverable"]
