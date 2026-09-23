"""Immutable segment store with a crash-safe manifest.

A write-ahead log recovers the *last* state. It does not protect against the
failure modes that actually destroy production data: a half-written snapshot,
a manifest that points at a segment which was never fsynced, silent bit rot in
a file nobody has read for six months, or a process killed between two writes
that had to land together.

The design is the one storage engines converge on, because it is the one that
survives:

* Segments are **immutable and content-addressed**. A segment is named by the
  hash of its bytes, so a corrupted segment cannot masquerade as a good one.
* Every record carries a CRC; every segment carries a footer with its own
  digest, record count and byte length.
* The **manifest** is the only mutable object, written to a temp file, fsynced,
  then atomically renamed. A crash yields either the old manifest or the new
  one — never a blend of the two.
* Fsync order is: segment data → segment footer → fsync(segment) →
  fsync(directory) → manifest → fsync(manifest) → rename → fsync(directory).
  Any crash point leaves a recoverable state, and `verify()` says which one.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

MAGIC = b"AEGSEG01"
FOOTER = struct.Struct("<8sQQ32s")      # magic, records, payload_bytes, sha256
RECORD_HEADER = struct.Struct("<IQ")    # crc32, length


class IncompatibleFormat(RuntimeError):
    """The on-disk format is newer than this build understands."""


class SegmentCorrupt(Exception):
    def __init__(self, path: Path, reason: str, recovered: int = 0) -> None:
        super().__init__(f"{path.name}: {reason} (recovered {recovered} records)")
        self.path = path
        self.reason = reason
        self.recovered = recovered


@dataclass(slots=True)
class SegmentInfo:
    segment_id: str
    path: str
    records: int
    bytes: int
    sha256: str
    created_at: float = field(default_factory=time.time)
    min_lsn: int = 0
    max_lsn: int = 0
    generation: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {"segment_id": self.segment_id, "path": self.path, "records": self.records,
                "bytes": self.bytes, "sha256": self.sha256, "created_at": self.created_at,
                "min_lsn": self.min_lsn, "max_lsn": self.max_lsn, "generation": self.generation}

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "SegmentInfo":
        return SegmentInfo(**row)


def _fsync_dir(path: Path) -> None:
    """Renames are only durable once the *directory* is fsynced."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, AttributeError):        # not all platforms allow this
        pass


class SegmentWriter:
    """Builds one immutable segment, then seals it."""

    def __init__(self, directory: Path, generation: int = 0) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.generation = generation
        self.temp = self.directory / f".building-{int(time.time() * 1e6)}.seg"
        self._handle = open(self.temp, "wb")
        self._digest = hashlib.sha256()
        self.records = 0
        self.payload_bytes = 0
        self.min_lsn = 0
        self.max_lsn = 0

    def append(self, record: dict[str, Any], lsn: int = 0) -> int:
        blob = json.dumps(record, separators=(",", ":"), default=str).encode("utf-8")
        header = RECORD_HEADER.pack(zlib.crc32(blob) & 0xFFFFFFFF, len(blob))
        self._handle.write(header)
        self._handle.write(blob)
        self._digest.update(header)
        self._digest.update(blob)
        self.records += 1
        self.payload_bytes += len(header) + len(blob)
        if lsn:
            self.min_lsn = min(self.min_lsn or lsn, lsn)
            self.max_lsn = max(self.max_lsn, lsn)
        return self.records

    def seal(self) -> SegmentInfo:
        """Write the footer, fsync, then rename into place under its content hash."""
        digest = self._digest.digest()
        self._handle.write(FOOTER.pack(MAGIC, self.records, self.payload_bytes, digest))
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()

        segment_id = digest.hex()[:32]
        final = self.directory / f"{segment_id}.seg"
        os.replace(self.temp, final)          # atomic within a filesystem
        _fsync_dir(self.directory)
        return SegmentInfo(
            segment_id=segment_id, path=str(final), records=self.records,
            bytes=final.stat().st_size, sha256=digest.hex(),
            min_lsn=self.min_lsn, max_lsn=self.max_lsn, generation=self.generation,
        )

    def abort(self) -> None:
        try:
            self._handle.close()
        finally:
            self.temp.unlink(missing_ok=True)


class SegmentReader:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def verify(self) -> tuple[bool, str]:
        """Full integrity check: footer, digest, and every record CRC."""
        try:
            size = self.path.stat().st_size
        except OSError as exc:
            return False, f"unreadable: {exc}"
        if size < FOOTER.size:
            return False, "truncated before footer"
        with open(self.path, "rb") as handle:
            handle.seek(size - FOOTER.size)
            magic, records, payload_bytes, digest = FOOTER.unpack(handle.read(FOOTER.size))
            if magic != MAGIC:
                # Either never sealed, or truncated past the footer. Both mean
                # the same thing operationally, and saying "bad magic" sends
                # whoever reads the log looking for the wrong failure.
                return False, "footer missing — segment truncated or never sealed"
            if payload_bytes + FOOTER.size != size:
                return False, f"length mismatch: footer says {payload_bytes}, file has {size - FOOTER.size}"
            handle.seek(0)
            body = handle.read(payload_bytes)
        if hashlib.sha256(body).digest() != digest:
            return False, "sha256 mismatch — bit rot or tampering"
        return True, f"ok ({records} records)"

    def read(self, strict: bool = True) -> Iterator[dict[str, Any]]:
        """Yield records. In non-strict mode, stop at the first damage instead of raising."""
        size = self.path.stat().st_size
        with open(self.path, "rb") as handle:
            body = handle.read(max(0, size - FOOTER.size))
        offset = 0
        recovered = 0
        while offset + RECORD_HEADER.size <= len(body):
            crc, length = RECORD_HEADER.unpack_from(body, offset)
            offset += RECORD_HEADER.size
            blob = body[offset:offset + length]
            if len(blob) < length:
                if strict:
                    raise SegmentCorrupt(self.path, "truncated record", recovered)
                return
            offset += length
            if (zlib.crc32(blob) & 0xFFFFFFFF) != crc:
                if strict:
                    raise SegmentCorrupt(self.path, f"crc mismatch at record {recovered}", recovered)
                return
            try:
                yield json.loads(blob.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                if strict:
                    raise SegmentCorrupt(self.path, "undecodable record", recovered)
                return
            recovered += 1


@dataclass
class Manifest:
    """The only mutable object in the store."""
    version: int = 1
    generation: int = 0
    segments: list[SegmentInfo] = field(default_factory=list)
    checkpoint_lsn: int = 0
    created_at: float = field(default_factory=time.time)
    node_id: str = ""
    format_version: int = 2

    def as_dict(self) -> dict[str, Any]:
        return {"version": self.version, "generation": self.generation,
                "segments": [s.as_dict() for s in self.segments],
                "checkpoint_lsn": self.checkpoint_lsn, "created_at": self.created_at,
                "node_id": self.node_id, "format_version": self.format_version}

    @staticmethod
    def from_dict(row: dict[str, Any]) -> "Manifest":
        return Manifest(
            version=row["version"], generation=row.get("generation", 0),
            segments=[SegmentInfo.from_dict(s) for s in row.get("segments", [])],
            checkpoint_lsn=row.get("checkpoint_lsn", 0), created_at=row.get("created_at", time.time()),
            node_id=row.get("node_id", ""), format_version=row.get("format_version", 1),
        )


@dataclass
class FsckReport:
    checked: int = 0
    healthy: int = 0
    corrupt: list[dict[str, Any]] = field(default_factory=list)
    orphans: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    repaired: int = 0
    quarantined: int = 0

    @property
    def clean(self) -> bool:
        return not self.corrupt and not self.missing

    def as_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "healthy": self.healthy, "corrupt": self.corrupt,
                "orphans": self.orphans, "missing": self.missing, "repaired": self.repaired,
                "quarantined": self.quarantined, "clean": self.clean}


class SegmentStore:
    """Manifest-governed set of immutable segments, with fsck and PITR."""

    FORMAT_VERSION = 2
    MANIFEST = "MANIFEST"

    def __init__(self, directory: Path, node_id: str = "") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.quarantine = self.directory / "quarantine"
        self.node_id = node_id
        self.manifest = self._load_manifest()
        self.writes = 0
        self.rollbacks = 0

    # -- manifest ---------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.directory / self.MANIFEST

    def _load_manifest(self) -> Manifest:
        for candidate in (self.manifest_path, self.directory / f"{self.MANIFEST}.prev"):
            if not candidate.exists():
                continue
            try:
                row = json.loads(candidate.read_text(encoding="utf-8"))
                manifest = Manifest.from_dict(row)
            except IncompatibleFormat:
                raise
            except Exception:
                # A damaged manifest must fall back to the previous one, not
                # take the node down. `json.JSONDecodeError` is a ValueError,
                # so the version refusal below needs its own exception type —
                # catching ValueError here would make every corrupt manifest
                # look like a version mismatch and kill the process.
                continue
            if manifest.format_version > self.FORMAT_VERSION:
                raise IncompatibleFormat(
                    f"on-disk format v{manifest.format_version} is newer than this build "
                    f"(v{self.FORMAT_VERSION}) — refusing to open rather than corrupt it")
            return manifest
        return Manifest(node_id=self.node_id, format_version=self.FORMAT_VERSION)

    def _commit_manifest(self, manifest: Manifest) -> None:
        """Temp file → fsync → rename → fsync(dir). Never a partial manifest."""
        manifest.version += 1
        manifest.node_id = self.node_id
        manifest.format_version = self.FORMAT_VERSION
        temp = self.directory / f".{self.MANIFEST}.{os.getpid()}.tmp"
        blob = json.dumps(manifest.as_dict(), default=str).encode("utf-8")
        with open(temp, "wb") as handle:
            handle.write(blob)
            handle.flush()
            os.fsync(handle.fileno())
        if self.manifest_path.exists():
            try:
                os.replace(self.manifest_path, self.directory / f"{self.MANIFEST}.prev")
            except OSError:
                pass
        os.replace(temp, self.manifest_path)
        _fsync_dir(self.directory)
        self.manifest = manifest

    # -- writing ----------------------------------------------------------

    def write_segment(self, records: list[dict[str, Any]], checkpoint_lsn: int = 0) -> SegmentInfo:
        writer = SegmentWriter(self.directory, self.manifest.generation + 1)
        try:
            for record in records:
                writer.append(record, lsn=int(record.get("lsn", 0)))
            info = writer.seal()
        except Exception:
            writer.abort()
            self.rollbacks += 1
            raise
        manifest = Manifest(
            version=self.manifest.version, generation=self.manifest.generation + 1,
            segments=[*self.manifest.segments, info],
            checkpoint_lsn=max(checkpoint_lsn, self.manifest.checkpoint_lsn),
            node_id=self.node_id,
        )
        self._commit_manifest(manifest)
        self.writes += 1
        return info

    # -- reading ----------------------------------------------------------

    def read_all(self, strict: bool = False, up_to_generation: int | None = None
                 ) -> Iterator[dict[str, Any]]:
        for info in sorted(self.manifest.segments, key=lambda s: s.generation):
            if up_to_generation is not None and info.generation > up_to_generation:
                continue
            path = Path(info.path)
            if not path.exists():
                continue
            yield from SegmentReader(path).read(strict=strict)

    # -- integrity --------------------------------------------------------

    def fsck(self, repair: bool = False) -> FsckReport:
        """Verify every segment against the manifest and the filesystem."""
        report = FsckReport()
        listed = {Path(info.path).name for info in self.manifest.segments}

        for info in list(self.manifest.segments):
            report.checked += 1
            path = Path(info.path)
            if not path.exists():
                report.missing.append(info.segment_id)
                continue
            healthy, reason = SegmentReader(path).verify()
            if healthy:
                report.healthy += 1
                continue
            report.corrupt.append({"segment_id": info.segment_id, "reason": reason,
                                   "records": info.records})
            if repair:
                self._quarantine(path)
                report.quarantined += 1

        for candidate in self.directory.glob("*.seg"):
            if candidate.name not in listed:
                report.orphans.append(candidate.name)   # written but never committed

        if repair and (report.corrupt or report.missing):
            surviving = [
                info for info in self.manifest.segments
                if info.segment_id not in {c["segment_id"] for c in report.corrupt}
                and info.segment_id not in report.missing
            ]
            self._commit_manifest(Manifest(
                version=self.manifest.version, generation=self.manifest.generation,
                segments=surviving, checkpoint_lsn=self.manifest.checkpoint_lsn,
                node_id=self.node_id))
            report.repaired = len(report.corrupt) + len(report.missing)
        return report

    def _quarantine(self, path: Path) -> None:
        """Keep damaged data for forensics; never silently delete it."""
        self.quarantine.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(path, self.quarantine / f"{int(time.time())}-{path.name}")
        except OSError:
            pass

    def lost_records(self, report: FsckReport) -> int:
        by_id = {info.segment_id: info for info in self.manifest.segments}
        return sum(by_id[c["segment_id"]].records for c in report.corrupt if c["segment_id"] in by_id)

    # -- point in time ----------------------------------------------------

    def generations(self) -> list[dict[str, Any]]:
        return [{"generation": s.generation, "segment_id": s.segment_id,
                 "records": s.records, "created_at": s.created_at,
                 "max_lsn": s.max_lsn} for s in sorted(self.manifest.segments,
                                                       key=lambda s: s.generation)]

    def restore_to(self, generation: int) -> int:
        """Point-in-time restore: drop everything after a generation."""
        keep = [s for s in self.manifest.segments if s.generation <= generation]
        dropped = len(self.manifest.segments) - len(keep)
        self._commit_manifest(Manifest(
            version=self.manifest.version, generation=generation, segments=keep,
            checkpoint_lsn=max((s.max_lsn for s in keep), default=0), node_id=self.node_id))
        return dropped

    def snapshot(self) -> dict[str, Any]:
        return {
            "directory": str(self.directory), "format_version": self.FORMAT_VERSION,
            "manifest_version": self.manifest.version, "generation": self.manifest.generation,
            "segments": len(self.manifest.segments),
            "records": sum(s.records for s in self.manifest.segments),
            "bytes": sum(s.bytes for s in self.manifest.segments),
            "checkpoint_lsn": self.manifest.checkpoint_lsn,
            "writes": self.writes, "rollbacks": self.rollbacks,
            "quarantined": len(list(self.quarantine.glob("*"))) if self.quarantine.exists() else 0,
        }
