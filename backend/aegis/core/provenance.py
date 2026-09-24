"""Provenance: proof that the thing on stage is the thing in the repository.

A demo is an assertion. Somebody watching one has no way to tell a node
running the committed code from a node running a branch with the hard parts
stubbed out, and the usual answer — "here is the repository, take my word for
it" — is exactly the answer somebody would give either way.

So the node computes, at runtime, a Merkle root over its own source tree, the
weights it loaded, and the graphs it compiled from them. Clone the repository,
run one command, and compare two hex strings. If they match, the process
answering questions is the code that is public. If they do not, the difference
is named file by file.

This is ordinary supply-chain hygiene everywhere software is shipped, and
essentially unheard of in a demo, because it only helps the people whose demo
is real.

What it deliberately does *not* claim: this is integrity, not authenticity.
The receipt proves the running code matches a tree with this root; it does not
prove who wrote it or that the root is trustworthy, which would need a
signature and a key somebody has reason to trust. Overstating that distinction
is how attestation turns into theatre.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

# Everything the running node's behaviour actually depends on.
SOURCE_GLOBS = ("aegis/**/*.py", "config/*.yaml", "requirements.txt")
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".cache", ".git"}


def _hash_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def _interesting(root: Path) -> list[Path]:
    seen: set[Path] = set()
    for pattern in SOURCE_GLOBS:
        for path in root.glob(pattern):
            if not path.is_file() or any(p in SKIP_PARTS for p in path.parts):
                continue
            seen.add(path)
    return sorted(seen)


def merkle_root(entries: Iterable[tuple[str, str]]) -> str:
    """A root over (name, hash) pairs, order-independent by sorting first.

    A flat hash of a concatenation would do for a fingerprint, but a tree lets
    a mismatch be *located* rather than merely detected, which is the
    difference between "something differs" and "this file differs".
    """
    leaves = [hashlib.sha256(f"{name}\0{digest}".encode()).digest()
              for name, digest in sorted(entries)]
    if not leaves:
        return hashlib.sha256(b"empty").hexdigest()
    while len(leaves) > 1:
        nxt = []
        for i in range(0, len(leaves), 2):
            pair = leaves[i] + (leaves[i + 1] if i + 1 < len(leaves) else leaves[i])
            nxt.append(hashlib.sha256(pair).digest())
        leaves = nxt
    return leaves[0].hex()


def _git(root: Path, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=root, capture_output=True,
                             text=True, timeout=6)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


class Provenance:
    """A receipt for one running process."""

    def __init__(self, root: Path, bundle: Any = None) -> None:
        self.root = Path(root)
        self.bundle = bundle
        self.built_at = time.time()
        self._cache: dict[str, Any] | None = None

    # -- the pieces -------------------------------------------------------

    def source_files(self) -> dict[str, str]:
        return {str(p.relative_to(self.root)): _hash_file(p)
                for p in _interesting(self.root)}

    def model_artefacts(self) -> dict[str, str]:
        """The weights and the graphs compiled from them.

        Source hashes alone would miss the thing most worth substituting: a
        model. A node running the committed code against a different embedding
        table is a different system.
        """
        out: dict[str, str] = {}
        bundle = self.bundle
        if bundle is None:
            return out
        for attribute in ("embedder_path", "reranker_path", "tokenizer_path"):
            path = getattr(bundle, attribute, None)
            if path and Path(path).exists():
                out[f"model/{Path(path).name}"] = _hash_file(Path(path))
        digest = getattr(bundle, "sha256", None)
        if digest:
            out["model/weights.sha256"] = str(digest)
        return out

    def git(self) -> dict[str, Any]:
        repo = self.root.parent if (self.root.parent / ".git").exists() else self.root
        status = _git(repo, "status", "--porcelain")
        return {
            "commit": _git(repo, "rev-parse", "HEAD"),
            "short": _git(repo, "rev-parse", "--short", "HEAD"),
            "branch": _git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "committed_at": _git(repo, "log", "-1", "--format=%cI"),
            # A dirty tree is not a failure, but a receipt that hid it would be
            # worthless: the whole point is that the claim can be checked.
            "clean": status == "" if status is not None else None,
            "dirty_files": [l[3:] for l in (status or "").splitlines() if l][:40],
        }

    # -- the receipt ------------------------------------------------------

    def receipt(self, refresh: bool = False) -> dict[str, Any]:
        if self._cache is not None and not refresh:
            return self._cache
        sources = self.source_files()
        models = self.model_artefacts()
        source_root = merkle_root(sources.items())
        model_root = merkle_root(models.items())
        receipt = {
            "receipt_version": 1,
            "generated_at": time.time(),
            "source": {
                "files": len(sources),
                "merkle_root": source_root,
                "globs": list(SOURCE_GLOBS),
            },
            "models": {
                "artefacts": len(models),
                "merkle_root": model_root,
                "names": sorted(models),
            },
            "git": self.git(),
            "runtime": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "machine": platform.machine(),
                "pid": os.getpid(),
                "executable": sys.executable,
            },
            # One string to compare. Everything above folds into it, so a
            # single mismatch anywhere changes this and nothing else has to be
            # read to know that something did.
            "root": hashlib.sha256(
                f"{source_root}\0{model_root}".encode()).hexdigest(),
            "claims": (
                "This is integrity, not authenticity: it proves the running "
                "process matches a tree with this root, not who produced the "
                "tree. Recompute it from a clean clone with "
                "`python3 -m aegis.core.provenance` and compare `root`."
            ),
        }
        self._cache = receipt
        return receipt

    def verify_against(self, expected: dict[str, Any]) -> dict[str, Any]:
        """Compare a receipt to this process, and name what differs."""
        mine = self.receipt(refresh=True)
        mine_files = self.source_files()
        theirs = expected.get("files") or {}
        added = sorted(set(mine_files) - set(theirs))
        removed = sorted(set(theirs) - set(mine_files))
        changed = sorted(f for f in set(mine_files) & set(theirs)
                         if mine_files[f] != theirs[f])
        return {
            "match": mine["root"] == expected.get("root"),
            "expected_root": expected.get("root"),
            "actual_root": mine["root"],
            "changed_files": changed[:40],
            "added_files": added[:40],
            "removed_files": removed[:40],
            "verdict": ("the running process is the published tree"
                        if mine["root"] == expected.get("root")
                        else f"{len(changed)} changed, {len(added)} added, "
                             f"{len(removed)} removed"),
        }

    def full(self) -> dict[str, Any]:
        """The receipt plus every leaf, for writing to disk and comparing later."""
        return {**self.receipt(), "files": self.source_files(),
                "model_files": self.model_artefacts()}


def main() -> None:
    """`python3 -m aegis.core.provenance` — compute the root from a clone."""
    root = Path(__file__).resolve().parents[2]
    prov = Provenance(root)
    receipt = prov.receipt()
    print(json.dumps({k: v for k, v in receipt.items() if k != "claims"}, indent=2))
    print(f"\nroot: {receipt['root']}")
    print("Compare that against /api/v1/provenance on the running node.")


if __name__ == "__main__":
    main()
