"""Every third-party import the node makes is declared as a dependency.

`cryptography` was added to the node's start-up path and left out of
requirements.txt. Nothing local caught it, because it was already installed
here as somebody else's transitive dependency — the node booted, the tests
passed, and the repository was broken for anybody cloning it. CI on a clean
machine found it in fourteen seconds, which is the right answer but a slow
loop. This is the fast one.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "aegis"

# Import name -> distribution name, where they differ.
DISTRIBUTION = {
    "yaml": "PyYAML",
    "qdrant_client": "qdrant-client",
    "tritonclient": "tritonclient",
}

# Declared as optional in requirements.txt and imported behind a guard. Each
# is named here rather than pattern-matched, so adding one is a decision.
OPTIONAL = {"tritonclient"}


def _declared() -> set[str]:
    text = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.split(">=")[0].split("==")[0].split("[")[0].strip()
        names.add(name.lower())
    return names


def _third_party_imports() -> set[str]:
    found: set[str] = set()
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                found.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:      # relative imports are ours
                    found.add(node.module.split(".")[0])
    stdlib = set(sys.stdlib_module_names)
    return {name for name in found
            if name not in stdlib and name != "aegis" and not name.startswith("_")}


def test_every_third_party_import_is_declared():
    declared = _declared()
    missing = sorted(
        name for name in _third_party_imports()
        if name not in OPTIONAL
        and DISTRIBUTION.get(name, name).lower() not in declared)
    assert not missing, (
        f"imported by aegis/ but not in requirements.txt: {missing}. The node will "
        f"not start on a clean machine, and it will start here, which is how this "
        f"goes unnoticed.")


def test_the_test_dependencies_are_declared_too():
    dev = (ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "-r requirements.txt" in dev, (
        "requirements-dev.txt must include the runtime set, or a green CI run "
        "says nothing about whether a device can boot")
    assert "pytest" in dev


def test_the_check_would_notice_something_missing():
    """The control: a checker that can only pass is not a checker."""
    declared = _declared()
    assert "cryptography" in declared            # the one that was actually missing
    assert "definitely-not-a-real-package" not in declared


@pytest.mark.parametrize("module", ["aegis.node", "aegis.main", "aegis.sim.world",
                                    "aegis.sync.identity", "aegis.core.invariants"])
def test_the_entry_points_import(module):
    __import__(module)
