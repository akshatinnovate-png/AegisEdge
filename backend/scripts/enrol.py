"""Issue a fleet root, and certificates for the devices in it.

Trust-on-first-use is what a mesh does before anybody has provisioned
anything, and it has one gap that nothing inside the protocol can close: an
attacker present at a device's *very first* contact is believed, because
first contact is the only moment at which there is nothing to compare against.

An enrolment authority closes it by moving that moment off the mesh. The fleet
has one root key pair. Every device is issued a certificate — its id bound to
its public key, signed by the root — before it ships. A node holding the root
*public* key accepts a peer only on a valid certificate, so a stranger at
first contact is refused like any other: it cannot produce the root's
signature over a name it was never issued.

    python3 scripts/enrol.py init   --out fleet/              # once per fleet
    python3 scripts/enrol.py device --fleet fleet/ --id edge-01 --data /var/aegis/edge-01

The root private key is written once, to `fleet/root.pem`, and never leaves
that directory. It is not on any device and is not needed to run one — only to
enrol the next. Losing it means you cannot add devices; leaking it means
somebody else can, which is the whole of its threat model and the reason this
script refuses to print it.
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cryptography.hazmat.primitives import serialization                 # noqa: E402
from cryptography.hazmat.primitives.asymmetric.ed25519 import (          # noqa: E402
    Ed25519PrivateKey)

from aegis.sync.identity import (DeviceIdentity,                          # noqa: E402
                                 write_private_key)

O, R, B, D = "\033[38;5;208m", "\033[0m", "\033[1m", "\033[2m"


def _root_public_b64(private: Ed25519PrivateKey) -> str:
    return base64.b64encode(private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw)).decode()


def init(out: Path) -> int:
    out.mkdir(parents=True, exist_ok=True)
    root_pem = out / "root.pem"
    if root_pem.exists():
        print(f"{O}refusing to overwrite {root_pem}{R}")
        print(f"{D}  Every certificate already issued was signed by the key in that "
              f"file. Replacing it would invalidate the whole fleet at once.{R}")
        return 1
    private = Ed25519PrivateKey.generate()
    # The fleet root is the most sensitive file this project creates: whoever
    # holds it can enrol a device into the fleet. It is written with its mode
    # already set rather than chmod-ed afterwards, because the gap between the
    # two is a window in which it is readable by anything on the box.
    write_private_key(root_pem, private.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()))
    public = _root_public_b64(private)
    (out / "root.pub").write_text(public + "\n", encoding="utf-8")
    print(f"{O}{B}fleet root created{R}")
    print(f"  private  {root_pem}  {D}(0600 — keep it here, it goes on no device){R}")
    print(f"  public   {out / 'root.pub'}")
    print(f"\n  {B}AEGIS_FLEET_ROOT={public}{R}")
    return 0


def device(fleet: Path, device_id: str, data_dir: Path) -> int:
    root_pem = fleet / "root.pem"
    if not root_pem.exists():
        print(f"{O}no fleet root at {root_pem} — run `enrol.py init` first{R}")
        return 1
    private = serialization.load_pem_private_key(root_pem.read_bytes(), password=None)
    if not isinstance(private, Ed25519PrivateKey):
        print(f"{O}{root_pem} is not an Ed25519 private key{R}")
        return 1

    # The device's own key, created here if it does not exist yet — the same
    # call the node makes at boot, so enrolling a device that has already run
    # certifies the key it is already using rather than handing it a new one
    # its peers would refuse as an identity change.
    identity = DeviceIdentity.load_or_create(device_id, data_dir / "device_key.pem")
    certificate = DeviceIdentity.issue(private, device_id, identity.public_key_b64)

    holder = DeviceIdentity(device_id, root_public_b64=_root_public_b64(private))
    if not holder.check_certificate(device_id, identity.public_key_b64, certificate):
        print(f"{O}the certificate this just issued does not verify — refusing to "
              f"hand out something that will be rejected{R}")
        return 1

    record = {"device_id": device_id, "public_key": identity.public_key_b64,
              "certificate": certificate}
    (fleet / f"{device_id}.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"{O}{B}{device_id} enrolled{R}")
    print(f"  key      {data_dir / 'device_key.pem'}")
    print(f"  record   {fleet / f'{device_id}.json'}")
    print(f"\n  {B}AEGIS_FLEET_ROOT={_root_public_b64(private)}{R}")
    print(f"  {B}AEGIS_DEVICE_CERT={certificate}{R}")
    print(f"\n{D}  Set both on this device. With both set the node requires every peer "
          f"to present a certificate;{R}")
    print(f"{D}  with either missing it falls back to trust-on-first-use and says so "
          f"in /api/v1/mesh/status.{R}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init", help="create the fleet root key pair")
    p_init.add_argument("--out", type=Path, default=Path("fleet"))

    p_dev = sub.add_parser("device", help="issue a certificate for one device")
    p_dev.add_argument("--fleet", type=Path, default=Path("fleet"))
    p_dev.add_argument("--id", required=True, dest="device_id")
    p_dev.add_argument("--data", type=Path, required=True, dest="data_dir")

    args = parser.parse_args()
    if args.command == "init":
        return init(args.out)
    return device(args.fleet, args.device_id, args.data_dir)


if __name__ == "__main__":
    sys.exit(main())
