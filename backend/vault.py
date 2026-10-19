"""At-rest protection and retention for the biometric gallery.

The gallery is a plain JSON file holding children's names beside 512-dimension
face embeddings, written with default permissions and kept forever. Anyone who
can read the file can read the biometrics, and nothing ever removed an
enrolment after the term it was made for.

Three measures, in increasing strength, because they defend against different
things and only one of them is free:

**Owner-only permissions**, always applied. Defends against other accounts on
a shared machine. Costs nothing and needs no configuration, so there is no
reason for it to be optional.

**A retention limit**, applied when asked. Defends against the indefinite
accumulation that makes a breach worse the longer the system runs. An
enrolment made for one term has no business surviving the next.

**Encryption**, when a passphrase is supplied. Defends against a stolen laptop
or a copied backup -- cases where file permissions are irrelevant because the
attacker has the disk. It is off by default because a key that lives beside
the data it protects secures nothing, so the passphrase has to come from
somewhere this module cannot invent: an operator, an environment variable, a
keystore.

What this deliberately does **not** do is derive a key from something fixed in
the repository, which would be theatre: every installation would share it, and
it would be readable by anyone who can read this file.
"""
from __future__ import annotations

import base64
import json
import os
import stat
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

#: Environment variable holding the gallery passphrase. Absent means the
#: gallery is stored in plain JSON, with permissions restricted.
KEY_ENV = "CLASSGRAPH_GALLERY_KEY"

#: Marks an encrypted gallery so loading can tell the two apart without
#: guessing from the bytes.
ENCRYPTED_MAGIC = "classgraph-encrypted-gallery-v1"

#: Iterations for key derivation. scrypt rather than a plain hash so a weak
#: passphrase costs an attacker real time per guess.
SCRYPT_N = 2 ** 15
SCRYPT_R = 8
SCRYPT_P = 1


def restrict_permissions(path: Path) -> bool:
    """Make a file readable only by its owner.

    Returns:
        Whether the restriction could be applied. On Windows, chmod does not
        express an ACL, so this reports False rather than claiming a
        protection it did not apply -- a security measure that silently does
        nothing is worse than one that is known to be absent.
    """
    path = Path(path)
    if not path.exists():
        return False
    if os.name == "nt":
        # icacls is the real mechanism here; attempting it keeps Windows
        # installations from being silently unprotected.
        import subprocess

        user = os.environ.get("USERNAME")
        if not user:
            return False
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r",
             "/grant:r", f"{user}:(R,W)"],
            capture_output=True, check=False)
        return result.returncode == 0
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return True


def _derive_key(passphrase: str, salt: bytes) -> bytes:
    """A Fernet key from a passphrase, via scrypt."""
    from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

    kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
    return base64.urlsafe_b64encode(kdf.derive(passphrase.encode("utf-8")))


def passphrase() -> str | None:
    """The configured passphrase, or None when encryption is off."""
    value = os.environ.get(KEY_ENV, "").strip()
    return value or None


def available() -> bool:
    """Whether the encryption dependency is installed."""
    try:
        import cryptography  # noqa: F401
    except ImportError:
        return False
    return True


def encrypt_payload(payload: dict[str, Any], secret: str) -> dict[str, Any]:
    """Wrap a gallery payload in an encrypted envelope.

    The salt travels with the ciphertext, which is standard: it is not a
    secret, it exists so the same passphrase on two machines does not derive
    the same key.
    """
    from cryptography.fernet import Fernet

    salt = os.urandom(16)
    token = Fernet(_derive_key(secret, salt)).encrypt(
        json.dumps(payload).encode("utf-8"))
    return {
        "format": ENCRYPTED_MAGIC,
        "salt": base64.b64encode(salt).decode("ascii"),
        "ciphertext": token.decode("ascii"),
    }


def decrypt_payload(envelope: dict[str, Any], secret: str) -> dict[str, Any]:
    """Unwrap an encrypted gallery.

    Raises:
        ValueError: If the passphrase is wrong, stated plainly. A wrong
            passphrase returning an empty gallery would look like a gallery
            with nobody in it, and the next enrolment would overwrite the
            real one.
    """
    from cryptography.fernet import Fernet, InvalidToken

    salt = base64.b64decode(envelope["salt"])
    try:
        plain = Fernet(_derive_key(secret, salt)).decrypt(
            envelope["ciphertext"].encode("ascii"))
    except InvalidToken as exc:
        raise ValueError(
            f"the gallery could not be decrypted -- check {KEY_ENV}") from exc
    return json.loads(plain.decode("utf-8"))


def is_encrypted(payload: dict[str, Any]) -> bool:
    """Whether a loaded file is an encrypted envelope."""
    return payload.get("format") == ENCRYPTED_MAGIC


def expired_people(payload: dict[str, Any], max_age_days: int,
                   now: datetime | None = None) -> list[dict[str, Any]]:
    """Enrolments older than the retention limit.

    Entries without an enrolment date are treated as expired when a limit is
    set. The alternative -- keeping anything undated forever -- means the
    oldest records, which are exactly the ones a retention policy exists to
    remove, are the ones it never reaches.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max_age_days)
    stale = []
    for person in payload.get("people", []):
        enrolled = person.get("enrolled_at")
        if not enrolled:
            stale.append(person)
            continue
        try:
            when = datetime.fromisoformat(enrolled)
        except ValueError:
            stale.append(person)
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when < cutoff:
            stale.append(person)
    return stale


def describe(path: Path) -> str:
    """One paragraph on how this gallery is currently protected."""
    path = Path(path)
    if not path.exists():
        return f"{path} does not exist."

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return f"{path} could not be read."

    lines = [f"{path}"]
    size = path.stat().st_size
    lines.append(f"  size            : {size:,} bytes")
    if is_encrypted(payload):
        lines.append("  at rest         : encrypted")
    else:
        n = len(payload.get("people", []))
        lines.append(f"  at rest         : PLAIN TEXT, {n} face record(s)")
        if not passphrase():
            lines.append(f"  to encrypt      : set {KEY_ENV} and re-save")
        elif not available():
            lines.append("  to encrypt      : pip install cryptography")

    if os.name != "nt":
        mode = stat.S_IMODE(path.stat().st_mode)
        others = mode & (stat.S_IRGRP | stat.S_IROTH)
        lines.append(f"  permissions     : {oct(mode)}"
                     f"{'  (readable by others)' if others else ''}")
    else:
        lines.append("  permissions     : see icacls; chmod does not apply")
    return "\n".join(lines)


def main() -> int:
    """Report on, restrict, encrypt or sweep the gallery."""
    import argparse

    ap = argparse.ArgumentParser(
        description="Inspect and protect the biometric gallery.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gallery", type=Path,
                    default=Path("outputs/enrollment/gallery.json"))
    ap.add_argument("--encrypt", action="store_true",
                    help=f"re-save encrypted, using {KEY_ENV}")
    ap.add_argument("--decrypt", action="store_true",
                    help="re-save as plain JSON")
    ap.add_argument("--expire-after", type=int, metavar="DAYS",
                    help="remove enrolments older than this many days")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = Path(args.gallery)
    if not path.exists():
        print(f"{path} does not exist.")
        return 1

    payload = json.loads(path.read_text(encoding="utf-8"))
    secret = passphrase()

    if is_encrypted(payload):
        if not secret:
            print(f"The gallery is encrypted. Set {KEY_ENV} to work with it.")
            return 1
        payload = decrypt_payload(payload, secret)

    changed = False
    if args.expire_after is not None:
        stale = expired_people(payload, args.expire_after)
        if stale:
            print(f"  {len(stale)} enrolment(s) past "
                  f"{args.expire_after} days:")
            for person in stale:
                print(f"      {person.get('name')} "
                      f"(enrolled {person.get('enrolled_at') or 'undated'})")
            if not args.dry_run:
                names = {id(p) for p in stale}
                payload["people"] = [p for p in payload["people"]
                                     if id(p) not in names]
                changed = True
        else:
            print(f"  nothing older than {args.expire_after} days.")

    if args.decrypt:
        changed = True
    elif args.encrypt:
        if not secret:
            print(f"Set {KEY_ENV} to a passphrase first.")
            return 1
        if not available():
            print("pip install cryptography")
            return 1
        payload = encrypt_payload(payload, secret)
        changed = True

    if changed and not args.dry_run:
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"  written -> {path}")
    elif args.dry_run:
        print("  dry run -- nothing written.")

    restricted = restrict_permissions(path)
    print(f"  owner-only permissions applied: {restricted}")
    print()
    print(describe(path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
