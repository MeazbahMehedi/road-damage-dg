"""Stable seed-dependent ordering independent of the process hash seed."""

from hashlib import sha256


def stable_rank(seed: int, key: str) -> bytes:
    return sha256(f"{seed}:{key}".encode("utf-8")).digest()
