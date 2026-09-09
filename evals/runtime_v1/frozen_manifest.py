"""The single byte-level identity for a frozen Runtime V1 manifest."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrozenManifest:
    path: Path
    raw_bytes: bytes
    document: dict[str, object]
    sha256: str


def manifest_sha256(raw_bytes: bytes) -> str:
    """Return SHA-256 of the exact frozen manifest bytes, never a re-serialization."""
    return hashlib.sha256(raw_bytes).hexdigest()


def load_frozen_manifest(path: Path) -> FrozenManifest:
    source = Path(path)
    raw_bytes = source.read_bytes()
    document = json.loads(raw_bytes.decode("utf-8"))
    if not isinstance(document, dict):
        raise ValueError("frozen manifest must be a JSON object")
    return FrozenManifest(source, raw_bytes, document, manifest_sha256(raw_bytes))
