"""Portable digests for the immutable, Windows-generated research CSV release.

The v2.2 source and output manifests were frozen on Windows, where text files
used CRLF newlines. Git stores CSV text with LF, and checks out native lines
on Linux. This reconstructs the original Windows CRLF *byte representation*
solely when verifying the archived, manifest-bound CSVs.

This is not normalization of numbers, dates, missing observations, headers,
whitespace within cells, or other content. Any such edits change the digest.
It does not replace ordinary SHA-256 for generic binary/JSON files.
"""
from __future__ import annotations
import hashlib
from pathlib import Path

def archived_file_sha256(path: str | Path) -> str:
    path=Path(path)
    raw=path.read_bytes()
    if path.suffix.lower()==".csv":
        if b"\r" in raw.replace(b"\r\n",b""):
            raise ValueError(f"Unexpected bare carriage return in archived CSV {path}")
        raw=raw.replace(b"\r\n",b"\n").replace(b"\n",b"\r\n")
    return hashlib.sha256(raw).hexdigest()
