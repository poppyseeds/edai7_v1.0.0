"""Prototype provenance fingerprint — not a cryptographic watermark."""

from __future__ import annotations

import hashlib

import pandas as pd

from app.schemas.schemas import ProvenanceFingerprint, utc_now


def provenance_fingerprint(
    df: pd.DataFrame,
    generator: str,
    random_seed: int,
) -> ProvenanceFingerprint:
    payload = df.to_csv(index=False).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    return ProvenanceFingerprint(
        dataset_hash=digest,
        generator=generator,
        random_seed=random_seed,
        timestamp=utc_now(),
        num_rows=len(df),
        columns=list(df.columns),
    )
