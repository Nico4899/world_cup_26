"""Off-site backup upload to S3-compatible storage.

Phase 10. Targets Cloudflare R2 by default (S3-compatible API, no egress
fees, 10 GB free tier), but works against AWS S3 / MinIO / any provider
boto3 can talk to by setting ``AWS_S3_ENDPOINT_URL``.

Environment variables:
    AWS_S3_BUCKET           required to enable upload (absent → no-op)
    AWS_S3_ENDPOINT_URL     optional; e.g. ``https://<account>.r2.cloudflarestorage.com``
    AWS_ACCESS_KEY_ID       boto3 picks this up automatically
    AWS_SECRET_ACCESS_KEY   boto3 picks this up automatically
    AWS_REGION              optional; defaults to ``auto`` (R2 convention)
    WC2026_BACKUP_PREFIX    object-key prefix; default ``wc2026/backups/``
    WC2026_BACKUP_RETENTION_DAYS  delete remote objects older than this; default 30

After a successful upload we sweep older objects in the same prefix so an
operator only ever sees the last month of dumps.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

ENV_BUCKET = "AWS_S3_BUCKET"
ENV_ENDPOINT = "AWS_S3_ENDPOINT_URL"
ENV_REGION = "AWS_REGION"
ENV_PREFIX = "WC2026_BACKUP_PREFIX"
ENV_RETENTION_DAYS = "WC2026_BACKUP_RETENTION_DAYS"

DEFAULT_PREFIX = "wc2026/backups/"
DEFAULT_RETENTION_DAYS = 30
DEFAULT_REGION = "auto"  # R2 convention; harmless for real S3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UploadResult:
    """Outcome of one upload + prune cycle."""

    uri: str
    key: str
    bytes_uploaded: int
    pruned_count: int


def _make_client(*, endpoint_url: str | None, region: str):
    """Instantiate a boto3 S3 client. Import lazily so the SDK isn't loaded on cold-start."""
    import boto3  # noqa: PLC0415

    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region,
    )


def upload_backup(
    local_path: Path,
    *,
    bucket: str | None = None,
    endpoint_url: str | None = None,
    region: str | None = None,
    prefix: str | None = None,
    retention_days: int | None = None,
    client=None,
) -> UploadResult | None:
    """Upload ``local_path`` to ``s3://{bucket}/{prefix}{filename}`` + prune older objects.

    Returns ``None`` (silent no-op) when no bucket is configured — that's the
    expected local-dev state. Surfaces unexpected boto3 errors so the
    scheduler row records the failure.
    """
    bucket = bucket or os.environ.get(ENV_BUCKET)
    if not bucket:
        logger.debug("s3_upload: no %s set; skipping off-site backup", ENV_BUCKET)
        return None
    if not local_path.exists():
        logger.warning("s3_upload: %s does not exist; nothing to upload", local_path)
        return None
    endpoint_url = endpoint_url or os.environ.get(ENV_ENDPOINT) or None
    region = region or os.environ.get(ENV_REGION) or DEFAULT_REGION
    prefix = prefix or os.environ.get(ENV_PREFIX) or DEFAULT_PREFIX
    if retention_days is None:
        retention_days = _int_env(ENV_RETENTION_DAYS, DEFAULT_RETENTION_DAYS)

    s3 = client or _make_client(endpoint_url=endpoint_url, region=region)
    key = f"{prefix.rstrip('/')}/{local_path.name}"
    size = local_path.stat().st_size
    s3.upload_file(str(local_path), bucket, key)
    pruned = prune_remote(
        s3,
        bucket=bucket,
        prefix=prefix,
        retention_days=retention_days,
    )
    uri = f"s3://{bucket}/{key}"
    logger.info("s3_upload: %s uploaded (%d bytes); pruned %d stale objects", uri, size, pruned)
    return UploadResult(uri=uri, key=key, bytes_uploaded=size, pruned_count=pruned)


def prune_remote(
    client,
    *,
    bucket: str,
    prefix: str,
    retention_days: int,
) -> int:
    """Delete objects under ``s3://bucket/prefix`` older than ``retention_days``.

    Returns the number of objects removed. Non-fatal: if a delete call raises
    we log and continue so a single stuck object doesn't block the rest.
    """
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    paginator = client.get_paginator("list_objects_v2")
    to_delete: list[dict[str, str]] = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents") or []:
            last_mod = obj.get("LastModified")
            if last_mod is None:
                continue
            if last_mod.tzinfo is None:
                last_mod = last_mod.replace(tzinfo=UTC)
            if last_mod < cutoff:
                to_delete.append({"Key": obj["Key"]})
    if not to_delete:
        return 0
    deleted = 0
    # S3 delete_objects caps at 1000 keys per call.
    for batch_start in range(0, len(to_delete), 1000):
        batch = to_delete[batch_start : batch_start + 1000]
        try:
            client.delete_objects(Bucket=bucket, Delete={"Objects": batch})
            deleted += len(batch)
        except Exception:
            logger.exception("s3_upload.prune_remote: batch delete failed (%d keys)", len(batch))
    return deleted


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; falling back to %d", name, raw, default)
        return default


__all__ = [
    "DEFAULT_PREFIX",
    "DEFAULT_REGION",
    "DEFAULT_RETENTION_DAYS",
    "ENV_BUCKET",
    "ENV_ENDPOINT",
    "ENV_PREFIX",
    "ENV_REGION",
    "ENV_RETENTION_DAYS",
    "UploadResult",
    "prune_remote",
    "upload_backup",
]
