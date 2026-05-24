"""Unit tests for the S3-compatible backup upload helper.

We never hit a real S3 — every test passes a stub client into
``upload_backup``/``prune_remote``. The retention math is the load-bearing
bit; we cover delete-old-keep-fresh + the empty-bucket branch.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from wc2026.observability import s3_upload


class _StubS3Client:
    """Minimal in-memory S3 lookalike for the methods upload_backup uses."""

    def __init__(self, existing: list[dict[str, Any]] | None = None):
        self.uploaded: list[tuple[str, str, str]] = []  # (local, bucket, key)
        self.deleted: list[list[str]] = []
        self._existing = existing or []

    def upload_file(self, local: str, bucket: str, key: str) -> None:
        self.uploaded.append((local, bucket, key))

    def get_paginator(self, _name: str):
        existing = self._existing

        class _P:
            def paginate(self, **_kw):
                return iter([{"Contents": existing}])

        return _P()

    def delete_objects(self, *, Bucket: str, Delete: dict) -> dict:
        keys = [obj["Key"] for obj in Delete["Objects"]]
        self.deleted.append(keys)
        _ = Bucket
        return {"Deleted": [{"Key": k} for k in keys]}


def _backup_file(
    tmp_path: Path, name: str = "wc2026-2026-06-20.sql.gz", payload: bytes = b"x"
) -> Path:
    p = tmp_path / name
    p.write_bytes(payload)
    return p


def test_upload_returns_none_when_bucket_unset(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("AWS_S3_BUCKET", raising=False)
    out = s3_upload.upload_backup(_backup_file(tmp_path))
    assert out is None


def test_upload_returns_none_when_local_file_missing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "wc2026-backups")
    out = s3_upload.upload_backup(tmp_path / "missing.sql.gz")
    assert out is None


def test_upload_writes_to_bucket_and_returns_result(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "wc2026-backups")
    client = _StubS3Client()
    backup = _backup_file(tmp_path, payload=b"some gzipped sql")
    out = s3_upload.upload_backup(backup, client=client)
    assert out is not None
    assert out.uri == "s3://wc2026-backups/wc2026/backups/wc2026-2026-06-20.sql.gz"
    assert out.key == "wc2026/backups/wc2026-2026-06-20.sql.gz"
    assert out.bytes_uploaded == len(b"some gzipped sql")
    assert client.uploaded == [
        (str(backup), "wc2026-backups", "wc2026/backups/wc2026-2026-06-20.sql.gz")
    ]


def test_upload_uses_custom_prefix_from_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "wc2026-backups")
    monkeypatch.setenv("WC2026_BACKUP_PREFIX", "nightly/")
    client = _StubS3Client()
    backup = _backup_file(tmp_path)
    out = s3_upload.upload_backup(backup, client=client)
    assert out is not None
    assert out.key.startswith("nightly/")


def test_prune_remote_deletes_objects_older_than_retention() -> None:
    now = datetime.now(UTC)
    old1 = now - timedelta(days=45)
    old2 = now - timedelta(days=31)
    fresh = now - timedelta(days=5)
    client = _StubS3Client(
        existing=[
            {"Key": "wc2026/backups/old1.sql.gz", "LastModified": old1},
            {"Key": "wc2026/backups/old2.sql.gz", "LastModified": old2},
            {"Key": "wc2026/backups/fresh.sql.gz", "LastModified": fresh},
        ]
    )
    removed = s3_upload.prune_remote(
        client, bucket="b", prefix="wc2026/backups/", retention_days=30
    )
    assert removed == 2
    assert client.deleted == [["wc2026/backups/old1.sql.gz", "wc2026/backups/old2.sql.gz"]]


def test_prune_remote_noop_when_nothing_old() -> None:
    now = datetime.now(UTC)
    client = _StubS3Client(
        existing=[
            {"Key": "wc2026/backups/fresh.sql.gz", "LastModified": now - timedelta(days=2)},
        ]
    )
    removed = s3_upload.prune_remote(
        client, bucket="b", prefix="wc2026/backups/", retention_days=30
    )
    assert removed == 0
    assert client.deleted == []


def test_prune_remote_noop_when_bucket_empty() -> None:
    client = _StubS3Client(existing=[])
    removed = s3_upload.prune_remote(
        client, bucket="b", prefix="wc2026/backups/", retention_days=30
    )
    assert removed == 0


def test_upload_invokes_prune_after_upload(monkeypatch, tmp_path) -> None:
    """Single-call surface: upload + sweep happen in one operator action."""
    monkeypatch.setenv("AWS_S3_BUCKET", "wc2026-backups")
    monkeypatch.setenv("WC2026_BACKUP_RETENTION_DAYS", "10")
    client = _StubS3Client(
        existing=[
            {
                "Key": "wc2026/backups/wc2026-2025-12-01.sql.gz",
                "LastModified": datetime.now(UTC) - timedelta(days=180),
            },
            {
                "Key": "wc2026/backups/wc2026-2026-06-15.sql.gz",
                "LastModified": datetime.now(UTC) - timedelta(days=5),
            },
        ]
    )
    out = s3_upload.upload_backup(_backup_file(tmp_path), client=client)
    assert out is not None
    assert out.pruned_count == 1
    assert client.deleted == [["wc2026/backups/wc2026-2025-12-01.sql.gz"]]


def test_upload_with_explicit_kwargs_overrides_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("AWS_S3_BUCKET", "wrong-bucket")
    monkeypatch.setenv("WC2026_BACKUP_PREFIX", "wrong/prefix/")
    client = _StubS3Client()
    out = s3_upload.upload_backup(
        _backup_file(tmp_path),
        bucket="correct-bucket",
        prefix="right/prefix/",
        retention_days=14,
        client=client,
    )
    assert out is not None
    assert out.uri.startswith("s3://correct-bucket/right/prefix/")


def test_int_env_falls_back_on_garbage(monkeypatch) -> None:
    monkeypatch.setenv("WC2026_BACKUP_RETENTION_DAYS", "not a number")
    assert s3_upload._int_env(s3_upload.ENV_RETENTION_DAYS, 42) == 42


def test_make_client_constructs_boto3_s3_client() -> None:
    """End-to-end sanity: the factory returns a real boto3 client with the
    endpoint_url + region_name we passed in. We don't make a real HTTP call —
    just inspect the client config."""
    client = s3_upload._make_client(endpoint_url="https://r2.example.test", region="auto")
    # boto3 stores the endpoint on the client's meta.endpoint_url
    assert client.meta.endpoint_url == "https://r2.example.test"
    assert client.meta.region_name == "auto"


def test_upload_backup_silent_when_client_factory_used(monkeypatch, tmp_path) -> None:
    """Sanity: if no client is passed, _make_client is invoked once. We patch the
    factory to avoid touching boto3 in the test."""
    monkeypatch.setenv("AWS_S3_BUCKET", "wc2026-backups")
    backup = _backup_file(tmp_path)
    stub = _StubS3Client()

    def fake_factory(*, endpoint_url=None, region=None):
        _ = endpoint_url, region
        return stub

    monkeypatch.setattr(s3_upload, "_make_client", fake_factory)
    out = s3_upload.upload_backup(backup)
    assert out is not None
    assert stub.uploaded
