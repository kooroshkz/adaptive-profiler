"""Pluggable artifact storage: S3 and local-filesystem backends.

The library uses the ArtifactStore protocol internally.  Users never need
to instantiate these directly — ``Profiler.from_yaml()`` creates the right
backend from the schema's ``model_store`` section.
"""

from __future__ import annotations

import io
import json
import os
import pickle
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config.schema import ModelStoreConfig

# Artifact key layout (Hive-style partitioning keeps it queryable with DuckDB/Athena):
#   partition={partition_key}/col={column}/latest.pkl
#   partition={partition_key}/col={column}/latest_meta.json


def _model_subkey(partition_key: str, column: str) -> str:
    return f"partition={partition_key}/col={column}/latest.pkl"


def _meta_subkey(partition_key: str, column: str) -> str:
    return f"partition={partition_key}/col={column}/latest_meta.json"


class ArtifactStore(ABC):
    """Abstract storage backend for trained model artifacts."""

    @abstractmethod
    def save(
        self,
        partition_key: str,
        column: str,
        artifact: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Persist the artifact dict and metadata for this partition/column."""

    @abstractmethod
    def load(self, partition_key: str, column: str) -> dict[str, Any] | None:
        """Return the artifact dict or None if not yet trained."""

    @abstractmethod
    def exists(self, partition_key: str, column: str) -> bool:
        """Return True if a trained artifact exists for this partition/column."""

    @abstractmethod
    def load_meta(self, partition_key: str, column: str) -> dict[str, Any] | None:
        """Return the metadata dict or None if absent."""

    def _enrich_meta(
        self, metadata: dict[str, Any], partition_key: str, column: str
    ) -> dict[str, Any]:
        return {
            **metadata,
            "partition_key": partition_key,
            "column": column,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }


class S3Store(ArtifactStore):
    """Store artifacts in Amazon S3.

    Requires ``boto3`` (``pip install adaptive-profiler[s3]``).
    AWS credentials are read from the standard boto3 credential chain
    (env vars, ~/.aws/credentials, IAM role, etc.).
    """

    def __init__(self, bucket: str, prefix: str) -> None:
        self._bucket = bucket
        self._prefix = prefix.rstrip("/")

    def _full_key(self, subkey: str) -> str:
        return f"{self._prefix}/{subkey}"

    def _client(self):
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for S3 storage. "
                "Install it with: pip install adaptive-profiler[s3]"
            ) from exc
        return boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION", "us-east-1"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID") or None,
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or None,
        )

    def save(
        self,
        partition_key: str,
        column: str,
        artifact: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        s3 = self._client()

        buf = io.BytesIO()
        pickle.dump(artifact, buf)
        buf.seek(0)
        s3.put_object(
            Bucket=self._bucket,
            Key=self._full_key(_model_subkey(partition_key, column)),
            Body=buf.read(),
        )

        enriched = self._enrich_meta(metadata, partition_key, column)
        s3.put_object(
            Bucket=self._bucket,
            Key=self._full_key(_meta_subkey(partition_key, column)),
            Body=json.dumps(enriched, indent=2).encode(),
            ContentType="application/json",
        )

    def load(self, partition_key: str, column: str) -> dict[str, Any] | None:
        try:
            from botocore.exceptions import ClientError
        except ImportError as exc:
            raise ImportError("botocore is required; pip install adaptive-profiler[s3]") from exc
        try:
            resp = self._client().get_object(
                Bucket=self._bucket,
                Key=self._full_key(_model_subkey(partition_key, column)),
            )
            return pickle.loads(resp["Body"].read())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            raise

    def exists(self, partition_key: str, column: str) -> bool:
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            return False
        try:
            self._client().head_object(
                Bucket=self._bucket,
                Key=self._full_key(_model_subkey(partition_key, column)),
            )
            return True
        except ClientError:
            return False

    def load_meta(self, partition_key: str, column: str) -> dict[str, Any] | None:
        try:
            from botocore.exceptions import ClientError
        except ImportError:
            return None
        try:
            resp = self._client().get_object(
                Bucket=self._bucket,
                Key=self._full_key(_meta_subkey(partition_key, column)),
            )
            return json.loads(resp["Body"].read())
        except ClientError:
            return None

    def __repr__(self) -> str:
        return f"S3Store(bucket={self._bucket!r}, prefix={self._prefix!r})"


class LocalStore(ArtifactStore):
    """Store artifacts on the local filesystem.

    Useful for development and testing without AWS credentials.
    """

    def __init__(self, base_dir: str | Path) -> None:
        self._base = Path(base_dir)

    def _model_path(self, partition_key: str, column: str) -> Path:
        return self._base / _model_subkey(partition_key, column)

    def _meta_path(self, partition_key: str, column: str) -> Path:
        return self._base / _meta_subkey(partition_key, column)

    def save(
        self,
        partition_key: str,
        column: str,
        artifact: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        mp = self._model_path(partition_key, column)
        mp.parent.mkdir(parents=True, exist_ok=True)
        with open(mp, "wb") as f:
            pickle.dump(artifact, f)

        enriched = self._enrich_meta(metadata, partition_key, column)
        with open(self._meta_path(partition_key, column), "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2)

    def load(self, partition_key: str, column: str) -> dict[str, Any] | None:
        mp = self._model_path(partition_key, column)
        if not mp.exists():
            return None
        with open(mp, "rb") as f:
            return pickle.load(f)

    def exists(self, partition_key: str, column: str) -> bool:
        return self._model_path(partition_key, column).exists()

    def load_meta(self, partition_key: str, column: str) -> dict[str, Any] | None:
        mp = self._meta_path(partition_key, column)
        if not mp.exists():
            return None
        with open(mp, encoding="utf-8") as f:
            return json.load(f)

    def __repr__(self) -> str:
        return f"LocalStore(base_dir={str(self._base)!r})"


def make_store(config: ModelStoreConfig) -> ArtifactStore:
    """Factory: create the right ArtifactStore from schema config."""
    if config.backend == "s3":
        if not config.bucket:
            raise ValueError("model_store.bucket is required when backend=s3")
        return S3Store(bucket=config.bucket, prefix=config.prefix)
    if config.backend == "local":
        return LocalStore(base_dir=config.local_dir)
    raise ValueError(
        f"Unknown model_store.backend: {config.backend!r}. Expected 's3' or 'local'."
    )
