"""Sealed CC Switch manifest builder and one-shot model discovery.

Discovery mode consumes one accepted snapshot and performs exactly one bounded
``GET /v1/models`` for each fixed source, emitting a closed redacted result.
Builder mode creates and atomically publishes only the secret-free authorization
manifest. Neither mode accepts endpoint or credential arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hmac
import hashlib
import json
import math
import os
from pathlib import Path
import re
import secrets
import ssl
import stat
import shutil
import tempfile
import select
import subprocess
import signal
import sys
import time
from typing import Any, Final, Mapping, NoReturn
from urllib.parse import urlsplit, urlunsplit

# ``python -I -B tools/cc_switch_model_discovery.py`` intentionally removes the
# checkout from sys.path.  Restore only this tool's own checked-out root; no
# current-working-directory or environment path is trusted.
if __package__ in {None, ""}:  # pragma: no cover - exercised by the CLI process.
    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

import certifi
import httpx

from tools.cc_switch_omp_source import SourceInventoryError, _migration_apply_source


__all__ = ["build_authorization_manifest", "main"]


_AUTHORIZATION_SCHEMA_VERSION: Final = 1
_TOTAL_TIMEOUT_SECONDS: Final = 15.0
_WATCHDOG_CLEANUP_RESERVE_SECONDS: Final = 3.0
_MAX_AUTHORIZATION_BYTES: Final = 64 * 1024
_MAX_BODY_BYTES: Final = 1 << 20
_MAX_SOURCE_DATABASE_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_WAL_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_SHM_BYTES: Final = 8 * 1024 * 1024
_MAX_JSON_DEPTH: Final = 32
_MAX_JSON_NODES: Final = 8192
_MAX_MODELS: Final = 2048
_MAX_ENDPOINT_TYPES: Final = 16
_MAX_CREDENTIAL_BYTES: Final = 8192
_HASH_CHUNK_BYTES: Final = 128 * 1024
_MAX_CHILD_OUTPUT_BYTES: Final = 2 * _MAX_BODY_BYTES

_DY_SOURCE_ID: Final = "6becb67f-1730-438d-b8ad-9429675d1090"
_XUANJU_SOURCE_ID: Final = "616ccdfd-8de6-4c82-a4cc-24f71edf0c49"
_PROVIDERS: Final = (
    ("dy", _DY_SOURCE_ID),
    ("xuanju", _XUANJU_SOURCE_ID),
)
_SOURCE_IDS: Final = frozenset(source_id for _label, source_id in _PROVIDERS)
_SOURCE_API_FAMILIES: Final = frozenset(
    {"anthropic", "openai_chat", "openai_responses"}
)

# The response field is remote, untrusted data.  This deliberately recognizes
# only exact, documented family-shaped tokens; a synonym or a mixed list is
# never guessed into a usable family.
_ENDPOINT_TYPE_TO_API_FAMILY: Final = {
    "anthropic-messages": "anthropic-messages",
    "openai-completions": "openai-completions",
    "openai-responses": "openai-responses",
}

_MODEL_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_ENDPOINT_TYPE_RE: Final = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_ENDPOINT_PATH_SEGMENT_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._~-]{0,127}$")
_DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_RE: Final = re.compile(r"^[0-9]+$")

_FILE_MANIFEST_KEYS: Final = frozenset(
    {
        "device",
        "fileId",
        "owner",
        "mode",
        "size",
        "mtimeNs",
        "ctimeNs",
        "sha256",
    }
)
_SOURCE_PRECONDITION_KEYS: Final = frozenset(
    {"realpath", "database", "wal", "shm"}
)
_SOURCE_SEAM_PRECONDITION_KEYS: Final = frozenset(
    {
        "realpath",
        "device",
        "fileId",
        "owner",
        "mode",
        "size",
        "mtimeNs",
        "ctimeNs",
        "sha256",
        "snapshotSha256",
        "wal",
        "shm",
    }
)

_WATCHDOG_ENV: Final = "HERMES_CC_SWITCH_DISCOVERY_WATCHDOG"
_WATCHDOG_FD_ARG_PREFIX: Final = "--__cc-switch-watchdog-fd="
_WATCHDOG_PARENT_ARG_PREFIX: Final = "--__cc-switch-watchdog-parent="
_WATCHDOG_TOKEN_ARG_PREFIX: Final = "--__cc-switch-watchdog-token="
_WATCHDOG_TOKEN_BYTES: Final = 32


class _DiscoveryBlocked(RuntimeError):
    """Refuse without retaining or emitting a reason string."""


class _ResponseTooLarge(_DiscoveryBlocked):
    """A bounded response exceeded the declared one-mebibyte ceiling."""


def _block() -> NoReturn:
    raise _DiscoveryBlocked()


def _too_large() -> NoReturn:
    raise _ResponseTooLarge()
def _is_isolated_runtime() -> bool:
    return bool(sys.flags.isolated and sys.flags.dont_write_bytecode)


def _require_isolated_runtime() -> None:
    key_log_path = os.environ.pop("SSLKEYLOGFILE", None)
    if key_log_path is not None or not _is_isolated_runtime():
        _block()




def _exact_mapping(value: Any, keys: frozenset[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        _block()
    return value


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST_RE.fullmatch(value) is not None


def _nonnegative_int(value: Any) -> int:
    if type(value) is not int or value < 0:
        _block()
    return value


def _validate_manifest(value: Any) -> dict[str, Any]:
    manifest = _exact_mapping(value, _FILE_MANIFEST_KEYS)
    normalized = {key: _nonnegative_int(manifest[key]) for key in _FILE_MANIFEST_KEYS - {"sha256"}}
    digest = manifest["sha256"]
    if not _is_digest(digest):
        _block()
    if normalized["fileId"] == 0 or normalized["mode"] > 0o777:
        _block()
    if normalized["mode"] & 0o077:
        _block()
    normalized["sha256"] = digest
    return normalized


def _strict_absolute_path(value: Any) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        _block()
    if not os.path.isabs(value) or os.path.normpath(value) != value:
        _block()
    return value


def _validate_authorization(value: Any) -> dict[str, Any]:
    authorization = _exact_mapping(
        value,
        frozenset(
            {
                "schemaVersion",
                "sourceSnapshotSha256",
                "sourcePrecondition",
                "providers",
            }
        ),
    )
    if authorization["schemaVersion"] != _AUTHORIZATION_SCHEMA_VERSION:
        _block()
    snapshot = authorization["sourceSnapshotSha256"]
    if not _is_digest(snapshot):
        _block()

    precondition = _exact_mapping(
        authorization["sourcePrecondition"], _SOURCE_PRECONDITION_KEYS
    )
    normalized_precondition = {
        "realpath": _strict_absolute_path(precondition["realpath"]),
        "database": _validate_manifest(precondition["database"]),
        "wal": None
        if precondition["wal"] is None
        else _validate_manifest(precondition["wal"]),
        "shm": None
        if precondition["shm"] is None
        else _validate_manifest(precondition["shm"]),
    }

    providers = authorization["providers"]
    if not isinstance(providers, list) or len(providers) != len(_PROVIDERS):
        _block()
    normalized_providers: list[dict[str, str]] = []
    for record, (label, source_id) in zip(providers, _PROVIDERS, strict=True):
        record = _exact_mapping(
            record,
            frozenset(
                {
                    "provider",
                    "sourceId",
                    "sourceApiFamily",
                    "endpointSha256",
                    "applyBindingSha256",
                }
            ),
        )
        api_family = record["sourceApiFamily"]
        if (
            record["provider"] != label
            or record["sourceId"] != source_id
            or api_family not in _SOURCE_API_FAMILIES
            or not _is_digest(record["endpointSha256"])
            or not _is_digest(record["applyBindingSha256"])
        ):
            _block()
        normalized_providers.append(
            {
                "provider": label,
                "sourceId": source_id,
                "sourceApiFamily": api_family,
                "endpointSha256": record["endpointSha256"],
                "applyBindingSha256": record["applyBindingSha256"],
            }
        )
    return {
        "sourceSnapshotSha256": snapshot,
        "sourcePrecondition": normalized_precondition,
        "providers": normalized_providers,
    }


def _open_flags(*, directory: bool = False) -> int:
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int):
        _block()
    flags = os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    if not isinstance(close_on_exec, int):
        _block()
    flags |= close_on_exec
    if directory:
        directory_flag = getattr(os, "O_DIRECTORY", None)
        if not isinstance(directory_flag, int):
            _block()
        flags |= directory_flag
    return flags


def _same_stat(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_uid,
        stat.S_IMODE(left.st_mode),
        left.st_nlink,
        left.st_size,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_uid,
        stat.S_IMODE(right.st_mode),
        right.st_nlink,
        right.st_size,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _current_uid() -> int:
    try:
        return os.getuid()
    except AttributeError:
        _block()


def _validate_directory(details: os.stat_result) -> None:
    mode = stat.S_IMODE(details.st_mode)
    if (
        not stat.S_ISDIR(details.st_mode)
        or details.st_uid not in {0, _current_uid()}
        or mode & 0o022
    ):
        _block()


def _validate_regular(details: os.stat_result, maximum_bytes: int) -> None:
    mode = stat.S_IMODE(details.st_mode)
    if (
        not stat.S_ISREG(details.st_mode)
        or details.st_uid != _current_uid()
        or details.st_nlink != 1
        or mode & 0o077
        or details.st_size < 0
        or details.st_size > maximum_bytes
    ):
        _block()


def _normalised_input_path(value: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(value)
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            _block()
        return os.path.normpath(os.path.abspath(raw))
    except (OSError, TypeError, ValueError):
        _block()


def _open_parent(path: str) -> tuple[int, str]:
    descriptor = -1
    try:
        components = tuple(component for component in path.split(os.sep) if component)
        if not components:
            _block()
        descriptor = os.open(os.path.sep, _open_flags(directory=True))
        _validate_directory(os.fstat(descriptor))
        for component in components[:-1]:
            child = os.open(component, _open_flags(directory=True), dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
            _validate_directory(os.fstat(descriptor))
        return descriptor, components[-1]
    except (OSError, ValueError):
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        _block()


def _lstat_at(parent_fd: int, name: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except (OSError, ValueError):
        _block()


def _manifest(details: os.stat_result, digest: str) -> dict[str, Any]:
    return {
        "device": details.st_dev,
        "fileId": details.st_ino,
        "owner": details.st_uid,
        "mode": stat.S_IMODE(details.st_mode),
        "size": details.st_size,
        "mtimeNs": details.st_mtime_ns,
        "ctimeNs": details.st_ctime_ns,
        "sha256": digest,
    }


def _hash_descriptor(descriptor: int, size: int) -> str:
    digest = hashlib.sha256()
    offset = 0
    try:
        while offset < size:
            chunk = os.pread(descriptor, min(_HASH_CHUNK_BYTES, size - offset), offset)
            if not chunk:
                _block()
            digest.update(chunk)
            offset += len(chunk)
    except (AttributeError, OSError, ValueError):
        _block()
    return digest.hexdigest()


def _capture_regular(
    parent_fd: int, name: str, maximum_bytes: int
) -> dict[str, Any]:
    descriptor = -1
    try:
        before = _lstat_at(parent_fd, name)
        _validate_regular(before, maximum_bytes)
        descriptor = os.open(name, _open_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        _validate_regular(opened, maximum_bytes)
        if not _same_stat(before, opened):
            _block()
        digest = _hash_descriptor(descriptor, opened.st_size)
        after = _lstat_at(parent_fd, name)
        final = os.fstat(descriptor)
        _validate_regular(final, maximum_bytes)
        if not _same_stat(opened, final) or not _same_stat(after, final):
            _block()
        return _manifest(final, digest)
    except (OSError, ValueError):
        _block()
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _capture_optional_regular(
    parent_fd: int, name: str, maximum_bytes: int
) -> dict[str, Any] | None:
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        _block()
    return _capture_regular(parent_fd, name, maximum_bytes)


def _capture_source_precondition(db_path: str | os.PathLike[str]) -> dict[str, Any]:
    path = _normalised_input_path(db_path)
    parent_fd = -1
    try:
        parent_fd, basename = _open_parent(path)
        try:
            os.stat(basename + "-journal", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        except (OSError, ValueError):
            _block()
        else:
            _block()
        result = {
            "realpath": path,
            "database": _capture_regular(
                parent_fd, basename, _MAX_SOURCE_DATABASE_BYTES
            ),
            "wal": _capture_optional_regular(
                parent_fd, basename + "-wal", _MAX_SOURCE_WAL_BYTES
            ),
            "shm": _capture_optional_regular(
                parent_fd, basename + "-shm", _MAX_SOURCE_SHM_BYTES
            ),
        }
        try:
            os.stat(basename + "-journal", dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return result
        except (OSError, ValueError):
            _block()
        _block()
    finally:
        if parent_fd != -1:
            with contextlib.suppress(OSError):
                os.close(parent_fd)


def _read_secure_authorization(path: str | os.PathLike[str]) -> bytes:
    absolute = _normalised_input_path(path)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd, basename = _open_parent(absolute)
        before = _lstat_at(parent_fd, basename)
        _validate_regular(before, _MAX_AUTHORIZATION_BYTES)
        descriptor = os.open(basename, _open_flags(), dir_fd=parent_fd)
        opened = os.fstat(descriptor)
        _validate_regular(opened, _MAX_AUTHORIZATION_BYTES)
        if not _same_stat(before, opened):
            _block()
        body = bytearray()
        offset = 0
        while offset < opened.st_size:
            chunk = os.pread(descriptor, min(_HASH_CHUNK_BYTES, opened.st_size - offset), offset)
            if not chunk:
                _block()
            body.extend(chunk)
            offset += len(chunk)
        final = os.fstat(descriptor)
        after = _lstat_at(parent_fd, basename)
        if not _same_stat(opened, final) or not _same_stat(after, final):
            _block()
        return bytes(body)
    except (OSError, ValueError):
        _block()
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if parent_fd != -1:
            with contextlib.suppress(OSError):
                os.close(parent_fd)


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if not isinstance(key, str) or key in result:
            raise ValueError
        result[key] = item
    return result


def _reject_json_number(_value: str) -> NoReturn:
    raise ValueError


def _check_json_limits(value: Any) -> None:
    nodes = 0
    stack: list[tuple[Any, int]] = [(value, 1)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            _block()
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    _block()
                stack.append((child, depth + 1))
        elif isinstance(item, list):
            stack.extend((child, depth + 1) for child in item)
        elif item is None or type(item) in {bool, int, str}:
            continue
        elif type(item) is float and math.isfinite(item):
            continue
        else:
            _block()


def _parse_json(body: bytearray) -> Any:
    try:
        text = body.decode("utf-8", "strict")
        parsed = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_number,
        )
        _check_json_limits(parsed)
        return parsed
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        _block()


def _normalise_models_url(endpoint: str) -> str:
    """Append the fixed models suffix to one explicit, canonical HTTPS base."""
    if (
        not isinstance(endpoint, str)
        or not endpoint
        or any(ord(character) < 0x21 for character in endpoint)
        or "\\" in endpoint
    ):
        _block()
    try:
        parts = urlsplit(endpoint)
        if (
            parts.scheme.lower() != "https"
            or not parts.netloc
            or "@" in parts.netloc
            or parts.query
            or parts.fragment
        ):
            _block()
        raw_path = parts.path
        if not raw_path or raw_path == "/":
            base_segments: list[str] = []
        else:
            if not raw_path.startswith("/"):
                _block()
            base_segments = raw_path.split("/")[1:]
            if base_segments[-1] == "":
                base_segments.pop()
            if (
                not base_segments
                or len(base_segments) > 16
                or len(raw_path) > 512
                or any(
                    segment in {"", ".", ".."}
                    or _ENDPOINT_PATH_SEGMENT_RE.fullmatch(segment) is None
                    for segment in base_segments
                )
            ):
                _block()
        host = parts.hostname
        port = parts.port
        if not host or not host.isascii() or any(character.isspace() for character in host):
            _block()
        if port is not None and not 1 <= port <= 65535:
            _block()
        rendered_host = f"[{host.lower()}]" if ":" in host else host.lower()
        rendered_netloc = rendered_host
        if port is not None and port != 443:
            rendered_netloc += f":{port}"
        if base_segments and base_segments[-1] == "v1":
            target_segments = [*base_segments, "models"]
        else:
            target_segments = [*base_segments, "v1", "models"]
        return urlunsplit(
            ("https", rendered_netloc, "/" + "/".join(target_segments), "", "")
        )
    except (TypeError, ValueError):
        _block()


def _validate_credential(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_CREDENTIAL_BYTES
        or any(ord(character) < 0x21 or ord(character) > 0x7E for character in value)
    ):
        _block()
    return value


def _source_precondition_matches(
    source_precondition: Any, expected: Mapping[str, Any], snapshot: str
) -> bool:
    if not isinstance(source_precondition, dict) or set(source_precondition) != _SOURCE_SEAM_PRECONDITION_KEYS:
        return False
    if (
        source_precondition.get("realpath") != expected["realpath"]
        or source_precondition.get("snapshotSha256") != snapshot
    ):
        return False
    source_database = {
        key: source_precondition.get(key) for key in _FILE_MANIFEST_KEYS
    }
    if source_database != expected["database"]:
        return False
    if source_precondition.get("wal") != expected["wal"]:
        return False
    if source_precondition.get("shm") != expected["shm"]:
        return False
    return True


def _validate_private_source(source: Any, authorization: Mapping[str, Any]) -> list[dict[str, Any]]:
    source = _exact_mapping(
        source,
        frozenset(
            {
                "source_app_type",
                "source_snapshot_sha256",
                "precondition",
                "providers",
            }
        ),
    )
    snapshot = authorization["sourceSnapshotSha256"]
    if (
        source["source_app_type"] != "claude"
        or source["source_snapshot_sha256"] != snapshot
        or not _source_precondition_matches(
            source["precondition"], authorization["sourcePrecondition"], snapshot
        )
    ):
        _block()
    private_records = source["providers"]
    if not isinstance(private_records, list) or len(private_records) != len(_PROVIDERS):
        _block()
    by_source_id: dict[str, dict[str, Any]] = {}
    expected_keys = frozenset(
        {
            "source_id",
            "api_family",
            "endpoint",
            "endpoint_sha256",
            "apply_binding_sha256",
            "credential_value",
        }
    )
    for record in private_records:
        record = _exact_mapping(record, expected_keys)
        source_id = record["source_id"]
        if not isinstance(source_id, str) or source_id in by_source_id:
            _block()
        by_source_id[source_id] = record
    if set(by_source_id) != _SOURCE_IDS:
        _block()

    requests: list[dict[str, Any]] = []
    for expected in authorization["providers"]:
        record = by_source_id[expected["sourceId"]]
        endpoint = record["endpoint"]
        credential = _validate_credential(record["credential_value"])
        if (
            record["api_family"] != expected["sourceApiFamily"]
            or record["api_family"] not in _SOURCE_API_FAMILIES
            or not isinstance(endpoint, str)
            or not _is_digest(record["endpoint_sha256"])
            or not _is_digest(record["apply_binding_sha256"])
            or hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            != record["endpoint_sha256"]
            or record["endpoint_sha256"] != expected["endpointSha256"]
            or record["apply_binding_sha256"] != expected["applyBindingSha256"]
        ):
            _block()
        requests.append(
            {
                "provider": expected["provider"],
                "source_api_family": record["api_family"],
                "endpoint": endpoint,
                "credential": credential,
                "models_url": _normalise_models_url(endpoint),
            }
        )
    return requests


def _authorization_from_private_source(
    source: Any, source_precondition: Mapping[str, Any]
) -> dict[str, Any]:
    source = _exact_mapping(
        source,
        frozenset(
            {
                "source_app_type",
                "source_snapshot_sha256",
                "precondition",
                "providers",
            }
        ),
    )
    snapshot = source["source_snapshot_sha256"]
    if (
        source["source_app_type"] != "claude"
        or not _is_digest(snapshot)
        or not _source_precondition_matches(
            source["precondition"], source_precondition, snapshot
        )
    ):
        _block()
    private_records = source["providers"]
    if not isinstance(private_records, list) or len(private_records) != len(_PROVIDERS):
        _block()
    expected_keys = frozenset(
        {
            "source_id",
            "api_family",
            "endpoint",
            "endpoint_sha256",
            "apply_binding_sha256",
            "credential_value",
        }
    )
    by_source_id: dict[str, dict[str, Any]] = {}
    for record in private_records:
        record = _exact_mapping(record, expected_keys)
        source_id = record["source_id"]
        if not isinstance(source_id, str) or source_id in by_source_id:
            _block()
        by_source_id[source_id] = record
    if set(by_source_id) != _SOURCE_IDS:
        _block()

    providers: list[dict[str, str]] = []
    for label, source_id in _PROVIDERS:
        record = by_source_id[source_id]
        endpoint = record["endpoint"]
        credential = _validate_credential(record["credential_value"])
        if (
            record["api_family"] not in _SOURCE_API_FAMILIES
            or not isinstance(endpoint, str)
            or not _is_digest(record["endpoint_sha256"])
            or not _is_digest(record["apply_binding_sha256"])
            or hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            != record["endpoint_sha256"]
        ):
            _block()
        _normalise_models_url(endpoint)
        providers.append(
            {
                "provider": label,
                "sourceId": source_id,
                "sourceApiFamily": record["api_family"],
                "endpointSha256": record["endpoint_sha256"],
                "applyBindingSha256": record["apply_binding_sha256"],
            }
        )
        # Drop this local reference as soon as its binding has been verified.
        del credential
    return {
        "schemaVersion": _AUTHORIZATION_SCHEMA_VERSION,
        "sourceSnapshotSha256": snapshot,
        "sourcePrecondition": dict(source_precondition),
        "providers": providers,
    }


def build_authorization_manifest(
    db_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Create an unsigned, secret-free authorization candidate from one snapshot."""
    _require_isolated_runtime()
    source: Any = None
    try:
        source_precondition = _capture_source_precondition(db_path)
        source = _migration_apply_source(
            db_path,
            selected_source_ids=_SOURCE_IDS,
            approved_origin_source_ids=_SOURCE_IDS,
        )
        if _capture_source_precondition(db_path) != source_precondition:
            _block()
        return _authorization_from_private_source(source, source_precondition)
    except Exception:
        _block()
    finally:
        _discard_private_source(source, [])


def _write_authorization_manifest(
    output_path: str | os.PathLike[str], authorization: Mapping[str, Any]
) -> tuple[str, str]:
    """Publish a validated manifest once, without replacing an existing file."""
    payload = json.dumps(
        authorization,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    _validate_authorization(json.loads(payload.decode("utf-8")))
    absolute = _normalised_input_path(output_path)
    parent_fd = -1
    temporary_name: str | None = None
    temporary_fd = -1
    try:
        parent_fd, basename = _open_parent(absolute)
        try:
            os.stat(basename, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            _block()
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(8):
            candidate = (
                f".{basename}.tmp-{os.getpid()}-{secrets.token_hex(12)}"
            )
            try:
                temporary_fd = os.open(
                    candidate, flags, 0o600, dir_fd=parent_fd
                )
            except FileExistsError:
                continue
            temporary_name = candidate
            break
        if temporary_fd == -1 or temporary_name is None:
            _block()
        remaining = memoryview(payload)
        while remaining:
            written = os.write(temporary_fd, remaining)
            if written <= 0:
                _block()
            remaining = remaining[written:]
        os.fchmod(temporary_fd, 0o600)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = -1
        try:
            os.link(
                temporary_name,
                basename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except FileExistsError:
            _block()
        os.unlink(temporary_name, dir_fd=parent_fd)
        temporary_name = None
        os.fsync(parent_fd)
        return absolute, hashlib.sha256(payload).hexdigest()
    except (OSError, TypeError, ValueError):
        _block()
    finally:
        if temporary_fd != -1:
            with contextlib.suppress(OSError):
                os.close(temporary_fd)
        if parent_fd != -1 and temporary_name is not None:
            with contextlib.suppress(OSError):
                os.unlink(temporary_name, dir_fd=parent_fd)
        if parent_fd != -1:
            with contextlib.suppress(OSError):
                os.close(parent_fd)

def _new_transport() -> httpx.AsyncBaseTransport:
    context = ssl.create_default_context(cafile=certifi.where())
    if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
        _block()
    return httpx.AsyncHTTPTransport(
        verify=context,
        retries=0,
        trust_env=False,
        http1=True,
        http2=False,
    )


def _result(
    status_category: str,
    *,
    http_status: int | None = None,
    models: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    returned_models = [] if models is None else models
    return {
        "statusCategory": status_category,
        "httpStatus": http_status,
        "modelCount": len(returned_models),
        "models": returned_models,
    }


def _source_blocked_output() -> dict[str, Any]:
    return {
        label: _result("source_blocked") for label, _source_id in _PROVIDERS
    }


def _response_models(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, dict) or not isinstance(value.get("data"), list):
        _block()
    data = value["data"]
    if len(data) > _MAX_MODELS:
        _block()
    seen_ids: set[str] = set()
    models: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            _block()
        model_id = item.get("id")
        if (
            not isinstance(model_id, str)
            or _MODEL_ID_RE.fullmatch(model_id) is None
            or any(segment in {"", ".", ".."} for segment in model_id.split("/"))
            or model_id in seen_ids
        ):
            _block()
        seen_ids.add(model_id)
        endpoint_types_present = "supported_endpoint_types" in item
        derived_api_family = "unknown"
        if endpoint_types_present:
            raw_types = item["supported_endpoint_types"]
            if not isinstance(raw_types, list) or len(raw_types) > _MAX_ENDPOINT_TYPES:
                _block()
            endpoint_types: set[str] = set()
            for endpoint_type in raw_types:
                if (
                    not isinstance(endpoint_type, str)
                    or _ENDPOINT_TYPE_RE.fullmatch(endpoint_type) is None
                    or endpoint_type in endpoint_types
                ):
                    _block()
                endpoint_types.add(endpoint_type)
            mapped = {
                _ENDPOINT_TYPE_TO_API_FAMILY[endpoint_type]
                for endpoint_type in endpoint_types
                if endpoint_type in _ENDPOINT_TYPE_TO_API_FAMILY
            }
            if len(mapped) == 1 and len(mapped) == len(endpoint_types):
                derived_api_family = next(iter(mapped))
        models.append(
            {
                "id": model_id,
                "supportedEndpointTypesPresent": endpoint_types_present,
                "derivedApiFamily": derived_api_family,
            }
        )
    return models


def _identity_headers(record: Mapping[str, Any]) -> dict[str, str]:
    credential = record["credential"]
    api_family = record["source_api_family"]
    if api_family in {"openai_chat", "openai_responses"}:
        authorization = f"Bearer {credential}"
        return {
            "accept": "application/json",
            "accept-encoding": "identity",
            "authorization": authorization,
        }
    if api_family == "anthropic":
        return {
            "accept": "application/json",
            "accept-encoding": "identity",
            "anthropic-version": "2023-06-01",
            "x-api-key": credential,
        }
    _block()


def _content_length(response: httpx.Response) -> int | None:
    raw = response.headers.get("content-length")
    if raw is None:
        return None
    if _DECIMAL_RE.fullmatch(raw) is None:
        _block()
    size = int(raw)
    if size > _MAX_BODY_BYTES:
        _too_large()
    return size


async def _read_response_body(response: httpx.Response) -> bytearray:
    encoding = response.headers.get("content-encoding")
    if encoding is not None and encoding.strip().lower() != "identity":
        _block()
    _content_length(response)
    body = bytearray()
    try:
        async for chunk in response.aiter_raw():
            if len(chunk) > _MAX_BODY_BYTES - len(body):
                _too_large()
            body.extend(chunk)
        return body
    except _DiscoveryBlocked:
        body.clear()
        raise
    except (httpx.HTTPError, OSError, ValueError):
        body.clear()
        _block()

def _deadline_remaining(deadline: float) -> float:
    return deadline - time.monotonic()


async def _close_client_bounded(
    client: httpx.AsyncClient, deadline: float
) -> None:
    close_task = asyncio.create_task(client.aclose())
    remaining = _deadline_remaining(deadline)
    if remaining <= 0:
        close_task.cancel()
        with contextlib.suppress(BaseException):
            await close_task
        return
    try:
        await asyncio.wait_for(close_task, timeout=remaining)
    except BaseException:
        close_task.cancel()
        with contextlib.suppress(BaseException):
            await close_task


async def _fetch_one(
    client: httpx.AsyncClient, record: Mapping[str, Any], deadline: float
) -> dict[str, Any]:
    headers = _identity_headers(record)
    body: bytearray | None = None
    http_status: int | None = None
    try:
        remaining = _deadline_remaining(deadline)
        if remaining <= 0:
            return _result("timeout")
        try:
            async with asyncio.timeout(remaining):
                async with client.stream(
                    "GET", record["models_url"], headers=headers
                ) as response:
                    if _deadline_remaining(deadline) <= 0:
                        return _result("timeout")
                    status = response.status_code
                    if 300 <= status < 400:
                        return _result("redirect_blocked", http_status=status)
                    if status != 200:
                        return _result("http_error", http_status=status)
                    http_status = status
                    body = await _read_response_body(response)
                    try:
                        if _deadline_remaining(deadline) <= 0:
                            return _result("timeout")
                        models = _response_models(_parse_json(body))
                        if _deadline_remaining(deadline) <= 0:
                            return _result("timeout")
                    finally:
                        body.clear()
                        body = None
                    return _result("ok", http_status=http_status, models=models)
        except TimeoutError:
            return _result("timeout")
        except _ResponseTooLarge:
            return _result("response_too_large", http_status=http_status)
        except _DiscoveryBlocked:
            return _result("response_invalid", http_status=http_status)
        except httpx.HTTPError:
            return _result("transport_error")
        except Exception:
            return _result("transport_error")
    finally:
        if body is not None:
            body.clear()
        headers.clear()


def _record_taint_values(
    record: Mapping[str, Any],
) -> tuple[tuple[str, bool], ...]:
    endpoint = record["endpoint"]
    models_url = record["models_url"]
    credential = record["credential"]
    try:
        parts = urlsplit(endpoint)
    except (TypeError, ValueError):
        _block()
    values: list[tuple[str, bool]] = [
        (endpoint, True),
        (models_url, True),
        (credential, False),
    ]
    if parts.netloc:
        values.append((parts.netloc, True))
        values.append((f"{parts.scheme}://{parts.netloc}", True))
    if parts.hostname is not None:
        values.append((parts.hostname, True))
    return tuple(value for value in values if isinstance(value[0], str) and value[0])


def _contains_taint(
    value: Mapping[str, Any], secrets: tuple[tuple[str, bool], ...]
) -> bool:
    serialized = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    for secret, case_insensitive in secrets:
        if not secret:
            continue
        escaped = json.dumps(secret, ensure_ascii=True)[1:-1]
        haystack = serialized.casefold() if case_insensitive else serialized
        if case_insensitive:
            if secret.casefold() in haystack or escaped.casefold() in haystack:
                return True
        elif secret in serialized or escaped in serialized:
            return True
    return False


def _redact_tainted_output(
    results: list[dict[str, Any]], secrets: tuple[tuple[str, bool], ...]
) -> dict[str, Any]:
    output = {
        label: (
            _result("redaction_blocked")
            if _contains_taint(item, secrets)
            else item
        )
        for item, (label, _source_id) in zip(results, _PROVIDERS, strict=True)
    }
    if _contains_taint(output, secrets):
        return {
            label: _result("redaction_blocked")
            for label, _source_id in _PROVIDERS
        }
    return output


def _discard_private_source(source: Any, requests: list[dict[str, Any]]) -> None:
    for request in requests:
        request.clear()
    if isinstance(source, dict):
        providers = source.get("providers")
        if isinstance(providers, list):
            for provider in providers:
                if isinstance(provider, dict):
                    provider.clear()
            providers.clear()
        source.clear()


async def run_model_discovery(
    db_path: str | os.PathLike[str],
    authorization_path: str | os.PathLike[str],
    *,
    accepted_authorization_sha256: str,
) -> dict[str, Any]:
    # Establish the single wall-clock budget before isolation, authorization,
    # source capture, and either provider request. Synchronous source work is
    # checked at every phase boundary because asyncio cannot cancel it.
    deadline = time.monotonic() + _TOTAL_TIMEOUT_SECONDS
    source: Any = None
    requests: list[dict[str, Any]] = []
    try:
        _require_isolated_runtime()
        if _deadline_remaining(deadline) <= 0:
            _block()
        if not _is_digest(accepted_authorization_sha256):
            _block()
        raw_authorization = _read_secure_authorization(authorization_path)
        if _deadline_remaining(deadline) <= 0:
            _block()
        if hashlib.sha256(raw_authorization).hexdigest() != accepted_authorization_sha256:
            _block()
        authorization = _validate_authorization(_parse_json(bytearray(raw_authorization)))
        if _deadline_remaining(deadline) <= 0:
            _block()
        if _capture_source_precondition(db_path) != authorization["sourcePrecondition"]:
            _block()
        if _deadline_remaining(deadline) <= 0:
            _block()
        source = _migration_apply_source(
            db_path,
            selected_source_ids=_SOURCE_IDS,
            approved_origin_source_ids=_SOURCE_IDS,
        )
        if _deadline_remaining(deadline) <= 0:
            _block()
        # Re-capture after the private source seam. This catches same-inode
        # rewrites and sidecar changes that happen during source extraction.
        if _capture_source_precondition(db_path) != authorization["sourcePrecondition"]:
            _block()
        requests = _validate_private_source(source, authorization)
        secrets = tuple(
            value
            for request in requests
            for value in _record_taint_values(request)
        )
        results: list[dict[str, Any]] = []
        for request in requests:
            if _deadline_remaining(deadline) <= 0:
                results.append(_result("timeout"))
                continue
            request_transport = _new_transport()
            if _deadline_remaining(deadline) <= 0:
                results.append(_result("timeout"))
                continue
            client = httpx.AsyncClient(
                transport=request_transport,
                trust_env=False,
                follow_redirects=False,
                http1=True,
                http2=False,
                timeout=httpx.Timeout(_TOTAL_TIMEOUT_SECONDS),
            )
            try:
                results.append(await _fetch_one(client, request, deadline))
            finally:
                await _close_client_bounded(client, deadline)
        return _redact_tainted_output(results, secrets)
    except Exception:
        return _source_blocked_output()
    finally:
        _discard_private_source(source, requests)


class _SafeArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        _block()


def _parser() -> argparse.ArgumentParser:
    parser = _SafeArgumentParser(add_help=False)
    parser.add_argument("--source-db")
    parser.add_argument("--authorization")
    parser.add_argument("--accepted-authorization-sha256")
    parser.add_argument("--build-authorization", action="store_true")
    parser.add_argument("--output")
    return parser


def _harden_process() -> None:
    try:
        os.umask(0o077)
        import resource

        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    except (ImportError, OSError, ValueError):
        _block()


def _emit(value: Mapping[str, Any]) -> None:
    try:
        sys.stdout.write(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n")
        sys.stdout.flush()
    except (OSError, TypeError, ValueError):
        pass


def _validate_watchdog_result(value: Any, *, builder: bool) -> dict[str, Any]:
    if not isinstance(value, dict):
        _block()
    if builder:
        if set(value) != {
            "status",
            "path",
            "authorizationSha256",
            "providerCount",
        }:
            _block()
        if (
            value["status"] != "authorization_written"
            or not isinstance(value["path"], str)
            or not _is_digest(value["authorizationSha256"])
            or value["providerCount"] != len(_PROVIDERS)
        ):
            _block()
        return value
    if set(value) != {label for label, _source_id in _PROVIDERS}:
        _block()
    expected_result_keys = {"statusCategory", "httpStatus", "modelCount", "models"}
    for item in value.values():
        if not isinstance(item, dict) or set(item) != expected_result_keys:
            _block()
    return value


def _read_watchdog_capability(
    read_fd: int, expected_token: bytes, deadline: float
) -> None:
    if read_fd < 3 or len(expected_token) != _WATCHDOG_TOKEN_BYTES:
        _block()
    try:
        details = os.fstat(read_fd)
        if not stat.S_ISFIFO(details.st_mode):
            _block()
        token = bytearray()
        while len(token) < _WATCHDOG_TOKEN_BYTES:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _block()
            ready, _, _ = select.select([read_fd], [], [], remaining)
            if not ready:
                _block()
            chunk = os.read(read_fd, _WATCHDOG_TOKEN_BYTES - len(token))
            if not chunk:
                _block()
            token.extend(chunk)
    except (OSError, ValueError):
        _block()
    finally:
        with contextlib.suppress(OSError):
            os.close(read_fd)
    if not hmac.compare_digest(token, expected_token):
        _block()


def _kill_watchdog_pid(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        with contextlib.suppress(OSError):
            os.kill(pid, signal.SIGKILL)


def _wait_watchdog_pid(pid: int, deadline: float) -> int | None:
    while True:
        try:
            waited_pid, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return None
        if waited_pid == pid:
            return status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return None
        time.sleep(min(0.01, remaining))


def _read_watchdog_output(
    pid: int, stdout_fd: int, stderr_fd: int, deadline: float
) -> tuple[bytes, bytes, int]:
    buffers = {stdout_fd: bytearray(), stderr_fd: bytearray()}
    stdout_buffer = buffers[stdout_fd]
    stderr_buffer = buffers[stderr_fd]
    while buffers:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _kill_watchdog_pid(pid)
            for descriptor in tuple(buffers):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            _wait_watchdog_pid(pid, deadline)
            _block()
        try:
            ready, _, _ = select.select(tuple(buffers), [], [], remaining)
        except (OSError, ValueError):
            _kill_watchdog_pid(pid)
            _wait_watchdog_pid(pid, deadline)
            _block()
        if not ready:
            _kill_watchdog_pid(pid)
            for descriptor in tuple(buffers):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            _wait_watchdog_pid(pid, deadline)
            _block()
        for descriptor in ready:
            try:
                chunk = os.read(descriptor, 65536)
            except OSError:
                _kill_watchdog_pid(pid)
                _wait_watchdog_pid(pid, deadline)
                _block()
            if not chunk:
                os.close(descriptor)
                buffers.pop(descriptor)
                continue
            buffers[descriptor].extend(chunk)
            if len(buffers[descriptor]) > _MAX_CHILD_OUTPUT_BYTES:
                _kill_watchdog_pid(pid)
                for open_descriptor in tuple(buffers):
                    with contextlib.suppress(OSError):
                        os.close(open_descriptor)
                _wait_watchdog_pid(pid, deadline)
                _block()
    status = _wait_watchdog_pid(pid, deadline)
    if status is None:
        _kill_watchdog_pid(pid)
        _block()
    return bytes(stdout_buffer), bytes(stderr_buffer), status


def _cleanup_control_root_bounded(path: str, deadline: float) -> bool:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        return False
    cleanup_code = "import shutil,sys; shutil.rmtree(sys.argv[1], ignore_errors=False)"
    environment = os.environ.copy()
    environment.pop(_WATCHDOG_ENV, None)
    try:
        cleanup_process = subprocess.Popen(
            [sys.executable, "-I", "-B", "-c", cleanup_code, path],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return False
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        _kill_watchdog_pid(cleanup_process.pid)
        _wait_watchdog_pid(cleanup_process.pid, deadline)
        return False
    try:
        status = cleanup_process.wait(timeout=remaining)
    except subprocess.TimeoutExpired:
        _kill_watchdog_pid(cleanup_process.pid)
        _wait_watchdog_pid(cleanup_process.pid, deadline)
        return False
    return status == 0


def _run_cli_payload(argv_values: list[str]) -> dict[str, Any]:
    args = _parser().parse_args(argv_values)
    if args.build_authorization:
        if (
            not args.source_db
            or not args.output
            or args.authorization is not None
            or args.accepted_authorization_sha256 is not None
        ):
            _block()
        manifest = build_authorization_manifest(args.source_db)
        path, digest = _write_authorization_manifest(args.output, manifest)
        return {
            "status": "authorization_written",
            "path": path,
            "authorizationSha256": digest,
            "providerCount": len(_PROVIDERS),
        }
    if (
        not args.source_db
        or not args.authorization
        or not args.accepted_authorization_sha256
        or args.output is not None
    ):
        _block()
    return asyncio.run(
        run_model_discovery(
            args.source_db,
            args.authorization,
            accepted_authorization_sha256=args.accepted_authorization_sha256,
        )
    )


def _run_watchdog_child(
    argv: list[str],
    expected_token: bytes,
    capability_fd: int,
    stdout_fd: int,
    stderr_fd: int,
    deadline: float,
) -> NoReturn:
    try:
        os.setsid()
        os.dup2(stdout_fd, 1)
        os.dup2(stderr_fd, 2)
        with contextlib.suppress(OSError):
            os.close(stdout_fd)
        with contextlib.suppress(OSError):
            os.close(stderr_fd)
        _read_watchdog_capability(capability_fd, expected_token, deadline)
        _harden_process()
        result = _run_cli_payload(argv)
    except BaseException:
        result = _source_blocked_output()
    _emit(result)
    os._exit(0)

def _run_watchdog(argv: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    outer_deadline = started + _TOTAL_TIMEOUT_SECONDS
    if _TOTAL_TIMEOUT_SECONDS <= _WATCHDOG_CLEANUP_RESERVE_SECONDS:
        _block()
    child_deadline = outer_deadline - _WATCHDOG_CLEANUP_RESERVE_SECONDS
    control_root = ""
    capability_read = -1
    capability_write = -1
    stdout_read = -1
    stdout_write = -1
    stderr_read = -1
    stderr_write = -1
    child_pid = -1
    old_tempdir = tempfile.tempdir
    try:
        control_root = tempfile.mkdtemp(prefix=".hermes-cc-watchdog-")
        os.chmod(control_root, 0o700)
        root_details = os.lstat(control_root)
        if (
            not stat.S_ISDIR(root_details.st_mode)
            or root_details.st_uid != os.getuid()
            or stat.S_IMODE(root_details.st_mode) & 0o077
        ):
            _block()
        capability_read, capability_write = os.pipe()
        stdout_read, stdout_write = os.pipe()
        stderr_read, stderr_write = os.pipe()
        expected_token = secrets.token_bytes(_WATCHDOG_TOKEN_BYTES)
        remaining_token = memoryview(expected_token)
        while remaining_token:
            written = os.write(capability_write, remaining_token)
            if written <= 0:
                _block()
            remaining_token = remaining_token[written:]
        os.close(capability_write)
        capability_write = -1
        tempfile.tempdir = control_root
        child_pid = os.fork()
        if child_pid == 0:
            for descriptor in (capability_write, stdout_read, stderr_read):
                if descriptor != -1:
                    with contextlib.suppress(OSError):
                        os.close(descriptor)
            _run_watchdog_child(
                argv,
                expected_token,
                capability_read,
                stdout_write,
                stderr_write,
                child_deadline,
            )
        os.close(capability_read)
        capability_read = -1
        os.close(stdout_write)
        stdout_write = -1
        os.close(stderr_write)
        stderr_write = -1
        stdout, stderr, status = _read_watchdog_output(
            child_pid, stdout_read, stderr_read, child_deadline
        )
        stdout_read = -1
        stderr_read = -1
        if status != 0 or stderr:
            _block()
        parsed = _parse_json(bytearray(stdout))
        return _validate_watchdog_result(
            parsed, builder="--build-authorization" in argv
        )
    except OSError:
        if child_pid != -1:
            _kill_watchdog_pid(child_pid)
            _wait_watchdog_pid(child_pid, child_deadline)
        _block()
    finally:
        tempfile.tempdir = old_tempdir
        for descriptor in (
            capability_read,
            capability_write,
            stdout_read,
            stdout_write,
            stderr_read,
            stderr_write,
        ):
            if descriptor != -1:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        if control_root and not _cleanup_control_root_bounded(
            control_root, outer_deadline
        ):
            _block()


def main(argv: list[str] | None = None) -> int:
    """Run the sealed manifest builder or discovery runner behind a watchdog."""
    result = _source_blocked_output()
    try:
        _harden_process()
        if _WATCHDOG_ENV in os.environ:
            _block()
        argv_values = list(sys.argv[1:] if argv is None else argv)
        result = _run_watchdog(argv_values)
    except BaseException:
        result = _source_blocked_output()
    _emit(result)
    return 0


if __name__ == "__main__":  # pragma: no cover - covered in an isolated process.
    raise SystemExit(main())
