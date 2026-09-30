"""Hardened, secret-free CC Switch Claude source seam.

``inventory_source`` is the only public, serializable inventory surface.  The
materializer uses ``materialization_source`` for a normalized in-process view;
it carries an endpoint and a credential *environment-variable name*, but never
a credential value or raw source configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
import re
import sqlite3
import stat
import tempfile
from typing import Any, Final
from urllib.parse import quote, urlsplit, urlunsplit

__all__ = [
    "SourceInventoryError",
    "inventory_source",
    "materialization_source",
    "migration_planning_source",
]

_SOURCE_APP_TYPE: Final = "claude"
_SUPPORTED_API_FAMILIES: Final = frozenset({
    "anthropic",
    "openai_chat",
    "openai_responses",
})
_LEGACY_API_FORMAT_KEYS: Final = ("apiFormat", "api_format")
_REQUIRED_COLUMNS: Final = frozenset({
    "id",
    "app_type",
    "name",
    "settings_config",
    "meta",
    "is_current",
    "sort_index",
})
_PROVIDER_COLUMNS: Final = frozenset({
    "id",
    "app_type",
    "name",
    "settings_config",
    "meta",
    "is_current",
    "sort_index",
})
_SCHEMA_COLUMNS: Final = frozenset({
    "type",
    "name",
    "tbl_name",
    "rootpage",
    "sql",
    "ROWID",
})
_SIDECAR_SUFFIXES: Final = ("-journal", "-wal", "-shm")
_SCHEMA_QUERY: Final = (
    "SELECT type, sql FROM main.sqlite_schema WHERE name = 'providers'"
)
_PROVIDER_QUERY: Final = (
    "SELECT id, app_type, name, settings_config, meta, is_current, sort_index "
    "FROM main.providers WHERE app_type = ? LIMIT ?"
)

# Environment evidence is public inventory only: it never augments the declared
# model set consumed by materialization.  The default key patterns are anchored
# so unrelated environment fields, including credentials and endpoints, are
# never inspected as model data.
_ENV_CONFIGURED_MODEL_KEYS: Final = (
    "ANTHROPIC_MODEL",
    "ANTHROPIC_REASONING_MODEL",
)
_ENV_DEFAULT_MODEL_KEY_RE: Final = re.compile(
    r"^ANTHROPIC_DEFAULT_[A-Z0-9]+(?:_[A-Z0-9]+)*_MODEL$"
)
_ENV_DEFAULT_MODEL_NAME_KEY_RE: Final = re.compile(
    r"^ANTHROPIC_DEFAULT_[A-Z0-9]+(?:_[A-Z0-9]+)*_MODEL_NAME$"
)
_CREDENTIAL_ENV_KEYS: Final = (
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)
_BASE_URL_ENV_KEYS: Final = ("ANTHROPIC_BASE_URL", "OPENAI_BASE_URL")
_MODEL_DEFINITION_FIELDS: Final = (
    "api",
    "baseUrl",
    "compactionModel",
    "compat",
    "contextPromotionTarget",
    "contextWindow",
    "cost",
    "headers",
    "id",
    "imageInputDecoder",
    "input",
    "maxTokens",
    "name",
    "omitMaxOutputTokens",
    "preferWebsockets",
    "premiumMultiplier",
    "reasoning",
    "supportsTools",
    "thinking",
    "tokenizer",
)
_MODEL_DEFINITION_FIELD_SET: Final = frozenset(_MODEL_DEFINITION_FIELDS)
_OMP_APIS: Final = frozenset({
    "anthropic-messages",
    "azure-openai-responses",
    "bedrock-converse-stream",
    "google-gemini-cli",
    "google-generative-ai",
    "google-vertex",
    "openai-codex-responses",
    "openai-completions",
    "openai-responses",
})
_COMPAT_BOOLEAN_FIELDS: Final = frozenset({
    "allowAnthropicHeaderOverrides",
    "allowsSyntheticReasoningContentForToolCalls",
    "alwaysSendMaxTokens",
    "disableReasoningOnForcedToolChoice",
    "disableReasoningOnToolChoice",
    "qwenTemplateReasoningEffort",
    "replayUnsignedThinking",
    "requiresAssistantAfterToolResult",
    "requiresAssistantContentForToolCalls",
    "requiresMistralToolIds",
    "requiresReasoningContentForToolCalls",
    "requiresThinkingAsText",
    "requiresToolResultId",
    "requiresToolResultName",
    "strictResponsesPairing",
    "stripImageInput",
    "supportsContextManagement",
    "supportsDeveloperRole",
    "supportsEagerToolInputStreaming",
    "supportsForcedToolChoice",
    "supportsImageDetailOriginal",
    "supportsLongPromptCacheRetention",
    "supportsMultipleSystemMessages",
    "supportsReasoningEffort",
    "supportsReasoningParams",
    "supportsReasoningSummary",
    "supportsStore",
    "supportsStrictMode",
    "supportsToolChoice",
    "supportsUsageInStreaming",
})
_COMPAT_FIELDS: Final = _COMPAT_BOOLEAN_FIELDS | frozenset({
    "cacheControlFormat",
    "extraBody",
    "maxTokensField",
    "openRouterRouting",
    "promptCacheMaximumCheckpoints",
    "promptCacheMinimumTokens",
    "promptCacheMode",
    "reasoningContentField",
    "reasoningEffortMap",
    "streamIdleTimeoutMs",
    "streamMarkupHealingPattern",
    "thinkingFormat",
    "toolStrictMode",
    "vercelGatewayRouting",
    "whenThinking",
})
_COMPAT_WHEN_THINKING_FIELDS: Final = _COMPAT_FIELDS - frozenset({
    "promptCacheMaximumCheckpoints",
    "promptCacheMinimumTokens",
    "promptCacheMode",
    "whenThinking",
})
_TOKENIZERS: Final = frozenset({
    "claude-v3",
    "claude-v47",
    "claude-v5",
    "claude-v5-sonnet",
    "deepseek-v3",
    "glm5",
    "kimi-k2",
    "qwen3",
})
_THINKING_FIELDS: Final = frozenset({
    "mode",
    "efforts",
    "levels",
    "minLevel",
    "maxLevel",
    "defaultLevel",
    "effortMap",
    "supportsDisplay",
    "requiresEffort",
})
_THINKING_MODES: Final = frozenset({
    "effort",
    "budget",
    "google-level",
    "anthropic-adaptive",
    "anthropic-budget-effort",
})
_THINKING_LEVELS: Final = frozenset({
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
})
_EFFORT_MAP_FIELDS: Final = _THINKING_LEVELS
_ROUTING_FIELDS: Final = frozenset({"only", "order"})
_SOURCE_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODEL_ID_RE: Final = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MODEL_CONTEXT_SUFFIX_RE: Final = re.compile(r"(?i:\[1m\])$")
_CONFIGURED_MODEL_VALUE_RE: Final = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}(?i:\[1m\])?$"
)
_MAX_SOURCE_TEXT_LENGTH: Final = 512
_MAX_SOURCE_DB_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_WAL_BYTES: Final = 256 * 1024 * 1024
_MAX_SOURCE_SHM_BYTES: Final = 8 * 1024 * 1024
_MAX_SOURCE_CAPTURE_BYTES: Final = 384 * 1024 * 1024
_MAX_PROVIDER_ROWS: Final = 256
_MAX_PROVIDER_FIELD_BYTES: Final = 256 * 1024
_MAX_PROVIDER_ROW_BYTES: Final = _MAX_PROVIDER_FIELD_BYTES * 3
_MAX_JSON_BYTES: Final = 128 * 1024
_MAX_JSON_NODES: Final = 4096
_MAX_JSON_DEPTH: Final = 32
_MAX_SQL_LENGTH: Final = 64 * 1024
_MAX_SQLITE_COLUMNS: Final = 64
_HASH_CHUNK_SIZE: Final = 128 * 1024

_ERR_PATH: Final = "blocked-db-path"
_ERR_SYMLINK: Final = "blocked-db-symlink"
_ERR_TYPE: Final = "blocked-db-type"
_ERR_OWNER: Final = "blocked-db-owner"
_ERR_SIDECAR: Final = "blocked-db-sidecar"
_ERR_SECURITY: Final = "blocked-db-security"
_ERR_SCHEMA: Final = "blocked-db-schema"
_ERR_READ: Final = "blocked-db-read"
_ERR_DATA: Final = "blocked-provider-data"
_ERR_JSON: Final = "blocked-provider-json"
_ERR_FILTER: Final = "blocked-provider-filter"
_ERR_SELECTION: Final = "blocked-source-selection"
_ERR_ORIGIN: Final = "blocked-origin-approval"


class SourceInventoryError(RuntimeError):
    """A source read was refused; ``code`` is intentionally the only detail."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class _FileIdentity:
    device: int
    file_id: int
    owner: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int


@dataclass
class _CapturedSource:
    descriptor: int
    identity: _FileIdentity
    wal_descriptor: int
    wal_identity: _FileIdentity | None
    shm_descriptor: int
    shm_identity: _FileIdentity | None
    source_sha256: str
    wal_sha256: str | None
    shm_sha256: str | None
    temporary: tempfile.TemporaryDirectory[str]
    snapshot_path: str


def _same_file(left: _FileIdentity, right: _FileIdentity) -> bool:
    return left.device == right.device and left.file_id == right.file_id


def _same_snapshot(left: _FileIdentity, right: _FileIdentity) -> bool:
    return (
        _same_file(left, right)
        and left.owner == right.owner
        and left.mode == right.mode
        and left.size == right.size
        and left.mtime_ns == right.mtime_ns
        and left.ctime_ns == right.ctime_ns
    )


def _source_path(db_path: str | os.PathLike[str]) -> str:
    try:
        raw_path = os.fspath(db_path)
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        if not isinstance(raw_path, str) or not raw_path:
            raise ValueError
        return os.path.abspath(raw_path)
    except (OSError, TypeError, ValueError):
        raise SourceInventoryError(_ERR_PATH) from None


def _allowed_owner_ids() -> set[int]:
    owner_ids = {os.getuid()}
    get_effective_uid = getattr(os, "geteuid", None)
    if get_effective_uid is not None:
        owner_ids.add(get_effective_uid())
    return owner_ids


def _checked_file_identity(
    details: os.stat_result, *, path_entry: bool, maximum_bytes: int
) -> _FileIdentity:
    if path_entry and stat.S_ISLNK(details.st_mode):
        raise SourceInventoryError(_ERR_SYMLINK)
    if not stat.S_ISREG(details.st_mode):
        raise SourceInventoryError(_ERR_TYPE)
    if details.st_uid not in _allowed_owner_ids():
        raise SourceInventoryError(_ERR_OWNER)
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise SourceInventoryError(_ERR_SECURITY)
    if details.st_size < 0 or details.st_size > maximum_bytes:
        raise SourceInventoryError(_ERR_READ)
    return _FileIdentity(
        device=details.st_dev,
        file_id=details.st_ino,
        owner=details.st_uid,
        mode=details.st_mode,
        size=details.st_size,
        mtime_ns=details.st_mtime_ns,
        ctime_ns=details.st_ctime_ns,
    )


def _database_identity(details: os.stat_result, *, path_entry: bool) -> _FileIdentity:
    return _checked_file_identity(
        details, path_entry=path_entry, maximum_bytes=_MAX_SOURCE_DB_BYTES
    )


def _sidecar_identity(
    details: os.stat_result, *, path_entry: bool, suffix: str
) -> _FileIdentity:
    maximum = _MAX_SOURCE_WAL_BYTES if suffix == "-wal" else _MAX_SOURCE_SHM_BYTES
    return _checked_file_identity(details, path_entry=path_entry, maximum_bytes=maximum)


def _lstat_database(path: str) -> _FileIdentity:
    try:
        details = os.lstat(path)
    except (OSError, ValueError):
        raise SourceInventoryError(_ERR_PATH) from None
    return _database_identity(details, path_entry=True)


def _fstat_database(descriptor: int) -> _FileIdentity:
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise SourceInventoryError(_ERR_SECURITY) from None
    return _database_identity(details, path_entry=False)


def _lstat_sidecar(path: str, suffix: str) -> _FileIdentity | None:
    try:
        details = os.lstat(path + suffix)
    except FileNotFoundError:
        return None
    except (OSError, ValueError):
        raise SourceInventoryError(_ERR_SIDECAR) from None
    return _sidecar_identity(details, path_entry=True, suffix=suffix)


def _fstat_sidecar(descriptor: int, suffix: str) -> _FileIdentity:
    try:
        details = os.fstat(descriptor)
    except OSError:
        raise SourceInventoryError(_ERR_SECURITY) from None
    return _sidecar_identity(details, path_entry=False, suffix=suffix)


def _assert_no_rollback_journal(path: str) -> None:
    try:
        os.lstat(path + "-journal")
    except FileNotFoundError:
        return
    except (OSError, ValueError):
        raise SourceInventoryError(_ERR_SIDECAR) from None
    raise SourceInventoryError(_ERR_SIDECAR)


def _read_flags() -> int:
    nofollow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(nofollow, int):
        raise SourceInventoryError(_ERR_SECURITY)
    nonblock = getattr(os, "O_NONBLOCK", 0)
    if not isinstance(nonblock, int):
        nonblock = 0
    return os.O_RDONLY | nofollow | nonblock | getattr(os, "O_CLOEXEC", 0)


def _open_source_descriptor(path: str) -> tuple[int, _FileIdentity]:
    """Open one immutable source identity without following its final path entry."""
    before = _lstat_database(path)
    try:
        descriptor = os.open(path, _read_flags())
    except OSError:
        raise SourceInventoryError(_ERR_PATH) from None
    try:
        opened = _fstat_database(descriptor)
        after = _lstat_database(path)
        if not _same_snapshot(before, opened) or not _same_snapshot(after, opened):
            raise SourceInventoryError(_ERR_SECURITY)
        return descriptor, opened
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_wal_descriptor(path: str) -> tuple[int, _FileIdentity | None]:
    before = _lstat_sidecar(path, "-wal")
    if before is None:
        return -1, None
    try:
        descriptor = os.open(path + "-wal", _read_flags())
    except OSError:
        raise SourceInventoryError(_ERR_SIDECAR) from None
    try:
        opened = _fstat_sidecar(descriptor, "-wal")
        after = _lstat_sidecar(path, "-wal")
        if (
            after is None
            or not _same_snapshot(before, opened)
            or not _same_snapshot(after, opened)
        ):
            raise SourceInventoryError(_ERR_SECURITY)
        return descriptor, opened
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_shm_descriptor(path: str) -> tuple[int, _FileIdentity | None]:
    before = _lstat_sidecar(path, "-shm")
    if before is None:
        return -1, None
    try:
        descriptor = os.open(path + "-shm", _read_flags())
    except OSError:
        raise SourceInventoryError(_ERR_SIDECAR) from None
    try:
        opened = _fstat_sidecar(descriptor, "-shm")
        after = _lstat_sidecar(path, "-shm")
        if (
            after is None
            or not _same_snapshot(before, opened)
            or not _same_snapshot(after, opened)
        ):
            raise SourceInventoryError(_ERR_SECURITY)
        return descriptor, opened
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _assert_source_stable(path: str, descriptor: int, identity: _FileIdentity) -> None:
    opened = _fstat_database(descriptor)
    current = _lstat_database(path)
    if not _same_snapshot(opened, identity) or not _same_snapshot(current, identity):
        raise SourceInventoryError(_ERR_SECURITY)


def _same_sidecar_binding(
    left: _FileIdentity | None, right: _FileIdentity | None
) -> bool:
    if left is None or right is None:
        return left is right
    return _same_snapshot(left, right)


def _assert_wal_stable(
    path: str, descriptor: int, identity: _FileIdentity | None
) -> None:
    current = _lstat_sidecar(path, "-wal")
    if identity is None:
        if descriptor != -1 or current is not None:
            raise SourceInventoryError(_ERR_SECURITY)
        return
    if descriptor == -1 or current is None:
        raise SourceInventoryError(_ERR_SECURITY)
    opened = _fstat_sidecar(descriptor, "-wal")
    if not _same_snapshot(opened, identity) or not _same_snapshot(current, identity):
        raise SourceInventoryError(_ERR_SECURITY)

def _assert_shm_stable(
    path: str, descriptor: int, identity: _FileIdentity | None
) -> None:
    current = _validate_shm_sidecar(path)
    if identity is None:
        if descriptor != -1 or current is not None:
            raise SourceInventoryError(_ERR_SECURITY)
        return
    if descriptor == -1 or current is None:
        raise SourceInventoryError(_ERR_SECURITY)
    opened = _fstat_sidecar(descriptor, "-shm")
    if (
        not _same_sidecar_binding(opened, identity)
        or not _same_sidecar_binding(current, identity)
    ):
        raise SourceInventoryError(_ERR_SECURITY)


def _validate_shm_sidecar(path: str) -> _FileIdentity | None:
    return _lstat_sidecar(path, "-shm")


def _snapshot_uri(path: str) -> str:
    return f"file:{quote(path, safe='/')}?mode=rw&cache=private"


def _stable_realpath(path: str, identity: _FileIdentity) -> str:
    try:
        resolved_path = os.path.realpath(path)
        resolved = _database_identity(os.stat(resolved_path), path_entry=False)
    except (OSError, ValueError):
        raise SourceInventoryError(_ERR_SECURITY) from None
    if not _same_file(resolved, identity):
        raise SourceInventoryError(_ERR_SECURITY)
    return resolved_path


def _file_sha256(descriptor: int, identity: _FileIdentity) -> str:
    try:
        digest = sha256()
        offset = 0
        while offset < identity.size:
            block = os.pread(
                descriptor, min(_HASH_CHUNK_SIZE, identity.size - offset), offset
            )
            if not block:
                raise OSError
            digest.update(block)
            offset += len(block)
        return digest.hexdigest()
    except (AttributeError, OSError, ValueError):
        raise SourceInventoryError(_ERR_SECURITY) from None


def _source_journal_mode(descriptor: int, identity: _FileIdentity) -> str:
    try:
        header = os.pread(descriptor, 20, 0)
    except (AttributeError, OSError, ValueError):
        raise SourceInventoryError(_ERR_SECURITY) from None
    if len(header) != 20 or header[:16] != b"SQLite format 3\x00":
        raise SourceInventoryError(_ERR_SCHEMA)
    versions = (header[18], header[19])
    if versions == (2, 2):
        return "wal"
    if versions == (1, 1):
        return "rollback"
    raise SourceInventoryError(_ERR_SCHEMA)


def _file_manifest(identity: _FileIdentity, digest: str) -> dict[str, Any]:
    return {
        "device": identity.device,
        "fileId": identity.file_id,
        "owner": identity.owner,
        "mode": stat.S_IMODE(identity.mode),
        "size": identity.size,
        "mtimeNs": identity.mtime_ns,
        "ctimeNs": identity.ctime_ns,
        "sha256": digest,
    }


def _shm_manifest(
    identity: _FileIdentity | None, digest: str | None
) -> dict[str, Any] | None:
    if identity is None:
        if digest is not None:
            raise SourceInventoryError(_ERR_SECURITY)
        return None
    if digest is None:
        raise SourceInventoryError(_ERR_SECURITY)
    return _file_manifest(identity, digest)


def _assert_capture_stable(path: str, capture: _CapturedSource) -> None:
    _assert_source_stable(path, capture.descriptor, capture.identity)
    _assert_wal_stable(path, capture.wal_descriptor, capture.wal_identity)
    _assert_shm_stable(path, capture.shm_descriptor, capture.shm_identity)
    if _file_sha256(capture.descriptor, capture.identity) != capture.source_sha256:
        raise SourceInventoryError(_ERR_SECURITY)
    if capture.wal_identity is not None and (
        capture.wal_sha256 is None
        or _file_sha256(capture.wal_descriptor, capture.wal_identity)
        != capture.wal_sha256
    ):
        raise SourceInventoryError(_ERR_SECURITY)
    if capture.shm_identity is not None and (
        capture.shm_sha256 is None
        or _file_sha256(capture.shm_descriptor, capture.shm_identity)
        != capture.shm_sha256
    ):
        raise SourceInventoryError(_ERR_SECURITY)
    _assert_source_stable(path, capture.descriptor, capture.identity)
    _assert_wal_stable(path, capture.wal_descriptor, capture.wal_identity)
    _assert_shm_stable(path, capture.shm_descriptor, capture.shm_identity)
    _assert_no_rollback_journal(path)


def _write_all(descriptor: int, payload: bytes) -> None:
    remaining = memoryview(payload)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise OSError
        remaining = remaining[written:]


def _copy_descriptor(
    source_descriptor: int,
    identity: _FileIdentity,
    destination_path: str,
) -> str:
    destination = -1
    try:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        destination = os.open(destination_path, flags, 0o600)
        digest = sha256()
        offset = 0
        while offset < identity.size:
            block = os.pread(
                source_descriptor,
                min(_HASH_CHUNK_SIZE, identity.size - offset),
                offset,
            )
            if not block:
                raise OSError
            _write_all(destination, block)
            digest.update(block)
            offset += len(block)
        os.fchmod(destination, 0o600)
        return digest.hexdigest()
    except (AttributeError, OSError, ValueError):
        raise SourceInventoryError(_ERR_SECURITY) from None
    finally:
        if destination != -1:
            try:
                os.close(destination)
            except OSError:
                pass


def _secure_snapshot_directory() -> tempfile.TemporaryDirectory[str]:
    try:
        temporary = tempfile.TemporaryDirectory(prefix="hermes-cc-switch-")
        details = os.lstat(temporary.name)
        if (
            not stat.S_ISDIR(details.st_mode)
            or details.st_uid not in _allowed_owner_ids()
            or stat.S_IMODE(details.st_mode) & 0o077
        ):
            raise OSError
        return temporary
    except OSError:
        raise SourceInventoryError(_ERR_SECURITY) from None


def _capture_source_snapshot(path: str) -> _CapturedSource:
    descriptor = -1
    wal_descriptor = -1
    shm_descriptor = -1
    temporary: tempfile.TemporaryDirectory[str] | None = None
    try:
        _assert_no_rollback_journal(path)
        descriptor, identity = _open_source_descriptor(path)
        wal_descriptor, wal_identity = _open_wal_descriptor(path)
        shm_descriptor, shm_identity = _open_shm_descriptor(path)
        journal_mode = _source_journal_mode(descriptor, identity)
        if (journal_mode == "wal") != (wal_identity is not None):
            raise SourceInventoryError(_ERR_SIDECAR)
        captured_bytes = identity.size + (
            0 if wal_identity is None else wal_identity.size
        ) + (0 if shm_identity is None else shm_identity.size)
        if captured_bytes > _MAX_SOURCE_CAPTURE_BYTES:
            raise SourceInventoryError(_ERR_READ)
        _assert_no_rollback_journal(path)
        _assert_shm_stable(path, shm_descriptor, shm_identity)

        temporary = _secure_snapshot_directory()
        snapshot_path = os.path.join(temporary.name, "source.db")
        source_digest = _copy_descriptor(descriptor, identity, snapshot_path)
        wal_digest = None
        if wal_identity is not None:
            wal_digest = _copy_descriptor(
                wal_descriptor, wal_identity, snapshot_path + "-wal"
            )

        if _file_sha256(descriptor, identity) != source_digest:
            raise SourceInventoryError(_ERR_SECURITY)
        if wal_identity is not None and (
            wal_digest is None
            or _file_sha256(wal_descriptor, wal_identity) != wal_digest
        ):
            raise SourceInventoryError(_ERR_SECURITY)
        shm_digest = (
            None
            if shm_identity is None
            else _file_sha256(shm_descriptor, shm_identity)
        )
        capture = _CapturedSource(
            descriptor=descriptor,
            identity=identity,
            wal_descriptor=wal_descriptor,
            wal_identity=wal_identity,
            shm_descriptor=shm_descriptor,
            shm_identity=shm_identity,
            source_sha256=source_digest,
            wal_sha256=wal_digest,
            shm_sha256=shm_digest,
            temporary=temporary,
            snapshot_path=snapshot_path,
        )
        _assert_capture_stable(path, capture)
        return capture
    except BaseException:
        for open_descriptor in (shm_descriptor, wal_descriptor, descriptor):
            if open_descriptor != -1:
                try:
                    os.close(open_descriptor)
                except OSError:
                    pass
        if temporary is not None:
            temporary.cleanup()
        raise


def _readonly_authorizer(
    action: int,
    parameter_1: str | None,
    parameter_2: str | None,
    _database_name: str | None,
    _trigger_or_view: str | None,
) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        if parameter_1 == "providers" and parameter_2 in _PROVIDER_COLUMNS:
            return sqlite3.SQLITE_OK
        if (
            parameter_1 in {"sqlite_master", "sqlite_schema"}
            and parameter_2 in _SCHEMA_COLUMNS
        ):
            return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_PRAGMA:
        if parameter_1 == "table_info" and parameter_2 == "providers":
            return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_TRANSACTION and parameter_1 in {"BEGIN", "ROLLBACK"}:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def _set_sqlite_read_limits(connection: sqlite3.Connection) -> None:
    set_limit = getattr(connection, "setlimit", None)
    if not callable(set_limit):
        return
    try:
        for name, value in (
            ("SQLITE_LIMIT_LENGTH", _MAX_PROVIDER_ROW_BYTES),
            ("SQLITE_LIMIT_SQL_LENGTH", _MAX_SQL_LENGTH),
            ("SQLITE_LIMIT_COLUMN", _MAX_SQLITE_COLUMNS),
        ):
            category = getattr(sqlite3, name, None)
            if isinstance(category, int):
                set_limit(category, value)
    except (sqlite3.Error, TypeError, ValueError):
        raise SourceInventoryError(_ERR_SECURITY) from None


def _configure_readonly_connection(connection: sqlite3.Connection) -> None:
    try:
        connection.enable_load_extension(False)
        _set_sqlite_read_limits(connection)
        connection.execute("PRAGMA query_only = ON")
        connection.execute("PRAGMA trusted_schema = OFF")
        connection.execute("PRAGMA temp_store = MEMORY")
        connection.set_authorizer(_readonly_authorizer)
    except (AttributeError, sqlite3.Error, TypeError):
        raise SourceInventoryError(_ERR_SECURITY) from None


def _validate_schema(connection: sqlite3.Connection) -> None:
    try:
        schema_row = connection.execute(_SCHEMA_QUERY).fetchone()
        if schema_row is None or schema_row[0] != "table":
            raise SourceInventoryError(_ERR_SCHEMA)
        table_sql = schema_row[1]
        if not isinstance(
            table_sql, str
        ) or not table_sql.lstrip().casefold().startswith("create table"):
            raise SourceInventoryError(_ERR_SCHEMA)
        columns = connection.execute("PRAGMA table_info(providers)").fetchall()
    except SourceInventoryError:
        raise
    except sqlite3.Error:
        raise SourceInventoryError(_ERR_SCHEMA) from None
    try:
        by_name = {
            column[1]: column
            for column in columns
            if isinstance(column, tuple)
            and len(column) > 5
            and isinstance(column[1], str)
            and isinstance(column[5], int)
        }
        primary_key = tuple(
            column[1]
            for column in sorted(by_name.values(), key=lambda column: column[5])
            if column[5]
        )
    except (IndexError, TypeError):
        raise SourceInventoryError(_ERR_SCHEMA) from None
    if not _REQUIRED_COLUMNS.issubset(by_name) or primary_key != (
        "id",
        "app_type",
    ):
        raise SourceInventoryError(_ERR_SCHEMA)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def _reject_json_constant(_constant: str) -> None:
    raise ValueError


def _utf8_length(value: str, code: str) -> int:
    try:
        return len(value.encode("utf-8"))
    except UnicodeError:
        raise SourceInventoryError(code) from None


def _bounded_provider_text(value: Any) -> str:
    if (
        not isinstance(value, str)
        or _utf8_length(value, _ERR_DATA) > _MAX_PROVIDER_FIELD_BYTES
    ):
        raise SourceInventoryError(_ERR_DATA)
    return value


def _validate_json_limits(value: Any) -> None:
    nodes = 0
    pending: list[tuple[Any, int]] = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES or depth > _MAX_JSON_DEPTH:
            raise SourceInventoryError(_ERR_JSON)
        if isinstance(item, dict):
            for key, child in item.items():
                if (
                    not isinstance(key, str)
                    or _utf8_length(key, _ERR_JSON) > _MAX_PROVIDER_FIELD_BYTES
                ):
                    raise SourceInventoryError(_ERR_JSON)
                pending.append((child, depth + 1))
        elif isinstance(item, list):
            pending.extend((child, depth + 1) for child in item)
        elif isinstance(item, str):
            if _utf8_length(item, _ERR_JSON) > _MAX_PROVIDER_FIELD_BYTES:
                raise SourceInventoryError(_ERR_JSON)
        elif item is None or isinstance(item, (bool, int)):
            continue
        elif isinstance(item, float):
            if item != item or item in {float("inf"), float("-inf")}:
                raise SourceInventoryError(_ERR_JSON)
        else:
            raise SourceInventoryError(_ERR_JSON)


def _json_object(raw_json: Any) -> dict[str, Any]:
    if (
        not isinstance(raw_json, str)
        or _utf8_length(raw_json, _ERR_JSON) > _MAX_JSON_BYTES
    ):
        raise SourceInventoryError(_ERR_JSON)
    try:
        value = json.loads(
            raw_json,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, UnicodeError, ValueError):
        raise SourceInventoryError(_ERR_JSON) from None
    _validate_json_limits(value)
    if not isinstance(value, dict):
        raise SourceInventoryError(_ERR_JSON)
    return value


def _source_text(value: Any) -> str:
    value = _bounded_provider_text(value)
    if (
        not value
        or len(value) > _MAX_SOURCE_TEXT_LENGTH
        or any(ord(character) < 32 for character in value)
    ):
        raise SourceInventoryError(_ERR_DATA)
    return value


def _optional_string(mapping: dict[str, Any], key: str) -> str | None:
    if key not in mapping:
        return None
    value = mapping[key]
    if not isinstance(value, str):
        raise SourceInventoryError(_ERR_DATA)
    return value or None


def _optional_object(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in mapping:
        return {}
    value = mapping[key]
    if not isinstance(value, dict):
        raise SourceInventoryError(_ERR_DATA)
    return value


def _supported_api_family(value: Any) -> str | None:
    if isinstance(value, str) and value in _SUPPORTED_API_FAMILIES:
        return value
    return None


def _api_family(settings_config: dict[str, Any], meta: dict[str, Any]) -> str | None:
    if "apiFormat" in meta:
        return _supported_api_family(meta["apiFormat"])
    for key in _LEGACY_API_FORMAT_KEYS:
        if key in settings_config:
            return _supported_api_family(settings_config[key])
    return "anthropic"


def _base_url(
    settings_config: dict[str, Any], meta: dict[str, Any], environment: dict[str, Any]
) -> str | None:
    # Claude rows can use an Anthropic-named transport variable even when their
    # OMP API family is selected by ``meta.apiFormat == 'openai_responses'``.
    environment_candidates = [
        _optional_string(environment, key) for key in _BASE_URL_ENV_KEYS
    ]
    for candidate in environment_candidates:
        if candidate is not None:
            return candidate
    configured_candidates = [
        _optional_string(settings_config, key) for key in ("base_url", "baseUrl")
    ]
    for candidate in configured_candidates:
        if candidate is not None:
            return candidate
    usage_script = _optional_object(meta, "usage_script")
    return _optional_string(usage_script, "baseUrl")


def _endpoint_facts(endpoint: str | None) -> tuple[bool, bool, bool, bool, str | None]:
    if endpoint is None:
        return False, False, False, False, None
    if endpoint != endpoint.strip() or any(
        ord(character) < 32 for character in endpoint
    ):
        return False, False, False, False, None
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        return False, False, False, False, None
    before_fragment = endpoint.split("#", 1)[0]
    endpoint_https = parsed.scheme.casefold() == "https" and bool(parsed.netloc)
    endpoint_userinfo = "@" in parsed.netloc
    endpoint_query = "?" in before_fragment
    endpoint_fragment = "#" in endpoint
    if not (
        endpoint_https
        and not endpoint_userinfo
        and not endpoint_query
        and not endpoint_fragment
        and parsed.hostname is not None
        and (port is None or port > 0)
    ):
        return (
            endpoint_https,
            endpoint_userinfo,
            endpoint_query,
            endpoint_fragment,
            None,
        )
    hostname = parsed.hostname.casefold()
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and port != 443:
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") if parsed.path != "/" else ""
    return (
        endpoint_https,
        endpoint_userinfo,
        endpoint_query,
        endpoint_fragment,
        urlunsplit(("https", host, path, "", "")),
    )


def _model_id(value: Any) -> str:
    if not isinstance(value, str):
        raise SourceInventoryError(_ERR_DATA)
    normalized = _MODEL_CONTEXT_SUFFIX_RE.sub("", value)
    if (
        not _MODEL_ID_RE.fullmatch(normalized)
        or any(segment in {"", ".", ".."} for segment in normalized.split("/"))
        or normalized.endswith("/")
    ):
        raise SourceInventoryError(_ERR_DATA)
    return normalized


def _configured_model_value(value: Any) -> str:
    """Validate an environment-configured model value without normalizing it."""
    if (
        not isinstance(value, str)
        or not _CONFIGURED_MODEL_VALUE_RE.fullmatch(value)
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
        or value.endswith("/")
    ):
        raise SourceInventoryError(_ERR_DATA)
    return value


def _source_identifier(value: Any) -> str:
    value = _source_text(value)
    if not _SOURCE_ID_RE.fullmatch(value) or value in {".", ".."}:
        raise SourceInventoryError(_ERR_DATA)
    return value


def _clean_metadata_string(value: Any) -> str:
    if (
        not isinstance(value, str)
        or not value
        or _utf8_length(value, _ERR_DATA) > 8192
        or any(ord(character) < 32 for character in value)
    ):
        raise SourceInventoryError(_ERR_DATA)
    return value


def _positive_int(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise SourceInventoryError(_ERR_DATA)
    return value


def _nonnegative_int(value: Any) -> int:
    if type(value) is not int or value < 0:
        raise SourceInventoryError(_ERR_DATA)
    return value


def _nonnegative_number(value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SourceInventoryError(_ERR_DATA)
    if isinstance(value, float) and (
        value != value or value in {float("inf"), float("-inf")}
    ):
        raise SourceInventoryError(_ERR_DATA)
    if value < 0:
        raise SourceInventoryError(_ERR_DATA)
    return value


def _safe_model_endpoint(value: Any) -> str:
    endpoint = _clean_metadata_string(value)
    if _utf8_length(endpoint, _ERR_DATA) > 2048:
        raise SourceInventoryError(_ERR_DATA)
    https, userinfo, query, fragment, normalized = _endpoint_facts(endpoint)
    if not https or userinfo or query or fragment or normalized is None:
        raise SourceInventoryError(_ERR_DATA)
    return normalized


def _thinking_levels(value: Any) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            not isinstance(item, str) or item not in _THINKING_LEVELS for item in value
        )
        or len(set(value)) != len(value)
    ):
        raise SourceInventoryError(_ERR_DATA)
    return list(value)


def _effort_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or not set(value).issubset(_EFFORT_MAP_FIELDS):
        raise SourceInventoryError(_ERR_DATA)
    return {key: _clean_metadata_string(item) for key, item in value.items()}


def _routing(value: Any) -> dict[str, list[str]]:
    if not isinstance(value, dict) or not set(value).issubset(_ROUTING_FIELDS):
        raise SourceInventoryError(_ERR_DATA)
    normalized: dict[str, list[str]] = {}
    for key, item in value.items():
        if not isinstance(item, list):
            raise SourceInventoryError(_ERR_DATA)
        normalized[key] = [_clean_metadata_string(entry) for entry in item]
    return normalized


def _compat(value: Any, *, when_thinking: bool = False) -> dict[str, Any]:
    allowed = _COMPAT_WHEN_THINKING_FIELDS if when_thinking else _COMPAT_FIELDS
    if not isinstance(value, dict) or not set(value).issubset(allowed):
        raise SourceInventoryError(_ERR_DATA)
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key in _COMPAT_BOOLEAN_FIELDS:
            if type(item) is not bool:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "cacheControlFormat":
            if item != "anthropic":
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "extraBody":
            if not isinstance(item, dict) or item:
                raise SourceInventoryError(_ERR_DATA)
        elif key == "maxTokensField":
            if not isinstance(item, str) or item not in {
                "max_completion_tokens",
                "max_tokens",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key in {"openRouterRouting", "vercelGatewayRouting"}:
            normalized[key] = _routing(item)
        elif key in {
            "promptCacheMaximumCheckpoints",
            "promptCacheMinimumTokens",
            "streamIdleTimeoutMs",
        }:
            normalized[key] = _nonnegative_int(item)
        elif key == "promptCacheMode":
            if not isinstance(item, str) or item not in {
                "automatic",
                "explicit",
                "none",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "reasoningContentField":
            if not isinstance(item, str) or item not in {
                "reasoning",
                "reasoning_content",
                "reasoning_text",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "reasoningEffortMap":
            normalized[key] = _effort_map(item)
        elif key == "streamMarkupHealingPattern":
            if not isinstance(item, str) or item not in {
                "dsml",
                "kimi",
                "qwen",
                "thinking",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "thinkingFormat":
            if not isinstance(item, str) or item not in {
                "openai",
                "openrouter",
                "qwen",
                "qwen-chat-template",
                "zai",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "toolStrictMode":
            if not isinstance(item, str) or item not in {"all_strict", "none"}:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "whenThinking":
            normalized[key] = _compat(item, when_thinking=True)
        else:
            raise SourceInventoryError(_ERR_DATA)
    return normalized


def _thinking(value: Any) -> dict[str, Any]:
    mode = value.get("mode") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or not set(value).issubset(_THINKING_FIELDS)
        or not isinstance(mode, str)
        or mode not in _THINKING_MODES
    ):
        raise SourceInventoryError(_ERR_DATA)
    normalized: dict[str, Any] = {"mode": mode}
    for key, item in value.items():
        if key == "mode":
            continue
        if key in {"efforts", "levels"}:
            normalized[key] = _thinking_levels(item)
        elif key in {"minLevel", "maxLevel", "defaultLevel"}:
            if not isinstance(item, str) or item not in _THINKING_LEVELS:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        elif key == "effortMap":
            normalized[key] = _effort_map(item)
        elif key in {"supportsDisplay", "requiresEffort"}:
            if type(item) is not bool:
                raise SourceInventoryError(_ERR_DATA)
            normalized[key] = item
        else:
            raise SourceInventoryError(_ERR_DATA)
    if ("minLevel" in normalized) != ("maxLevel" in normalized):
        raise SourceInventoryError(_ERR_DATA)
    return normalized


def _model_definition(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - _MODEL_DEFINITION_FIELD_SET:
        raise SourceInventoryError(_ERR_DATA)
    model = _model_id(value.get("id"))
    normalized: dict[str, Any] = {"id": model}
    for field in _MODEL_DEFINITION_FIELDS:
        if field == "id" or field not in value:
            continue
        item = value[field]
        if field == "api":
            if not isinstance(item, str) or item not in _OMP_APIS:
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = item
        elif field == "baseUrl":
            normalized[field] = _safe_model_endpoint(item)
        elif field in {"compactionModel", "contextPromotionTarget", "name"}:
            normalized[field] = _clean_metadata_string(item)
        elif field == "compat":
            normalized[field] = _compat(item)
        elif field in {"contextWindow", "maxTokens"}:
            normalized[field] = _positive_int(item)
        elif field == "cost":
            if not isinstance(item, dict) or set(item) != {
                "cacheRead",
                "cacheWrite",
                "input",
                "output",
            }:
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = {
                key: _nonnegative_number(amount) for key, amount in item.items()
            }
        elif field == "headers":
            if not isinstance(item, dict) or item:
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = {}
        elif field == "imageInputDecoder":
            if item != "stb":
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = item
        elif field == "input":
            if (
                not isinstance(item, list)
                or not item
                or any(
                    not isinstance(modality, str) or modality not in {"text", "image"}
                    for modality in item
                )
                or len(set(item)) != len(item)
            ):
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = list(item)
        elif field in {
            "omitMaxOutputTokens",
            "preferWebsockets",
            "reasoning",
            "supportsTools",
        }:
            if type(item) is not bool:
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = item
        elif field == "premiumMultiplier":
            normalized[field] = _nonnegative_number(item)
        elif field == "thinking":
            normalized[field] = _thinking(item)
        elif field == "tokenizer":
            if not isinstance(item, str) or item not in _TOKENIZERS:
                raise SourceInventoryError(_ERR_DATA)
            normalized[field] = item
        else:
            raise SourceInventoryError(_ERR_DATA)
    return normalized


def _model_data(
    settings_config: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]]]:
    declared: list[str] = []
    definitions_by_id: dict[str, dict[str, Any]] = {}
    primary_model = _optional_string(settings_config, "model")
    if primary_model is not None:
        declared.append(_model_id(primary_model))

    configured_models = settings_config.get("models")
    if configured_models is not None:
        if not isinstance(configured_models, list):
            raise SourceInventoryError(_ERR_DATA)
        for configured_model in configured_models:
            definition = _model_definition(configured_model)
            model = definition["id"]
            if model in definitions_by_id:
                raise SourceInventoryError(_ERR_DATA)
            declared.append(model)
            definitions_by_id[model] = definition

    unique_declared = list(dict.fromkeys(declared))
    return (
        unique_declared,
        [definitions_by_id.get(model, {"id": model}) for model in unique_declared],
    )


def _environment_model_evidence(environment: dict[str, Any]) -> dict[str, Any]:
    """Return only allowlisted model facts from a provider environment."""
    configured_value_by_key: dict[str, str] = {}
    default_configured_value_keys = sorted(
        key
        for key in environment
        if _ENV_DEFAULT_MODEL_KEY_RE.fullmatch(key) is not None
    )
    for key in (*_ENV_CONFIGURED_MODEL_KEYS, *default_configured_value_keys):
        configured_value = _optional_string(environment, key)
        if configured_value is not None:
            configured_value_by_key[key] = _configured_model_value(configured_value)

    name_to_configured_value: dict[str, str] = {}
    default_name_keys = sorted(
        key
        for key in environment
        if _ENV_DEFAULT_MODEL_NAME_KEY_RE.fullmatch(key) is not None
    )
    for name_key in default_name_keys:
        name = _optional_string(environment, name_key)
        if name is None:
            continue
        configured_value = configured_value_by_key.get(name_key.removesuffix("_NAME"))
        if configured_value is None:
            continue
        name = _configured_model_value(name)
        existing = name_to_configured_value.setdefault(name, configured_value)
        if existing != configured_value:
            raise SourceInventoryError(_ERR_DATA)

    return {
        "configured_model_values": sorted(set(configured_value_by_key.values())),
        "name_to_configured_value": dict(sorted(name_to_configured_value.items())),
    }


def _credential_reference(environment: dict[str, Any]) -> str | None:
    reference: str | None = None
    for key in _CREDENTIAL_ENV_KEYS:
        if key not in environment:
            continue
        credential = environment[key]
        if not isinstance(credential, str):
            raise SourceInventoryError(_ERR_DATA)
        if credential.strip() and reference is None:
            reference = key
    return reference


def _credential_value(
    environment: dict[str, Any], credential_reference: str | None
) -> str | None:
    """Keep the credential value private to the migration apply boundary."""
    if credential_reference is None:
        return None
    value = environment.get(credential_reference)
    if not isinstance(value, str) or not value.strip():
        raise SourceInventoryError(_ERR_DATA)
    return value


def _migration_apply_binding_sha256(
    *,
    source_id: str,
    api_family: str | None,
    endpoint: str | None,
    credential_reference: str | None,
    credential_value: str | None,
) -> str | None:
    """Hash apply-only origin/credential facts without serializing either."""
    if not all(
        isinstance(value, str) and value
        for value in (source_id, api_family, endpoint, credential_reference, credential_value)
    ):
        return None
    payload = {
        "apiFamily": api_family,
        "credentialReference": credential_reference,
        "credentialValue": credential_value,
        "endpoint": endpoint,
        "sourceId": source_id,
    }
    return sha256(
        json.dumps(
            payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _normalise_provider(row: tuple[Any, ...]) -> dict[str, Any]:
    if len(row) != 7:
        raise SourceInventoryError(_ERR_DATA)
    (
        source_id,
        app_type,
        source_name,
        raw_settings,
        raw_meta,
        raw_current,
        sort_index,
    ) = row
    source_id = _source_identifier(source_id)
    source_name = _source_text(source_name)
    app_type = _bounded_provider_text(app_type)
    if app_type != _SOURCE_APP_TYPE:
        raise SourceInventoryError(_ERR_DATA)
    if type(raw_current) is not int or raw_current not in {0, 1}:
        raise SourceInventoryError(_ERR_DATA)
    if sort_index is not None and type(sort_index) is not int:
        raise SourceInventoryError(_ERR_DATA)

    settings_config = _json_object(raw_settings)
    meta = _json_object(raw_meta)
    environment = _optional_object(settings_config, "env")
    api_family = _api_family(settings_config, meta)
    endpoint_https, endpoint_userinfo, endpoint_query, endpoint_fragment, endpoint = (
        _endpoint_facts(_base_url(settings_config, meta, environment))
    )
    declared_models, model_definitions = _model_data(settings_config)
    environment_models = _environment_model_evidence(environment)
    credential_reference = _credential_reference(environment)
    credential_value = _credential_value(environment, credential_reference)
    apply_binding_sha256 = _migration_apply_binding_sha256(
        source_id=source_id,
        api_family=api_family,
        endpoint=endpoint,
        credential_reference=credential_reference,
        credential_value=credential_value,
    )
    blocker_codes: list[str] = []
    if not endpoint_https:
        blocker_codes.append("blocked-insecure-scheme")
    if endpoint_userinfo:
        blocker_codes.append("blocked-url-userinfo")
    if endpoint_query:
        blocker_codes.append("blocked-url-query")
    if endpoint_fragment:
        blocker_codes.append("blocked-url-fragment")
    if credential_reference is None:
        blocker_codes.append("blocked-credential-reference-missing")
    if api_family is None:
        blocker_codes.append("blocked-unsupported-api")
    if not declared_models:
        blocker_codes.append("blocked-source-model-missing")

    return {
        "_sort_index": sort_index,
        "source_id": source_id,
        "app_type": _SOURCE_APP_TYPE,
        "source_name": source_name,
        "is_current": raw_current == 1,
        "api_family": api_family,
        "endpoint": endpoint,
        "credential_reference": credential_reference,
        "_credential_value": credential_value,
        "_migration_apply_binding_sha256": apply_binding_sha256,
        "declared_models": declared_models,
        "environment_models": environment_models,
        "model_definitions": model_definitions,
        "endpoint_https": endpoint_https,
        "endpoint_userinfo_present": endpoint_userinfo,
        "endpoint_query_present": endpoint_query,
        "endpoint_fragment_present": endpoint_fragment,
        "credential_reference_present": credential_reference is not None,
        "eligibility_status": "candidate" if not blocker_codes else "blocked",
        "blocker_codes": blocker_codes,
    }


def _provider_sort_key(provider: dict[str, Any]) -> tuple[Any, ...]:
    sort_index = provider["_sort_index"]
    return (
        sort_index is None,
        0 if sort_index is None else sort_index,
        provider["source_name"].casefold(),
        provider["source_id"].casefold(),
        provider["source_name"],
        provider["source_id"],
    )


def _source_snapshot(providers: list[dict[str, Any]]) -> str:
    canonical = {
        "source_app_type": _SOURCE_APP_TYPE,
        "providers": [
            {
                "source_id": provider["source_id"],
                "app_type": provider["app_type"],
                "source_name": provider["source_name"],
                "is_current": provider["is_current"],
                "api_family": provider["api_family"],
                "endpoint": provider["endpoint"],
                "credential_reference": provider["credential_reference"],
                "declared_models": provider["declared_models"],
                "environment_models": provider["environment_models"],
                "model_definitions": provider["model_definitions"],
                "blocker_codes": provider["blocker_codes"],
            }
            for provider in providers
        ],
    }
    encoded = json.dumps(
        canonical, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _public_provider(provider: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": provider["source_id"],
        "source_name": provider["source_name"],
        "is_current": provider["is_current"],
        "api_family": provider["api_family"],
        "declared_models": list(provider["declared_models"]),
        "environment_models": {
            "configured_model_values": list(
                provider["environment_models"]["configured_model_values"]
            ),
            "name_to_configured_value": dict(
                provider["environment_models"]["name_to_configured_value"]
            ),
        },
        "endpoint_https": provider["endpoint_https"],
        "endpoint_userinfo_present": provider["endpoint_userinfo_present"],
        "endpoint_query_present": provider["endpoint_query_present"],
        "endpoint_fragment_present": provider["endpoint_fragment_present"],
        "credential_reference_present": provider["credential_reference_present"],
        "eligibility_status": provider["eligibility_status"],
        "blocker_codes": list(provider["blocker_codes"]),
    }


def _materialization_provider(provider: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": provider["source_id"],
        "app_type": provider["app_type"],
        "is_current": provider["is_current"],
        "api_family": provider["api_family"],
        "endpoint": provider["endpoint"],
        "credential_reference": provider["credential_reference"],
        "declared_models": list(provider["declared_models"]),
        "model_definitions": [
            dict(model_definition) for model_definition in provider["model_definitions"]
        ],
        "blocker_codes": list(provider["blocker_codes"]),
    }


def _read_normalized_source(
    db_path: str | os.PathLike[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read Claude rows from a bounded, stable copy of the main DB and sidecars."""
    capture: _CapturedSource | None = None
    connection: sqlite3.Connection | None = None
    path = ""
    try:
        path = _source_path(db_path)
        capture = _capture_source_snapshot(path)
        connection = sqlite3.connect(
            _snapshot_uri(capture.snapshot_path),
            uri=True,
            isolation_level=None,
            timeout=1.0,
        )
        _configure_readonly_connection(connection)
        connection.execute("BEGIN")
        _validate_schema(connection)
        rows = connection.execute(
            _PROVIDER_QUERY, (_SOURCE_APP_TYPE, _MAX_PROVIDER_ROWS + 1)
        ).fetchall()
        if len(rows) > _MAX_PROVIDER_ROWS:
            raise SourceInventoryError(_ERR_DATA)
        providers = [_normalise_provider(tuple(row)) for row in rows]
        source_ids = [provider["source_id"] for provider in providers]
        if len(source_ids) != len(set(source_ids)):
            raise SourceInventoryError(_ERR_DATA)
        providers.sort(key=_provider_sort_key)
        snapshot_sha256 = _source_snapshot(providers)
        _assert_capture_stable(path, capture)
        precondition = {
            "realpath": _stable_realpath(path, capture.identity),
            **_file_manifest(capture.identity, capture.source_sha256),
            "snapshotSha256": snapshot_sha256,
            "wal": (
                None
                if capture.wal_identity is None or capture.wal_sha256 is None
                else _file_manifest(capture.wal_identity, capture.wal_sha256)
            ),
            "shm": _shm_manifest(capture.shm_identity, capture.shm_sha256),
        }
        return providers, precondition
    except SourceInventoryError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise SourceInventoryError(_ERR_READ) from None
    finally:
        stability_error: SourceInventoryError | None = None
        if capture is not None:
            try:
                _assert_capture_stable(path, capture)
            except SourceInventoryError as error:
                stability_error = error
        close_error = False
        if connection is not None:
            try:
                connection.close()
            except sqlite3.Error:
                close_error = True
        if capture is not None:
            for open_descriptor in (
                capture.shm_descriptor,
                capture.wal_descriptor,
                capture.descriptor,
            ):
                if open_descriptor != -1:
                    try:
                        os.close(open_descriptor)
                    except OSError:
                        if stability_error is None:
                            stability_error = SourceInventoryError(_ERR_SECURITY)
            try:
                capture.temporary.cleanup()
            except OSError:
                if stability_error is None:
                    stability_error = SourceInventoryError(_ERR_SECURITY)
        if stability_error is not None:
            raise stability_error
        if close_error:
            raise SourceInventoryError(_ERR_READ)


def _source_id_set(value: Any, code: str) -> frozenset[str]:
    if not isinstance(value, (set, frozenset)) or not value:
        raise SourceInventoryError(code)
    source_ids: set[str] = set()
    for source_id in value:
        source_ids.add(_source_identifier(source_id))
    return frozenset(source_ids)


def inventory_source(
    db_path: str | os.PathLike[str], provider_filter: str | None = None
) -> dict[str, Any]:
    """Return safe Claude-provider facts without source configuration or credentials."""
    try:
        if provider_filter is not None and not isinstance(provider_filter, str):
            raise SourceInventoryError(_ERR_FILTER)
        providers, _precondition = _read_normalized_source(db_path)
        snapshot = _source_snapshot(providers)
        if provider_filter is not None:
            providers = [
                provider
                for provider in providers
                if provider_filter == provider["source_id"]
                or provider_filter == provider["source_name"]
            ]
        return {
            "status": "inventory",
            "source_app_type": _SOURCE_APP_TYPE,
            "source_snapshot_sha256": snapshot,
            "providers": [_public_provider(provider) for provider in providers],
        }
    except SourceInventoryError:
        raise
    except (TypeError, ValueError):
        raise SourceInventoryError(_ERR_READ) from None


def materialization_source(
    db_path: str | os.PathLike[str],
    *,
    selected_source_ids: set[str] | frozenset[str],
    approved_origin_source_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Return selected, normalized in-process materialization facts.

    Every selected source ID must be explicitly approved.  The return value is
    intentionally not an inventory or plan payload: it has a private endpoint
    and a credential environment-variable name for the materializer, but no
    raw JSON, environment mapping, or credential value.
    """
    try:
        selected = _source_id_set(selected_source_ids, _ERR_SELECTION)
        approved = _source_id_set(approved_origin_source_ids, _ERR_ORIGIN)
        if not selected.issubset(approved):
            raise SourceInventoryError(_ERR_ORIGIN)
        providers, precondition = _read_normalized_source(db_path)
        by_source_id = {provider["source_id"]: provider for provider in providers}
        if any(source_id not in by_source_id for source_id in selected):
            raise SourceInventoryError(_ERR_SELECTION)
        selected_providers = [
            provider for provider in providers if provider["source_id"] in selected
        ]
        return {
            "source_app_type": _SOURCE_APP_TYPE,
            "source_snapshot_sha256": _source_snapshot(providers),
            "precondition": precondition,
            "providers": [
                _materialization_provider(provider) for provider in selected_providers
            ],
        }
    except SourceInventoryError:
        raise
    except (TypeError, ValueError):
        raise SourceInventoryError(_ERR_READ) from None


def _selected_normalized_providers(
    db_path: str | os.PathLike[str],
    *,
    selected_source_ids: set[str] | frozenset[str],
    approved_origin_source_ids: set[str] | frozenset[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Read one approved source snapshot without widening the source scope."""
    selected = _source_id_set(selected_source_ids, _ERR_SELECTION)
    approved = _source_id_set(approved_origin_source_ids, _ERR_ORIGIN)
    if not selected.issubset(approved):
        raise SourceInventoryError(_ERR_ORIGIN)
    providers, precondition = _read_normalized_source(db_path)
    by_source_id = {provider["source_id"]: provider for provider in providers}
    if any(source_id not in by_source_id for source_id in selected):
        raise SourceInventoryError(_ERR_SELECTION)
    return (
        [provider for provider in providers if provider["source_id"] in selected],
        precondition,
        _source_snapshot(providers),
    )


def migration_planning_source(
    db_path: str | os.PathLike[str],
    *,
    selected_source_ids: set[str] | frozenset[str],
    approved_origin_source_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Return a redacted migration-planning snapshot.

    This surface deliberately never returns a raw endpoint or credential.  The
    endpoint digest binds the later apply-time extraction to the exact CC Switch
    snapshot while keeping plans and inventory output safe to serialize.
    """
    try:
        selected, precondition, snapshot = _selected_normalized_providers(
            db_path,
            selected_source_ids=selected_source_ids,
            approved_origin_source_ids=approved_origin_source_ids,
        )
        providers: list[dict[str, Any]] = []
        for provider in selected:
            endpoint = provider["endpoint"]
            if not isinstance(endpoint, str):
                raise SourceInventoryError(_ERR_DATA)
            providers.append(
                {
                    "source_id": provider["source_id"],
                    "app_type": provider["app_type"],
                    "is_current": provider["is_current"],
                    "api_family": provider["api_family"],
                    "endpoint_https": provider["endpoint_https"],
                    "endpoint_sha256": sha256(endpoint.encode("utf-8")).hexdigest(),
                    "apply_binding_sha256": provider[
                        "_migration_apply_binding_sha256"
                    ],
                    "credential_reference_present": provider[
                        "credential_reference_present"
                    ],
                    "declared_models": list(provider["declared_models"]),
                    "environment_models": {
                        "configured_model_values": list(
                            provider["environment_models"][
                                "configured_model_values"
                            ]
                        ),
                        "name_to_configured_value": dict(
                            provider["environment_models"][
                                "name_to_configured_value"
                            ]
                        ),
                    },
                    "blocker_codes": list(provider["blocker_codes"]),
                }
            )
        return {
            "source_app_type": _SOURCE_APP_TYPE,
            "source_snapshot_sha256": snapshot,
            "precondition": precondition,
            "providers": providers,
        }
    except SourceInventoryError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise SourceInventoryError(_ERR_READ) from None


def _migration_apply_source(
    db_path: str | os.PathLike[str],
    *,
    selected_source_ids: set[str] | frozenset[str],
    approved_origin_source_ids: set[str] | frozenset[str],
) -> dict[str, Any]:
    """Return apply-only endpoint and credential bytes from one stable snapshot.

    Callers must never serialize this value.  It is intentionally separate from
    ``migration_planning_source`` so a plan cannot obtain a private origin or a
    credential value.
    """
    try:
        selected, precondition, snapshot = _selected_normalized_providers(
            db_path,
            selected_source_ids=selected_source_ids,
            approved_origin_source_ids=approved_origin_source_ids,
        )
        providers: list[dict[str, Any]] = []
        for provider in selected:
            endpoint = provider["endpoint"]
            credential = provider["_credential_value"]
            if not isinstance(endpoint, str) or not isinstance(credential, str):
                raise SourceInventoryError(_ERR_DATA)
            providers.append(
                {
                    "source_id": provider["source_id"],
                    "api_family": provider["api_family"],
                    "endpoint": endpoint,
                    "endpoint_sha256": sha256(endpoint.encode("utf-8")).hexdigest(),
                    "apply_binding_sha256": provider[
                        "_migration_apply_binding_sha256"
                    ],
                    "credential_value": credential,
                }
            )
        return {
            "source_app_type": _SOURCE_APP_TYPE,
            "source_snapshot_sha256": snapshot,
            "precondition": precondition,
            "providers": providers,
        }
    except SourceInventoryError:
        raise
    except (TypeError, ValueError, UnicodeError):
        raise SourceInventoryError(_ERR_READ) from None
