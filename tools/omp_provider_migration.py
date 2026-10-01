"""Opt-in, secret-free planning and recoverable apply for one CC Switch migration.

The ordinary projection seam deliberately remains unchanged.  This module is a
separate, closed contract for the explicitly requested ``cc-xuanju`` creation:
plans contain only model identifiers, audited digests, and HTTPS facts.  A raw
origin and a credential are available only inside the apply transaction after
all plan preconditions have been checked from one fresh CC Switch snapshot.
"""

from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
import json
import hashlib
import hmac
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

import yaml

import tools.omp_model_materializer as _materializer
from tools.cc_switch_omp_source import (
    SourceInventoryError,
    _migration_apply_source,
    materialization_source,
    migration_planning_source,
)
from tools.omp_config_handoff import (
    ConfigHandoffError,
    build_provider_migration_handoff,
    render_provider_migration_config,
)
from tools.omp_model_materializer_schema import validate_document


__all__ = [
    "MigrationPlanError",
    "apply_provider_migration_plan",
    "build_provider_migration_plan",
    "load_provider_migration_plan",
    "validate_provider_migration_plan",
    "verify_provider_migration",
]
_SCHEMA_VERSION = 4
_PROBE_SCHEMA_VERSION = 2
_OMP_VERSION = "18.2.0"
_OMP_PACKAGE_NAME = "@oh-my-pi/pi-coding-agent"
_OMP_PACKAGE_PATH = Path(
    "/Users/chiyuchou/.npm-global/lib/node_modules/@oh-my-pi/pi-coding-agent/package.json"
)
_MAX_FILE_BYTES = 16 << 20
_MAX_PROBE_BYTES = 1 << 20
_DY_SOURCE_ID = "6becb67f-1730-438d-b8ad-9429675d1090"
_XUANJU_SOURCE_ID = "616ccdfd-8de6-4c82-a4cc-24f71edf0c49"
_DY_TARGET = "cc-dy-codex"
_XUANJU_TARGET = "cc-xuanju"
_DEEPSEEK_ENV_KEY = "CC_SWITCH_DEEPSEEK_KEY"
_DY_ENV_KEY = "CC_SWITCH_DY_CODEX_KEY"
_XUANJU_ENV_KEY = "CC_SWITCH_XUANJU_KEY"
_DY_MODELS = ("gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra")
_XUANJU_MODELS = ("GLM-5.3-flash",)
_XUANJU_PROBE_MODELS = (
    "GLM-5.3-flash",
    "Kimi-K3",
    "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash",
    "GLM-5.3",
)
_XUANJU_PROBE_MODES = ("text", "tool")
_XUANJU_ARTIFACT_KIND = "xuanju-openai-responses-observation"
_DY_OFFICIAL_HOST = "www.api-codex.cn"
_DY_PROVIDER_BASE_URL = f"https://{_DY_OFFICIAL_HOST}/v1"
_DY_PROVIDER_BASE_SHA256 = hashlib.sha256(
    _DY_PROVIDER_BASE_URL.encode("utf-8")
).hexdigest()
_XUANJU_REQUEST_API = "openai-responses"
_XUANJU_SOURCE_API_FAMILIES = frozenset({"anthropic", "openai_responses"})
_XUANJU_OFFICIAL_HOST = "hs32.com"
_XUANJU_SOURCE_ENDPOINT_PATH = "/v1/responses"
_XUANJU_ANTHROPIC_ENDPOINT_PATH = "/api/anthropic"
_XUANJU_PROVIDER_BASE_URL = f"https://{_XUANJU_OFFICIAL_HOST}/v1"
_XUANJU_PROVIDER_BASE_SHA256 = hashlib.sha256(
    _XUANJU_PROVIDER_BASE_URL.encode("utf-8")
).hexdigest()
_XUANJU_SOURCE_ENDPOINT = (
    f"https://{_XUANJU_OFFICIAL_HOST}{_XUANJU_SOURCE_ENDPOINT_PATH}"
)
_XUANJU_SOURCE_ENDPOINTS = {
    "anthropic": f"https://{_XUANJU_OFFICIAL_HOST}{_XUANJU_ANTHROPIC_ENDPOINT_PATH}",
    "openai_responses": _XUANJU_SOURCE_ENDPOINT,
}
_NEW_SELECTORS = (
    "cc-xuanju/GLM-5.3-flash",
)
_DY_DISCOVERY = {
    "type": "openai-models-list",
    "injectV1": True,
}
_SOURCE_API_TARGET = {
    "anthropic": "anthropic-messages",
    "openai_chat": "openai-completions",
    "openai_responses": "openai-responses",
}
_PROBE_FAILURE_STATUSES = frozenset(
    {
        "blocked",
        "blocked-api-mismatch",
        "blocked-artifact",
        "blocked-child-capability",
        "blocked-credential-reference-missing",
        "blocked-credential-unavailable",
        "blocked-invalid-arguments",
        "blocked-model-not-listed",
        "blocked-no-follow-unavailable",
        "blocked-origin-approval-required",
        "blocked-output-directory",
        "blocked-output-exists",
        "blocked-output-path",
        "blocked-output-write",
        "blocked-preflight",
        "blocked-runner",
        "blocked-source-changed",
        "blocked-source-path",
        "blocked-source-unavailable",
        "blocked-transport-policy",
        "blocked-trusted-installation",
        "identity-mismatch",
        "identity-unverified",
        "interrupted",
        "temporary",
        "temporary-timeout",
        "unusable",
        "unusable-empty-text",
        "unusable-text-result",
        "unusable-tool-result",
        "unusable-worker-crash",
        "unusable-worker-launch",
        "unusable-worker-output",
        "blocked-worker-reap",
    }
)
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class MigrationPlanError(ValueError):
    """A migration contract was refused without including private context."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _Snapshot:
    raw: bytes
    parsed: Any
    precondition: dict[str, Any]
    details: os.stat_result | None = None


@dataclass
class _LockedFile:
    kind: str
    location: Any
    parent_fd: int
    descriptor: int
    snapshot: _Snapshot


@dataclass
class _PublishedFile:
    locked: _LockedFile
    replacement: Any


def _reject(code: str) -> None:
    raise MigrationPlanError(code)


class _JcsError(ValueError):
    """The deliberately narrow RFC 8785 value subset was not respected."""


_JCS_SAFE_INTEGER_MAX = (1 << 53) - 1
_DECIMAL_RE = re.compile(r"^(?:0|[1-9][0-9]*)$")


def _jcs_string(value: str) -> str:
    """Serialize a Unicode string exactly as ECMAScript JSON.stringify does."""
    try:
        value.encode("utf-8")
    except UnicodeError as error:
        raise _JcsError from error
    escaped: list[str] = ['"']
    for character in value:
        codepoint = ord(character)
        if character == '"':
            escaped.append('\\"')
        elif character == "\\":
            escaped.append("\\\\")
        elif character == "\b":
            escaped.append("\\b")
        elif character == "\t":
            escaped.append("\\t")
        elif character == "\n":
            escaped.append("\\n")
        elif character == "\f":
            escaped.append("\\f")
        elif character == "\r":
            escaped.append("\\r")
        elif codepoint < 0x20:
            escaped.append(f"\\u{codepoint:04x}")
        else:
            escaped.append(character)
    escaped.append('"')
    return "".join(escaped)


def _jcs_append(value: Any, output: list[str], active: set[int]) -> None:
    """Emit the safe-integer, non-float JCS subset used by this contract."""
    if value is None:
        output.append("null")
        return
    if type(value) is bool:
        output.append("true" if value else "false")
        return
    if type(value) is int:
        if abs(value) > _JCS_SAFE_INTEGER_MAX:
            raise _JcsError
        output.append(str(value))
        return
    if type(value) is str:
        output.append(_jcs_string(value))
        return
    if type(value) is list:
        identity = id(value)
        if identity in active:
            raise _JcsError
        active.add(identity)
        try:
            output.append("[")
            for index, item in enumerate(value):
                if index:
                    output.append(",")
                _jcs_append(item, output, active)
            output.append("]")
        finally:
            active.remove(identity)
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in active:
            raise _JcsError
        active.add(identity)
        try:
            items = list(value.items())
            keys: set[str] = set()
            for key, _item in items:
                if type(key) is not str or key in keys:
                    raise _JcsError
                _jcs_string(key)
                keys.add(key)
            items.sort(key=lambda item: item[0].encode("utf-16be"))
            output.append("{")
            for index, (key, item) in enumerate(items):
                if index:
                    output.append(",")
                output.append(_jcs_string(key))
                output.append(":")
                _jcs_append(item, output, active)
            output.append("}")
        finally:
            active.remove(identity)
        return
    raise _JcsError


def _canonical(value: Any, code: str) -> bytes:
    """RFC 8785 JCS for JSON values with exact safe integers only.

    Plans never need floating point.  Rejecting it avoids the implementation-
    dependent number rendering RFC 8785 otherwise inherits from ECMAScript.
    Filesystem metadata is represented as closed decimal strings before this
    boundary, so platform-sized inode and timestamp values remain portable.
    """
    try:
        output: list[str] = []
        _jcs_append(value, output, set())
        return "".join(output).encode("utf-8")
    except (AttributeError, TypeError, UnicodeError, RecursionError, _JcsError):
        _reject(code)


def _digest(value: Any, code: str) -> str:
    return hashlib.sha256(_canonical(value, code)).hexdigest()


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return _digest({key: value for key, value in plan.items() if key != "planSha256"}, "blocked-plan-schema")


def _exact(value: Any, keys: set[str], code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        _reject(code)
    return value


def _closed_decimal(value: Any, code: str) -> str:
    """Return an unsigned filesystem integer in its sole portable spelling."""
    if type(value) is not int or value < 0:
        _reject(code)
    return str(value)


def _decimal_value(value: Any) -> bool:
    return type(value) is str and _DECIMAL_RE.fullmatch(value) is not None


def _file_parent_precondition(value: Any, code: str) -> dict[str, str]:
    parent = _exact(value, {"device", "fileId", "ctimeNs"}, code)
    return {
        "device": _closed_decimal(parent["device"], code),
        "fileId": _closed_decimal(parent["fileId"], code),
        "ctimeNs": _closed_decimal(parent["ctimeNs"], code),
    }

def _source_main_identity(value: Any, code: str) -> dict[str, str]:
    """Project one strict source capture onto replacement/security identity.

    The source seam proves the complete main/WAL/SHM capture internally.  Only
    the main database identity crosses the plan boundary: physical bytes,
    timestamps, and sidecars are intentionally excluded so telemetry WAL
    churn cannot invalidate an otherwise unchanged selected binding.
    """
    if not isinstance(value, Mapping):
        _reject(code)
    realpath = value.get("realpath")
    if (
        not isinstance(realpath, str)
        or not realpath
        or "\x00" in realpath
        or not os.path.isabs(realpath)
        or os.path.normpath(realpath) != realpath
    ):
        _reject(code)
    try:
        identity = {
            "realpath": realpath,
            "device": _closed_decimal(value["device"], code),
            "fileId": _closed_decimal(value["fileId"], code),
            "owner": _closed_decimal(value["owner"], code),
            "mode": _closed_decimal(value["mode"], code),
        }
    except (KeyError, TypeError):
        _reject(code)
    if int(identity["fileId"]) == 0 or int(identity["mode"]) > 0o777:
        _reject(code)
    return identity


def _identifier(value: Any, code: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        _reject(code)
    return value


def _model(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or "[" in value
        or "]" in value
        or not _MODEL_RE.fullmatch(value)
    ):
        _reject(code)
    return value
def _selector_key(value: str) -> str:
    provider, separator, model = value.partition("/")
    if not separator:
        return value.strip().casefold()
    return f"{provider.strip().casefold()}/{model.strip().casefold()}"


def _is_xuanju_flash_selector(value: str) -> bool:
    return _selector_key(value) == _NEW_SELECTORS[0].casefold()


def _is_xuanju_selector(value: str) -> bool:
    return _selector_key(value).partition("/")[0] == _XUANJU_TARGET


def _is_canonical_target_provider(value: str) -> bool:
    normalized = value.casefold()
    return normalized in {_DY_TARGET, _XUANJU_TARGET} and value == normalized


def _validate_target_provider_order(value: Sequence[Any]) -> None:
    normalized = [
        provider.casefold() if isinstance(provider, str) else None
        for provider in value
    ]
    if (
        any(provider is None for provider in normalized)
        or len(normalized) != len(set(normalized))
        or any(
            isinstance(provider, str)
            and provider.casefold() in {_DY_TARGET, _XUANJU_TARGET}
            and not _is_canonical_target_provider(provider)
            for provider in value
        )
    ):
        _reject("blocked-plan-schema")


def _validate_enabled_selector_routes(value: Sequence[Any]) -> None:
    normalized = [
        _selector_key(selector) for selector in value if isinstance(selector, str)
    ]
    if len(normalized) != len(value):
        _reject("blocked-plan-schema")
    if any(
        _is_xuanju_selector(selector) and not _is_xuanju_flash_selector(selector)
        for selector in value
    ):
        _reject("blocked-plan-schema")



def _model_list(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _reject(code)
    values = [_model(item, code) for item in value]
    if not values or len(values) != len(set(values)):
        _reject(code)
    return values



def _model_list_allow_empty(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _reject(code)
    values = [_model(item, code) for item in value]
    if len(values) != len(set(values)):
        _reject(code)
    return values

def _configured_model_values_allow_context(value: Any, code: str) -> list[str]:
    """Validate bounded source values while preserving context suffixes opaquely."""
    if not isinstance(value, list) or len(value) > 256:
        _reject(code)
    values: list[str] = []
    for item in value:
        if (
            not isinstance(item, str)
            or not item
            or len(item) > 512
            or any(ord(character) < 32 for character in item)
        ):
            _reject(code)
        values.append(item)
    if len(values) != len(set(values)):
        _reject(code)
    return values


def _require_secure_platform() -> None:
    try:
        _materializer._require_secure_platform()
    except Exception as error:
        _reject(getattr(error, "code", "blocked-unsupported-platform"))


def _translate_target_error(error: Exception, kind: str) -> None:
    code = getattr(error, "code", "")
    if code == "blocked-target-symlink":
        _reject(f"blocked-{kind}-symlink")
    if code in {"blocked-target-changed", "blocked-target-path"}:
        _reject(f"blocked-{kind}-path")
    _reject(f"blocked-{kind}-path")


def _location(value: str | Path, kind: str) -> Any:
    try:
        return _materializer._target_location(value)
    except Exception as error:
        _translate_target_error(error, kind)


def _read_bytes(descriptor: int, maximum: int, code: str) -> bytes:
    try:
        return _materializer._read_descriptor(descriptor, maximum, code)
    except Exception:
        _reject(code)


def _validate_file_stat(details: os.stat_result, kind: str, *, exact_mode: bool) -> None:
    try:
        owned = _materializer._owned_by_current_user(details.st_uid)
    except Exception:
        owned = False
    if not stat.S_ISREG(details.st_mode) or details.st_nlink != 1 or not owned:
        _reject(f"blocked-{kind}-path")
    mode = stat.S_IMODE(details.st_mode)
    if mode & 0o022:
        _reject(f"blocked-{kind}-permissions")
    if exact_mode and mode != 0o600:
        _reject("blocked-env-mode" if kind == "environment" else f"blocked-{kind}-mode")
    if details.st_size < 0 or details.st_size > _MAX_FILE_BYTES:
        _reject(f"blocked-{kind}-size")


def _same_state(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_nlink,
        left.st_size,
        left.st_mode,
        left.st_mtime_ns,
        left.st_ctime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_nlink,
        right.st_size,
        right.st_mode,
        right.st_mtime_ns,
        right.st_ctime_ns,
    )


def _file_precondition(
    location: Any,
    parent_fd: int,
    details: os.stat_result,
    raw: bytes,
    kind: str,
) -> dict[str, Any]:
    code = f"blocked-{kind}-read"
    return {
        "realpath": location.realpath,
        "device": _closed_decimal(details.st_dev, code),
        "fileId": _closed_decimal(details.st_ino, code),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": _closed_decimal(stat.S_IMODE(details.st_mode), code),
        "ctimeNs": _closed_decimal(details.st_ctime_ns, code),
        "parent": _file_parent_precondition(
            _materializer._target_parent_identity(parent_fd), code
        ),
    }


def _snapshot_descriptor(
    parent_fd: int,
    location: Any,
    descriptor: int,
    kind: str,
    parser: Callable[[bytes], Any],
    *,
    exact_mode: bool,
) -> _Snapshot:
    try:
        before = os.fstat(descriptor)
        _validate_file_stat(before, kind, exact_mode=exact_mode)
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = _read_bytes(descriptor, _MAX_FILE_BYTES, f"blocked-{kind}-read")
        after = os.fstat(descriptor)
        _validate_file_stat(after, kind, exact_mode=exact_mode)
        if not _same_state(before, after):
            _reject(f"blocked-{kind}-changed")
        _materializer._assert_target_path_matches_descriptor(parent_fd, location, descriptor)
        parsed = parser(raw)
        return _Snapshot(
            raw,
            parsed,
            _file_precondition(location, parent_fd, after, raw, kind),
            after,
        )
    except MigrationPlanError:
        raise
    except Exception:
        _reject(f"blocked-{kind}-parse")


def _snapshot_file(
    value: str | Path,
    kind: str,
    parser: Callable[[bytes], Any],
    *,
    exact_mode: bool,
) -> tuple[Any, _Snapshot]:
    location = _location(value, kind)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd = _materializer._open_target_parent(location)
        descriptor = _materializer._open_target_descriptor(parent_fd, location)
        snapshot = _snapshot_descriptor(
            parent_fd, location, descriptor, kind, parser, exact_mode=exact_mode
        )
        return location, snapshot
    except MigrationPlanError:
        raise
    except Exception as error:
        _translate_target_error(error, kind)
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if parent_fd != -1:
            with contextlib.suppress(OSError):
                os.close(parent_fd)


def _omp_package_parser(raw: bytes) -> dict[str, str]:
    try:
        package = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, TypeError, ValueError):
        _reject("blocked-omp-package")
    if not isinstance(package, Mapping):
        _reject("blocked-omp-package")
    name = package.get("name")
    version = package.get("version")
    if name != _OMP_PACKAGE_NAME:
        _reject("blocked-omp-package")
    if version != _OMP_VERSION:
        _reject("blocked-omp-version")
    return {"name": name, "version": version}


def _omp_package_precondition(snapshot: _Snapshot) -> dict[str, str]:
    details = snapshot.details
    if details is None:
        _reject("blocked-omp-package")
    code = "blocked-omp-package"
    return {
        "realpath": snapshot.precondition["realpath"],
        "device": snapshot.precondition["device"],
        "fileId": snapshot.precondition["fileId"],
        "owner": _closed_decimal(details.st_uid, code),
        "mode": snapshot.precondition["mode"],
        "size": _closed_decimal(details.st_size, code),
        "ctimeNs": snapshot.precondition["ctimeNs"],
        "sha256": snapshot.precondition["sha256"],
    }


def _snapshot_omp_package(
    value: str | Path,
) -> tuple[Any, _Snapshot, dict[str, str]]:
    location, snapshot = _snapshot_file(
        value, "omp-package", _omp_package_parser, exact_mode=False
    )
    return location, snapshot, _omp_package_precondition(snapshot)


def _open_locked(
    value: str | Path,
    kind: str,
    parser: Callable[[bytes], Any],
    *,
    exact_mode: bool,
) -> _LockedFile:
    location = _location(value, kind)
    parent_fd = -1
    descriptor = -1
    try:
        parent_fd = _materializer._open_target_parent(location)
        descriptor = _materializer._open_target_descriptor(parent_fd, location)
        _materializer._acquire_file_lock(descriptor, f"blocked-{kind}-lock")
        snapshot = _snapshot_descriptor(
            parent_fd, location, descriptor, kind, parser, exact_mode=exact_mode
        )
        result = _LockedFile(kind, location, parent_fd, descriptor, snapshot)
        descriptor = -1
        parent_fd = -1
        return result
    except MigrationPlanError:
        raise
    except Exception as error:
        _translate_target_error(error, kind)
    finally:
        if descriptor != -1:
            _materializer._release_file_lock(descriptor)
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if parent_fd != -1:
            with contextlib.suppress(OSError):
                os.close(parent_fd)


def _close_locked(locked: _LockedFile) -> None:
    _materializer._release_file_lock(locked.descriptor)
    with contextlib.suppress(OSError):
        os.close(locked.descriptor)
    with contextlib.suppress(OSError):
        os.close(locked.parent_fd)


def _assert_precondition(actual: Mapping[str, Any], expected: Any, kind: str) -> None:
    if not isinstance(expected, Mapping) or dict(actual) != dict(expected):
        _reject(f"blocked-{kind}-changed")


def _assert_publishable(locked: _LockedFile) -> None:
    """Recheck the locked target inode without rejecting our own backup writes."""
    try:
        _materializer._assert_parent_path_matches_descriptor(
            locked.location, locked.parent_fd
        )
        _materializer._assert_target_path_matches_descriptor(
            locked.parent_fd, locked.location, locked.descriptor
        )
        current = _snapshot_descriptor(
            locked.parent_fd,
            locked.location,
            locked.descriptor,
            locked.kind,
            _parser_for_kind(locked.kind),
            exact_mode=False,
        )
    except MigrationPlanError:
        raise
    except Exception:
        _reject(f"blocked-{locked.kind}-changed")
    # Parent directory ctime necessarily changes when this transaction creates
    # its opaque backup and stages replacements.  The no-follow parent identity
    # and the locked target inode/hash/ctime are the relevant CAS guard here.
    for key in {"realpath", "device", "fileId", "sha256", "mode", "ctimeNs"}:
        if current.precondition.get(key) != locked.snapshot.precondition.get(key):
            _reject(f"blocked-{locked.kind}-changed")


def _models_parser(raw: bytes) -> dict[str, Any]:
    try:
        document = _materializer._read_generated_target(raw)
        validate_document(document, "blocked-models-parse")
        return document
    except MigrationPlanError:
        raise
    except Exception:
        _reject("blocked-models-parse")


def _config_parser(raw: bytes) -> dict[str, Any]:
    try:
        from tools import omp_config_handoff

        return omp_config_handoff._load_config(raw)
    except ConfigHandoffError as error:
        _reject(error.code)
    except Exception:
        _reject("blocked-config-parse")


def _parse_environment(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _reject("blocked-environment-parse")
    if "\x00" in text:
        _reject("blocked-environment-parse")
    lines = text.splitlines(keepends=True)
    keys: dict[str, int] = {}
    values: dict[str, str] = {}
    for index, line in enumerate(lines):
        body = line.rstrip("\r\n")
        stripped = body.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = body.partition("=")
        if not separator or not _ENV_KEY_RE.fullmatch(key):
            _reject("blocked-environment-parse")
        if key in keys:
            _reject("blocked-env-duplicate-key")
        keys[key] = index
        values[key] = value
    return {"lines": lines, "keys": keys, "values": values}


def _environment_value(parsed: Mapping[str, Any], key: str) -> str | None:
    values = parsed.get("values")
    if not isinstance(values, Mapping):
        return None
    raw = values.get(key)
    return raw if isinstance(raw, str) else None

def _parser_for_kind(kind: str) -> Callable[[bytes], Any]:
    return {
        "models": _models_parser,
        "config": _config_parser,
        "environment": _parse_environment,
    }[kind]


def _read_probe(path: str | Path) -> dict[str, Any]:
    descriptor = -1
    try:
        raw_path = os.fspath(path)
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise OSError
        before = os.lstat(raw_path)
        if stat.S_ISLNK(before.st_mode):
            _reject("blocked-probe-evidence")
        descriptor = os.open(
            raw_path,
            os.O_RDONLY
            | _materializer._close_on_exec_flag()
            | _materializer._no_follow_flag()
            | _materializer._non_blocking_flag(),
        )
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_size > _MAX_PROBE_BYTES
        ):
            _reject("blocked-probe-evidence")
        raw = _read_bytes(descriptor, _MAX_PROBE_BYTES, "blocked-probe-evidence")
        after = os.fstat(descriptor)
        if not _same_state(details, after) or os.lstat(raw_path).st_ino != after.st_ino:
            _reject("blocked-probe-evidence")
        parsed = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        if not isinstance(parsed, dict):
            _reject("blocked-probe-evidence")
        normalized = _validate_probe(parsed)
        canonical = _canonical(normalized, "blocked-probe-evidence")
        if raw not in {canonical, canonical + b"\n"}:
            _reject("blocked-probe-evidence")
        return normalized
    except MigrationPlanError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        _reject("blocked-probe-evidence")
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _read_probes(value: str | Path | Sequence[str | Path]) -> list[dict[str, Any]]:
    """Read exactly one canonical ten-call Responses artifact.

    The runner publishes one complete artifact.  Accepting multiple partial
    artifacts would make call ordering and freshness ambiguous, so the migration
    boundary deliberately refuses that legacy merge behavior.
    """
    if isinstance(value, (str, Path)):
        paths: list[str | Path] = [value]
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        paths = list(value)
    else:
        _reject("blocked-probe-evidence")
    if len(paths) != 1:
        _reject("blocked-probe-evidence")
    return [_read_probe(paths[0])]


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError
        result[key] = value
    return result


def _reject_json_constant(_value: str) -> None:
    raise ValueError


def _probe_token_usage(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict)
        and bool(value)
        and set(value).issubset({"input_tokens", "output_tokens", "total_tokens"})
        and all(
            type(item) is int and 0 <= item <= 1_000_000_000
            for item in value.values()
        )
    )


def _probe_observation(
    value: Any,
    *,
    source_id: str,
    expected_model: str,
    expected_mode: str,
) -> dict[str, Any]:
    _exact(
        value,
        {
            "sourceId",
            "apiFamily",
            "configuredModelValue",
            "model",
            "observedModel",
            "mode",
            "identityStatus",
            "status",
            "toolResult",
            "tokenUsage",
            "latencyMs",
            "usable",
        },
        "blocked-probe-evidence",
    )
    configured = value["configuredModelValue"]
    model = value["model"]
    observed = value["observedModel"]
    if (
        value["sourceId"] != source_id
        or value["apiFamily"] != _XUANJU_REQUEST_API
        or configured != expected_model
        or model != expected_model
        or value["mode"] != expected_mode
        or (
            observed is not None
            and observed != expected_model
        )
        or value["identityStatus"]
        not in {"not-observed", "identity-unverified", "identity-mismatch", "observed-match"}
        or value["status"] not in _PROBE_FAILURE_STATUSES | {"usable"}
        or not _probe_token_usage(value["tokenUsage"])
        or type(value["latencyMs"]) is not int
        or not 0 <= value["latencyMs"] <= 180_000
        or type(value["usable"]) is not bool
    ):
        _reject("blocked-probe-evidence")
    tool_result = value["toolResult"]
    if expected_mode == "text":
        if tool_result is not None:
            _reject("blocked-probe-evidence")
    elif tool_result not in {"not-run", "rejected", "smoke_ok"}:
        _reject("blocked-probe-evidence")
    if value["identityStatus"] == "observed-match":
        if observed != expected_model:
            _reject("blocked-probe-evidence")
    elif observed is not None:
        _reject("blocked-probe-evidence")
    if value["status"] == "usable":
        if (
            value["usable"] is not True
            or value["identityStatus"] != "observed-match"
            or tool_result != ("smoke_ok" if expected_mode == "tool" else None)
        ):
            _reject("blocked-probe-evidence")
    else:
        if value["usable"] is not False:
            _reject("blocked-probe-evidence")
    return {
        "sourceId": source_id,
        "apiFamily": _XUANJU_REQUEST_API,
        "configuredModelValue": expected_model,
        "model": expected_model,
        "observedModel": observed,
        "mode": expected_mode,
        "identityStatus": value["identityStatus"],
        "status": value["status"],
        "toolResult": tool_result,
        "tokenUsage": value["tokenUsage"],
        "latencyMs": value["latencyMs"],
        "usable": value["usable"],
    }


def _validate_probe(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact(
        value,
        {
            "schemaVersion",
            "artifactKind",
            "requestApi",
            "sourceApiFamily",
            "semanticBindingSha256",
            "providerEvidence",
            "observations",
            "artifactSha256",
        },
        "blocked-probe-evidence",
    )
    if (
        value["schemaVersion"] != _PROBE_SCHEMA_VERSION
        or value["artifactKind"] != _XUANJU_ARTIFACT_KIND
        or value["requestApi"] != _XUANJU_REQUEST_API
        or value["sourceApiFamily"] not in _XUANJU_SOURCE_API_FAMILIES
        or not _is_digest(value["semanticBindingSha256"])
    ):
        _reject("blocked-probe-evidence")
    provider_evidence = value["providerEvidence"]
    if (
        not isinstance(provider_evidence, list)
        or len(provider_evidence) != 1
        or not isinstance(provider_evidence[0], Mapping)
    ):
        _reject("blocked-probe-evidence")
    evidence = _exact(
        provider_evidence[0],
        {"sourceId", "semanticBindingSha256"},
        "blocked-probe-evidence",
    )
    source_id = _identifier(evidence["sourceId"], "blocked-probe-evidence")
    if (
        evidence["semanticBindingSha256"] != value["semanticBindingSha256"]
        or not _is_digest(evidence["semanticBindingSha256"])
    ):
        _reject("blocked-probe-evidence")
    observations = value["observations"]
    if not isinstance(observations, list) or len(observations) != len(
        _XUANJU_PROBE_MODELS
    ) * len(_XUANJU_PROBE_MODES):
        _reject("blocked-probe-evidence")
    normalized: list[dict[str, Any]] = []
    for observation, (model, mode) in zip(
        observations,
        (
            (model, mode)
            for model in _XUANJU_PROBE_MODELS
            for mode in _XUANJU_PROBE_MODES
        ),
    ):
        normalized.append(
            _probe_observation(
                observation,
                source_id=source_id,
                expected_model=model,
                expected_mode=mode,
            )
        )
    artifact_sha = value["artifactSha256"]
    body = {
        "schemaVersion": _PROBE_SCHEMA_VERSION,
        "artifactKind": _XUANJU_ARTIFACT_KIND,
        "requestApi": _XUANJU_REQUEST_API,
        "sourceApiFamily": value["sourceApiFamily"],
        "semanticBindingSha256": value["semanticBindingSha256"],
        "providerEvidence": [
            {
                "sourceId": source_id,
                "semanticBindingSha256": value["semanticBindingSha256"],
            }
        ],
        "observations": normalized,
    }
    if (
        not _is_digest(artifact_sha)
        or artifact_sha
        != _digest(body, "blocked-probe-evidence")
    ):
        _reject("blocked-probe-evidence")
    return {**body, "artifactSha256": artifact_sha}


def _planning_source(
    db_path: str | Path, source_ids: set[str]
) -> dict[str, Any]:
    try:
        source = migration_planning_source(
            db_path,
            selected_source_ids=source_ids,
            approved_origin_source_ids=source_ids,
        )
    except SourceInventoryError as error:
        _reject(error.code)
    except Exception:
        _reject("blocked-source-read")
    _exact(
        source,
        {"source_app_type", "source_snapshot_sha256", "precondition", "providers"},
        "blocked-source-read",
    )
    if source["source_app_type"] != "claude" or not isinstance(
        source["providers"], list
    ):
        _reject("blocked-source-read")
    if set(
        item.get("source_id")
        for item in source["providers"]
        if isinstance(item, Mapping)
    ) != set(source_ids):
        _reject("blocked-source-read")
    _source_main_identity(source["precondition"], "blocked-source-read")
    return source


def _apply_source(db_path: str | Path, source_ids: set[str]) -> dict[str, Any]:
    try:
        source = _migration_apply_source(
            db_path,
            selected_source_ids=source_ids,
            approved_origin_source_ids=source_ids,
        )
    except SourceInventoryError as error:
        _reject(error.code)
    except Exception:
        _reject("blocked-source-read")
    _exact(
        source,
        {"source_app_type", "source_snapshot_sha256", "precondition", "providers"},
        "blocked-source-read",
    )
    if source["source_app_type"] != "claude" or not isinstance(
        source["providers"], list
    ):
        _reject("blocked-source-read")
    if set(
        item.get("source_id")
        for item in source["providers"]
        if isinstance(item, Mapping)
    ) != set(source_ids):
        _reject("blocked-source-read")
    _source_main_identity(source["precondition"], "blocked-source-read")
    return source


def _xuanju_site_root(
    endpoint: Any, code: str, api_family: Any | None = None
) -> str:
    """Return the fixed provider base for one reviewed source family/endpoint pair."""
    if api_family is None:
        valid_endpoint = endpoint in _XUANJU_SOURCE_ENDPOINTS.values()
    else:
        valid_endpoint = (
            isinstance(api_family, str)
            and endpoint == _XUANJU_SOURCE_ENDPOINTS.get(api_family)
        )
    if not valid_endpoint:
        _reject(code)
    return _XUANJU_PROVIDER_BASE_URL


def _dy_provider_base(endpoint: Any, code: str) -> str:
    """Project a reviewed DY origin into the OMP ``/v1`` provider base."""
    if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip():
        _reject(code)
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        _reject(code)
    if (
        parsed.scheme != "https"
        or parsed.netloc != _DY_OFFICIAL_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or not isinstance(parsed.hostname, str)
        or parsed.hostname != _DY_OFFICIAL_HOST
        or not parsed.hostname
        or (port is not None and port <= 0)
        or parsed.path not in {"", "/v1"}
    ):
        _reject(code)
    return _DY_PROVIDER_BASE_URL


def _semantic_endpoint(endpoint: Any, code: str) -> str:
    """Normalize an endpoint exactly as the Responses probe's binding seam."""
    if not isinstance(endpoint, str) or not endpoint:
        _reject(code)
    try:
        parsed = urlsplit(endpoint)
        port = parsed.port
    except (TypeError, ValueError):
        _reject(code)
    if not isinstance(parsed.hostname, str) or not parsed.hostname:
        _reject(code)
    normalized_port = "" if port in (None, 443) else f":{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return f"{parsed.scheme.casefold()}://{parsed.hostname.casefold()}{normalized_port}{path}"


def _source_record(source: Mapping[str, Any], source_id: str) -> Mapping[str, Any]:
    records = source.get("providers")
    if not isinstance(records, list):
        _reject("blocked-source-read")
    record = next(
        (
            item
            for item in records
            if isinstance(item, Mapping) and item.get("source_id") == source_id
        ),
        None,
    )
    if not isinstance(record, Mapping):
        _reject("blocked-source-read")
    _exact(
        record,
        {
            "source_id",
            "app_type",
            "is_current",
            "api_family",
            "endpoint_https",
            "endpoint_sha256",
            "apply_binding_sha256",
            "credential_reference_present",
            "declared_models",
            "environment_models",
            "blocker_codes",
        },
        "blocked-source-read",
    )
    return record


def _source_public_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    return all(left.get(key) == right.get(key) for key in (
        "source_id",
        "app_type",
        "is_current",
        "api_family",
        "endpoint_https",
        "endpoint_sha256",
        "apply_binding_sha256",
        "credential_reference_present",
        "declared_models",
        "environment_models",
        "blocker_codes",
    ))
def _source_provider_record(
    source: Mapping[str, Any], source_id: str, keys: set[str], code: str
) -> Mapping[str, Any]:
    records = source.get("providers")
    if not isinstance(records, list):
        _reject(code)
    record = next(
        (
            item
            for item in records
            if isinstance(item, Mapping) and item.get("source_id") == source_id
        ),
        None,
    )
    if not isinstance(record, Mapping):
        _reject(code)
    _exact(record, keys, code)
    return record



def _selected_binding_context(
    db_path: str | Path,
    source: Mapping[str, Any],
    source_id: str,
    *,
    current_public_source: Mapping[str, Any] | None = None,
    materialized_source: Mapping[str, Any] | None = None,
    applied_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Read and clamp one selected provider without carrying physical churn.

    Every source call still has its own strict source-seam capture.  The
    continuity contract compares only the main replacement identity and the
    selected provider's public/private semantic binding.  Optional preloaded
    views let the final apply clamp bracket one private source read with
    complete semantic views for both providers.
    """
    if source_id not in {_DY_SOURCE_ID, _XUANJU_SOURCE_ID}:
        _reject("blocked-source-read")
    initial = _source_record(source, source_id)
    expected_identity = _source_main_identity(
        source.get("precondition"), "blocked-source-read"
    )
    owns_current_public = current_public_source is None
    owns_materialized = materialized_source is None
    owns_applied = applied_source is None
    materialized_raw: Mapping[str, Any] | None = materialized_source
    applied_raw: Mapping[str, Any] | None = applied_source
    try:
        if current_public_source is None:
            current_public_source = _planning_source(db_path, {source_id})
        current_identity = _source_main_identity(
            current_public_source.get("precondition"), "blocked-source-read"
        )
        current = _source_record(current_public_source, source_id)
        if current_identity != expected_identity or not _source_public_equal(
            initial, current
        ):
            _reject("blocked-source-changed")

        try:
            if materialized_raw is None:
                materialized_raw = materialization_source(
                    db_path,
                    selected_source_ids={source_id},
                    approved_origin_source_ids={source_id},
                )
            if applied_raw is None:
                applied_raw = _apply_source(db_path, {source_id})
        except SourceInventoryError as error:
            _reject(error.code)
        except Exception:
            _reject("blocked-source-read")
        for value in (materialized_raw, applied_raw):
            _exact(
                value,
                {"source_app_type", "source_snapshot_sha256", "precondition", "providers"},
                "blocked-source-read",
            )
            if value["source_app_type"] != "claude" or not isinstance(
                value["providers"], list
            ):
                _reject("blocked-source-read")
            if _source_main_identity(value["precondition"], "blocked-source-read") != expected_identity:
                _reject("blocked-source-changed")

        materialized = _source_provider_record(
            materialized_raw,
            source_id,
            {
                "source_id",
                "app_type",
                "is_current",
                "api_family",
                "endpoint",
                "credential_reference",
                "declared_models",
                "model_definitions",
                "blocker_codes",
            },
            "blocked-source-read",
        )
        applied = _source_provider_record(
            applied_raw,
            source_id,
            {
                "source_id",
                "api_family",
                "endpoint",
                "endpoint_sha256",
                "apply_binding_sha256",
                "credential_value",
            },
            "blocked-source-read",
        )
        if (
            materialized["source_id"] != source_id
            or applied["source_id"] != source_id
            or materialized["app_type"] != current["app_type"]
            or materialized["is_current"] != current["is_current"]
            or materialized["api_family"] != current["api_family"]
            or materialized["endpoint"] != applied["endpoint"]
            or current["endpoint_sha256"]
            != hashlib.sha256(applied["endpoint"].encode("utf-8")).hexdigest()
            or current["apply_binding_sha256"] != applied["apply_binding_sha256"]
            or materialized["declared_models"] != current["declared_models"]
            or materialized["blocker_codes"] != current["blocker_codes"]
            or not isinstance(materialized["credential_reference"], str)
            or not materialized["credential_reference"]
            or not isinstance(applied["credential_value"], str)
            or not applied["credential_value"]
        ):
            _reject("blocked-source-changed")

        endpoint = applied["endpoint"]
        if source_id == _XUANJU_SOURCE_ID:
            target_endpoint = _xuanju_site_root(
                endpoint, "blocked-source-url", materialized["api_family"]
            )
        else:
            target_endpoint = _dy_provider_base(endpoint, "blocked-source-url")
        normalized_endpoint = _semantic_endpoint(endpoint, "blocked-source-url")
        endpoint_origin_sha256 = hashlib.sha256(
            normalized_endpoint.encode("utf-8")
        ).hexdigest()
        semantic = {
            "sourceId": source_id,
            "appType": materialized["app_type"],
            "isCurrent": materialized["is_current"],
            "apiFamily": materialized["api_family"],
            "endpointOriginSha256": endpoint_origin_sha256,
            "applyBindingSha256": applied["apply_binding_sha256"],
            "credentialReferencePresent": current["credential_reference_present"],
            "declaredModels": list(materialized["declared_models"]),
            "environmentModels": copy.deepcopy(current["environment_models"]),
            "modelDefinitions": [
                copy.deepcopy(item) for item in materialized["model_definitions"]
            ],
            "blockerCodes": list(materialized["blocker_codes"]),
        }
        semantic_binding = _digest(semantic, "blocked-source-read")
        binding = {
            "sourceId": source_id,
            "appType": materialized["app_type"],
            "isCurrent": materialized["is_current"],
            "apiFamily": materialized["api_family"],
            "endpointHttps": current["endpoint_https"],
            "endpointSha256": current["endpoint_sha256"],
            "endpointOriginSha256": endpoint_origin_sha256,
            "credentialReferencePresent": current["credential_reference_present"],
            "declaredModels": list(materialized["declared_models"]),
            "environmentModels": copy.deepcopy(current["environment_models"]),
            "modelDefinitions": [
                copy.deepcopy(item) for item in materialized["model_definitions"]
            ],
            "blockerCodes": list(materialized["blocker_codes"]),
            "applyBindingSha256": applied["apply_binding_sha256"],
            "semanticBindingSha256": semantic_binding,
        }
        return {
            "endpoint": endpoint,
            "providerBase": target_endpoint if source_id == _DY_SOURCE_ID else None,
            "siteRoot": target_endpoint if source_id == _XUANJU_SOURCE_ID else None,
            "applyBindingSha256": applied["apply_binding_sha256"],
            "semanticBindingSha256": semantic_binding,
            "binding": binding,
        }
    finally:
        for value, owned in (
            (materialized_raw, owns_materialized),
            (applied_raw, owns_applied),
        ):
            if owned and isinstance(value, dict):
                for item in value.get("providers", []):
                    if isinstance(item, dict):
                        item.clear()
                value.clear()


def _xuanju_binding_context(
    db_path: str | Path,
    source: Mapping[str, Any],
    xuanju_source: str,
) -> dict[str, Any]:
    return _selected_binding_context(db_path, source, xuanju_source)
def _final_selected_binding_contexts(
    db_path: str | Path,
    source: Mapping[str, Any],
    source_ids: set[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Build v4 bindings around one final private source capture.

    Public semantic views are captured both before and after the private
    apply view.  The post-capture views are the ones published, while the
    private capture is retained for rendering credentials from that same
    source read.  This closes the final-window race where selected semantic
    state changed during the private capture.
    """
    if source_ids != {_DY_SOURCE_ID, _XUANJU_SOURCE_ID}:
        _reject("blocked-source-read")
    current_public_before: dict[str, Any] | None = None
    materialized_before: dict[str, Any] | None = None
    current_public_after: dict[str, Any] | None = None
    materialized_after: dict[str, Any] | None = None
    applied: dict[str, Any] | None = None
    try:
        current_public_before = _planning_source(db_path, source_ids)
        try:
            materialized_before = materialization_source(
                db_path,
                selected_source_ids=source_ids,
                approved_origin_source_ids=source_ids,
            )
            applied = _apply_source(db_path, source_ids)
            current_public_after = _planning_source(db_path, source_ids)
            materialized_after = materialization_source(
                db_path,
                selected_source_ids=source_ids,
                approved_origin_source_ids=source_ids,
            )
        except SourceInventoryError as error:
            _reject(error.code)
        except Exception:
            _reject("blocked-source-read")
        contexts = {
            source_id: _selected_binding_context(
                db_path,
                source,
                source_id,
                current_public_source=current_public_after,
                materialized_source=materialized_after,
                applied_source=applied,
            )
            for source_id in (_DY_SOURCE_ID, _XUANJU_SOURCE_ID)
        }
        if not isinstance(applied, dict):
            _reject("blocked-source-read")
        retained = applied
        applied = None
        return contexts, retained
    finally:
        for value in (
            current_public_before,
            materialized_before,
            current_public_after,
            materialized_after,
            applied,
        ):
            if isinstance(value, dict):
                for item in value.get("providers", []):
                    if isinstance(item, dict):
                        item.clear()
                value.clear()


def _dy_binding_context(
    db_path: str | Path,
    source: Mapping[str, Any],
    dy_source: str,
) -> dict[str, Any]:
    return _selected_binding_context(db_path, source, dy_source)


def _migration_inputs(
    provider_map: Mapping[str, str],
    selected_models: Mapping[str, Sequence[str]],
    approved: set[str] | frozenset[str],
    enable_xuanju: bool,
) -> tuple[str, str]:
    if enable_xuanju is not True:
        _reject("blocked-xuanju-opt-in")
    if not isinstance(provider_map, Mapping) or not isinstance(selected_models, Mapping):
        _reject("blocked-provider-mapping")
    normalized_map = {
        _identifier(source, "blocked-provider-mapping"): _identifier(
            target, "blocked-provider-mapping"
        )
        for source, target in provider_map.items()
    }
    if (
        len(normalized_map) != 2
        or len(normalized_map) != len(provider_map)
        or set(normalized_map.values()) != {_DY_TARGET, _XUANJU_TARGET}
    ):
        _reject("blocked-provider-mapping")
    dy_source = next(
        source for source, target in normalized_map.items() if target == _DY_TARGET
    )
    if dy_source != _DY_SOURCE_ID:
        _reject("blocked-provider-mapping")
    xuanju_source = next(
        source for source, target in normalized_map.items() if target == _XUANJU_TARGET
    )
    if xuanju_source != _XUANJU_SOURCE_ID:
        _reject("blocked-provider-mapping")
    if set(selected_models) != {dy_source, xuanju_source}:
        _reject("blocked-model-selection")
    # DY is a discovery-only provider.  The caller may retain an existing
    # selection list for audit context (or pass an empty list), but no DY model
    # identifier is copied into the target projection or config handoff.
    _model_list_allow_empty(selected_models[dy_source], "blocked-model-selection")
    if list(selected_models[xuanju_source]) != list(_XUANJU_MODELS):
        _reject("blocked-model-selection")
    if not isinstance(approved, (set, frozenset)) or set(approved) != {dy_source, xuanju_source}:
        _reject("blocked-origin-approval")
    return dy_source, xuanju_source


def _planning_actions(
    source: Mapping[str, Any],
    dy_source: str,
    xuanju_source: str,
    *,
    target_document: Mapping[str, Any],
    dy_binding: Mapping[str, str],
    xuanju_binding: Mapping[str, str],
) -> list[dict[str, Any]]:
    if dy_source != _DY_SOURCE_ID or xuanju_source != _XUANJU_SOURCE_ID:
        _reject("blocked-source-read")
    records = source["providers"]
    by_id: dict[str, Mapping[str, Any]] = {}
    for record in records:
        _exact(
            record,
            {
                "source_id",
                "app_type",
                "is_current",
                "api_family",
                "endpoint_https",
                "endpoint_sha256",
                "apply_binding_sha256",
                "credential_reference_present",
                "declared_models",
                "environment_models",
                "blocker_codes",
            },
            "blocked-source-read",
        )
        source_id = _identifier(record["source_id"], "blocked-source-read")
        if source_id in by_id or record["app_type"] != "claude":
            _reject("blocked-source-read")
        by_id[source_id] = record
    if set(by_id) != {dy_source, xuanju_source}:
        _reject("blocked-source-read")
    providers = target_document.get("providers")
    if not isinstance(providers, Mapping):
        _reject("blocked-models-migration-state")
    actions: list[dict[str, Any]] = []
    for source_id, target, expected_family in (
        (dy_source, _DY_TARGET, "openai_responses"),
        (xuanju_source, _XUANJU_TARGET, None),
    ):
        record = by_id[source_id]
        if (
            (
                record["is_current"] is not True
                and not (
                    target == _DY_TARGET
                    and source_id == _DY_SOURCE_ID
                )
            )
            or not isinstance(record["blocker_codes"], list)
        ):
            _reject("blocked-source-provider")
        blockers = record["blocker_codes"]
        allowed_blockers = (
            not blockers or blockers == ["blocked-source-model-missing"]
        )
        if (
            not allowed_blockers
            or record["endpoint_https"] is not True
            or record["credential_reference_present"] is not True
            or (
                expected_family is not None
                and record["api_family"] != expected_family
            )
            or (
                expected_family is None
                and record["api_family"] not in _XUANJU_SOURCE_API_FAMILIES
            )
            or not _is_digest(record["endpoint_sha256"])
            or not _is_digest(record["apply_binding_sha256"])
        ):
            _reject(
                "blocked-api-family-mismatch"
                if (
                    expected_family is not None
                    and record["api_family"] != expected_family
                )
                else "blocked-source-provider"
            )
        declared = _model_list_allow_empty(
            record["declared_models"], "blocked-source-read"
        )
        if target == _XUANJU_TARGET:
            endpoint = xuanju_binding.get("siteRoot")
            semantic_binding = xuanju_binding.get("semanticBindingSha256")
            if (
                not isinstance(endpoint, str)
                or not _is_digest(semantic_binding)
                or hashlib.sha256(
                    xuanju_binding.get("endpoint", "").encode("utf-8")
                ).hexdigest()
                != record["endpoint_sha256"]
            ):
                _reject("blocked-source-changed")
            target_provider = providers.get(target)
            desired = {
                "baseUrl": endpoint,
                "apiKey": _XUANJU_ENV_KEY,
                "api": _XUANJU_REQUEST_API,
                "models": [{"id": _XUANJU_MODELS[0]}],
            }
            operation = (
                "create"
                if target_provider is None
                else "retain"
                if target_provider == desired
                else "update"
            )
            endpoint_sha256 = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
        else:
            endpoint = dy_binding.get("providerBase")
            if not isinstance(endpoint, str):
                _reject("blocked-source-changed")
            target_provider = providers.get(target)
            operation = "update"
            endpoint_sha256 = hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            semantic_binding = None
        action: dict[str, Any] = {
            "action": operation,
            "sourceId": source_id,
            "target": target,
            "sourceApiFamily": record["api_family"],
            "targetApi": (
                _XUANJU_REQUEST_API
                if target == _XUANJU_TARGET
                else _SOURCE_API_TARGET[record["api_family"]]
            ),
            "endpoint": {"https": True, "sha256": endpoint_sha256},
            "applyBindingSha256": record["apply_binding_sha256"],
            "sourceProviderSha256": _digest(record, "blocked-source-read"),
            "declaredModels": declared,
            "probeVerifiedModels": (
                list(_XUANJU_MODELS) if target == _XUANJU_TARGET else []
            ),
        }
        if target == _DY_TARGET:
            action["discovery"] = dict(_DY_DISCOVERY)
        else:
            action["semanticBindingSha256"] = semantic_binding
        actions.append(action)
    return actions
def _selected_source_contract(
    source: Mapping[str, Any],
    dy_binding: Mapping[str, Any],
    xuanju_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the v4 source fragment and its selected-provider precondition."""
    try:
        dy_record = dy_binding["binding"]
        xuanju_record = xuanju_binding["binding"]
    except (KeyError, TypeError):
        _reject("blocked-source-read")
    selected_bindings = {
        _DY_SOURCE_ID: copy.deepcopy(dy_record),
        _XUANJU_SOURCE_ID: copy.deepcopy(xuanju_record),
    }
    selected_semantic = _digest(selected_bindings, "blocked-source-read")
    return (
        {"appType": "claude", "selectedSemanticSha256": selected_semantic},
        {
            "mainIdentity": _source_main_identity(
                source.get("precondition"), "blocked-source-read"
            ),
            "selectedBindings": selected_bindings,
        },
    )


def _actions_match(
    expected: Sequence[Mapping[str, Any]],
    actual: Sequence[Mapping[str, Any]],
    *,
    allow_retain: bool = False,
) -> bool:
    if len(expected) != len(actual):
        return False
    for left, right in zip(expected, actual):
        if dict(left) == dict(right):
            continue
        if not allow_retain or right.get("action") != "retain":
            return False
        left_copy = dict(left)
        right_copy = dict(right)
        left_copy.pop("action", None)
        right_copy.pop("action", None)
        if left_copy != right_copy:
            return False
    return True


def _assert_probe_coverage(
    probes: Sequence[Mapping[str, Any]],
    actions: Sequence[Mapping[str, Any]],
) -> None:
    """Require the current ten-call Responses artifact and flash success."""
    if len(probes) != 1:
        _reject("blocked-probe-evidence")
    xuanju_action = next(
        (action for action in actions if action["target"] == _XUANJU_TARGET),
        None,
    )
    if xuanju_action is None:
        _reject("blocked-probe-evidence")
    probe = probes[0]
    evidence = probe.get("providerEvidence")
    if (
        not isinstance(evidence, list)
        or len(evidence) != 1
        or not isinstance(evidence[0], Mapping)
        or evidence[0].get("sourceId") != xuanju_action["sourceId"]
        or evidence[0].get("semanticBindingSha256")
        != xuanju_action.get("semanticBindingSha256")
        or probe.get("semanticBindingSha256")
        != xuanju_action.get("semanticBindingSha256")
    ):
        _reject("blocked-probe-source-changed")
    observations = probe.get("observations")
    if not isinstance(observations, list):
        _reject("blocked-probe-evidence")
    required = {
        (xuanju_action["sourceId"], _XUANJU_MODELS[0], mode)
        for mode in _XUANJU_PROBE_MODES
    }
    successful: set[tuple[str, str, str]] = set()
    for observation in observations:
        if (
            observation.get("status") == "usable"
            and observation.get("usable") is True
            and observation.get("identityStatus") == "observed-match"
            and observation.get("observedModel") == _XUANJU_MODELS[0]
            and observation.get("toolResult")
            == (
                "smoke_ok"
                if observation.get("mode") == "tool"
                else None
            )
        ):
            successful.add(
                (
                    observation.get("sourceId"),
                    observation.get("model"),
                    observation.get("mode"),
                )
            )
    if not required.issubset(successful):
        _reject("blocked-probe-evidence")


def _env_keys(snapshot: _Snapshot) -> Mapping[str, int]:
    parsed = snapshot.parsed
    if not isinstance(parsed, Mapping) or not isinstance(parsed.get("keys"), Mapping):
        _reject("blocked-environment-parse")
    return parsed["keys"]


def _provider_api_key_references(
    document: Mapping[str, Any], code: str
) -> dict[str, list[str]]:
    """Return every provider credential/header value resolved by OMP.

    OMP resolves provider ``apiKey`` plus provider, static-model, and
    model-override header values through its environment resolver.  Inventory
    all of those values before replacing either migration key so an unrelated
    route cannot inherit the new credential or lose the old one.
    """
    providers = document.get("providers")
    if not isinstance(providers, Mapping):
        _reject(code)
    references: dict[str, list[str]] = {}
    provider_keys: set[str] = set()

    def collect_string(values: list[str], value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, str):
            _reject(code)
        values.append(value)

    def collect_headers(values: list[str], value: Any) -> None:
        if value is None:
            return
        if not isinstance(value, Mapping):
            _reject(code)
        for header_value in value.values():
            collect_string(values, header_value)

    for provider_name, provider in providers.items():
        if not isinstance(provider_name, str) or not isinstance(provider, Mapping):
            _reject(code)
        normalized_name = provider_name.casefold()
        if (
            normalized_name in provider_keys
            or (
                normalized_name in {_DY_TARGET, _XUANJU_TARGET}
                and provider_name != normalized_name
            )
        ):
            _reject(code)
        provider_keys.add(normalized_name)
        values: list[str] = []
        collect_string(values, provider.get("apiKey"))
        collect_headers(values, provider.get("headers"))

        models = provider.get("models", [])
        if not isinstance(models, list):
            _reject(code)
        for model in models:
            if not isinstance(model, Mapping):
                _reject(code)
            collect_headers(values, model.get("headers"))

        overrides = provider.get("modelOverrides", {})
        if not isinstance(overrides, Mapping):
            _reject(code)
        for override in overrides.values():
            if not isinstance(override, Mapping):
                _reject(code)
            collect_headers(values, override.get("headers"))
        references[provider_name] = values
    return references


def _validate_provider_api_key_references(
    document: Mapping[str, Any], *, post_migration: bool
) -> None:
    references = _provider_api_key_references(
        document, "blocked-models-migration-state"
    )
    for provider_name, values in references.items():
        for api_key in values:
            if api_key == _DY_ENV_KEY and provider_name != _DY_TARGET:
                _reject("blocked-environment-migration-state")
            if api_key == _XUANJU_ENV_KEY and provider_name != _XUANJU_TARGET:
                _reject("blocked-environment-migration-state")
            if api_key == _DEEPSEEK_ENV_KEY and (
                post_migration or provider_name != _XUANJU_TARGET
            ):
                _reject("blocked-environment-migration-state")


def _validate_initial_targets(
    models: _Snapshot,
    environment: _Snapshot,
    actions: Sequence[Mapping[str, Any]],
) -> None:
    document = models.parsed
    providers = document.get("providers") if isinstance(document, Mapping) else None
    if (
        not isinstance(providers, Mapping)
        or _DY_TARGET not in providers
        or not isinstance(providers[_DY_TARGET], Mapping)
    ):
        _reject("blocked-models-migration-state")
    dy_action = next(action for action in actions if action["target"] == _DY_TARGET)
    if (
        providers[_DY_TARGET].get("api") != dy_action["targetApi"]
        or providers[_DY_TARGET].get("apiKey") != _DY_ENV_KEY
    ):
        _reject("blocked-api-family-mismatch")
    keys = _env_keys(environment)
    if (
        _DEEPSEEK_ENV_KEY not in keys
        or _DY_ENV_KEY not in keys
        or _XUANJU_ENV_KEY in keys
    ):
        _reject("blocked-environment-migration-state")
    _validate_provider_api_key_references(document, post_migration=False)



def _config_handoff(config_path: str | Path, config_snapshot: _Snapshot) -> dict[str, Any]:
    try:
        handoff = build_provider_migration_handoff(config_path, omp_version=_OMP_VERSION)
    except ConfigHandoffError as error:
        _reject(error.code)
    _exact(handoff, {"preconditions", "changes"}, "blocked-config-read")
    target = handoff["preconditions"].get("target") if isinstance(handoff["preconditions"], Mapping) else None
    if not isinstance(target, Mapping):
        _reject("blocked-config-read")
    for key in {"realpath", "device", "fileId", "sha256", "mode", "ctimeNs"}:
        observed = target.get(key)
        if key in {"device", "fileId", "mode", "ctimeNs"}:
            observed = _closed_decimal(observed, "blocked-config-read")
        if observed != config_snapshot.precondition.get(key):
            _reject("blocked-config-changed")
    return dict(handoff["changes"])


def build_provider_migration_plan(
    db_path: str | Path,
    models_path: str | Path,
    config_path: str | Path,
    environment_path: str | Path,
    *,
    provider_map: Mapping[str, str],
    selected_models: Mapping[str, Sequence[str]],
    probe_evidence_path: str | Path | Sequence[str | Path],
    approved_current_origin_source_ids: set[str] | frozenset[str],
    enable_xuanju: bool,
    omp_version: str,
) -> dict[str, Any]:
    """Build a closed, redacted no-write migration plan."""
    _require_secure_platform()
    if omp_version != _OMP_VERSION:
        _reject("blocked-omp-version")
    package_path: str | Path = _OMP_PACKAGE_PATH
    _package_location, omp_package, omp_package_precondition = _snapshot_omp_package(
        package_path
    )
    dy_source, xuanju_source = _migration_inputs(
        provider_map,
        selected_models,
        approved_current_origin_source_ids,
        enable_xuanju,
    )
    source_ids = {dy_source, xuanju_source}
    source = _planning_source(db_path, source_ids)
    probes = _read_probes(probe_evidence_path)
    models_location, models = _snapshot_file(
        models_path, "models", _models_parser, exact_mode=True
    )
    config_location, config = _snapshot_file(
        config_path, "config", _config_parser, exact_mode=True
    )
    environment_location, environment = _snapshot_file(
        environment_path, "environment", _parse_environment, exact_mode=True
    )
    if len(
        {
            models_location.absolute_path,
            config_location.absolute_path,
            environment_location.absolute_path,
        }
    ) != 3:
        _reject("blocked-target-path")
    dy_binding = _dy_binding_context(db_path, source, dy_source)
    xuanju_binding = _xuanju_binding_context(db_path, source, xuanju_source)
    source_fragment, source_precondition = _selected_source_contract(
        source, dy_binding, xuanju_binding
    )
    actions = _planning_actions(
        source,
        dy_source,
        xuanju_source,
        target_document=models.parsed,
        dy_binding=dy_binding,
        xuanju_binding=xuanju_binding,
    )
    _assert_probe_coverage(probes, actions)
    _validate_initial_targets(models, environment, actions)
    config_changes = _config_handoff(config_path, config)
    plan: dict[str, Any] = {
        "schemaVersion": _SCHEMA_VERSION,
        "status": "planned",
        "planSha256": "",
        "ompPackage": {
            "name": omp_package.parsed["name"],
            "version": omp_package.parsed["version"],
            "precondition": omp_package_precondition,
        },
        "source": source_fragment,
        "probeEvidence": [
            {
                "schemaVersion": _PROBE_SCHEMA_VERSION,
                "artifactKind": probe["artifactKind"],
                "sourceId": probe["providerEvidence"][0]["sourceId"],
                "semanticBindingSha256": probe["semanticBindingSha256"],
                "sha256": probe["artifactSha256"],
            }
            for probe in probes
        ],
        "preconditions": {
            "source": source_precondition,
            "models": models.precondition,
            "config": config.precondition,
            "environment": environment.precondition,
        },
        "providerActions": actions,
        "credentialMigration": {
            "action": "replace",
            "removeEnvKey": _DEEPSEEK_ENV_KEY,
            "sourceId": xuanju_source,
            "sourceCredentialPresent": True,
            "writeEnvKey": _XUANJU_ENV_KEY,
        },
        "configActions": config_changes,
        "policy": {"allowOpaqueBackup": False, "enableXuanju": True, "require0600": True},
    }
    plan["planSha256"] = _plan_hash(plan)
    return validate_provider_migration_plan(plan)


def _validate_file_precondition(value: Any, code: str, *, exact_mode: bool) -> None:
    _exact(
        value,
        {"realpath", "device", "fileId", "sha256", "mode", "ctimeNs", "parent"},
        code,
    )
    if (
        type(value["realpath"]) is not str
        or not _is_digest(value["sha256"])
        or any(
            not _decimal_value(value[key])
            for key in ("device", "fileId", "mode", "ctimeNs")
        )
        or (exact_mode and value["mode"] != str(0o600))
    ):
        _reject(code)
    _exact(value["parent"], {"device", "fileId", "ctimeNs"}, code)
    if any(not _decimal_value(value["parent"][key]) for key in value["parent"]):
        _reject(code)


def _validate_omp_package_precondition(value: Any) -> None:
    _exact(
        value,
        {
            "realpath",
            "device",
            "fileId",
            "owner",
            "mode",
            "size",
            "ctimeNs",
            "sha256",
        },
        "blocked-plan-schema",
    )
    if (
        type(value["realpath"]) is not str
        or not _is_digest(value["sha256"])
        or any(
            not _decimal_value(value[key])
            for key in ("device", "fileId", "owner", "mode", "size", "ctimeNs")
        )
    ):
        _reject("blocked-plan-schema")


def _assert_omp_package_current(accepted: Mapping[str, Any]) -> None:
    package = accepted.get("ompPackage")
    if not isinstance(package, Mapping):
        _reject("blocked-omp-version")
    precondition = package.get("precondition")
    if not isinstance(precondition, Mapping):
        _reject("blocked-omp-version")
    try:
        location, snapshot, current = _snapshot_omp_package(_OMP_PACKAGE_PATH)
    except MigrationPlanError:
        _reject("blocked-omp-version")
    if (
        location.realpath != precondition.get("realpath")
        or snapshot.parsed
        != {"name": _OMP_PACKAGE_NAME, "version": _OMP_VERSION}
        or current != dict(precondition)
        or package.get("name") != _OMP_PACKAGE_NAME
        or package.get("version") != _OMP_VERSION
    ):
        _reject("blocked-omp-version")




def _validate_omp_package_runtime(plan: Mapping[str, Any]) -> None:
    _assert_omp_package_current(plan)


def _validate_source_main_identity(value: Any, code: str) -> None:
    _exact(value, {"realpath", "device", "fileId", "owner", "mode"}, code)
    if (
        not isinstance(value["realpath"], str)
        or not value["realpath"]
        or "\x00" in value["realpath"]
        or not os.path.isabs(value["realpath"])
        or os.path.normpath(value["realpath"]) != value["realpath"]
        or any(not _decimal_value(value[key]) for key in ("device", "fileId", "owner", "mode"))
        or int(value["fileId"]) == 0
        or int(value["mode"]) > 0o777
    ):
        _reject(code)


def _validate_source_model_definitions(value: Any, code: str) -> None:
    if not isinstance(value, list):
        _reject(code)
    try:
        indexed = _materializer._model_index(value, code)
        for model in indexed.values():
            _materializer._assert_source_model_secret_free(model, code)
    except MigrationPlanError:
        raise
    except Exception:
        _reject(code)


def _validate_source_environment_models(value: Any, code: str) -> None:
    _exact(value, {"configured_model_values", "name_to_configured_value"}, code)
    configured = _configured_model_values_allow_context(
        value["configured_model_values"], code
    )
    if configured != value["configured_model_values"]:
        _reject(code)
    names = value["name_to_configured_value"]
    if not isinstance(names, Mapping):
        _reject(code)
    for key, item in names.items():
        _configured_model_values_allow_context([key], code)
        _configured_model_values_allow_context([item], code)


def _validate_selected_binding(
    value: Any, source_id: str, code: str
) -> None:
    _exact(
        value,
        {
            "sourceId",
            "appType",
            "isCurrent",
            "apiFamily",
            "endpointHttps",
            "endpointSha256",
            "endpointOriginSha256",
            "credentialReferencePresent",
            "declaredModels",
            "environmentModels",
            "modelDefinitions",
            "blockerCodes",
            "applyBindingSha256",
            "semanticBindingSha256",
        },
        code,
    )
    if (
        value["sourceId"] != source_id
        or value["appType"] != "claude"
        or type(value["isCurrent"]) is not bool
        or (source_id == _XUANJU_SOURCE_ID and value["isCurrent"] is not True)
        or type(value["endpointHttps"]) is not bool
        or value["endpointHttps"] is not True
        or type(value["credentialReferencePresent"]) is not bool
        or value["credentialReferencePresent"] is not True
        or (
            source_id == _DY_SOURCE_ID
            and value["apiFamily"] != "openai_responses"
        )
        or (
            source_id == _XUANJU_SOURCE_ID
            and value["apiFamily"] not in _XUANJU_SOURCE_API_FAMILIES
        )
        or not _is_digest(value["endpointSha256"])
        or not _is_digest(value["endpointOriginSha256"])
        or not _is_digest(value["applyBindingSha256"])
        or not _is_digest(value["semanticBindingSha256"])
    ):
        _reject(code)
    declared = _model_list_allow_empty(value["declaredModels"], code)
    if declared != value["declaredModels"]:
        _reject(code)
    _validate_source_environment_models(value["environmentModels"], code)
    _validate_source_model_definitions(value["modelDefinitions"], code)
    blockers = value["blockerCodes"]
    if (
        not isinstance(blockers, list)
        or len(blockers) != len(set(blockers))
        or any(
            not isinstance(item, str)
            or not item.startswith("blocked-")
            or not _IDENTIFIER_RE.fullmatch(item)
            for item in blockers
        )
    ):
        _reject(code)
    semantic = {
        "sourceId": value["sourceId"],
        "appType": value["appType"],
        "isCurrent": value["isCurrent"],
        "apiFamily": value["apiFamily"],
        "endpointOriginSha256": value["endpointOriginSha256"],
        "applyBindingSha256": value["applyBindingSha256"],
        "credentialReferencePresent": value["credentialReferencePresent"],
        "declaredModels": list(value["declaredModels"]),
        "environmentModels": copy.deepcopy(value["environmentModels"]),
        "modelDefinitions": [
            copy.deepcopy(item) for item in value["modelDefinitions"]
        ],
        "blockerCodes": list(value["blockerCodes"]),
    }
    if value["semanticBindingSha256"] != _digest(semantic, code):
        _reject(code)


def _validate_source_precondition(value: Any) -> None:
    _exact(value, {"mainIdentity", "selectedBindings"}, "blocked-plan-schema")
    _validate_source_main_identity(value["mainIdentity"], "blocked-plan-schema")
    selected = value["selectedBindings"]
    if not isinstance(selected, Mapping) or set(selected) != {
        _DY_SOURCE_ID,
        _XUANJU_SOURCE_ID,
    }:
        _reject("blocked-plan-schema")
    for source_id in (_DY_SOURCE_ID, _XUANJU_SOURCE_ID):
        _validate_selected_binding(selected[source_id], source_id, "blocked-plan-schema")


def validate_provider_migration_plan(
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly validate a serializable provider migration plan."""
    if not isinstance(plan, Mapping):
        _reject("blocked-plan-schema")
    try:
        normalized = json.loads(
            _canonical(plan, "blocked-plan-schema").decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError, RecursionError):
        _reject("blocked-plan-schema")
    _exact(
        normalized,
        {
            "schemaVersion",
            "status",
            "planSha256",
            "ompPackage",
            "source",
            "probeEvidence",
            "preconditions",
            "providerActions",
            "credentialMigration",
            "configActions",
            "policy",
        },
        "blocked-plan-schema",
    )
    if (
        normalized["schemaVersion"] != _SCHEMA_VERSION
        or normalized["status"] != "planned"
        or not _is_digest(normalized["planSha256"])
    ):
        _reject("blocked-plan-schema")
    omp_package = _exact(
        normalized["ompPackage"],
        {"name", "version", "precondition"},
        "blocked-plan-schema",
    )
    if (
        omp_package["name"] != _OMP_PACKAGE_NAME
        or omp_package["version"] != _OMP_VERSION
    ):
        _reject("blocked-plan-schema")
    _validate_omp_package_precondition(omp_package["precondition"])
    _validate_omp_package_runtime(normalized)
    if normalized["policy"] != {
        "allowOpaqueBackup": False,
        "enableXuanju": True,
        "require0600": True,
    }:
        _reject("blocked-plan-schema")
    source = _exact(
        normalized["source"],
        {"appType", "selectedSemanticSha256"},
        "blocked-plan-schema",
    )
    if source["appType"] != "claude" or not _is_digest(
        source["selectedSemanticSha256"]
    ):
        _reject("blocked-plan-schema")
    probes = normalized["probeEvidence"]
    if not isinstance(probes, list) or len(probes) != 1:
        _reject("blocked-plan-schema")
    probe = _exact(
        probes[0],
        {
            "schemaVersion",
            "artifactKind",
            "sourceId",
            "semanticBindingSha256",
            "sha256",
        },
        "blocked-plan-schema",
    )
    if (
        probe["schemaVersion"] != _PROBE_SCHEMA_VERSION
        or probe["artifactKind"] != _XUANJU_ARTIFACT_KIND
        or not _is_digest(probe["semanticBindingSha256"])
        or not _is_digest(probe["sha256"])
    ):
        _reject("blocked-plan-schema")
    probe_source_id = _identifier(probe["sourceId"], "blocked-plan-schema")
    if probe_source_id != _XUANJU_SOURCE_ID:
        _reject("blocked-plan-schema")
    preconditions = _exact(
        normalized["preconditions"],
        {"source", "models", "config", "environment"},
        "blocked-plan-schema",
    )
    _validate_source_precondition(preconditions["source"])
    if source["selectedSemanticSha256"] != _digest(
        preconditions["source"]["selectedBindings"], "blocked-plan-schema"
    ):
        _reject("blocked-plan-schema")
    for kind in ("models", "config", "environment"):
        _validate_file_precondition(
            preconditions[kind], "blocked-plan-schema", exact_mode=True
        )
    actions = normalized["providerActions"]
    if not isinstance(actions, list) or len(actions) != 2:
        _reject("blocked-plan-schema")
    expected_targets = (_DY_TARGET, _XUANJU_TARGET)
    source_ids: set[str] = set()
    xuanju_action: Mapping[str, Any] | None = None
    for action, target in zip(actions, expected_targets):
        base_keys = {
            "action",
            "sourceId",
            "target",
            "sourceApiFamily",
            "targetApi",
            "endpoint",
            "applyBindingSha256",
            "sourceProviderSha256",
            "declaredModels",
            "probeVerifiedModels",
        }
        required_keys = (
            base_keys | {"discovery"}
            if target == _DY_TARGET
            else base_keys | {"semanticBindingSha256"}
        )
        _exact(action, required_keys, "blocked-plan-schema")
        source_id = _identifier(action["sourceId"], "blocked-plan-schema")
        expected_source_id = (
            _DY_SOURCE_ID if target == _DY_TARGET else _XUANJU_SOURCE_ID
        )
        if source_id != expected_source_id or source_id in source_ids:
            _reject("blocked-plan-schema")
        source_ids.add(source_id)
        if action["target"] != target:
            _reject("blocked-plan-schema")
        if target == _DY_TARGET:
            if (
                action["action"] not in {"update", "retain"}
                or action["sourceApiFamily"] != "openai_responses"
                or action["targetApi"] != _XUANJU_REQUEST_API
                or action["probeVerifiedModels"] != []
                or action["discovery"] != _DY_DISCOVERY
            ):
                _reject("blocked-plan-schema")
        else:
            xuanju_action = action
            if (
                action["action"] not in {"create", "update", "retain"}
                or action["sourceApiFamily"] not in _XUANJU_SOURCE_API_FAMILIES
                or action["targetApi"] != _XUANJU_REQUEST_API
                or action["probeVerifiedModels"] != list(_XUANJU_MODELS)
                or not _is_digest(action["semanticBindingSha256"])
            ):
                _reject("blocked-plan-schema")
        endpoint = _exact(
            action["endpoint"],
            {"https", "sha256"},
            "blocked-plan-schema",
        )
        if (
            endpoint["https"] is not True
            or not _is_digest(endpoint["sha256"])
            or endpoint["sha256"]
            != (
                _DY_PROVIDER_BASE_SHA256
                if target == _DY_TARGET
                else _XUANJU_PROVIDER_BASE_SHA256
            )
            or not _is_digest(action["applyBindingSha256"])
            or not _is_digest(action["sourceProviderSha256"])
        ):
            _reject("blocked-plan-schema")
        declared = _model_list_allow_empty(
            action["declaredModels"], "blocked-plan-schema"
        )
        if declared != action["declaredModels"]:
            _reject("blocked-plan-schema")
        verified = _model_list_allow_empty(
            action["probeVerifiedModels"], "blocked-plan-schema"
        )
        if verified != action["probeVerifiedModels"]:
            _reject("blocked-plan-schema")
        selected_binding = preconditions["source"]["selectedBindings"][source_id]
        if (
            action["sourceApiFamily"] != selected_binding["apiFamily"]
            or action["applyBindingSha256"]
            != selected_binding["applyBindingSha256"]
            or action["declaredModels"] != selected_binding["declaredModels"]
            or (
                target == _XUANJU_TARGET
                and action["semanticBindingSha256"]
                != selected_binding["semanticBindingSha256"]
            )
        ):
            _reject("blocked-plan-schema")
    if (
        xuanju_action is None
        or probe_source_id != xuanju_action["sourceId"]
        or probe["semanticBindingSha256"]
        != xuanju_action["semanticBindingSha256"]
    ):
        _reject("blocked-probe-source-changed")
    credential = _exact(
        normalized["credentialMigration"],
        {
            "action",
            "removeEnvKey",
            "sourceId",
            "sourceCredentialPresent",
            "writeEnvKey",
        },
        "blocked-plan-schema",
    )
    if credential != {
        "action": "replace",
        "removeEnvKey": _DEEPSEEK_ENV_KEY,
        "sourceId": actions[1]["sourceId"],
        "sourceCredentialPresent": True,
        "writeEnvKey": _XUANJU_ENV_KEY,
    }:
        _reject("blocked-plan-schema")
    changes = _exact(
        normalized["configActions"],
        {
            "addEnabledModels",
            "addProviderOrder",
            "enabledModels",
            "fallbackChainsSha256",
            "modelProviderOrder",
            "modelRolesSha256",
            "preserveFallbackChains",
            "preserveModelRoles",
        },
        "blocked-plan-schema",
    )
    if (
        changes["preserveFallbackChains"] is not True
        or changes["preserveModelRoles"] is not True
        or not _is_digest(changes["fallbackChainsSha256"])
        or not _is_digest(changes["modelRolesSha256"])
    ):
        _reject("blocked-plan-schema")
    add_enabled = changes["addEnabledModels"]
    add_order = changes["addProviderOrder"]
    if add_enabled not in ([], list(_NEW_SELECTORS)) or add_order not in (
        [],
        [_XUANJU_TARGET],
    ):
        _reject("blocked-plan-schema")
    enabled = _exact(
        changes["enabledModels"],
        {"operation", "selectors", "result"},
        "blocked-plan-schema",
    )
    order = _exact(
        changes["modelProviderOrder"],
        {"operation", "providers", "after", "result"},
        "blocked-plan-schema",
    )
    if (
        enabled["operation"] not in {"append", "retain"}
        or order["operation"] not in {"insert-after", "retain"}
        or not isinstance(enabled["result"], list)
        or not isinstance(order["result"], list)
        or order["after"] != _DY_TARGET
        or (
            enabled["selectors"] != list(_NEW_SELECTORS)
            if enabled["operation"] == "append"
            else enabled["selectors"] != []
        )
        or (
            order["providers"] != [_XUANJU_TARGET]
            if order["operation"] == "insert-after"
            else order["providers"] != []
        )
    ):
        _reject("blocked-plan-schema")
    _model_list_allow_empty(enabled["selectors"], "blocked-plan-schema")
    if any(
        not isinstance(provider, str) or not _IDENTIFIER_RE.fullmatch(provider)
        for provider in order["result"]
    ):
        _reject("blocked-plan-schema")
    if any(
        not isinstance(selector, str)
        or not _MODEL_RE.fullmatch(selector)
        for selector in enabled["result"]
    ):
        _reject("blocked-plan-schema")
    _validate_target_provider_order(order["result"])
    _validate_enabled_selector_routes(enabled["result"])
    order_keys = [provider.casefold() for provider in order["result"]]
    enabled_keys = {_selector_key(selector) for selector in enabled["result"]}
    if (
        _XUANJU_TARGET not in order_keys
        or order_keys.count(_XUANJU_TARGET) != 1
        or _DY_TARGET not in order_keys
        or (
            order["operation"] == "insert-after"
            and order_keys.index(_XUANJU_TARGET)
            != order_keys.index(_DY_TARGET) + 1
        )
        or _selector_key(_NEW_SELECTORS[0]) not in enabled_keys
        or sum(
            _selector_key(selector) == _selector_key(_NEW_SELECTORS[0])
            for selector in enabled["result"]
        )
        != 1
        or not all(
            _selector_key(selector) in enabled_keys
            for selector in (
                f"{_DY_TARGET}/gpt-5.6",
                f"{_DY_TARGET}/gpt-5.6-sol",
            )
        )
        or add_enabled
        != ([] if enabled["operation"] == "retain" else list(_NEW_SELECTORS))
        or add_order
        != ([] if order["operation"] == "retain" else [_XUANJU_TARGET])
    ):
        _reject("blocked-plan-schema")
    if normalized["planSha256"] != _plan_hash(normalized):
        _reject("blocked-plan-hash")
    return normalized


def load_provider_migration_plan(path: str | Path) -> dict[str, Any]:
    """Load an exact canonical migration plan from a no-follow file."""
    _require_secure_platform()
    descriptor = -1
    try:
        raw_path = os.fspath(path)
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        if not isinstance(raw_path, str) or not raw_path or "\x00" in raw_path:
            raise OSError
        descriptor = os.open(
            raw_path,
            os.O_RDONLY
            | _materializer._close_on_exec_flag()
            | _materializer._no_follow_flag()
            | _materializer._non_blocking_flag(),
        )
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_size > _MAX_PROBE_BYTES
            or stat.S_IMODE(details.st_mode) & 0o022
        ):
            _reject("blocked-plan-load")
        payload = _read_bytes(descriptor, _MAX_PROBE_BYTES, "blocked-plan-load")
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
        validated = validate_provider_migration_plan(parsed)
        canonical = _canonical(validated, "blocked-plan-schema")
        if payload not in {canonical, canonical + b"\n"}:
            _reject("blocked-plan-canonical")
        return validated
    except MigrationPlanError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        _reject("blocked-plan-load")
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _source_ids(plan: Mapping[str, Any]) -> tuple[str, str]:
    actions = plan["providerActions"]
    dy_source, xuanju_source = actions[0]["sourceId"], actions[1]["sourceId"]
    if (
        dy_source != _DY_SOURCE_ID
        or xuanju_source != _XUANJU_SOURCE_ID
    ):
        _reject("blocked-plan-schema")
    return dy_source, xuanju_source


def _assert_source_current(
    plan: Mapping[str, Any],
    source: Mapping[str, Any],
    selected_bindings: Mapping[str, Any] | None = None,
) -> None:
    expected = plan["preconditions"]["source"]
    try:
        actual_identity = _source_main_identity(
            source.get("precondition"), "blocked-source-read"
        )
    except MigrationPlanError:
        _reject("blocked-source-changed")
    if (
        source.get("source_app_type") != "claude"
        or actual_identity != expected["mainIdentity"]
        or (
            selected_bindings is not None
            and dict(selected_bindings) != dict(expected["selectedBindings"])
        )
    ):
        _reject("blocked-source-changed")


def _apply_records(source: Mapping[str, Any], plan: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    _assert_source_current(plan, source)
    records: dict[str, Mapping[str, Any]] = {}
    for record in source["providers"]:
        _exact(
            record,
            {
                "source_id",
                "api_family",
                "endpoint",
                "endpoint_sha256",
                "apply_binding_sha256",
                "credential_value",
            },
            "blocked-source-read",
        )
        source_id = _identifier(record["source_id"], "blocked-source-read")
        if (
            source_id in records
            or record["api_family"] not in _SOURCE_API_TARGET
            or not isinstance(record["endpoint"], str)
            or not isinstance(record["credential_value"], str)
            or not _is_digest(record["apply_binding_sha256"])
        ):
            _reject("blocked-source-read")
        if (
            hashlib.sha256(record["endpoint"].encode("utf-8")).hexdigest()
            != record["endpoint_sha256"]
        ):
            _reject("blocked-source-read")
        records[source_id] = record
    if set(records) != set(_source_ids(plan)):
        _reject("blocked-source-read")
    for action in plan["providerActions"]:
        record = records[action["sourceId"]]
        if action["target"] == _XUANJU_TARGET:
            endpoint = _xuanju_site_root(
                record["endpoint"], "blocked-source-url", record["api_family"]
            )
        else:
            endpoint = _dy_provider_base(record["endpoint"], "blocked-source-url")
        if (
            record["api_family"] != action["sourceApiFamily"]
            or hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            != action["endpoint"]["sha256"]
            or not _is_digest(record["apply_binding_sha256"])
            or record["apply_binding_sha256"] != action["applyBindingSha256"]
        ):
            _reject("blocked-source-changed")
    return records


def _render_models(
    raw: bytes, plan: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]
) -> bytes:
    document = _models_parser(raw)
    _validate_provider_api_key_references(document, post_migration=False)
    updated = copy.deepcopy(document)
    providers = updated.get("providers")
    if not isinstance(providers, dict) or _DY_TARGET not in providers:
        _reject("blocked-models-changed")
    try:
        dy_action = next(
            action for action in plan["providerActions"] if action["target"] == _DY_TARGET
        )
        xuanju_action = next(
            action
            for action in plan["providerActions"]
            if action["target"] == _XUANJU_TARGET
        )
    except (KeyError, StopIteration):
        _reject("blocked-models-changed")
    dy_source = records[dy_action["sourceId"]]
    xuanju_source = records[xuanju_action["sourceId"]]
    dy_existing = providers[_DY_TARGET]
    if (
        not isinstance(dy_existing, Mapping)
        or dy_existing.get("apiKey") != _DY_ENV_KEY
        or dy_action.get("discovery") != _DY_DISCOVERY
        or dy_action.get("targetApi") != _XUANJU_REQUEST_API
        or xuanju_action.get("targetApi") != _XUANJU_REQUEST_API
    ):
        _reject("blocked-models-changed")
    dy_endpoint = _dy_provider_base(dy_source["endpoint"], "blocked-source-url")
    if hashlib.sha256(dy_endpoint.encode("utf-8")).hexdigest() != dy_action[
        "endpoint"
    ]["sha256"]:
        _reject("blocked-source-changed")
    xuanju_endpoint = _xuanju_site_root(
        xuanju_source["endpoint"],
        "blocked-source-url",
        xuanju_source["api_family"],
    )
    if hashlib.sha256(xuanju_endpoint.encode("utf-8")).hexdigest() != xuanju_action[
        "endpoint"
    ]["sha256"]:
        _reject("blocked-source-changed")
    # Replace the selected providers with exact reviewed shapes.  DY remains
    # discovery-backed; Xuanju is a static single-model Responses provider.
    providers[_DY_TARGET] = {
        "baseUrl": dy_endpoint,
        "apiKey": _DY_ENV_KEY,
        "api": _XUANJU_REQUEST_API,
        "discovery": dict(_DY_DISCOVERY),
    }
    providers[_XUANJU_TARGET] = {
        "baseUrl": xuanju_endpoint,
        "apiKey": _XUANJU_ENV_KEY,
        "api": _XUANJU_REQUEST_API,
        "models": [{"id": _XUANJU_MODELS[0]}],
    }
    try:
        validate_document(updated, "blocked-models-render")
        rendered = (
            _materializer._GENERATED_HEADER
            + "\n"
            + yaml.safe_dump(updated, sort_keys=False, allow_unicode=True)
        ).encode("utf-8")
        _models_parser(rendered)
        return rendered
    except MigrationPlanError:
        raise
    except Exception:
        _reject("blocked-models-render")


def _render_environment(raw: bytes, credential: str) -> bytes:
    parsed = _parse_environment(raw)
    lines = parsed["lines"]
    keys = parsed["keys"]
    if _DEEPSEEK_ENV_KEY not in keys or _XUANJU_ENV_KEY in keys:
        _reject("blocked-environment-changed")
    if (
        not isinstance(credential, str)
        or not credential.strip()
        or "\x00" in credential
        or "\n" in credential
        or "\r" in credential
        or len(credential.encode("utf-8")) > 32_768
    ):
        _reject("blocked-source-credential")
    retained = [line for index, line in enumerate(lines) if index != keys[_DEEPSEEK_ENV_KEY]]
    text = "".join(retained)
    if text and not text.endswith("\n"):
        text += "\n"
    return (text + f"{_XUANJU_ENV_KEY}={shlex.quote(credential)}\n").encode("utf-8")


def _render_all(locked: Mapping[str, _LockedFile], plan: Mapping[str, Any], records: Mapping[str, Mapping[str, Any]]) -> dict[str, bytes]:
    try:
        config = render_provider_migration_config(
            locked["config"].snapshot.raw, plan["configActions"]
        )
    except ConfigHandoffError as error:
        _reject(error.code)
    return {
        "models": _render_models(locked["models"].snapshot.raw, plan, records),
        "config": config,
        "environment": _render_environment(
            locked["environment"].snapshot.raw,
            records[plan["credentialMigration"]["sourceId"]]["credential_value"],
        ),
    }


def _recovery_manifest(
    locked: Mapping[str, _LockedFile], plan_sha256: str
) -> bytes:
    if not _is_digest(plan_sha256):
        _reject("blocked-rollback-path")
    files: dict[str, dict[str, str]] = {}
    for kind in ("models", "config", "environment"):
        snapshot = locked[kind].snapshot
        mode = snapshot.precondition.get("mode")
        if not _decimal_value(mode):
            _reject("blocked-rollback-path")
        files[kind] = {
            "mode": mode,
            "sha256": hashlib.sha256(snapshot.raw).hexdigest(),
            "size": str(len(snapshot.raw)),
        }
    return _canonical(
        {
            "schemaVersion": 1,
            "planSha256": plan_sha256,
            "files": files,
        },
        "blocked-rollback-path",
    )


def _write_backup_set(
    locked: Mapping[str, _LockedFile], plan_sha256: str
) -> tuple[int, int, str, dict[str, tuple[int, str]]]:
    root_fd = -1
    run_fd = -1
    run_name = ""
    backups: dict[str, tuple[int, str]] = {}
    try:
        root_fd = _materializer._secure_rollback_root(locked["models"].parent_fd)
        run_name, run_fd = _materializer._create_run_directory(root_fd)
        for kind in ("models", "config", "environment"):
            name = f"{kind}.opaque"
            descriptor = _materializer._write_exclusive_file(
                run_fd, name, locked[kind].snapshot.raw, 0o600
            )
            backups[kind] = (descriptor, name)
        manifest_name = "manifest.json"
        manifest_descriptor = _materializer._write_exclusive_file(
            run_fd, manifest_name, _recovery_manifest(locked, plan_sha256), 0o600
        )
        backups["manifest"] = (manifest_descriptor, manifest_name)
        _materializer._fsync_directory(run_fd)
        _materializer._fsync_directory(root_fd)
        return root_fd, run_fd, run_name, backups
    except Exception:
        if run_fd != -1:
            for descriptor, name in backups.values():
                _materializer._unlink_at_quietly(run_fd, name)
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            with contextlib.suppress(OSError):
                _materializer._fsync_directory(run_fd)
            with contextlib.suppress(OSError):
                os.close(run_fd)
        if root_fd != -1:
            if run_name:
                with contextlib.suppress(OSError):
                    _materializer._rmdir_at(root_fd, run_name)
            with contextlib.suppress(OSError):
                _materializer._fsync_directory(root_fd)
            with contextlib.suppress(OSError):
                os.close(root_fd)
        _reject("blocked-rollback-path")


def _prepare_replacements(locked: Mapping[str, _LockedFile], rendered: Mapping[str, bytes]) -> dict[str, Any]:
    replacements: dict[str, Any] = {}
    current: Any | None = None
    current_kind = ""
    try:
        for kind in ("models", "config", "environment"):
            current_kind = kind
            current = _materializer._write_target_temp(
                locked[kind].parent_fd, locked[kind].location.basename, rendered[kind]
            )
            _materializer._acquire_file_lock(
                current.descriptor, f"blocked-{kind}-lock"
            )
            replacements[kind] = current
            current = None
        return replacements
    except Exception:
        if current is not None:
            if current.name is not None:
                _materializer._unlink_at_quietly(
                    locked[current_kind].parent_fd, current.name
                )
            _materializer._release_file_lock(current.descriptor)
            with contextlib.suppress(OSError):
                os.close(current.descriptor)
        for kind, replacement in replacements.items():
            if replacement.name is not None:
                _materializer._unlink_at_quietly(
                    locked[kind].parent_fd, replacement.name
                )
            _materializer._release_file_lock(replacement.descriptor)
            with contextlib.suppress(OSError):
                os.close(replacement.descriptor)
        _reject("blocked-target-write")


def _replace_published_target(locked: _LockedFile, replacement: Any) -> None:
    """Perform only rename; caller records recovery state before any fsync."""
    _assert_publishable(locked)
    if replacement.name is None:
        _reject("blocked-target-write")
    try:
        os.replace(
            replacement.name,
            locked.location.basename,
            src_dir_fd=locked.parent_fd,
            dst_dir_fd=locked.parent_fd,
        )
        replacement.name = None
    except OSError:
        _reject("blocked-target-write")


def _finalize_published_target(locked: _LockedFile, replacement: Any) -> None:
    """Durably finalize a target which is already recorded as published."""
    try:
        os.fchmod(replacement.descriptor, 0o600)
        os.fsync(replacement.descriptor)
        _materializer._fsync_directory(locked.parent_fd)
    except OSError:
        _reject("blocked-target-write")


def _backup_bytes(descriptor: int, kind: str) -> bytes:
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
        return _read_bytes(descriptor, _MAX_FILE_BYTES, "blocked-rollback")
    except MigrationPlanError:
        raise
    except OSError:
        _reject("blocked-rollback")


def _rollback(published: Sequence[_PublishedFile], backups: Mapping[str, tuple[int, str]]) -> None:
    for item in reversed(published):
        locked = item.locked
        replacement = item.replacement
        restore = None
        try:
            _materializer._assert_target_path_matches_descriptor(
                locked.parent_fd, locked.location, replacement.descriptor
            )
            restore = _materializer._write_target_temp(
                locked.parent_fd,
                locked.location.basename,
                _backup_bytes(backups[locked.kind][0], locked.kind),
            )
            _materializer._acquire_file_lock(restore.descriptor, "blocked-rollback")
            if restore.name is None:
                _reject("blocked-rollback")
            os.replace(
                restore.name,
                locked.location.basename,
                src_dir_fd=locked.parent_fd,
                dst_dir_fd=locked.parent_fd,
            )
            restore.name = None
            os.fchmod(
                restore.descriptor, int(locked.snapshot.precondition["mode"])
            )
            os.fsync(restore.descriptor)
            _materializer._fsync_directory(locked.parent_fd)
            restored = _snapshot_file(
                locked.location.absolute_path,
                locked.kind,
                _parser_for_kind(locked.kind),
                exact_mode=False,
            )[1]
            if restored.raw != locked.snapshot.raw:
                _reject("blocked-rollback")
        except MigrationPlanError:
            raise
        except Exception:
            _reject("blocked-rollback")
        finally:
            if restore is not None:
                if restore.name is not None:
                    _materializer._unlink_at_quietly(locked.parent_fd, restore.name)
                _materializer._release_file_lock(restore.descriptor)
                with contextlib.suppress(OSError):
                    os.close(restore.descriptor)




def _retain_backups(
    root_fd: int,
    run_fd: int,
    backups: Mapping[str, tuple[int, str]],
) -> None:
    """Close a failed transaction's opaque recovery set without deleting it."""
    for descriptor, _name in backups.values():
        with contextlib.suppress(OSError):
            os.close(descriptor)
    with contextlib.suppress(OSError):
        _materializer._fsync_directory(run_fd)
    with contextlib.suppress(OSError):
        os.close(run_fd)
    with contextlib.suppress(OSError):
        _materializer._fsync_directory(root_fd)
    with contextlib.suppress(OSError):
        os.close(root_fd)


def _name_absent(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _restore_cleanup_backups(
    run_fd: int,
    backups: Mapping[str, tuple[int, str]],
    removed: Sequence[str],
) -> bool:
    """Reconstitute every removed opaque file before reporting retention."""
    try:
        for kind in removed:
            descriptor, name = backups[kind]
            if not _name_absent(run_fd, name):
                continue
            restored = _materializer._write_exclusive_file(
                run_fd, name, _backup_bytes(descriptor, kind), 0o600
            )
            with contextlib.suppress(OSError):
                os.close(restored)
        _materializer._fsync_directory(run_fd)
        for _kind, (_descriptor, name) in backups.items():
            details = os.stat(name, dir_fd=run_fd, follow_symlinks=False)
            if (
                not stat.S_ISREG(details.st_mode)
                or details.st_nlink != 1
                or stat.S_IMODE(details.st_mode) != 0o600
            ):
                return False
        return True
    except Exception:
        return False


def _cleanup_backups(
    root_fd: int,
    run_fd: int,
    run_name: str,
    backups: Mapping[str, tuple[int, str]],
) -> bool:
    """Delete a complete recovery set, or restore it intact and report retention."""
    removed: list[str] = []
    try:
        for kind, (_descriptor, name) in backups.items():
            _materializer._unlink_at(run_fd, name)
            if _name_absent(run_fd, name):
                removed.append(kind)
                continue
            if _restore_cleanup_backups(run_fd, backups, removed):
                return False
            _reject("blocked-rollback-path")
        _materializer._fsync_directory(run_fd)
        _materializer._rmdir_at(root_fd, run_name)
        if _name_absent(root_fd, run_name):
            with contextlib.suppress(OSError):
                _materializer._fsync_directory(root_fd)
            return True
        if _restore_cleanup_backups(run_fd, backups, removed):
            return False
        _reject("blocked-rollback-path")
    except MigrationPlanError:
        raise
    except Exception:
        if _restore_cleanup_backups(run_fd, backups, removed):
            return False
        _reject("blocked-rollback-path")
    finally:
        for descriptor, _name in backups.values():
            with contextlib.suppress(OSError):
                os.close(descriptor)
        with contextlib.suppress(OSError):
            os.close(run_fd)
        with contextlib.suppress(OSError):
            os.close(root_fd)


def _close_replacements(replacements: Mapping[str, Any], locked: Mapping[str, _LockedFile]) -> None:
    for kind, replacement in replacements.items():
        if replacement.name is not None:
            _materializer._unlink_at_quietly(locked[kind].parent_fd, replacement.name)
        _materializer._release_file_lock(replacement.descriptor)
        with contextlib.suppress(OSError):
            os.close(replacement.descriptor)


def verify_provider_migration(
    plan: Mapping[str, Any],
    db_path: str | Path,
    models_path: str | Path,
    config_path: str | Path,
    environment_path: str | Path,
) -> bool:
    """Read-only confirmation of the exact post-migration state."""
    try:
        _require_secure_platform()
        accepted = validate_provider_migration_plan(plan)
        _assert_omp_package_current(accepted)
        dy_source, xuanju_source = _source_ids(accepted)
        source = _planning_source(db_path, {dy_source, xuanju_source})
        _assert_source_current(accepted, source)
        dy_binding = _dy_binding_context(db_path, source, dy_source)
        xuanju_binding = _xuanju_binding_context(db_path, source, xuanju_source)
        source_fragment, source_precondition = _selected_source_contract(
            source, dy_binding, xuanju_binding
        )
        if (
            source_fragment != accepted["source"]
            or source_precondition != accepted["preconditions"]["source"]
        ):
            return False
        _models_location, models = _snapshot_file(
            models_path, "models", _models_parser, exact_mode=True
        )
        _config_location, config = _snapshot_file(
            config_path, "config", _config_parser, exact_mode=True
        )
        _environment_location, environment = _snapshot_file(
            environment_path, "environment", _parse_environment, exact_mode=True
        )
        actions = _planning_actions(
            source,
            dy_source,
            xuanju_source,
            target_document=models.parsed,
            dy_binding=dy_binding,
            xuanju_binding=xuanju_binding,
        )
        if not _actions_match(
            accepted["providerActions"], actions, allow_retain=True
        ):
            return False
        providers = models.parsed.get("providers")
        if (
            not isinstance(providers, Mapping)
            or _DY_TARGET not in providers
            or _XUANJU_TARGET not in providers
        ):
            return False
        _validate_provider_api_key_references(models.parsed, post_migration=True)
        dy_action = next(
            action for action in accepted["providerActions"] if action["target"] == _DY_TARGET
        )
        xuanju_action = next(
            action
            for action in accepted["providerActions"]
            if action["target"] == _XUANJU_TARGET
        )
        dy_provider = providers[_DY_TARGET]
        xuanju_provider = providers[_XUANJU_TARGET]
        if (
            not isinstance(dy_provider, Mapping)
            or set(dy_provider)
            != {"baseUrl", "apiKey", "api", "discovery"}
            or dy_provider.get("api") != _XUANJU_REQUEST_API
            or dy_provider.get("discovery") != _DY_DISCOVERY
            or dy_provider.get("apiKey") != _DY_ENV_KEY
            or not isinstance(xuanju_provider, Mapping)
            or set(xuanju_provider) != {"baseUrl", "apiKey", "api", "models"}
            or xuanju_provider.get("api") != _XUANJU_REQUEST_API
            or xuanju_provider.get("models") != [{"id": _XUANJU_MODELS[0]}]
            or xuanju_provider.get("apiKey") != _XUANJU_ENV_KEY
            or xuanju_provider.get("baseUrl")
            != xuanju_binding["siteRoot"]
        ):
            return False
        dy_endpoint = dy_provider.get("baseUrl")
        if not isinstance(dy_endpoint, str):
            return False
        try:
            if _dy_provider_base(dy_endpoint, "blocked-source-url") != dy_endpoint:
                return False
        except MigrationPlanError:
            return False
        if hashlib.sha256(dy_endpoint.encode("utf-8")).hexdigest() != dy_action[
            "endpoint"
        ]["sha256"]:
            return False
        if (
            not isinstance(xuanju_provider.get("baseUrl"), str)
            or hashlib.sha256(xuanju_provider["baseUrl"].encode("utf-8")).hexdigest()
            != xuanju_action["endpoint"]["sha256"]
        ):
            return False
        env_keys = _env_keys(environment)
        if (
            _DEEPSEEK_ENV_KEY in env_keys
            or _DY_ENV_KEY not in env_keys
            or _XUANJU_ENV_KEY not in env_keys
        ):
            return False
        final_bindings, fresh_source = _final_selected_binding_contexts(
            db_path, source, {dy_source, xuanju_source}
        )
        final_fragment, final_precondition = _selected_source_contract(
            source,
            final_bindings[dy_source],
            final_bindings[xuanju_source],
        )
        if (
            final_fragment != accepted["source"]
            or final_precondition != accepted["preconditions"]["source"]
        ):
            return False
        fresh_records = _apply_records(fresh_source, accepted)
        fresh_record = fresh_records.get(xuanju_source)
        source_credential = (
            fresh_record.get("credential_value")
            if isinstance(fresh_record, Mapping)
            else None
        )
        _environment_location_after, environment_after = _snapshot_file(
            environment_path, "environment", _parse_environment, exact_mode=True
        )
        if environment_after.precondition != environment.precondition:
            return False
        target_credential = _environment_value(
            environment_after.parsed, _XUANJU_ENV_KEY
        )
        if (
            not isinstance(source_credential, str)
            or not isinstance(target_credential, str)
            or not hmac.compare_digest(
                target_credential.encode("utf-8"),
                shlex.quote(source_credential).encode("utf-8"),
            )
        ):
            return False
        from tools import omp_config_handoff
        omp_config_handoff._assert_migration_disabled_providers(config.parsed)

        roles, order, fallback, unrestricted, enabled = (
            omp_config_handoff._config_public_fields(config.parsed)
        )
        if (
            unrestricted
            or order != accepted["configActions"]["modelProviderOrder"]["result"]
            or enabled != accepted["configActions"]["enabledModels"]["result"]
            or _digest(roles, "blocked-config-read")
            != accepted["configActions"]["modelRolesSha256"]
            or _digest(fallback, "blocked-config-read")
            != accepted["configActions"]["fallbackChainsSha256"]
        ):
            return False
        return True
    except (
        MigrationPlanError,
        ConfigHandoffError,
        SourceInventoryError,
        OSError,
        TypeError,
        ValueError,
    ):
        return False


def apply_provider_migration_plan(
    plan: Mapping[str, Any],
    db_path: str | Path,
    models_path: str | Path,
    config_path: str | Path,
    environment_path: str | Path,
    *,
    accepted_plan_sha256: str,
    allow_opaque_backup: bool = False,
) -> dict[str, Any]:
    """Apply the accepted migration as one recoverable three-file transaction."""
    _require_secure_platform()
    accepted = validate_provider_migration_plan(plan)
    _assert_omp_package_current(accepted)
    if not _is_digest(accepted_plan_sha256) or accepted_plan_sha256 != accepted["planSha256"]:
        _reject("blocked-accepted-plan-sha256")
    if allow_opaque_backup is not True:
        _reject("blocked-backup-consent")
    dy_source, xuanju_source = _source_ids(accepted)
    planning_source = _planning_source(db_path, {dy_source, xuanju_source})
    _assert_source_current(accepted, planning_source)
    paths = {
        "models": models_path,
        "config": config_path,
        "environment": environment_path,
    }
    locked: dict[str, _LockedFile] = {}
    replacements: dict[str, Any] = {}
    backups: dict[str, tuple[int, str]] = {}
    root_fd = -1
    run_fd = -1
    run_name = ""
    published: list[_PublishedFile] = []
    completed = False
    rollback_succeeded = False
    report: dict[str, Any] | None = None
    try:
        ordered = sorted(paths, key=lambda kind: os.fspath(paths[kind]))
        for kind in ordered:
            locked[kind] = _open_locked(
                paths[kind], kind, _parser_for_kind(kind), exact_mode=True
            )
        for kind in paths:
            _assert_precondition(
                locked[kind].snapshot.precondition,
                accepted["preconditions"][kind],
                kind,
            )
        dy_binding = _dy_binding_context(db_path, planning_source, dy_source)
        xuanju_binding = _xuanju_binding_context(
            db_path, planning_source, xuanju_source
        )
        source_fragment, source_precondition = _selected_source_contract(
            planning_source, dy_binding, xuanju_binding
        )
        if (
            source_fragment != accepted["source"]
            or source_precondition != accepted["preconditions"]["source"]
        ):
            _reject("blocked-source-changed")
        if _planning_actions(
            planning_source,
            dy_source,
            xuanju_source,
            target_document=locked["models"].snapshot.parsed,
            dy_binding=dy_binding,
            xuanju_binding=xuanju_binding,
        ) != accepted["providerActions"]:
            _reject("blocked-source-changed")
        final_bindings, final_source = _final_selected_binding_contexts(
            db_path, planning_source, {dy_source, xuanju_source}
        )
        try:
            final_fragment, final_precondition = _selected_source_contract(
                planning_source,
                final_bindings[dy_source],
                final_bindings[xuanju_source],
            )
            if (
                final_fragment != accepted["source"]
                or final_precondition != accepted["preconditions"]["source"]
            ):
                _reject("blocked-source-changed")
            final_records = _apply_records(final_source, accepted)
            rendered = _render_all(locked, accepted, final_records)
        finally:
            for item in final_source.get("providers", []):
                if isinstance(item, dict):
                    item.clear()
            final_source.clear()
        _assert_omp_package_current(accepted)
        root_fd, run_fd, run_name, backups = _write_backup_set(
            locked, accepted["planSha256"]
        )
        replacements = _prepare_replacements(locked, rendered)
        for kind in ("models", "config", "environment"):
            _replace_published_target(locked[kind], replacements[kind])
            # A rename is visible before fchmod/fsync.  Record it first so
            # every post-rename failure is covered by the recovery branch.
            published.append(_PublishedFile(locked[kind], replacements[kind]))
            _finalize_published_target(locked[kind], replacements[kind])
        if not verify_provider_migration(
            accepted,
            db_path,
            models_path,
            config_path,
            environment_path,
        ):
            _reject("blocked-apply-verify")
        report = {
            "status": "applied",
            "updatedFiles": ["config", "environment", "models"],
            "rollback": {"opaqueBackup": "deleted", "withinRun": True},
        }
        completed = True
        return report
    except BaseException:
        if published:
            _rollback(published, backups)
            rollback_succeeded = True
        raise
    finally:
        try:
            _close_replacements(replacements, locked)
        finally:
            try:
                if run_fd != -1 and root_fd != -1:
                    if completed or rollback_succeeded or not published:
                        backup_deleted = _cleanup_backups(
                            root_fd, run_fd, run_name, backups
                        )
                        if report is not None and not backup_deleted:
                            report["rollback"]["opaqueBackup"] = "retained"
                    else:
                        _retain_backups(root_fd, run_fd, backups)
            finally:
                for item in locked.values():
                    _close_locked(item)
