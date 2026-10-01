"""Build a redacted, read-only OMP config handoff plan."""

from __future__ import annotations

import copy
import contextlib
import errno
import hashlib
import json
import os
import re
import stat
from collections.abc import Iterable, Mapping
from typing import Any

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode


class ConfigHandoffError(ValueError):
    """A handoff cannot be safely planned; ``code`` is machine-readable."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


_ACTIVE_PROVIDER = "cc-deepseek"
_MAX_CONFIG_BYTES = 1 << 20
_MAX_YAML_DEPTH = 32
_MAX_YAML_NODES = 4_096
_MAX_YAML_SCALARS = 2_048
_MAX_YAML_SCALAR_CHARS = 4_096
_MAX_CONFIG_LIST_ITEMS = 256
_MAX_CONFIG_MAPPING_ITEMS = 128
_MAX_MODEL_SEGMENTS = 16
_MAX_MODEL_LENGTH = 512
_MAX_PATH_LENGTH = 4_096
_LITERAL_SELECTOR_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_ALLOWED_YAML_TAGS = frozenset({
    "tag:yaml.org,2002:map",
    "tag:yaml.org,2002:seq",
    "tag:yaml.org,2002:str",
    "tag:yaml.org,2002:int",
    "tag:yaml.org,2002:float",
    "tag:yaml.org,2002:bool",
    "tag:yaml.org,2002:null",
})


class _UnsafeYamlError(yaml.YAMLError):
    """YAML exceeded the narrow data-only handoff surface."""


class _BoundedYamlLoader(yaml.SafeLoader):
    """Safe loader that rejects aliases, custom types, duplicate keys, and excess structure."""

    def __init__(self, stream: Any) -> None:
        super().__init__(stream)
        self._depth = 0
        self._node_count = 0
        self._scalar_count = 0

    def compose_node(self, parent: Node | None, index: Any) -> Node:
        event = self.peek_event()
        if isinstance(event, AliasEvent) or getattr(event, "anchor", None) is not None:
            raise _UnsafeYamlError("aliases and anchors are not supported")
        self._depth += 1
        self._node_count += 1
        try:
            if self._depth > _MAX_YAML_DEPTH or self._node_count > _MAX_YAML_NODES:
                raise _UnsafeYamlError("YAML structure limit exceeded")
            node = super().compose_node(parent, index)
        finally:
            self._depth -= 1
        if node.tag not in _ALLOWED_YAML_TAGS:
            raise _UnsafeYamlError("custom YAML tags are not supported")
        if isinstance(node, ScalarNode):
            self._scalar_count += 1
            if (
                self._scalar_count > _MAX_YAML_SCALARS
                or len(node.value) > _MAX_YAML_SCALAR_CHARS
            ):
                raise _UnsafeYamlError("YAML scalar limit exceeded")
        return node

    def construct_mapping(
        self, node: MappingNode, deep: bool = False
    ) -> dict[str, Any]:
        mapping: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if type(key) is not str or key in mapping:
                raise _UnsafeYamlError("duplicate or non-string mapping key")
            mapping[key] = self.construct_object(value_node, deep=deep)
        return mapping


def _canonical_plan_sha256(document: Mapping[str, Any]) -> str:
    """Hash the complete handoff document except its self-referential digest."""
    payload = {key: value for key, value in document.items() if key != "planSha256"}
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    except (TypeError, UnicodeEncodeError, ValueError):
        raise ConfigHandoffError("blocked-noncanonical-handoff") from None
    return hashlib.sha256(canonical).hexdigest()


def _selector_list(value: Any) -> list[str]:
    if type(value) is not list or not value or len(value) > _MAX_CONFIG_LIST_ITEMS:
        raise ConfigHandoffError("blocked-invalid-plan")
    selectors = [_selector(item) for item in value]
    if len(selectors) != len(set(selectors)):
        raise ConfigHandoffError("blocked-invalid-plan")
    return selectors


def _selector(value: Any) -> str:
    if type(value) is not dict or set(value) != {"provider", "model"}:
        raise ConfigHandoffError("blocked-invalid-model-selector")
    return _selector_from_parts(
        value["provider"], value["model"], "blocked-invalid-model-selector"
    )


def _selector_from_parts(provider: Any, model: Any, code: str) -> str:
    if type(provider) is not str or type(model) is not str:
        raise ConfigHandoffError(code)
    if (
        not _LITERAL_SELECTOR_PART.fullmatch(provider)
        or len(model) > _MAX_MODEL_LENGTH
        or not _safe_model_id(model)
    ):
        raise ConfigHandoffError("blocked-nonliteral-model-selector")
    return f"{provider}/{model}"


def _safe_model_id(value: str) -> bool:
    segments = value.split("/")
    return (
        bool(segments)
        and len(segments) <= _MAX_MODEL_SEGMENTS
        and all(
            _LITERAL_SELECTOR_PART.fullmatch(segment) is not None
            for segment in segments
        )
    )


def _public_selector(value: Any) -> str:
    if type(value) is not str:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    provider, separator, model = value.partition("/")
    if not separator:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    try:
        return _selector_from_parts(provider, model, "blocked-unsafe-config-value")
    except ConfigHandoffError as error:
        if error.code == "blocked-nonliteral-model-selector":
            raise ConfigHandoffError("blocked-unsafe-config-value") from None
        raise


def _provider_name(value: Any) -> str:
    if type(value) is not str or not _LITERAL_SELECTOR_PART.fullmatch(value):
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return value


def _role_name(value: Any) -> str:
    if type(value) is not str or not _LITERAL_SELECTOR_PART.fullmatch(value):
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return value


def _public_selector_list(value: Any) -> list[str]:
    if type(value) is not list or len(value) > _MAX_CONFIG_LIST_ITEMS:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    selectors = [_public_selector(item) for item in value]
    if len(selectors) != len(set(selectors)):
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return selectors


def _public_provider_order(value: Any) -> list[str]:
    if type(value) is not list or len(value) > _MAX_CONFIG_LIST_ITEMS:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    providers = [_provider_name(item) for item in value]
    normalized = [provider.casefold() for provider in providers]
    if len(normalized) != len(set(normalized)):
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return providers


def _public_model_roles(value: Any) -> dict[str, str]:
    if type(value) is not dict or len(value) > _MAX_CONFIG_MAPPING_ITEMS:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return {
        _role_name(name): _public_selector(selector) for name, selector in value.items()
    }


def _public_fallback_chains(value: Any) -> dict[str, list[str]]:
    if type(value) is not dict or len(value) > _MAX_CONFIG_MAPPING_ITEMS:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    return {
        _role_name(name): _public_selector_list(selectors)
        for name, selectors in value.items()
    }


def _projection_handoff_fragment(
    projection_plan: Mapping[str, Any],
) -> tuple[Mapping[str, Any], bool]:
    if not isinstance(projection_plan, Mapping):
        raise ConfigHandoffError("blocked-invalid-plan")
    try:
        handoff = projection_plan["configHandoff"]
        removals = projection_plan["removeProviders"]
    except (KeyError, TypeError):
        raise ConfigHandoffError("blocked-invalid-plan") from None
    if type(handoff) is not dict or set(handoff) != {"addModelSelectors"}:
        raise ConfigHandoffError("blocked-invalid-plan")
    if type(removals) is not list or removals not in ([], [_ACTIVE_PROVIDER]):
        raise ConfigHandoffError("blocked-invalid-plan")
    return handoff, removals == [_ACTIVE_PROVIDER]


def _verified_selector_set(value: Iterable[str]) -> set[str]:
    if isinstance(value, (str, bytes)):
        raise ConfigHandoffError("blocked-invalid-plan")
    try:
        iterator = iter(value)
    except TypeError:
        raise ConfigHandoffError("blocked-invalid-plan") from None
    verified: set[str] = set()
    for count, selector in enumerate(iterator, start=1):
        if count > _MAX_CONFIG_LIST_ITEMS:
            raise ConfigHandoffError("blocked-invalid-plan")
        try:
            verified.add(_public_selector(selector))
        except ConfigHandoffError:
            raise ConfigHandoffError("blocked-invalid-plan") from None
    return verified


def _selector_key(selector: str) -> str:
    provider, separator, model = selector.partition("/")
    if not separator:
        return selector.strip().casefold()
    return f"{provider.strip().casefold()}/{model.strip().casefold()}"


def _is_xuanju_flash_selector(selector: str) -> bool:
    return _selector_key(selector) == _MIGRATION_SELECTORS[0].casefold()


def _assert_route_token(
    value: str, *, require_qualified: bool, allow_switches: bool
) -> None:
    token = value.strip()
    if not token:
        return
    selector = _selector_key(token)
    provider, separator, model = selector.partition("/")
    if allow_switches and selector in {"on", "off", "true", "false"}:
        return
    if (
        provider == "pi"
        or provider == "*"
        or provider.startswith("@")
        or (
            provider == _MIGRATION_PROVIDER
            and (
                not separator
                or model != _MIGRATION_SELECTORS[0].partition("/")[2].casefold()
            )
        )
        or (require_qualified and not separator)
    ):
        raise ConfigHandoffError("blocked-config-migration-state")


def _assert_xuanju_route_tree(value: Any) -> None:
    """Reject unsupported routes and legacy aliases in migration fields."""
    if isinstance(value, str):
        for token in value.split(","):
            _assert_route_token(
                token, require_qualified=False, allow_switches=False
            )
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str):
                _assert_route_token(
                    key, require_qualified=False, allow_switches=False
                )
            _assert_xuanju_route_tree(item)
    elif isinstance(value, list):
        for item in value:
            _assert_xuanju_route_tree(item)



def _config_public_fields(
    config: dict[str, Any],
) -> tuple[dict[str, str], list[str], dict[str, list[str]], bool, list[str]]:
    roles = _public_model_roles(config.get("modelRoles", {}))
    order = _public_provider_order(config.get("modelProviderOrder", []))

    retry = config.get("retry", {})
    if type(retry) is not dict:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    fallback = _public_fallback_chains(retry.get("fallbackChains", {}))
    _assert_xuanju_route_tree(roles)
    _assert_xuanju_route_tree(fallback)

    if "enabledModels" not in config or config["enabledModels"] is None:
        return roles, order, fallback, True, []
    enabled_raw = config["enabledModels"]
    if type(enabled_raw) is not list:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    if not enabled_raw:
        return roles, order, fallback, True, []
    enabled = _public_selector_list(enabled_raw)
    _assert_xuanju_route_tree(enabled)
    return roles, order, fallback, False, enabled


def _without_active_selectors(values: list[str]) -> list[str]:
    return [value for value in values if _selector_provider(value) != _ACTIVE_PROVIDER]


def _selector_provider(selector: str) -> str:
    return selector.partition("/")[0].strip().casefold()


def _target_path(config_path: str | os.PathLike[str]) -> str:
    try:
        raw = os.fspath(config_path)
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        if type(raw) is not str or not raw or "\x00" in raw:
            raise ValueError
        absolute = os.path.abspath(raw)
        if len(absolute) > _MAX_PATH_LENGTH or _contains_control_character(absolute):
            raise ValueError
        return absolute
    except (OSError, TypeError, ValueError):
        raise ConfigHandoffError("blocked-target-not-file") from None


def _contains_control_character(value: str) -> bool:
    return any(
        codepoint < 0x20 or 0x7F <= codepoint <= 0x9F or 0xD800 <= codepoint <= 0xDFFF
        for codepoint in map(ord, value)
    )


def _snapshot_realpath(absolute: str, details: os.stat_result) -> str:
    try:
        realpath = os.path.realpath(absolute)
        realpath_details = os.lstat(realpath)
        _validate_target_stat(realpath_details)
        if (
            len(realpath) > _MAX_PATH_LENGTH
            or _contains_control_character(realpath)
            or _stat_identity(realpath_details) != _stat_identity(details)
        ):
            raise ValueError
        return realpath
    except (OSError, ValueError):
        raise ConfigHandoffError("blocked-target-changed") from None


def _snapshot_config(
    config_path: str | os.PathLike[str],
) -> tuple[str, os.stat_result, bytes]:
    absolute = _target_path(config_path)
    try:
        path_before = os.lstat(absolute)
    except OSError:
        raise ConfigHandoffError("blocked-target-not-file") from None
    if stat.S_ISLNK(path_before.st_mode):
        raise ConfigHandoffError("blocked-target-symlink")
    _validate_target_stat(path_before)

    descriptor = -1
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY
            | _close_on_exec_flag()
            | _no_follow_flag()
            | _nonblocking_flag(),
        )
        fd_before = os.fstat(descriptor)
        _validate_target_stat(fd_before)
        if _stat_identity(path_before) != _stat_identity(fd_before):
            raise ConfigHandoffError("blocked-target-changed")
        raw = _read_descriptor(descriptor, _MAX_CONFIG_BYTES)
        fd_after = os.fstat(descriptor)
        path_after = os.lstat(absolute)
        _validate_target_stat(fd_after)
        _validate_target_stat(path_after)
        if (
            _stat_identity(fd_before) != _stat_identity(fd_after)
            or _stat_identity(path_after) != _stat_identity(fd_after)
            or len(raw) != fd_after.st_size
        ):
            raise ConfigHandoffError("blocked-target-changed")
        return _snapshot_realpath(absolute, fd_after), fd_after, raw
    except ConfigHandoffError:
        raise
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ConfigHandoffError("blocked-target-symlink") from None
        raise ConfigHandoffError("blocked-target-not-file") from None
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)


def _validate_target_stat(details: os.stat_result) -> None:
    if not stat.S_ISREG(details.st_mode):
        raise ConfigHandoffError("blocked-target-not-file")
    if not _owned_by_current_user(details.st_uid):
        raise ConfigHandoffError("blocked-target-owner")
    if stat.S_IMODE(details.st_mode) & 0o022:
        raise ConfigHandoffError("blocked-target-permissions")
    if details.st_size < 0 or details.st_size > _MAX_CONFIG_BYTES:
        raise ConfigHandoffError("blocked-target-size")


def _owned_by_current_user(owner: int) -> bool:
    try:
        allowed = {os.getuid()}
        effective_uid = getattr(os, "geteuid", None)
        if effective_uid is not None:
            allowed.add(effective_uid())
    except (AttributeError, OSError):
        return False
    return owner in allowed


def _stat_identity(details: os.stat_result) -> tuple[int, int, int, int, int, int, int]:
    return (
        details.st_dev,
        details.st_ino,
        details.st_mode,
        details.st_uid,
        details.st_size,
        _mtime_ns(details),
        _ctime_ns(details),
    )


def _mtime_ns(details: os.stat_result) -> int:
    value = getattr(details, "st_mtime_ns", None)
    if type(value) is int:
        return value
    return int(details.st_mtime * 1_000_000_000)


def _ctime_ns(details: os.stat_result) -> int:
    value = getattr(details, "st_ctime_ns", None)
    if type(value) is int:
        return value
    return int(details.st_ctime * 1_000_000_000)


def _read_descriptor(descriptor: int, maximum: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, 65_536)
        except InterruptedError:
            continue
        except OSError:
            raise ConfigHandoffError("blocked-target-not-file") from None
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > maximum:
            raise ConfigHandoffError("blocked-target-size")
        chunks.append(chunk)


def _close_on_exec_flag() -> int:
    flag = getattr(os, "O_CLOEXEC", 0)
    return flag if type(flag) is int else 0


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if type(flag) is not int or flag == 0:
        raise ConfigHandoffError("blocked-no-follow")
    return flag


def _nonblocking_flag() -> int:
    flag = getattr(os, "O_NONBLOCK", None)
    if type(flag) is not int or flag == 0:
        raise ConfigHandoffError("blocked-nonblocking")
    return flag


def _load_config(raw: bytes) -> dict[str, Any]:
    try:
        config = yaml.load(raw, Loader=_BoundedYamlLoader)
    except (RecursionError, TypeError, UnicodeDecodeError, ValueError, yaml.YAMLError):
        raise ConfigHandoffError("blocked-invalid-yaml") from None
    if type(config) is not dict:
        raise ConfigHandoffError("blocked-config-not-mapping")
    return config


def build_config_handoff(
    projection_plan: Mapping[str, Any],
    config_path: str | os.PathLike[str],
    verified_selectors: Iterable[str],
    omp_version: str = "18.2.0",
) -> dict[str, Any]:
    """Return a no-write, fully hashed, redacted plan for one OMP config snapshot."""
    if omp_version != "18.2.0":
        raise ConfigHandoffError("blocked-omp-version")


    handoff, remove_active = _projection_handoff_fragment(projection_plan)
    selectors = _selector_list(handoff["addModelSelectors"])
    verified = _verified_selector_set(verified_selectors)
    if any(selector not in verified for selector in selectors):
        raise ConfigHandoffError("blocked-unverified-model")
    if remove_active and any(
        _selector_provider(selector) == _ACTIVE_PROVIDER for selector in selectors
    ):
        raise ConfigHandoffError("blocked-active-provider-selector")

    realpath, details, raw = _snapshot_config(config_path)
    config = _load_config(raw)
    roles, order, fallback, unrestricted, enabled = _config_public_fields(config)

    if remove_active and any(
        _selector_provider(selector) == _ACTIVE_PROVIDER for selector in roles.values()
    ):
        raise ConfigHandoffError("blocked-active-role-removal")

    if unrestricted:
        enabled_change: dict[str, Any] = {"operation": "preserve-unrestricted"}
        add_enabled: list[str] = []
    else:
        result = _without_active_selectors(enabled) if remove_active else list(enabled)
        for selector in selectors:
            if selector not in result:
                result.append(selector)
        enabled_change = {
            "operation": "append",
            "selectors": list(selectors),
            "result": result,
        }
        add_enabled = list(selectors)

    changes = {
        "addEnabledModels": add_enabled,
        "enabledModels": enabled_change,
        "modelProviderOrder": [
            provider
            for provider in order
            if not remove_active or provider != _ACTIVE_PROVIDER
        ],
        "fallbackChains": {
            name: _without_active_selectors(chain) if remove_active else list(chain)
            for name, chain in fallback.items()
        },
        "modelRoles": dict(roles),
    }
    document = {
        "preconditions": {
            "target": {
                "realpath": realpath,
                "device": details.st_dev,
                "fileId": details.st_ino,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": stat.S_IMODE(details.st_mode),
                "ctimeNs": _ctime_ns(details),
                "size": len(raw),
            }
        },
        "changes": changes,
    }
    document["planSha256"] = _canonical_plan_sha256(document)
    return document

_MIGRATION_PROVIDER = "cc-xuanju"
_MIGRATION_SELECTORS = (
    "cc-xuanju/GLM-5.3-flash",
)
_MIGRATION_AFTER_PROVIDER = "cc-dy-codex"


def _public_state_sha256(value: Any) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise ConfigHandoffError("blocked-unsafe-config-value") from None
    return hashlib.sha256(encoded).hexdigest()


def _assert_migration_disabled_providers(config: Mapping[str, Any]) -> None:
    """Keep both projected providers available for the migration contract."""
    if "disabledProviders" not in config:
        return
    disabled = config["disabledProviders"]
    if type(disabled) is not list:
        raise ConfigHandoffError("blocked-unsafe-config-value")
    for provider in disabled:
        if not isinstance(provider, str):
            raise ConfigHandoffError("blocked-unsafe-config-value")
        if provider.strip().casefold() in {
            _MIGRATION_PROVIDER,
            _MIGRATION_AFTER_PROVIDER,
        }:
            raise ConfigHandoffError("blocked-config-migration-state")


def _provider_migration_changes(config: dict[str, Any]) -> dict[str, Any]:
    """Derive only the additive Xuanju provider/order/model changes."""
    _assert_migration_disabled_providers(config)
    roles, order, fallback, unrestricted, enabled = _config_public_fields(config)
    if unrestricted:
        raise ConfigHandoffError("blocked-enabled-models-unrestricted")
    if len(order) != len(set(order)):
        raise ConfigHandoffError("blocked-config-migration-state")
    normalized_order = [provider.casefold() for provider in order]
    if len(normalized_order) != len(set(normalized_order)):
        raise ConfigHandoffError("blocked-config-migration-state")
    for provider, normalized in zip(order, normalized_order):
        if (
            normalized in {_MIGRATION_PROVIDER, _MIGRATION_AFTER_PROVIDER}
            and provider != normalized
        ):
            raise ConfigHandoffError("blocked-config-migration-state")
    existing_xuanju_selectors = [
        selector
        for selector in enabled
        if _selector_provider(selector) == _MIGRATION_PROVIDER
    ]
    if any(not _is_xuanju_flash_selector(selector) for selector in existing_xuanju_selectors):
        raise ConfigHandoffError("blocked-config-migration-state")
    if _MIGRATION_AFTER_PROVIDER not in order:
        raise ConfigHandoffError("blocked-config-migration-state")
    enabled_keys = {_selector_key(selector) for selector in enabled}
    existing_dy_selectors = {
        f"{_MIGRATION_AFTER_PROVIDER}/gpt-5.6",
        f"{_MIGRATION_AFTER_PROVIDER}/gpt-5.6-sol",
    }
    if not {_selector_key(selector) for selector in existing_dy_selectors}.issubset(
        enabled_keys
    ):
        raise ConfigHandoffError("blocked-config-migration-state")
    provider_present = _MIGRATION_PROVIDER in order
    add_order = [] if provider_present else [_MIGRATION_PROVIDER]
    provider_order_result = (
        list(order)
        if provider_present
        else [
            *order[: order.index(_MIGRATION_AFTER_PROVIDER) + 1],
            _MIGRATION_PROVIDER,
            *order[order.index(_MIGRATION_AFTER_PROVIDER) + 1 :],
        ]
    )
    selector_present = _MIGRATION_SELECTORS[0].casefold() in enabled_keys
    add_enabled = [] if selector_present else list(_MIGRATION_SELECTORS)
    enabled_result = list(enabled) if selector_present else [*enabled, *_MIGRATION_SELECTORS]
    return {
        "addEnabledModels": add_enabled,
        "addProviderOrder": add_order,
        "enabledModels": {
            "operation": "retain" if selector_present else "append",
            "selectors": add_enabled,
            "result": enabled_result,
        },
        "fallbackChainsSha256": _public_state_sha256(fallback),
        "modelProviderOrder": {
            "operation": "retain" if provider_present else "insert-after",
            "providers": add_order,
            "after": _MIGRATION_AFTER_PROVIDER,
            "result": provider_order_result,
        },
        "modelRolesSha256": _public_state_sha256(roles),
        "preserveFallbackChains": True,
        "preserveModelRoles": True,
    }


def build_provider_migration_handoff(
    config_path: str | os.PathLike[str],
    *,
    omp_version: str = "18.2.0",
) -> dict[str, Any]:
    """Return a read-only, redacted handoff for the fixed additive migration."""
    if omp_version != "18.2.0":
        raise ConfigHandoffError("blocked-omp-version")
    realpath, details, raw = _snapshot_config(config_path)
    config = _load_config(raw)
    return {
        "preconditions": {
            "target": {
                "realpath": realpath,
                "device": details.st_dev,
                "fileId": details.st_ino,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "mode": stat.S_IMODE(details.st_mode),
                "ctimeNs": _ctime_ns(details),
                "size": len(raw),
            }
        },
        "changes": _provider_migration_changes(config),
    }


def render_provider_migration_config(raw: bytes, changes: Mapping[str, Any]) -> bytes:
    """Render the fixed additive migration after rederiving its exact change.

    ``raw`` is an already no-follow, CAS-locked private target snapshot.  The
    returned bytes are never part of a plan or error path.
    """
    if not isinstance(raw, bytes) or not isinstance(changes, Mapping):
        raise ConfigHandoffError("blocked-config-render")
    config = _load_config(raw)
    expected = _provider_migration_changes(config)
    if dict(changes) != expected:
        raise ConfigHandoffError("blocked-config-changed")
    updated = copy.deepcopy(config)
    updated["modelProviderOrder"] = list(expected["modelProviderOrder"]["result"])
    updated["enabledModels"] = list(expected["enabledModels"]["result"])
    roles, _order, fallback, _unrestricted, _enabled = _config_public_fields(updated)
    if (
        _public_state_sha256(roles) != expected["modelRolesSha256"]
        or _public_state_sha256(fallback) != expected["fallbackChainsSha256"]
    ):
        raise ConfigHandoffError("blocked-config-render")
    try:
        rendered = yaml.safe_dump(
            updated, sort_keys=False, allow_unicode=True
        ).encode("utf-8")
    except (TypeError, ValueError, yaml.YAMLError):
        raise ConfigHandoffError("blocked-config-render") from None
    # Reparse through the bounded loader so an emitter regression cannot widen
    # the accepted config surface.
    rendered_config = _load_config(rendered)
    rendered_roles, _order, rendered_fallback, _unrestricted, _enabled = (
        _config_public_fields(rendered_config)
    )
    if (
        _public_state_sha256(rendered_roles) != expected["modelRolesSha256"]
        or _public_state_sha256(rendered_fallback)
        != expected["fallbackChainsSha256"]
    ):
        raise ConfigHandoffError("blocked-config-render")
    return rendered
