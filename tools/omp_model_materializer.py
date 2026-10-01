"""Plan and atomically apply a constrained CC Switch projection into OMP.

The only database boundary in this module is ``materialization_source``.  It owns
all SQLite access, including its no-follow read-only snapshot.  This module owns
the other half of the transaction: a closed, redacted plan schema and a secure
no-follow target-file transaction.
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import os
import secrets
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

try:  # Native Windows has no advisory file-locking module.
    import fcntl
except ImportError:  # pragma: no cover - exercised on native Windows.
    fcntl = None  # type: ignore[assignment]

import yaml
from yaml.events import AliasEvent
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from tools.cc_switch_omp_source import SourceInventoryError, materialization_source
from tools.omp_model_materializer_schema import (
    ProjectionPlanError,
    assert_selected_model_transport as _assert_selected_model_transport,
    assert_selected_provider_transport as _assert_selected_provider_transport,
    assert_source_model_secret_free as _assert_source_model_secret_free,
    canonical_https_url as _canonical_https_url,
    canonical_json as _canonical_json,
    document_sha256 as _document_sha256,
    effective_model_index as _effective_model_index,
    ensure_json_value as _ensure_json_value,
    is_digest as _is_digest,
    json_clone as _json_clone,
    json_sha256 as _json_sha256,
    model_index as _model_index,
    require_env_reference as _require_env_reference,
    require_exact_keys as _require_exact_keys,
    require_model_id as _require_model_id,
    require_provider_id as _require_provider_id,
    source_model_matches as _source_model_matches,
    validate_distinct_model_id_list as _validate_distinct_model_id_list,
    validate_document as _validate_document,
    validate_id_list as _validate_id_list,
    validate_model_id_list as _validate_model_id_list,
    validate_provider as _validate_provider,
    validate_selected_models as _validate_selected_models,
)


__all__ = [
    "ProjectionPlanError",
    "apply_projection_plan",
    "apply_provider_migration_plan",
    "build_projection_plan",
    "build_provider_migration_plan",
    "load_projection_plan",
    "load_provider_migration_plan",
    "validate_projection_plan",
    "validate_provider_migration_plan",
    "verify_projection",
    "verify_provider_migration",
]


_GENERATED_HEADER = "# CC Switch source projection; generated for OMP."
_SUPPORTED_OMP_VERSION = "18.2.0"
_SCHEMA_VERSION = 2
_MAX_PLAN_BYTES = 1 << 20
_MAX_TARGET_BYTES = 16 << 20
_MAX_YAML_DEPTH = 64
_MAX_YAML_NODES = 20_000

_SOURCE_API_FAMILIES = {
    "anthropic": "anthropic-messages",
    "openai_chat": "openai-completions",
    "openai_responses": "openai-responses",
}


@dataclass(frozen=True)
class _TargetLocation:
    absolute_path: str
    parent_components: tuple[str, ...]
    basename: str
    realpath: str


@dataclass(frozen=True)
class _TargetSnapshot:
    precondition: dict[str, Any]
    raw: bytes
    document: dict[str, Any]


@dataclass
class _LockedTarget:
    descriptor: int
    snapshot: _TargetSnapshot


@dataclass
class _TrustedTemp:
    name: str | None
    descriptor: int


class _BoundedUniqueKeySafeLoader(yaml.SafeLoader):
    """Safe loader with duplicate-key, alias, depth, and node-count limits."""

    def __init__(self, stream: str) -> None:
        super().__init__(stream)
        self._yaml_depth = 0
        self._yaml_nodes = 0

    def compose_node(self, parent: Node | None, index: object) -> Node:
        if self.check_event(AliasEvent):
            raise yaml.YAMLError("aliases are not permitted")
        self._yaml_depth += 1
        self._yaml_nodes += 1
        try:
            if self._yaml_depth > _MAX_YAML_DEPTH or self._yaml_nodes > _MAX_YAML_NODES:
                raise yaml.YAMLError("YAML structure exceeds materializer bounds")
            return super().compose_node(parent, index)
        finally:
            self._yaml_depth -= 1


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.Node, deep: bool = False
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str) or key in mapping:
            raise yaml.YAMLError("duplicate or non-string mapping key")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_BoundedUniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def build_projection_plan(
    db_path: str | Path,
    target_path: str | Path,
    *,
    provider_map: Mapping[str, str],
    selected_models: Mapping[str, Sequence[str]],
    excluded_source_ids: set[str] | frozenset[str],
    approved_current_origin_source_ids: set[str] | frozenset[str],
    omp_version: str,
) -> dict[str, Any]:
    """Return one closed, redacted projection plan without changing either input."""
    _require_secure_platform()
    if omp_version != _SUPPORTED_OMP_VERSION:
        _reject("blocked-omp-version")

    source_ids, provider_targets, selections, exclusions, approvals = (
        _validate_planning_inputs(
            provider_map,
            selected_models,
            excluded_source_ids,
            approved_current_origin_source_ids,
        )
    )
    location = _target_location(target_path)
    parent_fd = _open_target_parent(location)
    try:
        target = _snapshot_target_at(parent_fd, location)
    finally:
        os.close(parent_fd)

    source = _source_snapshot(db_path, source_ids, approvals)
    derived = _derive_operations(
        source, target, provider_targets, selections, exclusions
    )
    plan: dict[str, Any] = {
        "approvedOriginSourceIds": approvals,
        "configHandoff": derived["configHandoff"],
        "excludedSourceIds": exclusions,
        "planSha256": "",
        "policy": {
            # Consent is deliberately not controlled by the editable plan.
            "allowOpaqueBackup": False,
            "requireTargetHeader": True,
        },
        "preconditions": {
            "source": source["precondition"],
            "target": target.precondition,
        },
        "providerActions": derived["providerActions"],
        "removeProviders": derived["removeProviders"],
        "result": derived["result"],
        "schemaVersion": _SCHEMA_VERSION,
        "source": derived["source"],
        "status": "planned",
    }
    plan["planSha256"] = _plan_hash(plan)
    return validate_projection_plan(plan)


def load_projection_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load one no-follow JSON plan and validate its complete closed schema."""
    _require_secure_platform()
    descriptor = -1
    try:
        raw_path = os.fspath(path)
        if isinstance(raw_path, bytes):
            raw_path = os.fsdecode(raw_path)
        if not isinstance(raw_path, str) or not raw_path:
            raise OSError
        flags = (
            os.O_RDONLY
            | _close_on_exec_flag()
            | _no_follow_flag()
            | _non_blocking_flag()
        )
        descriptor = os.open(raw_path, flags)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_nlink != 1
            or details.st_size > _MAX_PLAN_BYTES
            or stat.S_IMODE(details.st_mode) & 0o022
        ):
            _reject("blocked-plan-load")
        payload = _read_descriptor(descriptor, _MAX_PLAN_BYTES, "blocked-plan-load")
    except ProjectionPlanError:
        raise
    except (OSError, TypeError, ValueError):
        _reject("blocked-plan-load")
    finally:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)

    try:
        parsed = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        RecursionError,
        TypeError,
        ValueError,
    ):
        _reject("blocked-plan-load")
    validated = validate_projection_plan(parsed)
    canonical = _canonical_json(validated, "blocked-plan-schema")
    if payload not in {canonical, canonical + b"\n"}:
        _reject("blocked-plan-canonical")
    return validated


def validate_projection_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Return an immutable-by-copy, canonicalized plan or raise a specific refusal."""
    if not isinstance(plan, Mapping):
        _reject("blocked-plan-schema")
    try:
        normalized = json.loads(
            _canonical_json(plan, "blocked-plan-schema").decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        _reject("blocked-plan-schema")
    if not isinstance(normalized, dict):
        _reject("blocked-plan-schema")

    _require_exact_keys(
        normalized,
        {
            "approvedOriginSourceIds",
            "configHandoff",
            "excludedSourceIds",
            "planSha256",
            "policy",
            "preconditions",
            "providerActions",
            "removeProviders",
            "result",
            "schemaVersion",
            "source",
            "status",
        },
        "blocked-plan-schema",
    )
    if (
        normalized["status"] != "planned"
        or normalized["schemaVersion"] != _SCHEMA_VERSION
    ):
        _reject("blocked-plan-schema")
    if not _is_digest(normalized["planSha256"]):
        _reject("blocked-plan-schema")

    source_ids = _validate_id_list(
        normalized["approvedOriginSourceIds"], "blocked-plan-schema"
    )
    exclusions = _validate_id_list(
        normalized["excludedSourceIds"], "blocked-plan-schema"
    )
    if set(source_ids) & set(exclusions):
        _reject("blocked-exclusion-intersection")

    actions = normalized["providerActions"]
    if not isinstance(actions, list) or not actions:
        _reject("blocked-plan-schema")
    action_source_ids: list[str] = []
    action_providers: list[str] = []
    for action in actions:
        _require_exact_keys(
            action,
            {
                "action",
                "models",
                "provider",
                "sourceId",
                "sourceProviderSha256",
                "targetProviderSha256",
            },
            "blocked-plan-schema",
        )
        if action["action"] != "retain":
            _reject("blocked-plan-action")
        source_id = _require_provider_id(action["sourceId"], "blocked-plan-schema")
        provider = _require_provider_id(action["provider"], "blocked-plan-schema")
        models = _validate_model_id_list(action["models"], "blocked-plan-schema")
        if (
            not models
            or not _is_digest(action["sourceProviderSha256"])
            or not _is_digest(action["targetProviderSha256"])
        ):
            _reject("blocked-plan-schema")
        action_source_ids.append(source_id)
        action_providers.append(provider)
    if action_source_ids != sorted(action_source_ids) or len(
        set(action_source_ids)
    ) != len(action_source_ids):
        _reject("blocked-plan-schema")
    if len(set(action_providers)) != len(action_providers):
        _reject("blocked-plan-action")
    if source_ids != action_source_ids:
        _reject("blocked-origin-approval")

    _require_exact_keys(
        normalized["source"], {"appType", "snapshotSha256"}, "blocked-plan-schema"
    )
    if normalized["source"]["appType"] != "claude" or not _is_digest(
        normalized["source"]["snapshotSha256"]
    ):
        _reject("blocked-plan-schema")

    _require_exact_keys(
        normalized["preconditions"], {"source", "target"}, "blocked-plan-schema"
    )
    _validate_source_precondition(
        normalized["preconditions"]["source"], "blocked-plan-schema"
    )
    if (
        normalized["preconditions"]["source"]["snapshotSha256"]
        != normalized["source"]["snapshotSha256"]
    ):
        _reject("blocked-plan-schema")
    _validate_target_precondition(
        normalized["preconditions"]["target"], "blocked-plan-schema"
    )

    _require_exact_keys(
        normalized["policy"],
        {"allowOpaqueBackup", "requireTargetHeader"},
        "blocked-plan-schema",
    )
    if normalized["policy"] != {
        "allowOpaqueBackup": False,
        "requireTargetHeader": True,
    }:
        _reject("blocked-plan-schema")

    removals = normalized["removeProviders"]
    if (
        not isinstance(removals, list)
        or any(not isinstance(item, str) for item in removals)
        or removals != sorted(set(removals))
    ):
        _reject("blocked-plan-schema")
    if any(item != "cc-deepseek" for item in removals):
        _reject("blocked-plan-action")

    selectors = _selectors_from_actions(actions)
    _require_exact_keys(
        normalized["configHandoff"], {"addModelSelectors"}, "blocked-plan-schema"
    )
    if normalized["configHandoff"]["addModelSelectors"] != selectors:
        _reject("blocked-plan-action")

    _require_exact_keys(
        normalized["result"],
        {"targetBytesSha256", "targetDocumentSha256"},
        "blocked-plan-schema",
    )
    if not all(_is_digest(normalized["result"][name]) for name in normalized["result"]):
        _reject("blocked-plan-schema")
    if normalized["planSha256"] != _plan_hash(normalized):
        _reject("blocked-plan-hash")
    return normalized


def verify_projection(
    plan: Mapping[str, Any], db_path: str | Path, target_path: str | Path
) -> bool:
    """Verify the exact prospective bytes and selected source-to-target mapping."""
    try:
        _require_secure_platform()
        accepted = validate_projection_plan(plan)
        source_ids, providers, selections, exclusions = _plan_inputs(accepted)
        source = _source_snapshot(
            db_path, source_ids, accepted["approvedOriginSourceIds"]
        )
        if source["precondition"] != accepted["preconditions"]["source"]:
            return False
        if _source_plan_fragment(source) != accepted["source"]:
            return False

        location = _target_location(target_path)
        parent_fd = _open_target_parent(location)
        try:
            target = _snapshot_target_at(parent_fd, location)
        finally:
            os.close(parent_fd)
        expected_target = accepted["preconditions"]["target"]
        if (
            target.precondition["realpath"] != expected_target["realpath"]
            or target.precondition["mode"] != expected_target["mode"]
            or not _same_parent_identity(
                target.precondition["parent"],
                expected_target["parent"],
                include_ctime=False,
            )
        ):
            return False
        if (
            hashlib.sha256(target.raw).hexdigest()
            != accepted["result"]["targetBytesSha256"]
        ):
            return False
        _validate_document(target.document, "blocked-target-provider")
        actions = _derive_actions(source, target.document, providers, selections)
        if actions != accepted["providerActions"]:
            return False
        if (
            _document_sha256(target.document)
            != accepted["result"]["targetDocumentSha256"]
        ):
            return False
        if (
            "cc-deepseek" in accepted["removeProviders"]
            and "cc-deepseek" in target.document["providers"]
        ):
            return False
        if set(accepted["removeProviders"]) - {"cc-deepseek"}:
            return False
        if set(source_ids) & set(exclusions):
            return False
        return True
    except (OSError, TypeError, ValueError, SourceInventoryError, ProjectionPlanError):
        return False


def apply_projection_plan(
    plan: Mapping[str, Any],
    db_path: str | Path,
    target_path: str | Path,
    *,
    accepted_plan_sha256: str,
    allow_opaque_backup: bool = False,
) -> dict[str, Any]:
    """Apply an accepted plan under locks on the old and replacement target inodes.

    Advisory locks protect cooperative writers.  A same-UID process that ignores
    those locks can still race the kernel's non-conditional ``rename`` primitive,
    so the last path/FD comparison is deliberately immediately before publish.
    """
    _require_secure_platform()
    accepted = validate_projection_plan(plan)
    if (
        not _is_digest(accepted_plan_sha256)
        or accepted_plan_sha256 != accepted["planSha256"]
    ):
        _reject("blocked-accepted-plan-sha256")
    if allow_opaque_backup is not True:
        _reject("blocked-backup-consent")

    location = _target_location(target_path)
    parent_fd = _open_target_parent(location)
    locked: _LockedTarget | None = None
    root_fd = -1
    run_fd = -1
    run_name: str | None = None
    backup_fd = -1
    backup_name: str | None = None
    replacement: _TrustedTemp | None = None
    rendered: bytes | None = None
    published = False
    rollback_succeeded = False
    report: dict[str, Any] | None = None
    try:
        locked = _open_locked_target(parent_fd, location)
        original = locked.snapshot
        _assert_precondition(
            original.precondition, accepted["preconditions"]["target"], "target"
        )

        source_ids, providers, selections, exclusions = _plan_inputs(accepted)
        source = _source_snapshot(
            db_path, source_ids, accepted["approvedOriginSourceIds"]
        )
        _assert_precondition(
            source["precondition"], accepted["preconditions"]["source"], "source"
        )
        derived = _derive_operations(
            source, original, providers, selections, exclusions
        )
        _assert_derived_operations(accepted, derived)
        rendered = derived["_rendered"]

        # No content operation means no backup/run-directory/temp/replace side effect.
        if rendered == original.raw:
            if not verify_projection(accepted, db_path, target_path):
                _reject("blocked-apply-verify")
            return {
                "status": "unchanged",
                "target": accepted["preconditions"]["target"]["realpath"],
                "withinRun": {"deleted": [], "unavailable": []},
                "rollback": {"retained": False, "withinRun": False},
                "cleanup": {
                    "backup": "not-created",
                    "runDirectory": "not-created",
                    "temporary": "not-created",
                },
            }

        root_fd = _secure_rollback_root(parent_fd)
        run_name, run_fd = _create_run_directory(root_fd)
        backup_name = "target.backup"
        backup_fd = _write_exclusive_file(run_fd, backup_name, original.raw, 0o600)
        _fsync_directory(run_fd)
        _fsync_directory(root_fd)
        _fsync_directory(parent_fd)

        replacement = _write_target_temp(parent_fd, location.basename, rendered)
        _acquire_file_lock(replacement.descriptor, "blocked-target-lock")
        publish_parent = _target_parent_identity(parent_fd)

        # Re-snapshot source and revalidate the path-to-old-inode relationship
        # immediately before the only publication operation.
        fresh_source = _source_snapshot(
            db_path, source_ids, accepted["approvedOriginSourceIds"]
        )
        _assert_precondition(
            fresh_source["precondition"], accepted["preconditions"]["source"], "source"
        )
        if _source_plan_fragment(fresh_source) != accepted["source"]:
            _reject("blocked-source-changed")
        _assert_target_publishable(parent_fd, location, locked, publish_parent)
        _fsync_directory(parent_fd)

        if replacement.name is None:
            _reject("blocked-target-write")
        try:
            os.replace(
                replacement.name,
                location.basename,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            replacement.name = None
            published = True
            _restore_target_mode(
                replacement.descriptor,
                original.precondition["mode"],
                "blocked-target-write",
            )
            os.fsync(replacement.descriptor)
            _fsync_directory(parent_fd)
        except OSError:
            _reject("blocked-target-write")

        if not verify_projection(accepted, db_path, target_path):
            _reject("blocked-apply-verify")
        report = {
            "status": "applied",
            "target": accepted["preconditions"]["target"]["realpath"],
            "withinRun": {"deleted": derived["removeProviders"], "unavailable": []},
            "rollback": {"retained": True, "withinRun": True},
            "cleanup": {
                "backup": "retained",
                "runDirectory": "retained",
                "temporary": "published",
            },
        }
        return report
    except BaseException:
        if published:
            if (
                locked is None
                or replacement is None
                or backup_fd == -1
                or rendered is None
            ):
                _reject("blocked-rollback")
            try:
                _rollback_from_backup(
                    parent_fd,
                    location,
                    backup_fd,
                    replacement.descriptor,
                    original=locked.snapshot,
                    published_sha256=hashlib.sha256(rendered).hexdigest(),
                )
                rollback_succeeded = True
            except (OSError, ProjectionPlanError, TypeError, ValueError):
                _reject("blocked-rollback")
        raise
    finally:
        if replacement is not None:
            if replacement.name is not None:
                _unlink_at_quietly(parent_fd, replacement.name)
            _release_file_lock(replacement.descriptor)
            with contextlib.suppress(OSError):
                os.close(replacement.descriptor)
        if run_fd != -1:
            retain_backup = published and report is None and not rollback_succeeded
            backup_deleted = False if retain_backup else True
            if backup_name is not None and not retain_backup:
                backup_deleted = _unlink_at(run_fd, backup_name)
            if report is not None:
                report["cleanup"]["backup"] = (
                    "deleted" if backup_deleted else "retained"
                )
                report["rollback"]["retained"] = not backup_deleted
            if backup_fd != -1:
                with contextlib.suppress(OSError):
                    os.close(backup_fd)
            with contextlib.suppress(OSError):
                _fsync_directory(run_fd)
            with contextlib.suppress(OSError):
                os.close(run_fd)
        if root_fd != -1 and run_name is not None:
            run_deleted = _rmdir_at(root_fd, run_name)
            if report is not None:
                report["cleanup"]["runDirectory"] = (
                    "deleted" if run_deleted else "retained"
                )
            with contextlib.suppress(OSError):
                _fsync_directory(root_fd)
        if root_fd != -1:
            with contextlib.suppress(OSError):
                os.close(root_fd)
        if locked is not None:
            _release_file_lock(locked.descriptor)
            with contextlib.suppress(OSError):
                os.close(locked.descriptor)
        with contextlib.suppress(OSError):
            _fsync_directory(parent_fd)
        with contextlib.suppress(OSError):
            os.close(parent_fd)


def _validate_planning_inputs(
    provider_map: Mapping[str, str],
    selected_models: Mapping[str, Sequence[str]],
    excluded_source_ids: set[str] | frozenset[str],
    approved_origin_source_ids: set[str] | frozenset[str],
) -> tuple[list[str], dict[str, str], dict[str, list[str]], list[str], list[str]]:
    if not isinstance(provider_map, Mapping) or not isinstance(
        selected_models, Mapping
    ):
        _reject("blocked-projection-contract")
    if not isinstance(excluded_source_ids, (set, frozenset)) or not isinstance(
        approved_origin_source_ids, (set, frozenset)
    ):
        _reject("blocked-projection-contract")

    provider_targets: dict[str, str] = {}
    for source_id, target in provider_map.items():
        source = _require_provider_id(source_id, "blocked-projection-contract")
        provider_targets[source] = _require_provider_id(
            target, "blocked-projection-contract"
        )
    source_ids = sorted(provider_targets)
    if not source_ids or len(provider_targets) != len(provider_map):
        _reject("blocked-projection-contract")
    if len(set(provider_targets.values())) != len(provider_targets):
        _reject("blocked-projection-contract")

    if set(selected_models) != set(source_ids):
        _reject("blocked-model-selection")
    selections: dict[str, list[str]] = {}
    for source_id in source_ids:
        selections[source_id] = _validate_selected_models(
            selected_models[source_id], "blocked-model-selection"
        )

    exclusions = sorted(
        _require_provider_id(value, "blocked-projection-contract")
        for value in excluded_source_ids
    )
    if len(exclusions) != len(excluded_source_ids):
        _reject("blocked-projection-contract")
    if set(exclusions) & set(source_ids):
        _reject("blocked-exclusion-intersection")

    approvals = sorted(
        _require_provider_id(value, "blocked-projection-contract")
        for value in approved_origin_source_ids
    )
    if len(approvals) != len(approved_origin_source_ids):
        _reject("blocked-projection-contract")
    if approvals != source_ids:
        _reject("blocked-origin-approval")
    return source_ids, provider_targets, selections, exclusions, approvals


def _plan_inputs(
    plan: Mapping[str, Any],
) -> tuple[list[str], dict[str, str], dict[str, list[str]], list[str]]:
    providers = {
        action["sourceId"]: action["provider"] for action in plan["providerActions"]
    }
    selections = {
        action["sourceId"]: list(action["models"]) for action in plan["providerActions"]
    }
    return sorted(providers), providers, selections, list(plan["excludedSourceIds"])


def _derive_operations(
    source: Mapping[str, Any],
    target: _TargetSnapshot,
    providers: Mapping[str, str],
    selections: Mapping[str, Sequence[str]],
    exclusions: Sequence[str],
) -> dict[str, Any]:
    document = target.document
    actions = _derive_actions(source, document, providers, selections)
    if set(providers) & set(exclusions):
        _reject("blocked-exclusion-intersection")
    retained = {action["provider"] for action in actions}
    target_providers = document["providers"]
    removals: list[str] = []
    if "cc-deepseek" in target_providers and "cc-deepseek" not in retained:
        removals.append("cc-deepseek")
    after = _document_after_removals(document, removals)
    _validate_document(after, "blocked-target-provider")
    rendered = _render_target_projection(target.raw, document, removals)
    rendered_document = _read_generated_target(rendered)
    _validate_document(rendered_document, "blocked-target-provider")
    if _document_sha256(rendered_document) != _document_sha256(after):
        _reject("blocked-target-splice")
    return {
        "_rendered": rendered,
        "configHandoff": {"addModelSelectors": _selectors_from_actions(actions)},
        "excludedSourceIds": list(exclusions),
        "providerActions": actions,
        "removeProviders": removals,
        "result": {
            "targetBytesSha256": hashlib.sha256(rendered).hexdigest(),
            "targetDocumentSha256": _document_sha256(rendered_document),
        },
        "source": _source_plan_fragment(source),
    }


def _derive_actions(
    source: Mapping[str, Any],
    document: Mapping[str, Any],
    providers: Mapping[str, str],
    selections: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    source_records = source["providers"]
    target_providers = document.get("providers")
    if not isinstance(target_providers, dict):
        _reject("blocked-target-parse")

    actions: list[dict[str, Any]] = []
    for source_id in sorted(providers):
        record = source_records.get(source_id)
        if not isinstance(record, dict):
            _reject("blocked-source-provider")
        target_name = providers[source_id]
        target_provider = target_providers.get(target_name)
        if not isinstance(target_provider, dict):
            _reject("blocked-target-provider")
        _validate_provider(target_provider, "blocked-target-provider")

        expected_api = _SOURCE_API_FAMILIES.get(record["api_family"])
        if expected_api is None:
            _reject("blocked-unsupported-api")
        source_endpoint = _canonical_https_url(record["endpoint"], "blocked-source-url")
        if target_provider.get("api") != expected_api:
            _reject("blocked-api-family-mismatch")
        if (
            _canonical_https_url(target_provider.get("baseUrl"), "blocked-target-url")
            != source_endpoint
        ):
            _reject("blocked-url-mismatch")
        _require_env_reference(record["credential_reference"], "blocked-env-reference")
        _require_env_reference(target_provider.get("apiKey"), "blocked-env-reference")
        _assert_selected_provider_transport(target_provider, "blocked-model-metadata")

        target_models = _effective_model_index(
            target_provider, "blocked-model-metadata"
        )
        direct_target_models = _model_index(
            target_provider.get("models", []), "blocked-model-metadata"
        )
        source_models = _model_index(
            record["model_definitions"], "blocked-model-metadata"
        )
        overrides = target_provider.get("modelOverrides", {})
        if not isinstance(overrides, dict):
            _reject("blocked-model-metadata")
        declared = set(record["declared_models"])
        chosen = _validate_selected_models(
            selections[source_id], "blocked-model-selection"
        )
        for model_id in chosen:
            if model_id not in declared:
                _reject("blocked-model-selection")
            source_model = source_models.get(model_id)
            direct_target_model = direct_target_models.get(model_id)
            target_model = target_models.get(model_id)
            override = overrides.get(model_id)
            if (
                source_model is None
                or direct_target_model is None
                or target_model is None
            ):
                _reject("blocked-model-metadata")
            _assert_selected_model_transport(
                source_model,
                direct_target_model,
                override,
                target_model,
                target_provider,
                expected_api,
                source_endpoint,
            )
            if not _source_model_matches(source_model, target_model):
                _reject("blocked-model-metadata")
            if target_model.get("api", target_provider["api"]) != expected_api:
                _reject("blocked-api-family-mismatch")
            if (
                _canonical_https_url(
                    target_model.get("baseUrl", target_provider["baseUrl"]),
                    "blocked-target-url",
                )
                != source_endpoint
            ):
                _reject("blocked-url-mismatch")

        actions.append({
            "action": "retain",
            "models": chosen,
            "provider": target_name,
            "sourceId": source_id,
            "sourceProviderSha256": _source_provider_sha256(record),
            "targetProviderSha256": _json_sha256(target_provider),
        })
    return actions


def _selectors_from_actions(
    actions: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {"provider": action["provider"], "model": model}
        for action in actions
        for model in action["models"]
    ]


def _source_snapshot(
    db_path: str | Path,
    source_ids: Sequence[str],
    approved_origin_source_ids: Sequence[str],
) -> dict[str, Any]:
    try:
        raw = materialization_source(
            db_path,
            selected_source_ids=frozenset(source_ids),
            approved_origin_source_ids=frozenset(approved_origin_source_ids),
        )
    except SourceInventoryError as error:
        _reject(error.code)
    except (OSError, TypeError, ValueError):
        _reject("blocked-source-read")

    _require_exact_keys(
        raw,
        {"precondition", "providers", "source_app_type", "source_snapshot_sha256"},
        "blocked-source-read",
    )
    if raw["source_app_type"] != "claude" or not _is_digest(
        raw["source_snapshot_sha256"]
    ):
        _reject("blocked-source-read")
    _validate_source_precondition(raw["precondition"], "blocked-source-read")
    raw_providers = raw["providers"]
    if not isinstance(raw_providers, list):
        _reject("blocked-source-read")

    records: dict[str, dict[str, Any]] = {}
    for record in raw_providers:
        _require_exact_keys(
            record,
            {
                "api_family",
                "app_type",
                "blocker_codes",
                "credential_reference",
                "declared_models",
                "endpoint",
                "is_current",
                "model_definitions",
                "source_id",
            },
            "blocked-source-read",
        )
        source_id = _require_provider_id(record["source_id"], "blocked-source-read")
        if (
            source_id in records
            or record["app_type"] != "claude"
            or type(record["is_current"]) is not bool
        ):
            _reject("blocked-source-provider")
        blockers = record["blocker_codes"]
        if not isinstance(blockers, list) or any(
            not isinstance(code, str) or not code.startswith("blocked-")
            for code in blockers
        ):
            _reject("blocked-source-read")
        if blockers:
            _reject(blockers[0])
        if record["api_family"] not in _SOURCE_API_FAMILIES:
            _reject("blocked-unsupported-api")
        _canonical_https_url(record["endpoint"], "blocked-source-url")
        _require_env_reference(record["credential_reference"], "blocked-env-reference")
        declared = _validate_distinct_model_id_list(
            record["declared_models"], "blocked-source-model-missing"
        )
        if not declared:
            _reject("blocked-source-model-missing")
        models = _model_index(record["model_definitions"], "blocked-model-metadata")
        if not set(models).issubset(declared):
            _reject("blocked-model-metadata")
        for model in models.values():
            _assert_source_model_secret_free(model, "blocked-model-metadata")
        records[source_id] = {
            "api_family": record["api_family"],
            "credential_reference": record["credential_reference"],
            "declared_models": declared,
            "endpoint": _canonical_https_url(record["endpoint"], "blocked-source-url"),
            "model_definitions": [models[model_id] for model_id in sorted(models)],
            "source_id": source_id,
        }
    if set(records) != set(source_ids):
        _reject("blocked-source-provider")
    return {
        "precondition": dict(raw["precondition"]),
        "providers": records,
        "source_app_type": "claude",
        "source_snapshot_sha256": raw["source_snapshot_sha256"],
    }


def _source_plan_fragment(source: Mapping[str, Any]) -> dict[str, str]:
    return {"appType": "claude", "snapshotSha256": source["source_snapshot_sha256"]}


def _source_provider_sha256(record: Mapping[str, Any]) -> str:
    return _json_sha256(record)


def _target_location(value: str | Path) -> _TargetLocation:
    try:
        raw = os.fspath(value)
        if isinstance(raw, bytes):
            raw = os.fsdecode(raw)
        if not isinstance(raw, str) or not raw or "\x00" in raw:
            raise ValueError
        absolute = os.path.normpath(os.path.abspath(raw))
        parts = tuple(part for part in absolute.split(os.sep) if part)
        if not parts:
            raise ValueError
        return _TargetLocation(
            absolute_path=absolute,
            parent_components=parts[:-1],
            basename=parts[-1],
            # Every component is opened with O_NOFOLLOW, so this is the only
            # meaningful stable path identity; resolving symlinks would reopen
            # attacker-controlled names.
            realpath=absolute,
        )
    except (OSError, TypeError, ValueError):
        _reject("blocked-target-path")


def _open_target_parent(location: _TargetLocation) -> int:
    descriptor = -1
    try:
        descriptor = os.open(
            os.path.sep, os.O_RDONLY | _directory_flag() | _close_on_exec_flag()
        )
        for index, component in enumerate(location.parent_components):
            child = os.open(
                component,
                os.O_RDONLY
                | _directory_flag()
                | _close_on_exec_flag()
                | _no_follow_flag(),
                dir_fd=descriptor,
            )
            os.close(descriptor)
            descriptor = child
            _validate_directory_component(
                os.fstat(descriptor),
                final=index == len(location.parent_components) - 1,
            )
        if not location.parent_components:
            _validate_directory_component(os.fstat(descriptor), final=True)
        return descriptor
    except OSError:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        _reject("blocked-target-path")


def _validate_directory_component(details: os.stat_result, *, final: bool) -> None:
    if not stat.S_ISDIR(details.st_mode):
        raise OSError
    mode = stat.S_IMODE(details.st_mode)
    if final:
        if not _owned_by_current_user(details.st_uid) or mode & 0o022:
            raise OSError
        return
    if details.st_uid not in _trusted_directory_owners():
        raise OSError
    if mode & 0o022 and not (details.st_mode & stat.S_ISVTX):
        raise OSError


def _target_parent_identity(parent_fd: int) -> dict[str, int]:
    details = os.fstat(parent_fd)
    if not stat.S_ISDIR(details.st_mode):
        _reject("blocked-target-path")
    return {
        "device": details.st_dev,
        "fileId": details.st_ino,
        "ctimeNs": details.st_ctime_ns,
    }


def _open_target_descriptor(parent_fd: int, location: _TargetLocation) -> int:
    try:
        descriptor = os.open(
            location.basename,
            os.O_RDONLY
            | _close_on_exec_flag()
            | _no_follow_flag()
            | _non_blocking_flag(),
            dir_fd=parent_fd,
        )
    except OSError as error:
        if error.errno == errno.ELOOP:
            _reject("blocked-target-symlink")
        _reject("blocked-target-path")
    try:
        _validate_target_descriptor(os.fstat(descriptor))
        return descriptor
    except BaseException:
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _validate_target_descriptor(details: os.stat_result) -> None:
    if (
        not stat.S_ISREG(details.st_mode)
        or not _owned_by_current_user(details.st_uid)
        or details.st_nlink != 1
        or details.st_size > _MAX_TARGET_BYTES
        or stat.S_IMODE(details.st_mode) & 0o022
    ):
        _reject("blocked-target-path")


def _snapshot_target_at(parent_fd: int, location: _TargetLocation) -> _TargetSnapshot:
    descriptor = _open_target_descriptor(parent_fd, location)
    try:
        return _snapshot_target_descriptor(parent_fd, location, descriptor)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _snapshot_target_descriptor(
    parent_fd: int, location: _TargetLocation, descriptor: int
) -> _TargetSnapshot:
    try:
        before = os.fstat(descriptor)
        _validate_target_descriptor(before)
        os.lseek(descriptor, 0, os.SEEK_SET)
        raw = _read_descriptor(descriptor, _MAX_TARGET_BYTES, "blocked-target-parse")
        after = os.fstat(descriptor)
        _validate_target_descriptor(after)
    except ProjectionPlanError:
        raise
    except OSError:
        _reject("blocked-target-path")
    if not _same_target_file_state(before, after):
        _reject("blocked-target-changed")

    document = _read_generated_target(raw)
    _ensure_json_value(document, "blocked-target-parse")
    precondition = {
        "realpath": location.realpath,
        "device": after.st_dev,
        "fileId": after.st_ino,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "mode": stat.S_IMODE(after.st_mode),
        "ctimeNs": after.st_ctime_ns,
        "parent": _target_parent_identity(parent_fd),
    }
    return _TargetSnapshot(precondition=precondition, raw=raw, document=document)


def _same_target_file_state(left: os.stat_result, right: os.stat_result) -> bool:
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


def _open_locked_target(parent_fd: int, location: _TargetLocation) -> _LockedTarget:
    descriptor = _open_target_descriptor(parent_fd, location)
    try:
        _acquire_file_lock(descriptor, "blocked-target-lock")
        snapshot = _snapshot_target_descriptor(parent_fd, location, descriptor)
        _assert_target_path_matches_descriptor(parent_fd, location, descriptor)
        return _LockedTarget(descriptor=descriptor, snapshot=snapshot)
    except BaseException:
        _release_file_lock(descriptor)
        with contextlib.suppress(OSError):
            os.close(descriptor)
        raise


def _assert_target_publishable(
    parent_fd: int,
    location: _TargetLocation,
    locked: _LockedTarget,
    expected_parent: Mapping[str, Any],
) -> None:
    _assert_parent_path_matches_descriptor(location, parent_fd)
    current = _snapshot_target_descriptor(parent_fd, location, locked.descriptor)
    expected = locked.snapshot.precondition
    current_target = current.precondition
    for key in {"realpath", "device", "fileId", "sha256", "mode", "ctimeNs"}:
        if current_target[key] != expected[key]:
            _reject("blocked-target-changed")
    if not _same_parent_identity(
        current_target["parent"], expected_parent, include_ctime=True
    ):
        _reject("blocked-target-changed")
    _assert_target_path_matches_descriptor(parent_fd, location, locked.descriptor)


def _assert_parent_path_matches_descriptor(
    location: _TargetLocation, descriptor: int
) -> None:
    current_fd = _open_target_parent(location)
    try:
        current = os.fstat(current_fd)
        expected = os.fstat(descriptor)
        if (current.st_dev, current.st_ino) != (expected.st_dev, expected.st_ino):
            _reject("blocked-target-changed")
    finally:
        with contextlib.suppress(OSError):
            os.close(current_fd)


def _assert_target_path_matches_descriptor(
    parent_fd: int, location: _TargetLocation, descriptor: int
) -> None:
    current_fd = _open_target_descriptor(parent_fd, location)
    try:
        current = os.fstat(current_fd)
        expected = os.fstat(descriptor)
        if (current.st_dev, current.st_ino, current.st_nlink) != (
            expected.st_dev,
            expected.st_ino,
            expected.st_nlink,
        ):
            _reject("blocked-target-changed")
    finally:
        with contextlib.suppress(OSError):
            os.close(current_fd)


def _read_generated_target(raw: bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _reject("blocked-target-parse")
    if not text.startswith(_GENERATED_HEADER + "\n"):
        _reject("blocked-target-header")
    document, _node = _load_bounded_yaml(
        text[len(_GENERATED_HEADER) + 1 :], "blocked-target-parse"
    )
    _require_exact_keys(document, {"providers"}, "blocked-target-parse")
    providers = document["providers"]
    if not isinstance(providers, dict) or any(
        not isinstance(name, str) for name in providers
    ):
        _reject("blocked-target-parse")
    return document


def _load_bounded_yaml(text: str, code: str) -> tuple[dict[str, Any], Node]:
    loader: _BoundedUniqueKeySafeLoader | None = None
    try:
        loader = _BoundedUniqueKeySafeLoader(text)
        node = loader.get_single_node()
        if node is None:
            _reject(code)
        document = loader.construct_document(node)
    except ProjectionPlanError:
        raise
    except (yaml.YAMLError, RecursionError, TypeError, ValueError):
        _reject(code)
    finally:
        if loader is not None:
            with contextlib.suppress(Exception):
                loader.dispose()
    if not isinstance(document, dict):
        _reject(code)
    return document, node


def _document_after_removals(
    document: Mapping[str, Any], removals: Sequence[str]
) -> dict[str, Any]:
    clone = _json_clone(document, "blocked-target-parse")
    providers = clone.get("providers")
    if not isinstance(providers, dict):
        _reject("blocked-target-parse")
    for provider in removals:
        if provider != "cc-deepseek":
            _reject("blocked-plan-action")
        providers.pop(provider, None)
    return clone


def _render_target_projection(
    raw: bytes, document: Mapping[str, Any], removals: Sequence[str]
) -> bytes:
    if not removals:
        return raw
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        _reject("blocked-target-parse")
    if not text.startswith(_GENERATED_HEADER + "\n"):
        _reject("blocked-target-header")
    body = text[len(_GENERATED_HEADER) + 1 :]
    parsed, root = _load_bounded_yaml(body, "blocked-target-splice")
    if _document_sha256(parsed) != _document_sha256(document):
        _reject("blocked-target-splice")
    spliced = _splice_provider_removals(body, root, removals)
    rendered = (_GENERATED_HEADER + "\n" + spliced).encode("utf-8")
    rendered_document = _read_generated_target(rendered)
    _ensure_json_value(rendered_document, "blocked-target-splice")
    return rendered


def _splice_provider_removals(body: str, root: Node, removals: Sequence[str]) -> str:
    if list(removals) != ["cc-deepseek"]:
        _reject("blocked-target-splice")
    if not isinstance(root, MappingNode) or root.flow_style is not False:
        _reject("blocked-target-splice")
    providers_key: ScalarNode | None = None
    providers_node: MappingNode | None = None
    for key_node, value_node in root.value:
        if not isinstance(key_node, ScalarNode):
            _reject("blocked-target-splice")
        if key_node.value == "providers":
            if (
                not isinstance(value_node, MappingNode)
                or value_node.flow_style is not False
            ):
                _reject("blocked-target-splice")
            providers_key = key_node
            providers_node = value_node
            break
    if providers_key is None or providers_node is None or len(providers_node.value) < 2:
        _reject("blocked-target-splice")

    entry_index: int | None = None
    for index, (key_node, _value_node) in enumerate(providers_node.value):
        if not isinstance(key_node, ScalarNode):
            _reject("blocked-target-splice")
        if key_node.value == "cc-deepseek":
            entry_index = index
            break
    if entry_index is None:
        _reject("blocked-target-splice")

    key_node, value_node = providers_node.value[entry_index]
    if key_node.start_mark.column <= providers_key.start_mark.column:
        _reject("blocked-target-splice")
    start = _line_start(body, key_node.start_mark.index)
    end = _line_end_after_syntactic_content(
        body, _last_syntactic_content_end(value_node)
    )
    if entry_index + 1 < len(providers_node.value):
        next_key, _next_value = providers_node.value[entry_index + 1]
        if not isinstance(next_key, ScalarNode):
            _reject("blocked-target-splice")
        next_start = _line_start(body, next_key.start_mark.index)
        if end > next_start:
            _reject("blocked-target-splice")
    if any(
        not character.isspace() for character in body[start : key_node.start_mark.index]
    ):
        _reject("blocked-target-splice")
    if start >= end:
        _reject("blocked-target-splice")
    return body[:start] + body[end:]


def _line_start(text: str, index: int) -> int:
    return text.rfind("\n", 0, index) + 1


def _line_end(text: str, index: int) -> int:
    ending = text.find("\n", index)
    return len(text) if ending == -1 else ending + 1


def _last_syntactic_content_end(node: Node) -> int:
    if isinstance(node, ScalarNode):
        return node.end_mark.index
    if isinstance(node, (MappingNode, SequenceNode)) and node.flow_style is True:
        return node.end_mark.index
    if isinstance(node, MappingNode):
        children = [child for entry in node.value for child in entry]
    elif isinstance(node, SequenceNode):
        children = list(node.value)
    else:
        _reject("blocked-target-splice")
    if not children:
        return node.end_mark.index
    return max(_last_syntactic_content_end(child) for child in children)


def _line_end_after_syntactic_content(text: str, index: int) -> int:
    if index < 0 or index > len(text):
        _reject("blocked-target-splice")
    while index > 0 and text[index - 1] in "\r\n":
        index -= 1
    return _line_end(text, index)


def _assert_derived_operations(
    plan: Mapping[str, Any], derived: Mapping[str, Any]
) -> None:
    for key in {
        "configHandoff",
        "providerActions",
        "removeProviders",
        "result",
        "source",
    }:
        if plan[key] != derived[key]:
            _reject("blocked-plan-action")
    if plan["excludedSourceIds"] != derived["excludedSourceIds"]:
        _reject("blocked-exclusion-intersection")


def _assert_precondition(
    actual: Mapping[str, Any], expected: Mapping[str, Any], kind: str
) -> None:
    if kind == "target":
        actual_without_parent = {
            key: value for key, value in actual.items() if key != "parent"
        }
        expected_without_parent = {
            key: value for key, value in expected.items() if key != "parent"
        }
        if (
            actual_without_parent != expected_without_parent
            or not _same_parent_identity(
                actual.get("parent"), expected.get("parent"), include_ctime=False
            )
        ):
            _reject("blocked-target-changed")
        return
    if actual != expected:
        _reject(f"blocked-{kind}-changed")


def _acquire_file_lock(descriptor: int, code: str) -> None:
    if fcntl is None:
        _reject("blocked-unsupported-platform")
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except (AttributeError, OSError):
        _reject(code)


def _release_file_lock(descriptor: int) -> None:
    if fcntl is not None:
        with contextlib.suppress(OSError):
            fcntl.flock(descriptor, fcntl.LOCK_UN)


def _secure_rollback_root(parent_fd: int) -> int:
    name = ".omp-projection-rollback"
    try:
        os.mkdir(name, 0o700, dir_fd=parent_fd)
        _fsync_directory(parent_fd)
    except FileExistsError:
        pass
    except OSError:
        _reject("blocked-rollback-path")
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDONLY | _directory_flag() | _close_on_exec_flag() | _no_follow_flag(),
            dir_fd=parent_fd,
        )
        details = os.fstat(descriptor)
        if (
            not stat.S_ISDIR(details.st_mode)
            or not _owned_by_current_user(details.st_uid)
            or stat.S_IMODE(details.st_mode) != 0o700
        ):
            raise OSError
        _fsync_directory(descriptor)
        return descriptor
    except OSError:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        _reject("blocked-rollback-path")


def _create_run_directory(root_fd: int) -> tuple[str, int]:
    for _ in range(32):
        name = "run-" + secrets.token_hex(16)
        try:
            os.mkdir(name, 0o700, dir_fd=root_fd)
        except FileExistsError:
            continue
        except OSError:
            _reject("blocked-rollback-path")
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDONLY
                | _directory_flag()
                | _close_on_exec_flag()
                | _no_follow_flag(),
                dir_fd=root_fd,
            )
            details = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(details.st_mode)
                or not _owned_by_current_user(details.st_uid)
                or stat.S_IMODE(details.st_mode) != 0o700
            ):
                raise OSError
            _fsync_directory(descriptor)
            _fsync_directory(root_fd)
            return name, descriptor
        except OSError:
            if descriptor != -1:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
            with contextlib.suppress(OSError):
                os.rmdir(name, dir_fd=root_fd)
            _reject("blocked-rollback-path")
    _reject("blocked-rollback-path")


def _write_exclusive_file(
    directory_fd: int, name: str, payload: bytes, mode: int
) -> int:
    descriptor = -1
    try:
        descriptor = os.open(
            name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | _close_on_exec_flag()
            | _no_follow_flag(),
            mode,
            dir_fd=directory_fd,
        )
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or not _owned_by_current_user(details.st_uid)
            or details.st_nlink != 1
        ):
            raise OSError
        os.fchmod(descriptor, 0o600)
        _write_all(descriptor, payload)
        os.fsync(descriptor)
        return descriptor
    except OSError:
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        _unlink_at_quietly(directory_fd, name)
        _reject("blocked-rollback-path")


def _restore_target_mode(descriptor: int, mode: Any, code: str) -> None:
    if not _is_nonnegative_int(mode) or mode > 0o777 or mode & 0o022:
        _reject(code)
    os.fchmod(descriptor, mode)


def _write_target_temp(parent_fd: int, basename: str, payload: bytes) -> _TrustedTemp:
    for _ in range(32):
        name = f".{basename}.omp-projection-{secrets.token_hex(16)}.tmp"
        descriptor = -1
        try:
            descriptor = os.open(
                name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | _close_on_exec_flag()
                | _no_follow_flag(),
                0o600,
                dir_fd=parent_fd,
            )
        except FileExistsError:
            continue
        except OSError:
            _reject("blocked-target-write")
        try:
            details = os.fstat(descriptor)
            if (
                not stat.S_ISREG(details.st_mode)
                or not _owned_by_current_user(details.st_uid)
                or details.st_nlink != 1
            ):
                raise OSError
            # Remain private while this entry is still a named temporary; only
            # restore the original mode after its atomic publication.
            os.fchmod(descriptor, 0o600)
            _write_all(descriptor, payload)
            os.fsync(descriptor)
            _fsync_directory(parent_fd)
            return _TrustedTemp(name=name, descriptor=descriptor)
        except OSError:
            _unlink_at_quietly(parent_fd, name)
            with contextlib.suppress(OSError):
                os.close(descriptor)
            _reject("blocked-target-write")
    _reject("blocked-target-write")


def _rollback_from_backup(
    parent_fd: int,
    location: _TargetLocation,
    backup_fd: int,
    replacement_fd: int,
    *,
    original: _TargetSnapshot,
    published_sha256: str,
) -> None:
    _assert_published_target(parent_fd, location, replacement_fd, published_sha256)
    details = os.fstat(backup_fd)
    if (
        not stat.S_ISREG(details.st_mode)
        or not _owned_by_current_user(details.st_uid)
        or stat.S_IMODE(details.st_mode) != 0o600
        or details.st_size > _MAX_TARGET_BYTES
    ):
        _reject("blocked-rollback")
    os.lseek(backup_fd, 0, os.SEEK_SET)
    payload = _read_descriptor(backup_fd, _MAX_TARGET_BYTES, "blocked-rollback")
    if hashlib.sha256(payload).hexdigest() != original.precondition["sha256"]:
        _reject("blocked-rollback")
    restore = _write_target_temp(parent_fd, location.basename, payload)
    try:
        _acquire_file_lock(restore.descriptor, "blocked-rollback")
        _assert_published_target(parent_fd, location, replacement_fd, published_sha256)
        if restore.name is None:
            _reject("blocked-rollback")
        os.replace(
            restore.name,
            location.basename,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
        )
        restore.name = None
        _restore_target_mode(
            restore.descriptor, original.precondition["mode"], "blocked-rollback"
        )
        os.fsync(restore.descriptor)
        _fsync_directory(parent_fd)
    except OSError:
        _reject("blocked-rollback")
    finally:
        if restore.name is not None:
            _unlink_at_quietly(parent_fd, restore.name)
        _release_file_lock(restore.descriptor)
        with contextlib.suppress(OSError):
            os.close(restore.descriptor)


def _assert_published_target(
    parent_fd: int, location: _TargetLocation, replacement_fd: int, expected_sha256: str
) -> None:
    _assert_target_path_matches_descriptor(parent_fd, location, replacement_fd)
    snapshot = _snapshot_target_descriptor(parent_fd, location, replacement_fd)
    if hashlib.sha256(snapshot.raw).hexdigest() != expected_sha256:
        _reject("blocked-rollback")


def _read_descriptor(descriptor: int, maximum: int, code: str) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        try:
            chunk = os.read(descriptor, 65536)
        except OSError:
            _reject(code)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if total > maximum:
            _reject(code)
        chunks.append(chunk)


def _write_all(descriptor: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise OSError("short write")
        view = view[written:]


def _fsync_directory(descriptor: int) -> None:
    os.fsync(descriptor)


def _unlink_at_quietly(directory_fd: int, name: str) -> None:
    _unlink_at(directory_fd, name)


def _unlink_at(directory_fd: int, name: str) -> bool:
    try:
        os.unlink(name, dir_fd=directory_fd)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _rmdir_at(directory_fd: int, name: str) -> bool:
    try:
        os.rmdir(name, dir_fd=directory_fd)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def _validate_source_precondition(value: Any, code: str) -> None:
    main_manifest_fields = {
        "ctimeNs",
        "device",
        "fileId",
        "mode",
        "mtimeNs",
        "owner",
        "sha256",
        "size",
    }
    _require_exact_keys(
        value,
        main_manifest_fields | {"realpath", "shm", "snapshotSha256", "wal"},
        code,
    )
    if (
        not isinstance(value["realpath"], str)
        or not value["realpath"]
        or not os.path.isabs(value["realpath"])
        or any(ord(char) < 32 for char in value["realpath"])
        or not _is_digest(value["snapshotSha256"])
    ):
        _reject(code)
    _validate_source_file_manifest(
        {field: value[field] for field in main_manifest_fields}, code
    )
    if value["wal"] is not None:
        _validate_source_file_manifest(value["wal"], code)
    if value["shm"] is not None:
        _validate_source_file_manifest(value["shm"], code)


def _validate_source_file_manifest(value: Any, code: str) -> None:
    fields = {
        "ctimeNs",
        "device",
        "fileId",
        "mode",
        "mtimeNs",
        "owner",
        "sha256",
        "size",
    }
    _require_exact_keys(value, fields, code)
    if (
        any(not _is_nonnegative_int(value[field]) for field in fields - {"sha256"})
        or value["mode"] > 0o777
        or value["mode"] & 0o022
        or not _is_digest(value["sha256"])
    ):
        _reject(code)


def _validate_target_precondition(value: Any, code: str) -> None:
    _validate_file_precondition(
        value,
        {"ctimeNs", "device", "fileId", "mode", "parent", "realpath", "sha256"},
        code,
    )
    _validate_parent_identity(value["parent"], code)
    if value["mode"] & 0o022:
        _reject(code)


def _validate_file_precondition(value: Any, keys: set[str], code: str) -> None:
    _require_exact_keys(value, keys, code)
    if (
        not isinstance(value["realpath"], str)
        or not value["realpath"]
        or not os.path.isabs(value["realpath"])
        or any(ord(char) < 32 for char in value["realpath"])
        or not _is_nonnegative_int(value["device"])
        or not _is_nonnegative_int(value["fileId"])
        or not _is_digest(value["sha256"])
        or not _is_nonnegative_int(value["mode"])
        or value["mode"] > 0o777
        or not _is_nonnegative_int(value["ctimeNs"])
    ):
        _reject(code)


def _validate_parent_identity(value: Any, code: str) -> None:
    _require_exact_keys(value, {"ctimeNs", "device", "fileId"}, code)
    if not all(_is_nonnegative_int(value[key]) for key in value):
        _reject(code)


def _same_parent_identity(
    left: Mapping[str, Any], right: Mapping[str, Any], *, include_ctime: bool
) -> bool:
    keys = {"device", "fileId", "ctimeNs"} if include_ctime else {"device", "fileId"}
    return all(left.get(key) == right.get(key) for key in keys)


def _trusted_directory_owners() -> set[int]:
    return _owned_user_ids() | {0}


def _owned_by_current_user(owner: int) -> bool:
    return owner in _owned_user_ids()


def _owned_user_ids() -> set[int]:
    allowed = {os.getuid()}
    effective_uid = getattr(os, "geteuid", None)
    if effective_uid is not None:
        allowed.add(effective_uid())
    return allowed


def _is_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _plan_hash(plan: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical_json(
            {key: value for key, value in plan.items() if key != "planSha256"},
            "blocked-plan-schema",
        )
    ).hexdigest()


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def _require_secure_platform() -> None:
    if fcntl is None:
        _reject("blocked-unsupported-platform")
    for flag in (getattr(os, "O_NOFOLLOW", None), getattr(os, "O_DIRECTORY", None)):
        if not isinstance(flag, int) or flag == 0:
            _reject("blocked-unsupported-platform")


def _reject_json_constant(_constant: str) -> None:
    raise ValueError


def _no_follow_flag() -> int:
    flag = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(flag, int) or flag == 0:
        _reject("blocked-no-follow")
    return flag


def _directory_flag() -> int:
    flag = getattr(os, "O_DIRECTORY", None)
    if not isinstance(flag, int) or flag == 0:
        _reject("blocked-no-follow")
    return flag


def _close_on_exec_flag() -> int:
    return getattr(os, "O_CLOEXEC", 0)


def _non_blocking_flag() -> int:
    flag = getattr(os, "O_NONBLOCK", 0)
    return flag if isinstance(flag, int) and flag > 0 else 0


def _reject(code: str) -> None:
    raise ProjectionPlanError(code)


def build_provider_migration_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Lazy bridge to the separate opt-in three-file migration contract."""
    from tools.omp_provider_migration import build_provider_migration_plan as build

    return build(*args, **kwargs)


def load_provider_migration_plan(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a strict provider-migration plan without changing legacy plans."""
    from tools.omp_provider_migration import load_provider_migration_plan as load

    return load(path)


def validate_provider_migration_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a strict schema-v4 provider-migration plan via its bridge."""
    from tools.omp_provider_migration import validate_provider_migration_plan as validate

    return validate(plan)


def verify_provider_migration(*args: Any, **kwargs: Any) -> bool:
    """Read-only bridge for the opt-in provider migration verifier."""
    from tools.omp_provider_migration import verify_provider_migration as verify

    return verify(*args, **kwargs)


def apply_provider_migration_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Apply only a separately accepted provider-migration plan."""
    from tools.omp_provider_migration import apply_provider_migration_plan as apply

    return apply(*args, **kwargs)
