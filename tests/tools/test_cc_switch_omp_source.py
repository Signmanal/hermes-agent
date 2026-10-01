"""Contract tests for the hardened CC Switch Claude source seam."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import stat
from pathlib import Path
from typing import Any

import pytest

import tools.cc_switch_omp_source as cc_switch_omp_source
from tools.cc_switch_omp_source import (
    SourceInventoryError,
    inventory_source,
    materialization_source,
)


_SECRET_MARKERS = (
    "marker-anthropic-auth-token",
    "marker-foreign-credential",
    "marker-meta-secret",
)
_PRIVATE_HOST_MARKERS = (
    "private-anthropic.invalid",
    "private-foreign.invalid",
)
_PUBLIC_TOP_LEVEL_FIELDS = {
    "status",
    "source_app_type",
    "source_snapshot_sha256",
    "providers",
}
_PUBLIC_PROVIDER_FIELDS = {
    "source_id",
    "source_name",
    "is_current",
    "api_family",
    "declared_models",
    "environment_models",
    "endpoint_https",
    "endpoint_userinfo_present",
    "endpoint_query_present",
    "endpoint_fragment_present",
    "credential_reference_present",
    "eligibility_status",
    "blocker_codes",
}


def _model_definition(model_id: str = "gpt-5.6-sol") -> dict[str, Any]:
    """A model object using every source-admitted 18.1.19 model field."""
    endpoint = "https://model-definition.invalid/v1"
    return {
        "api": "openai-responses",
        "baseUrl": endpoint,
        "compactionModel": "gpt-5.6-mini",
        "compat": {"supportsReasoningEffort": True},
        "contextPromotionTarget": "gpt-5.6-sol",
        "contextWindow": 200000,
        "cost": {"cacheRead": 0, "cacheWrite": 0, "input": 1, "output": 2},
        "headers": {},
        "id": model_id,
        "imageInputDecoder": "stb",
        "input": ["text", "image"],
        "maxTokens": 8192,
        "name": "GPT-5.6 Sol",
        "omitMaxOutputTokens": False,
        "preferWebsockets": True,
        "premiumMultiplier": 1.0,
        "reasoning": True,
        "supportsTools": True,
        "thinking": {
            "mode": "effort",
            "efforts": ["low", "high"],
            "defaultLevel": "high",
            "effortMap": {"high": "high"},
            "supportsDisplay": True,
            "requiresEffort": False,
        },
        "tokenizer": "claude-v5",
    }


def _claude_row(
    *,
    source_id: str = "shared-provider",
    is_current: Any = 1,
    sort_index: int | None = 1,
) -> dict[str, Any]:
    return {
        "id": source_id,
        "app_type": "claude",
        "name": "Claude Responses",
        "is_current": is_current,
        "sort_index": sort_index,
        "settings_config": {
            "model": "claude-sonnet-4-5-20250929",
            "models": [_model_definition()],
            # This stale field demonstrates that the real Claude environment
            # setting takes precedence even for openai_responses.
            "base_url": "http://ignored-source-shape.invalid/v1",
            "env": {
                "ANTHROPIC_BASE_URL": "https://private-anthropic.invalid/v1/",
                "ANTHROPIC_AUTH_TOKEN": "marker-anthropic-auth-token",
                "ANTHROPIC_MODEL": "claude-opus-4-6",
                "CLAUDE_CODE_SUBAGENT_MODEL": "claude-haiku-4-5-20251001",
                # A substring match must not turn this malicious value into a
                # selectable model.
                "MALICIOUS_MODEL": "gpt-*",
                "OTHER_OPTION": "must-not-appear",
            },
        },
        "meta": {
            "apiFormat": "openai_responses",
            "apiKey": "marker-meta-secret",
            "usage_script": {"baseUrl": "http://ignored-meta-shape.invalid/v1"},
        },
    }


def _foreign_row() -> dict[str, Any]:
    """Same ID under another app type; it must never enter Claude normalization."""
    return {
        "id": "shared-provider",
        "app_type": "openai",
        "name": "Foreign Duplicate",
        "is_current": 1,
        "sort_index": 0,
        "settings_config": {
            "model": "foreign-model",
            "env": {
                "OPENAI_BASE_URL": "https://private-foreign.invalid/v1",
                "OPENAI_API_KEY": "marker-foreign-credential",
            },
        },
        "meta": {"apiFormat": "openai_chat"},
    }


def _create_provider_db(
    tmp_path: Path,
    rows: list[dict[str, Any]],
    *,
    filename: str = "cc-switch-providers.db",
) -> Path:
    db_path = tmp_path / filename
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE providers (
                id TEXT NOT NULL,
                app_type TEXT NOT NULL,
                name TEXT NOT NULL,
                settings_config TEXT NOT NULL,
                meta TEXT NOT NULL,
                is_current INTEGER NOT NULL,
                sort_index INTEGER,
                PRIMARY KEY (id, app_type)
            );
            """
        )
        connection.executemany(
            """
            INSERT INTO providers (
                id, app_type, name, settings_config, meta, is_current, sort_index
            ) VALUES (
                :id, :app_type, :name, :settings_config, :meta, :is_current, :sort_index
            )
            """,
            [
                {
                    **row,
                    "settings_config": json.dumps(row["settings_config"]),
                    "meta": json.dumps(row["meta"]),
                }
                for row in rows
            ],
        )
    return db_path


@pytest.fixture
def inventory_db(tmp_path: Path) -> Path:
    return _create_provider_db(tmp_path, [_claude_row(), _foreign_row()])


def test_inventory_is_secret_free_and_filters_the_composite_source_scope(
    inventory_db: Path,
):
    result = inventory_source(str(inventory_db))

    assert result == inventory_source(str(inventory_db))
    assert set(result) == _PUBLIC_TOP_LEVEL_FIELDS
    assert result["status"] == "inventory"
    assert result["source_app_type"] == "claude"
    assert re.fullmatch(r"[0-9a-f]{64}", result["source_snapshot_sha256"])
    assert result["providers"] == [
        {
            "source_id": "shared-provider",
            "source_name": "Claude Responses",
            "is_current": True,
            "api_family": "openai_responses",
            "declared_models": [
                "claude-sonnet-4-5-20250929",
                "gpt-5.6-sol",
            ],
            "environment_models": {
                "configured_model_values": ["claude-opus-4-6"],
                "name_to_configured_value": {},
            },
            "endpoint_https": True,
            "endpoint_userinfo_present": False,
            "endpoint_query_present": False,
            "endpoint_fragment_present": False,
            "credential_reference_present": True,
            "eligibility_status": "candidate",
            "blocker_codes": [],
        }
    ]
    assert all(
        set(provider) == _PUBLIC_PROVIDER_FIELDS for provider in result["providers"]
    )

    serialized = json.dumps(result, sort_keys=True)
    for marker in (
        *_SECRET_MARKERS,
        *_PRIVATE_HOST_MARKERS,
        "MALICIOUS_MODEL",
        "gpt-*",
    ):
        assert marker not in serialized
    assert "settings_config" not in serialized
    assert '"meta"' not in serialized


@pytest.mark.parametrize(
    ("provider_filter", "expected_ids"),
    [
        ("shared-provider", ["shared-provider"]),
        ("Claude Responses", ["shared-provider"]),
        ("Foreign Duplicate", []),
        ("claude responses", []),
        ("shared", []),
    ],
)
def test_inventory_filters_only_exact_claude_source_id_or_name(
    inventory_db: Path, provider_filter: str, expected_ids: list[str]
):
    result = inventory_source(str(inventory_db), provider_filter=provider_filter)

    assert [provider["source_id"] for provider in result["providers"]] == expected_ids


def test_materialization_source_normalizes_real_claude_shape_without_secrets(
    inventory_db: Path,
):
    result = materialization_source(
        str(inventory_db),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert set(result) == {
        "source_app_type",
        "source_snapshot_sha256",
        "precondition",
        "providers",
    }
    assert result["source_app_type"] == "claude"
    assert re.fullmatch(r"[0-9a-f]{64}", result["source_snapshot_sha256"])
    details = inventory_db.stat()
    assert result["precondition"] == {
        "realpath": str(inventory_db.resolve()),
        "device": details.st_dev,
        "fileId": details.st_ino,
        "owner": details.st_uid,
        "mode": stat.S_IMODE(details.st_mode),
        "size": details.st_size,
        "mtimeNs": details.st_mtime_ns,
        "ctimeNs": details.st_ctime_ns,
        "sha256": hashlib.sha256(inventory_db.read_bytes()).hexdigest(),
        "snapshotSha256": result["source_snapshot_sha256"],
        "wal": None,
        "shm": None,
    }
    assert result["providers"] == [
        {
            "source_id": "shared-provider",
            "app_type": "claude",
            "is_current": True,
            "api_family": "openai_responses",
            "endpoint": "https://private-anthropic.invalid/v1",
            "credential_reference": "ANTHROPIC_AUTH_TOKEN",
            "declared_models": [
                "claude-sonnet-4-5-20250929",
                "gpt-5.6-sol",
            ],
            "model_definitions": [
                {"id": "claude-sonnet-4-5-20250929"},
                _model_definition(),
            ],
            "blocker_codes": [],
        }
    ]

    serialized = json.dumps(result, sort_keys=True)
    for marker in _SECRET_MARKERS:
        assert marker not in serialized
    assert "settings_config" not in serialized
    assert '"meta"' not in serialized


def test_materialization_source_accepts_anthropic_api_key_credentials(tmp_path: Path):
    row = _claude_row()
    environment = row["settings_config"]["env"]
    del environment["ANTHROPIC_AUTH_TOKEN"]
    environment["ANTHROPIC_API_KEY"] = "marker-anthropic-api-key"
    db_path = _create_provider_db(tmp_path, [row])

    result = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert result["providers"][0]["credential_reference"] == "ANTHROPIC_API_KEY"
    assert "marker-anthropic-api-key" not in json.dumps(result, sort_keys=True)


def test_materialization_skips_empty_credential_before_later_supported_key(
    tmp_path: Path,
):
    row = _claude_row()
    environment = row["settings_config"]["env"]
    environment["ANTHROPIC_AUTH_TOKEN"] = " \t "
    environment["ANTHROPIC_API_KEY"] = "marker-fallback-api-key"
    db_path = _create_provider_db(tmp_path, [row])

    result = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert result["providers"][0]["credential_reference"] == "ANTHROPIC_API_KEY"
    assert "marker-fallback-api-key" not in json.dumps(result, sort_keys=True)


def test_composite_provider_identity_prevents_cross_app_overwrite(inventory_db: Path):
    result = materialization_source(
        str(inventory_db),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    provider = result["providers"][0]
    assert provider["app_type"] == "claude"
    assert provider["endpoint"] == "https://private-anthropic.invalid/v1"
    assert provider["credential_reference"] == "ANTHROPIC_AUTH_TOKEN"
    assert "foreign-model" not in provider["declared_models"]


def test_materialization_requires_explicit_approval_for_every_selected_provider(
    tmp_path: Path,
):
    db_path = _create_provider_db(tmp_path, [_claude_row(is_current=0)])

    with pytest.raises(SourceInventoryError) as raised:
        materialization_source(
            str(db_path),
            selected_source_ids={"shared-provider"},
            approved_origin_source_ids=set(),
        )

    assert raised.value.code == "blocked-origin-approval"

    approved = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )
    assert approved["providers"][0]["is_current"] is False


@pytest.mark.parametrize("invalid_current", [2, -1, 1.5, "current"])
def test_inventory_fails_closed_unless_is_current_is_exactly_zero_or_one(
    tmp_path: Path, invalid_current: Any
):
    db_path = _create_provider_db(
        tmp_path,
        [_claude_row(is_current=invalid_current)],
        filename=f"invalid-current-{invalid_current!s}.db",
    )

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_only_allowlisted_environment_keys_contribute_inventory_evidence(
    inventory_db: Path,
):
    provider = inventory_source(str(inventory_db))["providers"][0]

    assert provider["declared_models"] == [
        "claude-sonnet-4-5-20250929",
        "gpt-5.6-sol",
    ]
    assert provider["environment_models"] == {
        "configured_model_values": ["claude-opus-4-6"],
        "name_to_configured_value": {},
    }
    assert "gpt-*" not in json.dumps(provider, sort_keys=True)


def test_inventory_exposes_only_allowlisted_environment_model_evidence(
    tmp_path: Path,
):
    row = _claude_row()
    row["settings_config"]["env"].update(
        {
            "ANTHROPIC_MODEL": "gpt-6-astra[1M]",
            "ANTHROPIC_REASONING_MODEL": "gpt-6-astra[1M]",
            "ANTHROPIC_DEFAULT_FABLE_MODEL": "gpt-6-astra[1M]",
            "ANTHROPIC_DEFAULT_FABLE_MODEL_NAME": "gpt-6-astra",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "claude-opus-4-6[1M]",
            "ANTHROPIC_DEFAULT_OPUS_MODEL_NAME": "opus",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "claude-sonnet-4-5-20250929",
            "ANTHROPIC_DEFAULT_SONNET_MODEL_NAME": "sonnet",
            "ANTHROPIC_API_KEY": "not-a-secret-api-key",
            "PAT": "not-a-secret-pat",
            "UNRELATED_URL": "https://not-a-secret.invalid",
        }
    )
    db_path = _create_provider_db(tmp_path, [row])

    provider = inventory_source(str(db_path))["providers"][0]

    assert provider["declared_models"] == [
        "claude-sonnet-4-5-20250929",
        "gpt-5.6-sol",
    ]
    assert provider["environment_models"] == {
        "configured_model_values": [
            "claude-opus-4-6[1M]",
            "claude-sonnet-4-5-20250929",
            "gpt-6-astra[1M]",
        ],
        "name_to_configured_value": {
            "gpt-6-astra": "gpt-6-astra[1M]",
            "opus": "claude-opus-4-6[1M]",
            "sonnet": "claude-sonnet-4-5-20250929",
        },
    }
    serialized = json.dumps(provider, sort_keys=True)
    for decoy in (
        "not-a-secret-api-key",
        "not-a-secret-pat",
        "https://not-a-secret.invalid",
        "claude-haiku-4-5-20251001",
    ):
        assert decoy not in serialized


def test_declared_models_canonicalize_context_suffixes_but_environment_evidence_does_not(
    tmp_path: Path,
):
    row = _claude_row()
    row["settings_config"]["model"] = "gpt-5.6-sol[1M]"
    row["settings_config"].pop("models")
    row["settings_config"]["env"] = {
        "ANTHROPIC_BASE_URL": "https://private-anthropic.invalid/v1",
        "ANTHROPIC_AUTH_TOKEN": "marker-anthropic-auth-token",
        "ANTHROPIC_MODEL": "gpt-5.6[1m]",
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "gpt-5.6-sol[1M]",
    }
    db_path = _create_provider_db(tmp_path, [row])

    provider = inventory_source(str(db_path))["providers"][0]

    assert provider["declared_models"] == ["gpt-5.6-sol"]
    assert provider["environment_models"] == {
        "configured_model_values": ["gpt-5.6-sol[1M]", "gpt-5.6[1m]"],
        "name_to_configured_value": {},
    }


def test_inventory_fails_closed_for_unsafe_explicit_or_allowed_model_ids(
    tmp_path: Path,
):
    variants: list[tuple[str, dict[str, Any]]] = []

    top_level = _claude_row()
    top_level["settings_config"]["model"] = "gpt-*"
    variants.append(("top-level", top_level))

    configured = _claude_row()
    configured["settings_config"]["models"] = [{"id": "../model"}]
    variants.append(("configured", configured))

    environment = _claude_row()
    environment["settings_config"]["env"]["ANTHROPIC_MODEL"] = "gpt-unsafe[2M]"
    variants.append(("environment", environment))

    for name, row in variants:
        db_path = _create_provider_db(
            tmp_path, [row], filename=f"unsafe-model-{name}.db"
        )
        with pytest.raises(SourceInventoryError) as raised:
            inventory_source(str(db_path))
        assert raised.value.code == "blocked-provider-data"


def test_inventory_preserves_safe_internal_model_id_segments(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["model"] = "vendor/primary"
    row["settings_config"]["models"] = [_model_definition("vendor/selected")]
    db_path = _create_provider_db(tmp_path, [row])

    declared = inventory_source(str(db_path))["providers"][0]["declared_models"]

    assert declared[:2] == ["vendor/primary", "vendor/selected"]


def test_inventory_fails_closed_for_unknown_model_definition_fields(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["untrustedField"] = "not-supported"
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_inventory_fails_closed_for_nonempty_model_headers(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["headers"] = {
        "Authorization": "Bearer marker-model-header-secret"
    }
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_inventory_rejects_nonempty_model_compat_extra_body(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["compat"]["extraBody"] = {
        "Authorization": "marker-nested-extra-body"
    }
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_inventory_rejects_any_model_remote_compaction(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["remoteCompaction"] = {}
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_materialization_accepts_when_thinking_empty_extra_body(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["compat"] = {
        "whenThinking": {"cacheControlFormat": "anthropic", "extraBody": {}}
    }
    db_path = _create_provider_db(tmp_path, [row])

    result = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert result["providers"][0]["model_definitions"][1]["compat"] == {
        "whenThinking": {"cacheControlFormat": "anthropic"}
    }


@pytest.mark.parametrize("extra_body", [{"unexpected": "value"}, "not-a-mapping"])
def test_inventory_rejects_invalid_when_thinking_extra_body(
    tmp_path: Path, extra_body: Any
):
    row = _claude_row()
    row["settings_config"]["models"][0]["compat"] = {
        "whenThinking": {"extraBody": extra_body}
    }
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_materialization_omits_empty_compat_extra_body(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["models"][0]["compat"]["extraBody"] = {}
    db_path = _create_provider_db(tmp_path, [row])

    result = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert "extraBody" not in json.dumps(result, sort_keys=True)


def test_inventory_fails_closed_for_malformed_known_environment_values(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["env"]["ANTHROPIC_API_KEY"] = ["not-a-credential"]
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_materialization_retains_semantic_blockers_for_the_planner(tmp_path: Path):
    row = _claude_row()
    row["meta"]["apiFormat"] = "vendor-private-api"
    db_path = _create_provider_db(tmp_path, [row])

    result = materialization_source(
        str(db_path),
        selected_source_ids={"shared-provider"},
        approved_origin_source_ids={"shared-provider"},
    )

    assert result["providers"][0]["api_family"] is None
    assert result["providers"][0]["blocker_codes"] == ["blocked-unsupported-api"]


@pytest.mark.parametrize(
    ("meta", "settings", "expected_api_family"),
    [
        (
            {"apiFormat": "openai_responses"},
            {"apiFormat": "openai_chat", "api_format": "anthropic"},
            "openai_responses",
        ),
        (
            {},
            {"apiFormat": "openai_chat", "api_format": "openai_responses"},
            "openai_chat",
        ),
        ({}, {"api_format": "openai_responses"}, "openai_responses"),
        ({}, {}, "anthropic"),
        (
            {"apiFormat": "vendor-private-api"},
            {"apiFormat": "openai_chat"},
            None,
        ),
        ({}, {"api_format": "vendor-private-api"}, None),
        (
            {},
            {"apiFormat": "vendor-private-api", "api_format": "openai_chat"},
            None,
        ),
    ],
)
def test_native_claude_api_format_precedence_and_default(
    tmp_path: Path,
    meta: dict[str, Any],
    settings: dict[str, Any],
    expected_api_family: str | None,
):
    row = _claude_row()
    row["meta"] = meta
    row["settings_config"].update(settings)
    db_path = _create_provider_db(tmp_path, [row])

    provider = inventory_source(str(db_path))["providers"][0]

    assert provider["api_family"] == expected_api_family
    assert provider["blocker_codes"] == (
        [] if expected_api_family is not None else ["blocked-unsupported-api"]
    )


def test_inventory_does_not_mutate_the_source_database(inventory_db: Path):
    before = inventory_db.read_bytes()

    result = inventory_source(str(inventory_db))

    assert result["status"] == "inventory"
    assert inventory_db.read_bytes() == before
    assert not Path(f"{inventory_db}-journal").exists()
    assert not Path(f"{inventory_db}-wal").exists()
    assert not Path(f"{inventory_db}-shm").exists()


def test_inventory_reads_committed_wal_without_mutating_source(inventory_db: Path):
    connection = sqlite3.connect(inventory_db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute(
            "UPDATE providers SET name = ? WHERE id = ? AND app_type = 'claude'",
            ("Claude Responses from WAL", "shared-provider"),
        )
        connection.commit()
        sidecars = [
            Path(f"{inventory_db}-wal"),
            Path(f"{inventory_db}-shm"),
        ]
        assert all(sidecar.exists() for sidecar in sidecars)
        before = {path.name: path.read_bytes() for path in [inventory_db, *sidecars]}

        result = inventory_source(str(inventory_db))

        assert result["providers"][0]["source_name"] == "Claude Responses from WAL"
        assert {
            path.name: path.read_bytes() for path in [inventory_db, *sidecars]
        } == before
    finally:
        connection.close()


def test_wal_unrelated_writes_preserve_logical_hash_but_change_exact_precondition(
    inventory_db: Path,
):
    connection = sqlite3.connect(inventory_db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        before = materialization_source(
            str(inventory_db),
            selected_source_ids={"shared-provider"},
            approved_origin_source_ids={"shared-provider"},
        )
        shm_path = Path(f"{inventory_db}-shm")
        shm_details = shm_path.stat()
        assert before["precondition"]["shm"] == {
            "device": shm_details.st_dev,
            "fileId": shm_details.st_ino,
            "owner": shm_details.st_uid,
            "mode": stat.S_IMODE(shm_details.st_mode),
            "size": shm_details.st_size,
            "mtimeNs": shm_details.st_mtime_ns,
            "ctimeNs": shm_details.st_ctime_ns,
            "sha256": hashlib.sha256(shm_path.read_bytes()).hexdigest(),
        }

        connection.execute("INSERT INTO source_activity (value) VALUES ('unrelated')")
        connection.commit()
        after = materialization_source(
            str(inventory_db),
            selected_source_ids={"shared-provider"},
            approved_origin_source_ids={"shared-provider"},
        )

        assert after["source_snapshot_sha256"] == before["source_snapshot_sha256"]
        assert after["precondition"] != before["precondition"]
    finally:
        connection.close()


def test_wal_provider_change_updates_logical_source_precondition(inventory_db: Path):
    connection = sqlite3.connect(inventory_db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        before = materialization_source(
            str(inventory_db),
            selected_source_ids={"shared-provider"},
            approved_origin_source_ids={"shared-provider"},
        )

        connection.execute(
            "UPDATE providers SET name = ? WHERE id = ? AND app_type = 'claude'",
            ("Changed provider", "shared-provider"),
        )
        connection.commit()
        after = materialization_source(
            str(inventory_db),
            selected_source_ids={"shared-provider"},
            approved_origin_source_ids={"shared-provider"},
        )

        assert after["source_snapshot_sha256"] != before["source_snapshot_sha256"]
        assert after["precondition"] != before["precondition"]
    finally:
        connection.close()


def test_inventory_rejects_wal_header_when_wal_file_is_missing(tmp_path: Path):
    db_path = _create_provider_db(tmp_path, [_claude_row()])
    connection = sqlite3.connect(db_path)
    assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
    connection.close()
    assert not Path(f"{db_path}-wal").exists()

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-sidecar"


def test_inventory_rejects_json_above_the_source_limit(tmp_path: Path):
    row = _claude_row()
    row["settings_config"]["unusedPadding"] = "x" * (
        cc_switch_omp_source._MAX_JSON_BYTES + 1
    )
    db_path = _create_provider_db(tmp_path, [row])

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-json"


def test_inventory_rejects_json_above_the_source_depth_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    row = _claude_row()
    row["settings_config"] = {
        "model": "claude-sonnet-4-5-20250929",
        "env": {
            "ANTHROPIC_BASE_URL": "https://private-anthropic.invalid/v1",
            "ANTHROPIC_AUTH_TOKEN": "marker-anthropic-auth-token",
        },
        "nested": {"one": {"two": {"three": {"four": "value"}}}},
    }
    row["meta"] = {}
    db_path = _create_provider_db(tmp_path, [row])
    monkeypatch.setattr(cc_switch_omp_source, "_MAX_JSON_DEPTH", 4)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-json"


def test_inventory_rejects_rows_above_the_source_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = _create_provider_db(
        tmp_path,
        [_claude_row(source_id="source-one"), _claude_row(source_id="source-two")],
    )
    monkeypatch.setattr(cc_switch_omp_source, "_MAX_PROVIDER_ROWS", 1)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-provider-data"


def test_inventory_rejects_database_above_the_source_byte_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = _create_provider_db(tmp_path, [_claude_row()])
    monkeypatch.setattr(
        cc_switch_omp_source, "_MAX_SOURCE_DB_BYTES", db_path.stat().st_size - 1
    )

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-read"


def test_inventory_rejects_main_and_wal_above_aggregate_capture_limit(
    inventory_db: Path, monkeypatch: pytest.MonkeyPatch
):
    connection = sqlite3.connect(inventory_db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        wal_path = Path(f"{inventory_db}-wal")
        monkeypatch.setattr(
            cc_switch_omp_source,
            "_MAX_SOURCE_CAPTURE_BYTES",
            inventory_db.stat().st_size + wal_path.stat().st_size - 1,
        )

        with pytest.raises(SourceInventoryError) as raised:
            inventory_source(str(inventory_db))

        assert raised.value.code == "blocked-db-read"
    finally:
        connection.close()


def test_inventory_rejects_in_place_source_change_after_snapshot_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = _create_provider_db(tmp_path, [_claude_row()])
    original_hash = cc_switch_omp_source._file_sha256

    def mutate_before_hash(descriptor: int, identity: Any) -> str:
        db_path.touch()
        return original_hash(descriptor, identity)

    monkeypatch.setattr(cc_switch_omp_source, "_file_sha256", mutate_before_hash)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-security"


def test_inventory_rejects_wal_append_during_final_digest(
    inventory_db: Path, monkeypatch: pytest.MonkeyPatch
):
    connection = sqlite3.connect(inventory_db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        wal_file_id = Path(f"{inventory_db}-wal").stat().st_ino
        original_hash = cc_switch_omp_source._file_sha256
        wal_hash_count = 0

        def append_after_final_wal_hash(descriptor: int, identity: Any) -> str:
            nonlocal wal_hash_count
            digest = original_hash(descriptor, identity)
            if identity.file_id == wal_file_id:
                wal_hash_count += 1
                if wal_hash_count == 4:
                    connection.execute(
                        "INSERT INTO source_activity (value) VALUES ('raced')"
                    )
                    connection.commit()
            return digest

        monkeypatch.setattr(
            cc_switch_omp_source, "_file_sha256", append_after_final_wal_hash
        )

        with pytest.raises(SourceInventoryError) as raised:
            inventory_source(str(inventory_db))

        assert wal_hash_count == 4
        assert raised.value.code == "blocked-db-security"
    finally:
        connection.close()


def test_inventory_fails_closed_when_a_rollback_journal_exists(inventory_db: Path):
    Path(f"{inventory_db}-journal").write_bytes(b"unexpected sidecar")

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(inventory_db))

    assert raised.value.code == "blocked-db-sidecar"


@pytest.mark.require_symlinks
@pytest.mark.parametrize("sidecar_suffix", ["-wal", "-shm"])
def test_inventory_rejects_symlinked_wal_sidecars(
    inventory_db: Path, tmp_path: Path, sidecar_suffix: str
):
    sidecar_target = tmp_path / f"outside{sidecar_suffix}"
    sidecar_target.write_bytes(b"untrusted sidecar")
    Path(f"{inventory_db}{sidecar_suffix}").symlink_to(sidecar_target)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(inventory_db))

    assert raised.value.code == "blocked-db-symlink"


def test_inventory_rejects_group_writable_wal_sidecar(inventory_db: Path):
    wal_path = Path(f"{inventory_db}-wal")
    wal_path.write_bytes(b"untrusted sidecar")
    wal_path.chmod(0o620)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(inventory_db))

    assert raised.value.code == "blocked-db-security"


@pytest.mark.require_symlinks
def test_inventory_fails_closed_for_a_database_symlink(
    inventory_db: Path, tmp_path: Path
):
    linked_path = tmp_path / "providers-link.db"
    linked_path.symlink_to(inventory_db)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(linked_path))

    assert raised.value.code == "blocked-db-symlink"


def test_inventory_fails_closed_for_a_non_regular_database_path(tmp_path: Path):
    directory_path = tmp_path / "not-a-database"
    directory_path.mkdir()

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(directory_path))

    assert raised.value.code == "blocked-db-type"


@pytest.mark.skipif(
    not callable(getattr(os, "mkfifo", None))
    or not isinstance(getattr(os, "O_NOFOLLOW", None), int)
    or not isinstance(getattr(os, "O_NONBLOCK", None), int),
    reason="FIFO final-component validation requires POSIX nonblocking opens",
)
def test_inventory_refuses_fifo_substituted_after_lstat_without_blocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    regular_path = _create_provider_db(tmp_path, [_claude_row()])
    fifo_path = tmp_path / "providers.fifo"
    os.mkfifo(fifo_path)
    original_lstat = cc_switch_omp_source._lstat_database

    def stale_lstat(path: str) -> Any:
        if path == str(fifo_path):
            return original_lstat(str(regular_path))
        return original_lstat(path)

    monkeypatch.setattr(cc_switch_omp_source, "_lstat_database", stale_lstat)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(fifo_path))

    assert raised.value.code == "blocked-db-type"


@pytest.mark.parametrize(
    "mode", [0o620, 0o602], ids=["group-writable", "world-writable"]
)
def test_inventory_rejects_group_or_world_writable_database_modes(
    tmp_path: Path, mode: int
):
    db_path = _create_provider_db(tmp_path, [_claude_row()])
    db_path.chmod(mode)

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-security"


def test_inventory_rejects_provider_schema_without_the_composite_key(tmp_path: Path):
    db_path = tmp_path / "legacy-schema.db"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE providers (
                id TEXT PRIMARY KEY,
                app_type TEXT NOT NULL,
                name TEXT NOT NULL,
                settings_config TEXT NOT NULL,
                meta TEXT NOT NULL,
                is_current INTEGER NOT NULL,
                sort_index INTEGER
            );
            """
        )

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-schema"


def test_inventory_reports_an_invalid_provider_schema_with_a_public_code(
    tmp_path: Path,
):
    db_path = tmp_path / "invalid-schema.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE providers (id TEXT PRIMARY KEY)")

    with pytest.raises(SourceInventoryError) as raised:
        inventory_source(str(db_path))

    assert raised.value.code == "blocked-db-schema"
