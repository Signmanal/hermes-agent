"""Contract tests for the hardened read-only OMP projection-plan seam."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest
import yaml

import tools.omp_model_materializer as materializer
from tools.omp_model_materializer import (
    ProjectionPlanError,
    build_projection_plan,
    load_projection_plan,
)


_DY_SOURCE_ID = "2e8d3c50-d9e0-4f5a-a40e-619b3ac1e751"
_DEEPSEEK_SOURCE_ID = "875a85dd-e68d-4d3d-b6d4-7e57d58ddb5a"
_DY_TARGET = "cc-dy-codex"
_DEEPSEEK_TARGET = "cc-deepseek"
_GENERATED_HEADER = "# CC Switch source projection; generated for OMP."
_SECRET_MARKER = "marker-dy-api-key-value"
_PRIVATE_HOST = "private-dy.invalid"


def _remote_compaction(*, enabled: bool) -> dict[str, Any]:
    endpoint = "https://unapproved-secondary.invalid/v1"
    return {
        "api": "openai-responses",
        "enabled": enabled,
        "endpoint": endpoint,
        "model": "unapproved-secondary-model",
        "streamingEndpoint": endpoint,
        "v2Endpoint": endpoint,
        "v2StreamingEnabled": False,
    }


def _models(*, endpoint: str, extended: bool = False) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = [
        {
            "id": "gpt-5.6-sol",
            "name": "GPT-5.6 Sol",
            "reasoning": True,
            "input": ["text", "image"],
            "contextWindow": 200000,
            "maxTokens": 8192,
        },
        {
            "id": "gpt-5.6",
            "name": "GPT-5.6",
            "reasoning": False,
            "input": ["text"],
            "contextWindow": 128000,
            "maxTokens": 4096,
        },
    ]
    if extended:
        models[0].update({
            "api": "openai-responses",
            "baseUrl": endpoint,
            "compactionModel": "gpt-5.6",
            "compat": {
                "alwaysSendMaxTokens": True,
                # The source seam permits only an empty body object, which
                # is schema-valid but carries no credential-like value.
                "extraBody": {},
                "openRouterRouting": {"only": ["preferred-route"]},
                "promptCacheMode": "automatic",
                "whenThinking": {"supportsReasoningEffort": True},
            },
            "contextPromotionTarget": "gpt-5.6",
            "cost": {"cacheRead": 0.1, "cacheWrite": 0.2, "input": 1.0, "output": 2.0},
            "headers": {},
            "imageInputDecoder": "stb",
            "omitMaxOutputTokens": True,
            "preferWebsockets": False,
            "premiumMultiplier": 1.5,
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
        })
    return models


def _source_settings(endpoint: str, models: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        # Real Claude Adapter rows often use Anthropic names even where meta
        # selects an OpenAI Responses wire format.
        "env": {
            "ANTHROPIC_BASE_URL": endpoint,
            "ANTHROPIC_AUTH_TOKEN": _SECRET_MARKER,
        },
        "models": models,
    }


def _write_canonical_db(
    tmp_path: Path,
    *,
    endpoint: str = f"https://{_PRIVATE_HOST}/v1",
    api_format: str = "openai_responses",
    extended_models: bool = False,
    model_context_window: int | None = None,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    db_path = tmp_path / "cc-switch-providers.db"
    dy_models = _models(endpoint=endpoint, extended=extended_models)
    if model_context_window is not None:
        dy_models[0]["contextWindow"] = model_context_window
    deepseek_models = [{"id": "deepseek-chat"}]
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
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    _DY_SOURCE_ID,
                    "claude",
                    "DY Codex",
                    json.dumps(_source_settings(endpoint, dy_models)),
                    json.dumps({
                        "apiFormat": api_format,
                        "private_note": "marker-private-meta-value",
                    }),
                    1,
                    1,
                ),
                (
                    _DEEPSEEK_SOURCE_ID,
                    "claude",
                    "DeepSeek (excluded)",
                    json.dumps({
                        "env": {
                            "ANTHROPIC_BASE_URL": "https://deepseek.invalid/v1",
                            "ANTHROPIC_API_KEY": "marker-deepseek-key",
                        },
                        "models": deepseek_models,
                    }),
                    json.dumps({"apiFormat": "openai_responses"}),
                    0,
                    2,
                ),
                # Same source ID in another app must not be selected for this
                # projection.  The source seam enforces app_type='claude'.
                (
                    _DY_SOURCE_ID,
                    "openai",
                    "Foreign duplicate",
                    json.dumps({"model": "foreign"}),
                    json.dumps({"apiFormat": "openai_responses"}),
                    1,
                    0,
                ),
            ],
        )
    return db_path


def _write_generated_models(
    tmp_path: Path,
    *,
    header: str = _GENERATED_HEADER,
    dy_api: str = "openai-responses",
    endpoint: str = f"https://{_PRIVATE_HOST}/v1",
    extended_models: bool = False,
) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    target_path = tmp_path / "models.yml"
    document = {
        "providers": {
            _DY_TARGET: {
                "baseUrl": endpoint,
                "apiKey": "CC_SWITCH_DY_CODEX_KEY",
                "api": dy_api,
                "models": _models(endpoint=endpoint, extended=extended_models),
            },
            _DEEPSEEK_TARGET: {
                "auth": "apiKey",
                "baseUrl": "https://deepseek.invalid/v1",
                "apiKey": "CC_SWITCH_DEEPSEEK_KEY",
                "api": "anthropic-messages",
                "models": [],
            },
        }
    }
    target_path.write_text(
        header + "\n" + yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )
    return target_path


@pytest.fixture
def projection_inputs(tmp_path: Path) -> tuple[Path, Path]:
    return _write_canonical_db(tmp_path), _write_generated_models(tmp_path)


def _build(db_path: Path, target_path: Path, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "provider_map": {_DY_SOURCE_ID: _DY_TARGET},
        "selected_models": {_DY_SOURCE_ID: ["gpt-5.6-sol", "gpt-5.6"]},
        "excluded_source_ids": {_DEEPSEEK_SOURCE_ID},
        "approved_current_origin_source_ids": {_DY_SOURCE_ID},
        "omp_version": "18.2.0",
    }
    arguments.update(overrides)
    return build_projection_plan(db_path, target_path, **arguments)


def _yaml_document(path: Path) -> dict[str, Any]:
    raw = path.read_text(encoding="utf-8")
    return yaml.safe_load(raw.split("\n", 1)[1])


def _write_yaml_document(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        _GENERATED_HEADER + "\n" + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def _replace_source_models(db_path: Path, models: list[dict[str, Any]]) -> None:
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ? AND app_type = 'claude'",
            (_DY_SOURCE_ID,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["models"] = models
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ? AND app_type = 'claude'",
            (json.dumps(settings), _DY_SOURCE_ID),
        )


def _load_plan_fifo_worker(path: str, results: Any) -> None:
    try:
        load_projection_plan(path)
    except ProjectionPlanError as error:
        results.put(("rejected", error.code))
    except BaseException as error:
        results.put(("unexpected", repr(error)))
    else:
        results.put(("accepted", None))


def _build_target_fifo_worker(db_path: str, target_path: str, results: Any) -> None:
    try:
        _build(Path(db_path), Path(target_path))
    except ProjectionPlanError as error:
        results.put(("rejected", error.code))
    except BaseException as error:
        results.put(("unexpected", repr(error)))
    else:
        results.put(("accepted", None))


def _fifo_rejection(worker: Any, *arguments: str) -> str:
    context = multiprocessing.get_context("fork")
    results = context.Queue()
    process = context.Process(target=worker, args=(*arguments, results))
    process.start()
    try:
        process.join(timeout=1)
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
            pytest.fail("untrusted FIFO read blocked")
        assert process.exitcode == 0
        status, result = results.get(timeout=1)
        assert status == "rejected", result
        return result
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=1)
        results.close()
        results.join_thread()


def test_build_projection_plan_maps_actual_claude_fields_and_exclusion(
    projection_inputs: tuple[Path, Path],
):
    db_path, target_path = projection_inputs

    plan = _build(db_path, target_path)

    assert plan["status"] == "planned"
    assert plan["schemaVersion"] == 2
    assert plan["approvedOriginSourceIds"] == [_DY_SOURCE_ID]
    assert plan["excludedSourceIds"] == [_DEEPSEEK_SOURCE_ID]
    assert plan["providerActions"] == [
        {
            "action": "retain",
            "models": ["gpt-5.6", "gpt-5.6-sol"],
            "provider": _DY_TARGET,
            "sourceId": _DY_SOURCE_ID,
            "sourceProviderSha256": plan["providerActions"][0]["sourceProviderSha256"],
            "targetProviderSha256": plan["providerActions"][0]["targetProviderSha256"],
        }
    ]
    assert plan["removeProviders"] == [_DEEPSEEK_TARGET]
    assert plan["configHandoff"]["addModelSelectors"] == [
        {"provider": _DY_TARGET, "model": "gpt-5.6"},
        {"provider": _DY_TARGET, "model": "gpt-5.6-sol"},
    ]


def test_plan_is_redacted_canonical_and_preserves_inputs(
    projection_inputs: tuple[Path, Path],
):
    db_path, target_path = projection_inputs
    before_db = hashlib.sha256(db_path.read_bytes()).hexdigest()
    before_target = hashlib.sha256(target_path.read_bytes()).hexdigest()

    plan = _build(db_path, target_path)
    serialized = json.dumps(plan, sort_keys=True)

    for forbidden in (
        _PRIVATE_HOST,
        _SECRET_MARKER,
        "marker-private-meta-value",
        "settings_config",
        '"meta"',
        "ANTHROPIC_AUTH_TOKEN",
    ):
        assert forbidden not in serialized
    assert plan["policy"] == {"allowOpaqueBackup": False, "requireTargetHeader": True}
    assert (
        plan["planSha256"]
        == hashlib.sha256(
            json.dumps(
                {key: value for key, value in plan.items() if key != "planSha256"},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
    )
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_db
    assert hashlib.sha256(target_path.read_bytes()).hexdigest() == before_target


@pytest.mark.parametrize(
    ("endpoint", "api_format", "target_api", "expected_code"),
    [
        (
            "http://private-dy.invalid/v1",
            "openai_responses",
            "openai-responses",
            "blocked-insecure-scheme",
        ),
        (
            f"https://{_PRIVATE_HOST}/v1",
            "vendor-private-api",
            "openai-responses",
            "blocked-unsupported-api",
        ),
        (
            f"https://{_PRIVATE_HOST}/v1",
            "openai_responses",
            "anthropic-messages",
            "blocked-api-family-mismatch",
        ),
    ],
)
def test_build_blocks_invalid_normalized_source_or_target_api(
    tmp_path: Path, endpoint: str, api_format: str, target_api: str, expected_code: str
):
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint, api_format=api_format)
    target_path = _write_generated_models(
        tmp_path, endpoint=endpoint, dy_api=target_api
    )

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == expected_code


def test_build_blocks_tampered_source_or_target_model_metadata(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path, model_context_window=123456)
    target_path = _write_generated_models(tmp_path)

    with pytest.raises(ProjectionPlanError) as source_raised:
        _build(db_path, target_path)
    assert source_raised.value.code == "blocked-model-metadata"

    db_path = _write_canonical_db(tmp_path / "target", model_context_window=200000)
    target_path = _write_generated_models(tmp_path / "target")
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["contextWindow"] = 123456
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as target_raised:
        _build(db_path, target_path)
    assert target_raised.value.code == "blocked-model-metadata"


def test_build_accepts_approved_omp_18119_model_field_allowlist(tmp_path: Path):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint, extended_models=True)
    target_path = _write_generated_models(
        tmp_path, endpoint=endpoint, extended_models=True
    )

    plan = _build(db_path, target_path)

    assert plan["providerActions"][0]["models"] == ["gpt-5.6", "gpt-5.6-sol"]


def test_build_fails_closed_for_unknown_target_model_field(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["futureModelField"] = "not-supported"
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


def test_build_requires_explicit_origin_for_every_selected_provider(
    projection_inputs: tuple[Path, Path],
):
    db_path, target_path = projection_inputs

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path, approved_current_origin_source_ids=set())

    assert raised.value.code == "blocked-origin-approval"


def test_build_rejects_exclusion_intersection_and_glob_selector(
    projection_inputs: tuple[Path, Path],
):
    db_path, target_path = projection_inputs

    with pytest.raises(ProjectionPlanError) as exclusion:
        _build(db_path, target_path, excluded_source_ids={_DY_SOURCE_ID})
    assert exclusion.value.code == "blocked-exclusion-intersection"

    with pytest.raises(ProjectionPlanError) as glob_selector:
        _build(db_path, target_path, selected_models={_DY_SOURCE_ID: ["gpt-*"]})
    assert glob_selector.value.code == "blocked-model-selection"


@pytest.mark.require_symlinks
def test_build_refuses_symlink_target(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    linked = tmp_path / "models-link.yml"
    linked.symlink_to(target_path)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, linked)

    assert raised.value.code == "blocked-target-symlink"


def test_source_scope_is_claude_even_with_same_id_in_other_app(
    projection_inputs: tuple[Path, Path],
):
    db_path, target_path = projection_inputs

    assert _build(db_path, target_path)["source"]["appType"] == "claude"


@pytest.mark.parametrize(
    "thinking",
    [
        {"mode": "enabled", "efforts": ["low"]},
        {"mode": "effort", "efforts": ["not-an-effort"]},
        {"mode": "effort", "efforts": ["low"], "requiresEffort": "true"},
    ],
)
def test_build_rejects_invalid_omp_thinking_metadata(
    tmp_path: Path, thinking: dict[str, Any]
):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["thinking"] = thinking
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


def test_build_accepts_real_legacy_thinking_range_shape(tmp_path: Path):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint)
    target_path = _write_generated_models(tmp_path, endpoint=endpoint)
    thinking = {
        "mode": "budget",
        "minLevel": "low",
        "maxLevel": "high",
        "supportsDisplay": True,
        "requiresEffort": False,
    }
    source_models = _models(endpoint=endpoint)
    source_models[0]["thinking"] = thinking
    _replace_source_models(db_path, source_models)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["thinking"] = thinking
    _write_yaml_document(target_path, document)

    assert _build(db_path, target_path)["providerActions"][0]["models"] == [
        "gpt-5.6",
        "gpt-5.6-sol",
    ]


def test_build_mirrors_nested_model_override_merge(tmp_path: Path):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint)
    target_path = _write_generated_models(tmp_path, endpoint=endpoint)
    source_models = _models(endpoint=endpoint)
    source_models[0]["compat"] = {
        "whenThinking": {
            "supportsReasoningEffort": True,
            "supportsStore": True,
        }
    }
    _replace_source_models(db_path, source_models)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["compat"] = {
        "whenThinking": {"supportsStore": True}
    }
    document["providers"][_DY_TARGET]["modelOverrides"] = {
        "gpt-5.6-sol": {"compat": {"whenThinking": {"supportsReasoningEffort": True}}}
    }
    _write_yaml_document(target_path, document)

    assert _build(db_path, target_path)["providerActions"][0]["models"] == [
        "gpt-5.6",
        "gpt-5.6-sol",
    ]


@pytest.mark.parametrize(
    "override",
    [
        {"api": "openai-responses"},
        {"baseUrl": f"https://{_PRIVATE_HOST}/v1"},
        {"transport": "pi-native"},
    ],
)
def test_build_rejects_transport_masking_model_override(
    tmp_path: Path, override: dict[str, Any]
):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["modelOverrides"] = {"gpt-5.6-sol": override}
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("placement", ["provider", "direct-model", "model-override"])
def test_build_rejects_selected_remote_compaction(
    tmp_path: Path, placement: str, enabled: bool
):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    remote_compaction = _remote_compaction(enabled=enabled)
    provider = document["providers"][_DY_TARGET]
    if placement == "provider":
        provider["remoteCompaction"] = remote_compaction
    elif placement == "direct-model":
        provider["models"][0]["remoteCompaction"] = remote_compaction
    else:
        provider["modelOverrides"] = {
            "gpt-5.6-sol": {"remoteCompaction": remote_compaction}
        }
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-model-metadata"


def test_build_preserves_unselected_provider_remote_compaction(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    remote_compaction = _remote_compaction(enabled=True)
    document["providers"]["unselected-remote"] = {
        "api": "openai-responses",
        "apiKey": "UNSELECTED_REMOTE_KEY",
        "baseUrl": "https://unselected.invalid/v1",
        "models": [],
        "remoteCompaction": remote_compaction,
    }
    _write_yaml_document(target_path, document)

    plan = _build(db_path, target_path)

    assert plan["status"] == "planned"
    assert plan["removeProviders"] == [_DEEPSEEK_TARGET]
    assert _yaml_document(target_path)["providers"]["unselected-remote"][
        "remoteCompaction"
    ] == (remote_compaction)


@pytest.mark.parametrize("extra_body", [7, ["not", "an", "object"]])
def test_build_rejects_nonobject_compat_extra_body(tmp_path: Path, extra_body: Any):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["compat"] = {"extraBody": extra_body}
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


def test_build_rejects_discovery_inject_v1_outside_openai_models_list(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["discovery"] = {
        "type": "ollama",
        "injectV1": False,
    }
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


def test_build_accepts_zero_valued_omp_compat_controls(tmp_path: Path):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    compat = {
        "promptCacheMaximumCheckpoints": 0,
        "promptCacheMinimumTokens": 0,
        "streamIdleTimeoutMs": 0,
    }
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint)
    target_path = _write_generated_models(tmp_path, endpoint=endpoint)
    source_models = _models(endpoint=endpoint)
    source_models[0]["compat"] = compat
    _replace_source_models(db_path, source_models)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["compat"] = compat
    _write_yaml_document(target_path, document)

    assert _build(db_path, target_path)["status"] == "planned"


def test_build_blocks_invalid_preserved_provider(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    document = _yaml_document(target_path)
    document["providers"]["preserved"] = {"models": "not-a-list"}
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-provider"


def test_build_accepts_literal_model_id_with_internal_slash(tmp_path: Path):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    slash_model = "vendor/gpt-5.6-sol"
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint)
    target_path = _write_generated_models(tmp_path, endpoint=endpoint)
    source_models = _models(endpoint=endpoint)
    source_models[0]["id"] = slash_model
    _replace_source_models(db_path, source_models)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0]["id"] = slash_model
    _write_yaml_document(target_path, document)

    plan = _build(
        db_path,
        target_path,
        selected_models={_DY_SOURCE_ID: [slash_model, "gpt-5.6"]},
    )

    assert plan["configHandoff"]["addModelSelectors"] == [
        {"provider": _DY_TARGET, "model": "gpt-5.6"},
        {"provider": _DY_TARGET, "model": slash_model},
    ]


@pytest.mark.parametrize("field", ["baseUrl", "headers"])
def test_build_rejects_extra_selected_model_transport(tmp_path: Path, field: str):
    endpoint = f"https://{_PRIVATE_HOST}/v1"
    db_path = _write_canonical_db(tmp_path, endpoint=endpoint)
    target_path = _write_generated_models(tmp_path, endpoint=endpoint)
    document = _yaml_document(target_path)
    document["providers"][_DY_TARGET]["models"][0][field] = (
        endpoint if field == "baseUrl" else {"X-Not-From-Source": "value"}
    )
    _write_yaml_document(target_path, document)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-model-metadata"


def test_build_refuses_yaml_aliases_and_bounded_structures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    target_path.write_text(
        _GENERATED_HEADER + "\nproviders: &providers\n  cc-dy-codex: *providers\n",
        encoding="utf-8",
    )
    with pytest.raises(ProjectionPlanError) as alias:
        _build(db_path, target_path)
    assert alias.value.code == "blocked-target-parse"

    nested = "providers:\n"
    indent = "  "
    for index in range(materializer._MAX_YAML_DEPTH + 2):
        nested += f"{indent}level{index}:\n"
        indent += "  "
    nested += f"{indent}leaf: true\n"
    target_path.write_text(_GENERATED_HEADER + "\n" + nested, encoding="utf-8")
    with pytest.raises(ProjectionPlanError) as depth:
        _build(db_path, target_path)
    assert depth.value.code == "blocked-target-parse"

    target_path = _write_generated_models(tmp_path / "nodes")
    monkeypatch.setattr(materializer, "_MAX_YAML_NODES", 1)
    with pytest.raises(ProjectionPlanError) as nodes:
        _build(db_path, target_path)
    assert nodes.value.code == "blocked-target-parse"


@pytest.mark.parametrize("unsafe_bits", [0o020, 0o002])
def test_build_refuses_group_or_world_writable_target(tmp_path: Path, unsafe_bits: int):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    original_mode = target_path.stat().st_mode & 0o777
    os.chmod(target_path, (original_mode & ~0o022) | unsafe_bits)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db_path, target_path)

    assert raised.value.code == "blocked-target-path"


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not getattr(os, "O_NONBLOCK", 0),
    reason="requires nonblocking POSIX FIFOs",
)
def test_load_projection_plan_refuses_fifo_without_blocking(tmp_path: Path):
    plan_path = tmp_path / "plan.fifo"
    os.mkfifo(plan_path)

    assert (
        _fifo_rejection(_load_plan_fifo_worker, str(plan_path)) == "blocked-plan-load"
    )


@pytest.mark.skipif(
    not hasattr(os, "mkfifo") or not getattr(os, "O_NONBLOCK", 0),
    reason="requires nonblocking POSIX FIFOs",
)
def test_build_projection_plan_refuses_target_fifo_without_blocking(tmp_path: Path):
    db_path = _write_canonical_db(tmp_path)
    target_path = tmp_path / "models.fifo"
    os.mkfifo(target_path)

    assert (
        _fifo_rejection(_build_target_fifo_worker, str(db_path), str(target_path))
        == "blocked-target-path"
    )


@pytest.mark.parametrize("unsafe_bits", [0o020, 0o002])
def test_load_projection_plan_refuses_group_or_world_writable_file(
    tmp_path: Path, unsafe_bits: int
):
    db_path = _write_canonical_db(tmp_path)
    target_path = _write_generated_models(tmp_path)
    plan_path = tmp_path / "plan.json"
    plan = _build(db_path, target_path)
    plan_path.write_bytes(
        json.dumps(
            plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    original_mode = plan_path.stat().st_mode & 0o777
    os.chmod(plan_path, (original_mode & ~0o022) | unsafe_bits)

    with pytest.raises(ProjectionPlanError) as raised:
        load_projection_plan(plan_path)

    assert raised.value.code == "blocked-plan-load"
