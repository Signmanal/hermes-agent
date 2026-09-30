"""Public-contract tests for the opt-in CC Switch provider migration seam.

The fixtures intentionally use synthetic origins and credentials.  Assertions only
inspect those values inside target files; plans, reports, and raised errors must
remain secret-free.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
import shlex
import shutil
import subprocess
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

import tools.omp_provider_migration as migration
from tools.cc_switch_omp_source import migration_planning_source
from tools.omp_provider_migration import (
    MigrationPlanError,
    apply_provider_migration_plan,
    build_provider_migration_plan,
    verify_provider_migration,
)


DY_SOURCE = "6becb67f-1730-438d-b8ad-9429675d1090"
XUANJU_SOURCE = "616ccdfd-8de6-4c82-a4cc-24f71edf0c49"
DY = "cc-dy-codex"
XUANJU = "cc-xuanju"
DEEPSEEK_KEY = "CC_SWITCH_DEEPSEEK_KEY"
XUANJU_KEY = "CC_SWITCH_XUANJU_KEY"
DY_MODELS = ["gpt-5.6", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"]
XUANJU_MODELS = ["GLM-5.3-flash"]
XUANJU_PROBE_MODELS = list(migration._XUANJU_PROBE_MODELS)
NEW_SELECTORS = [
    f"{XUANJU}/GLM-5.3-flash",
]
HEADER = "# CC Switch source projection; generated for OMP."
DY_HOST = "www.api-codex.cn"
XUANJU_HOST = "private-xuanju-migration.invalid"
DY_SECRET = "marker-dy-migration-secret"
XUANJU_SECRET = "marker-xuanju-migration-secret"
OLD_SECRET = "marker-old-deepseek-secret"


def _environment_for(models: list[str], *, base_url: str, credential: str) -> dict[str, str]:
    environment = {
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_AUTH_TOKEN": credential,
    }
    for index, model in enumerate(models):
        stem = f"ANTHROPIC_DEFAULT_MIGRATION_{index}"
        environment[f"{stem}_MODEL"] = model
        environment[f"{stem}_MODEL_NAME"] = model
    return environment


def _write_db(tmp_path: Path) -> Path:
    db = tmp_path / "cc-switch.db"
    dy_environment = _environment_for(
        DY_MODELS,
        base_url=f"https://{DY_HOST}",
        credential=DY_SECRET,
    )
    xuanju_environment = _environment_for(
        XUANJU_MODELS,
        base_url=f"https://{XUANJU_HOST}/v1",
        credential=XUANJU_SECRET,
    )
    rows = [
        (
            DY_SOURCE,
            "claude",
            "DY synthetic",
            json.dumps(
                {
                    "env": dy_environment,
                    # Declared data deliberately differs from the environment
                    # evidence so the plan must show both categories.
                    "models": [{"id": "gpt-5.6"}, {"id": "gpt-5.6-sol"}],
                }
            ),
            json.dumps({"apiFormat": "openai_responses"}),
            0,
            1,
        ),
        (
            XUANJU_SOURCE,
            "claude",
            "Xuanju synthetic",
            json.dumps(
                {
                    "env": {
                        **xuanju_environment,
                        "ANTHROPIC_BASE_URL": "https://hs32.com/api/anthropic",
                    },
                    "models": [{"id": model} for model in XUANJU_PROBE_MODELS],
                }
            ),
            json.dumps({"apiFormat": "anthropic"}),
            1,
            2,
        ),
    ]
    with sqlite3.connect(db) as connection:
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
            rows,
        )
    return db


def _write_models(tmp_path: Path) -> Path:
    path = tmp_path / "models.yml"
    document = {
        "providers": {
            DY: {
                "api": "openai-responses",
                "apiKey": "CC_SWITCH_DY_CODEX_KEY",
                "baseUrl": f"https://{DY_HOST}/old",
                "models": [{"id": "gpt-5.6"}, {"id": "gpt-5.6-sol"}],
            },
            "other": {
                "api": "openai-responses",
                "apiKey": "OTHER_KEY",
                "baseUrl": "https://other.synthetic.invalid/v1",
                "models": [{"id": "other-model"}],
            },
        }
    }
    path.write_text(HEADER + "\n" + yaml.safe_dump(document, sort_keys=False))
    path.chmod(0o600)
    return path


def _config() -> dict[str, Any]:
    return {
        "modelProviderOrder": ["openai-codex", DY],
        "enabledModels": [f"{DY}/gpt-5.6", f"{DY}/gpt-5.6-sol"],
        "modelRoles": {"default": "openai-codex/gpt-5.6-sol", "plan": f"{DY}/gpt-5.6-sol"},
        "retry": {"fallbackChains": {"default": ["openai-codex/gpt-5.6-sol", f"{DY}/gpt-5.6-sol"]}},
        "telemetry": {"token": "marker-unrelated-config-secret"},
    }


def _write_config(tmp_path: Path) -> Path:
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(_config(), sort_keys=False))
    path.chmod(0o600)
    return path


def _write_env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "# synthetic environment\n"
        f"{DEEPSEEK_KEY}={OLD_SECRET}\n"
        "CC_SWITCH_DY_CODEX_KEY=marker-existing-dy-secret\n"
    )
    path.chmod(0o600)
    return path


def _probe_payload(db: Path) -> dict[str, Any]:
    source = migration_planning_source(
        db,
        selected_source_ids={DY_SOURCE, XUANJU_SOURCE},
        approved_origin_source_ids={DY_SOURCE, XUANJU_SOURCE},
    )
    binding = migration._xuanju_binding_context(db, source, XUANJU_SOURCE)
    observations: list[dict[str, Any]] = []
    for model in XUANJU_PROBE_MODELS:
        for mode in ("text", "tool"):
            observations.append(
                {
                    "sourceId": XUANJU_SOURCE,
                    "apiFamily": "openai-responses",
                    "configuredModelValue": model,
                    "model": model,
                    "observedModel": model,
                    "mode": mode,
                    "identityStatus": "observed-match",
                    "status": "usable",
                    "toolResult": "smoke_ok" if mode == "tool" else None,
                    "tokenUsage": {"total_tokens": 2},
                    "latencyMs": 1,
                    "usable": True,
                }
            )
    document = {
        "schemaVersion": 2,
        "artifactKind": migration._XUANJU_ARTIFACT_KIND,
        "requestApi": "openai-responses",
        "sourceApiFamily": "anthropic",
        "semanticBindingSha256": binding["semanticBindingSha256"],
        "providerEvidence": [
            {
                "sourceId": XUANJU_SOURCE,
                "semanticBindingSha256": binding["semanticBindingSha256"],
            }
        ],
        "observations": observations,
        "artifactSha256": "",
    }
    document["artifactSha256"] = _artifact_digest(document)
    return document


def _artifact_digest(document: dict[str, Any]) -> str:
    return hashlib.sha256(
        migration._canonical(
            {key: value for key, value in document.items() if key != "artifactSha256"},
            "blocked-test-artifact",
        )
    ).hexdigest()


def _write_probe(
    db: Path, path: Path, payload: dict[str, Any] | None = None
) -> Path:
    document = _probe_payload(db) if payload is None else payload
    path.write_bytes(
        migration._canonical(document, "blocked-test-artifact") + b"\n"
    )
    path.chmod(0o600)
    return path


def _write_omp_package(tmp_path: Path, *, version: str = "18.2.0") -> Path:
    package = tmp_path / "package.json"
    package.write_text(
        json.dumps(
            {
                "name": "@oh-my-pi/pi-coding-agent",
                "version": version,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    package.chmod(0o600)
    return package


@pytest.fixture(autouse=True)
def _omp_runtime_manifest(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(migration, "_OMP_PACKAGE_PATH", _write_omp_package(tmp_path))


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    db = _write_db(tmp_path)
    return db, _write_models(tmp_path), _write_config(tmp_path), _write_env(tmp_path), _write_probe(db, tmp_path / "probe.json")


def _build(
    db: Path,
    models: Path,
    config: Path,
    environment: Path,
    probe: Path | list[Path],
    **overrides: Any,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "provider_map": {DY_SOURCE: DY, XUANJU_SOURCE: XUANJU},
        "selected_models": {DY_SOURCE: DY_MODELS, XUANJU_SOURCE: XUANJU_MODELS},
        "probe_evidence_path": probe,
        "approved_current_origin_source_ids": {DY_SOURCE, XUANJU_SOURCE},
        "enable_xuanju": True,
        "omp_version": "18.2.0",
    }
    arguments.update(overrides)
    return build_provider_migration_plan(db, models, config, environment, **arguments)


def _apply(plan: dict[str, Any], db: Path, models: Path, config: Path, environment: Path) -> dict[str, Any]:
    return apply_provider_migration_plan(
        plan,
        db,
        models,
        config,
        environment,
        accepted_plan_sha256=plan["planSha256"],
        allow_opaque_backup=True,
    )


def _assert_no_secret(value: Any) -> None:
    serialized = json.dumps(value, sort_keys=True)
    for marker in (DY_HOST, XUANJU_HOST, DY_SECRET, XUANJU_SECRET, OLD_SECRET):
        assert marker not in serialized


def test_plan_is_read_only_secret_free_and_distinguishes_declared_from_probe_verified(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    before = {path: path.read_bytes() for path in (db, models, config, environment, probe)}

    plan = _build(db, models, config, environment, probe)

    assert plan["status"] == "planned"
    assert plan["source"] == {
        "appType": "claude",
        "selectedSemanticSha256": migration._digest(
            plan["preconditions"]["source"]["selectedBindings"],
            "blocked-test-source",
        ),
    }
    assert set(plan["preconditions"]["source"]) == {
        "mainIdentity",
        "selectedBindings",
    }
    actions = {action["target"]: action for action in plan["providerActions"]}
    assert actions[DY]["action"] == "update"
    assert actions[DY]["declaredModels"] == ["gpt-5.6", "gpt-5.6-sol"]
    assert actions[DY]["probeVerifiedModels"] == []
    assert actions[DY]["discovery"] == {
        "type": "openai-models-list",
        "injectV1": True,
    }
    assert actions[XUANJU]["action"] == "create"
    assert actions[XUANJU]["sourceApiFamily"] == "anthropic"
    assert actions[XUANJU]["declaredModels"] == XUANJU_PROBE_MODELS
    assert actions[XUANJU]["probeVerifiedModels"] == XUANJU_MODELS
    assert actions[XUANJU]["targetApi"] == "openai-responses"
    assert "discovery" not in actions[XUANJU]
    assert "models" not in actions[XUANJU]
    assert len(actions[XUANJU]["semanticBindingSha256"]) == 64
    assert actions[XUANJU]["endpoint"]["sha256"] == hashlib.sha256(
        b"https://hs32.com/v1"
    ).hexdigest()
    assert all(
        len(action["applyBindingSha256"]) == 64
        for action in actions.values()
    )
    assert plan["credentialMigration"] == {
        "action": "replace",
        "removeEnvKey": DEEPSEEK_KEY,
        "sourceId": XUANJU_SOURCE,
        "sourceCredentialPresent": True,
        "writeEnvKey": XUANJU_KEY,
    }
    assert plan["configActions"]["addEnabledModels"] == NEW_SELECTORS
    assert plan["configActions"]["addProviderOrder"] == [XUANJU]
    assert plan["configActions"]["preserveFallbackChains"] is True
    assert plan["configActions"]["preserveModelRoles"] is True
    assert plan["policy"] == {"allowOpaqueBackup": False, "enableXuanju": True, "require0600": True}
    _assert_no_secret(plan)
    assert {path: path.read_bytes() for path in before} == before


def test_plan_encodes_physical_preconditions_as_closed_decimal_strings(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)

    for target in (
        plan["preconditions"]["models"],
        plan["preconditions"]["config"],
        plan["preconditions"]["environment"],
    ):
        assert all(
            isinstance(target[key], str) and target[key] == str(int(target[key]))
            for key in ("device", "fileId", "mode", "ctimeNs")
        )
        assert all(
            isinstance(target["parent"][key], str)
            and target["parent"][key] == str(int(target["parent"][key]))
            for key in ("device", "fileId", "ctimeNs")
        )
    source = plan["preconditions"]["source"]
    assert set(source) == {"mainIdentity", "selectedBindings"}
    assert set(source["mainIdentity"]) == {
        "realpath",
        "device",
        "fileId",
        "owner",
        "mode",
    }
    assert all(
        isinstance(source["mainIdentity"][key], str)
        and source["mainIdentity"][key] == str(int(source["mainIdentity"][key]))
        for key in ("device", "fileId", "owner", "mode")
    )
    assert set(source["selectedBindings"]) == {DY_SOURCE, XUANJU_SOURCE}
    for binding in source["selectedBindings"].values():
        assert set(binding) == {
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
        }
        assert all(
            isinstance(binding[key], str) and len(binding[key]) == 64
            for key in (
                "endpointSha256",
                "endpointOriginSha256",
                "applyBindingSha256",
                "semanticBindingSha256",
            )
        )

    malformed = copy.deepcopy(plan)
    malformed["preconditions"]["models"]["ctimeNs"] = 1 << 53
    with pytest.raises(MigrationPlanError) as raised:
        migration.validate_provider_migration_plan(malformed)
    assert raised.value.code == "blocked-plan-schema"


def test_jcs_fixed_vector_rejects_nonportable_numbers_and_matches_node(
    tmp_path: Path,
):
    del tmp_path
    vector = {
        "\ue000": "private",
        "\U0001f600": "grin",
        "a": "\u000f\n\"\\/",
        "z": [True, None, False],
    }
    expected = (
        '{"a":"\\u000f\\n\\"\\\\/","z":[true,null,false],"😀":"grin","":"private"}'
    ).encode("utf-8")

    canonical = migration._canonical(vector, "blocked-jcs")
    assert canonical == expected
    for invalid in ({"number": 1.5}, {"number": float("nan")}, {"number": 1 << 53}):
        with pytest.raises(MigrationPlanError) as raised:
            migration._canonical(invalid, "blocked-jcs")
        assert raised.value.code == "blocked-jcs"

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the cross-language JCS contract")
    result = subprocess.run(
        [
            node,
            "-e",
            (
                "const fs=require('fs');"
                "const sort=value=>Array.isArray(value)?value.map(sort):"
                "(value&&typeof value==='object'?Object.fromEntries("
                "Object.keys(value).sort().map(key=>[key,sort(value[key])])):value);"
                "process.stdout.write(JSON.stringify(sort(JSON.parse("
                "fs.readFileSync(0,'utf8')))));"
            ),
        ],
        input=json.dumps(vector, ensure_ascii=False).encode("utf-8"),
        capture_output=True,
        check=True,
    )
    assert result.stdout == canonical



def test_plan_requires_one_canonical_ten_call_responses_artifact(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    payload = _probe_payload(db)
    second = tmp_path / "probe-second.json"
    _write_probe(db, second, payload)
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, [probe, second])
    assert raised.value.code == "blocked-probe-evidence"

    assert payload["schemaVersion"] == 2
    assert payload["artifactKind"] == migration._XUANJU_ARTIFACT_KIND
    assert payload["requestApi"] == "openai-responses"
    assert payload["providerEvidence"] == [
        {
            "sourceId": XUANJU_SOURCE,
            "semanticBindingSha256": payload["semanticBindingSha256"],
        }
    ]
    assert len(payload["observations"]) == 10
    assert [
        (item["model"], item["mode"]) for item in payload["observations"]
    ] == [
        (model, mode)
        for model in XUANJU_PROBE_MODELS
        for mode in ("text", "tool")
    ]


def test_discovery_only_dy_does_not_require_functionality_probe(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    payload = _probe_payload(db)
    payload["observations"] = [
        observation
        for observation in payload["observations"]
        if observation["sourceId"] == XUANJU_SOURCE
    ]
    payload["providerEvidence"] = [
        evidence
        for evidence in payload["providerEvidence"]
        if evidence["sourceId"] == XUANJU_SOURCE
    ]
    payload["artifactSha256"] = _artifact_digest(payload)
    _write_probe(db, probe, payload)

    plan = _build(db, models, config, environment, probe)

    dy_action = next(
        action for action in plan["providerActions"] if action["target"] == DY
    )
    assert dy_action["probeVerifiedModels"] == []
    assert dy_action["discovery"] == {
        "type": "openai-models-list",
        "injectV1": True,
    }



def test_live_dy_context_suffix_values_are_ignored_for_discovery_only(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (DY_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        for key in settings["env"]:
            if key.endswith("_MODEL"):
                settings["env"][key] += "[1M]"
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), DY_SOURCE),
        )
    payload = _probe_payload(db)
    payload["observations"] = [
        observation
        for observation in payload["observations"]
        if observation["sourceId"] == XUANJU_SOURCE
    ]
    payload["providerEvidence"] = [
        evidence
        for evidence in payload["providerEvidence"]
        if evidence["sourceId"] == XUANJU_SOURCE
    ]
    payload["artifactSha256"] = _artifact_digest(payload)
    _write_probe(db, probe, payload)

    _build(
        db,
        models,
        config,
        environment,
        probe,
        selected_models={DY_SOURCE: [], XUANJU_SOURCE: XUANJU_MODELS},
    )


def test_xuanju_context_suffix_values_do_not_gate_dynamic_provider(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_DEFAULT_MIGRATION_0_MODEL"] = "GLM-5.3-flash[1M]"
        settings["env"]["ANTHROPIC_DEFAULT_MIGRATION_0_MODEL_NAME"] = "GLM-5.3-flash[1M]"
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), XUANJU_SOURCE),
        )
    _write_probe(db, probe)
    plan = _build(db, models, config, environment, probe)
    xuanju_action = next(
        action for action in plan["providerActions"] if action["target"] == XUANJU
    )
    assert xuanju_action["probeVerifiedModels"] == XUANJU_MODELS


def test_wal_source_precondition_accepts_hashed_shm(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    connection = sqlite3.connect(db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute(
            "CREATE TABLE migration_wal_activity (value TEXT NOT NULL)"
        )
        connection.commit()

        _write_probe(db, probe)
        plan = _build(db, models, config, environment, probe)

        assert plan["status"] == "planned"
    finally:
        connection.close()



def test_plan_requires_explicit_xuanju_opt_in_and_exact_source_target_contract(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)

    with pytest.raises(MigrationPlanError) as opt_in:
        _build(db, models, config, environment, probe, enable_xuanju=False)
    assert opt_in.value.code == "blocked-xuanju-opt-in"

    with pytest.raises(MigrationPlanError) as drift:
        _build(
            db,
            models,
            config,
            environment,
            probe,
            provider_map={DY_SOURCE: DY, XUANJU_SOURCE: "cc-xuanju-alias"},
        )
    assert drift.value.code == "blocked-provider-mapping"


@pytest.mark.parametrize(
    ("overrides", "expected_code"),
    [
        (
            {
                "provider_map": {
                    "wrong-dy-source": DY,
                    XUANJU_SOURCE: XUANJU,
                }
            },
            "blocked-provider-mapping",
        ),
        (
            {
                "selected_models": {
                    "wrong-dy-source": DY_MODELS,
                    XUANJU_SOURCE: XUANJU_MODELS,
                }
            },
            "blocked-model-selection",
        ),
        (
            {"approved_current_origin_source_ids": {XUANJU_SOURCE}},
            "blocked-origin-approval",
        ),
    ],
)
def test_plan_requires_fixed_selected_and_approved_source_ids(
    tmp_path: Path, overrides: dict[str, Any], expected_code: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe, **overrides)
    assert raised.value.code == expected_code


def test_plan_rejects_noncurrent_xuanju_source(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        connection.execute(
            "UPDATE providers SET is_current = 0 WHERE id = ?", (XUANJU_SOURCE,)
        )
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-source-provider"


def test_validator_rejects_rehashed_plan_with_wrong_dy_source_id(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    tampered = copy.deepcopy(plan)
    tampered["providerActions"][0]["sourceId"] = "wrong-dy-source"
    tampered["planSha256"] = migration._plan_hash(tampered)

    with pytest.raises(MigrationPlanError) as raised:
        migration.validate_provider_migration_plan(tampered)
    assert raised.value.code == "blocked-plan-schema"
@pytest.mark.parametrize(
    "endpoint",
    [
        "https://hs32.com",
        "https://hs32.com/v1",
        "https://hs32.com/v1/chat/completions",
    ],
)
def test_plan_rejects_noncanonical_xuanju_endpoint_path(
    tmp_path: Path, endpoint: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_BASE_URL"] = endpoint
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), XUANJU_SOURCE),
        )
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-source-url"


@pytest.mark.parametrize(
    ("api_format", "endpoint"),
    [
        ("anthropic", "https://hs32.com/v1/responses"),
        ("openai_responses", "https://hs32.com/api/anthropic"),
    ],
)
def test_plan_rejects_xuanju_family_endpoint_mismatch(
    tmp_path: Path, api_format: str, endpoint: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_BASE_URL"] = endpoint
        connection.execute(
            "UPDATE providers SET settings_config = ?, meta = ? WHERE id = ?",
            (json.dumps(settings), json.dumps({"apiFormat": api_format}), XUANJU_SOURCE),
        )

    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-source-url"


def test_plan_accepts_already_migrated_xuanju_responses_source(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_BASE_URL"] = "https://hs32.com/v1/responses"
        connection.execute(
            "UPDATE providers SET settings_config = ?, meta = ? WHERE id = ?",
            (
                json.dumps(settings),
                json.dumps({"apiFormat": "openai_responses"}),
                XUANJU_SOURCE,
            ),
        )
    _write_probe(db, probe)

    plan = _build(db, models, config, environment, probe)
    action = next(
        action for action in plan["providerActions"] if action["target"] == XUANJU
    )
    assert action["sourceApiFamily"] == "openai_responses"
    assert action["endpoint"]["sha256"] == hashlib.sha256(
        b"https://hs32.com/v1"
    ).hexdigest()


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://hs32.com/v1/responses/",
        "https://hs32.com/v1/responses?verified=1",
        "https://hs32.com/v1/responses#fragment",
        "HTTPS://hs32.com/v1/responses",
        "https://HS32.com/v1/responses",
        "https://hs32.com:443/v1/responses",
    ],
)
def test_xuanju_site_root_rejects_noncanonical_endpoint_spelling(endpoint: str):
    with pytest.raises(MigrationPlanError) as raised:
        migration._xuanju_site_root(endpoint, "blocked-source-url")
    assert raised.value.code == "blocked-source-url"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://evil.example",
        "https://www.api-codex.cn/v2",
        "https://www.api-codex.cn:8443",
    ],
)
def test_plan_rejects_noncanonical_dy_endpoint(
    tmp_path: Path, endpoint: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?", (DY_SOURCE,)
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_BASE_URL"] = endpoint
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), DY_SOURCE),
        )
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-source-url"


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://www.api-codex.cn?query=1",
        "https://user@www.api-codex.cn",
        "https://www.api-codex.cn/v1#fragment",
    ],
)
def test_dy_provider_base_rejects_url_ambiguity(endpoint: str):
    with pytest.raises(MigrationPlanError) as raised:
        migration._dy_provider_base(endpoint, "blocked-source-url")
    assert raised.value.code == "blocked-source-url"


@pytest.mark.parametrize(
    ("provider", "location", "api_key"),
    [
        ("other", "apiKey", XUANJU_KEY),
        (DY, "provider-header", XUANJU_KEY),
        ("other", "provider-header", XUANJU_KEY),
        (DY, "model-header", XUANJU_KEY),
        ("other", "model-header", XUANJU_KEY),
        (DY, "override-header", XUANJU_KEY),
        ("other", "override-header", XUANJU_KEY),
        ("other", "apiKey", DEEPSEEK_KEY),
        ("other", "provider-header", DEEPSEEK_KEY),
        ("other", "model-header", DEEPSEEK_KEY),
        ("other", "apiKey", migration._DY_ENV_KEY),
    ],
)
def test_plan_rejects_fixed_migration_key_provider_collisions(
    tmp_path: Path, provider: str, location: str, api_key: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    document = yaml.safe_load(models.read_text().split("\n", 1)[1])
    target = document["providers"][provider]
    if location == "apiKey":
        target["apiKey"] = api_key
    elif location == "provider-header":
        target["headers"] = {"Authorization": api_key}
    elif location == "model-header":
        target["models"][0]["headers"] = {"Authorization": api_key}
    else:
        target["modelOverrides"] = {
            target["models"][0]["id"]: {"headers": {"Authorization": api_key}}
        }
    models.write_text(HEADER + "\n" + yaml.safe_dump(document, sort_keys=False))
    models.chmod(0o600)

    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-environment-migration-state"


def test_plan_rejects_ambient_dy_credential_reference(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    document = yaml.safe_load(models.read_text().split("\n", 1)[1])
    document["providers"][DY]["apiKey"] = "AWS_SECRET_ACCESS_KEY"
    models.write_text(HEADER + "\n" + yaml.safe_dump(document, sort_keys=False))
    models.chmod(0o600)

    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-api-family-mismatch"


@pytest.mark.parametrize("provider", ["CC-XUANJU", "CC-DY-CODEX"])
def test_plan_rejects_noncanonical_target_provider_aliases(
    tmp_path: Path, provider: str
):
    db, models, config, environment, probe = _inputs(tmp_path)
    document = yaml.safe_load(models.read_text().split("\n", 1)[1])
    document["providers"][provider] = copy.deepcopy(document["providers"]["other"])
    models.write_text(HEADER + "\n" + yaml.safe_dump(document, sort_keys=False))
    models.chmod(0o600)

    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-models-migration-state"


def test_applied_provider_override_urls_match_omp_responses_and_discovery_paths(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)

    document = yaml.safe_load(models.read_text().split("\n", 1)[1])
    provider_override = document["providers"][XUANJU]
    assert provider_override == {
        "baseUrl": "https://hs32.com/v1",
        "apiKey": XUANJU_KEY,
        "api": "openai-responses",
        "models": [{"id": "GLM-5.3-flash"}],
    }

    # OMP 18.2.0 gives providerOverride.baseUrl precedence over the
    # discovered model baseUrl before openai-responses appends /responses.
    discovered_model = {"baseUrl": "https://discovery.invalid/v1"}
    effective_inference_base = (
        provider_override.get("baseUrl") or discovered_model["baseUrl"]
    )
    assert effective_inference_base.rstrip("/") + "/responses" == (
        "https://hs32.com/v1/responses"
    )

    dy_override = document["providers"][DY]


    assert dy_override["discovery"] == {
        "type": "openai-models-list",
        "injectV1": True,
    }
    discovery_base = dy_override["baseUrl"].rstrip("/")
    if migration._DY_DISCOVERY["injectV1"] and not discovery_base.endswith("/v1"):
        discovery_base += "/v1"
    assert discovery_base + "/models" == f"https://{DY_HOST}/v1/models"

def test_plan_binds_installed_omp_package_manifest(monkeypatch, tmp_path: Path):
    package = _write_omp_package(tmp_path)
    monkeypatch.setattr(migration, "_OMP_PACKAGE_PATH", package)
    db, models, config, environment, probe = _inputs(tmp_path)

    plan = _build(db, models, config, environment, probe)

    details = package.stat()
    precondition = plan["ompPackage"]["precondition"]
    assert plan["schemaVersion"] == 4
    assert plan["ompPackage"]["name"] == "@oh-my-pi/pi-coding-agent"
    assert plan["ompPackage"]["version"] == "18.2.0"
    assert precondition == {
        "realpath": str(package.resolve()),
        "device": str(details.st_dev),
        "fileId": str(details.st_ino),
        "owner": str(details.st_uid),
        "mode": str(stat.S_IMODE(details.st_mode)),
        "size": str(details.st_size),
        "ctimeNs": str(details.st_ctime_ns),
        "sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
    }
    assert migration.validate_provider_migration_plan(plan) == plan

    tampered = copy.deepcopy(plan)
    tampered["ompPackage"]["precondition"]["sha256"] = "0" * 64
    tampered["planSha256"] = migration._plan_hash(tampered)
    with pytest.raises(MigrationPlanError) as raised:
        migration.validate_provider_migration_plan(tampered)
    assert raised.value.code == "blocked-omp-version"


@pytest.mark.parametrize("version", ["18.1.21", "18.2.1"])
def test_plan_rejects_noninstalled_omp_manifest_version(
    monkeypatch, tmp_path: Path, version: str
):
    package = _write_omp_package(tmp_path, version=version)
    monkeypatch.setattr(migration, "_OMP_PACKAGE_PATH", package)
    db, models, config, environment, probe = _inputs(tmp_path)

    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-omp-version"


def test_apply_rejects_omp_manifest_drift_before_publication(monkeypatch, tmp_path: Path):
    package = _write_omp_package(tmp_path)
    monkeypatch.setattr(migration, "_OMP_PACKAGE_PATH", package)
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}

    package.write_text(
        '{"name":"@oh-my-pi/pi-coding-agent","version":"18.2.1"}',
        encoding="utf-8",
    )
    package.chmod(0o600)
    with pytest.raises(MigrationPlanError) as raised:
        _apply(plan, db, models, config, environment)

    assert raised.value.code == "blocked-omp-version"
    assert {path: path.read_bytes() for path in before} == before


def test_verify_rejects_omp_manifest_hash_drift(monkeypatch, tmp_path: Path):
    package = _write_omp_package(tmp_path)
    monkeypatch.setattr(migration, "_OMP_PACKAGE_PATH", package)
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)

    package.write_bytes(package.read_bytes() + b"\n")
    package.chmod(0o600)

    assert verify_provider_migration(plan, db, models, config, environment) is False


@pytest.mark.parametrize("version", ["18.1.21", "18.2.1", "18.2.2"])
def test_only_omp_1820_is_accepted(tmp_path: Path, version: str):
    db, models, config, environment, probe = _inputs(tmp_path)
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe, omp_version=version)
    assert raised.value.code == "blocked-omp-version"


def test_plan_rejects_unverified_alias_or_context_suffix_substitution(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    payload = _probe_payload(db)
    missing_tool = copy.deepcopy(payload)

    missing_tool["observations"] = [
        observation
        for observation in missing_tool["observations"]
        if not (
            observation["model"] == "GLM-5.3-flash"
            and observation["mode"] == "tool"
        )
    ]
    missing_tool["artifactSha256"] = _artifact_digest(missing_tool)
    _write_probe(db, probe, missing_tool)
    with pytest.raises(MigrationPlanError) as unverified:
        _build(db, models, config, environment, probe)
    assert unverified.value.code == "blocked-probe-evidence"

    alias = _probe_payload(db)
    for observation in alias["observations"]:
        if observation["model"] == "GLM-5.3-flash" and observation["mode"] == "text":
            observation["observedModel"] = "GLM-5.3-flash-alias"
            break
    else:
        raise AssertionError("flash text observation missing")
    alias["artifactSha256"] = _artifact_digest(alias)
    _write_probe(db, probe, alias)
    with pytest.raises(MigrationPlanError) as ambiguous:
        _build(db, models, config, environment, probe)
    assert ambiguous.value.code == "blocked-probe-evidence"

    suffix = _probe_payload(db)
    for observation in suffix["observations"]:
        if observation["model"] == "GLM-5.3-flash":
            observation["configuredModelValue"] = "GLM-5.3-flash[1M]"
    suffix["artifactSha256"] = _artifact_digest(suffix)
    _write_probe(db, probe, suffix)
    with pytest.raises(MigrationPlanError) as substituted:
        _build(db, models, config, environment, probe)
    assert substituted.value.code == "blocked-probe-evidence"

def test_plan_rejects_unusable_flash_probe_evidence(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    payload = _probe_payload(db)
    for observation in payload["observations"]:
        if observation["model"] == "GLM-5.3-flash" and observation["mode"] == "text":
            observation["observedModel"] = None
            observation["identityStatus"] = "identity-unverified"
            observation["status"] = "unusable"
            observation["usable"] = False
            break
    else:
        raise AssertionError("flash text observation missing")
    payload["artifactSha256"] = _artifact_digest(payload)
    _write_probe(db, probe, payload)
    with pytest.raises(MigrationPlanError) as raised:
        _build(db, models, config, environment, probe)
    assert raised.value.code == "blocked-probe-evidence"

def test_extra_xuanju_probe_models_are_not_enabled(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    enabled = plan["configActions"]["enabledModels"]["result"]
    assert [
        selector
        for selector in enabled
        if selector.partition("/")[0] == XUANJU
    ] == NEW_SELECTORS
    assert all(
        f"{XUANJU}/{model}" not in enabled
        for model in XUANJU_PROBE_MODELS
        if model != XUANJU_MODELS[0]
    )

 


def test_plan_rejects_probe_schema_hash_and_source_snapshot_drift(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    payload = _probe_payload(db)
    payload["unexpected"] = True
    payload["artifactSha256"] = _artifact_digest(payload)
    _write_probe(db, probe, payload)
    with pytest.raises(MigrationPlanError) as schema:
        _build(db, models, config, environment, probe)
    assert schema.value.code == "blocked-probe-evidence"

    legacy = _probe_payload(db)
    legacy["schemaVersion"] = 1
    del legacy["providerEvidence"]
    legacy["artifactSha256"] = _artifact_digest(legacy)
    _write_probe(db, probe, legacy)
    with pytest.raises(MigrationPlanError) as legacy_schema:
        _build(db, models, config, environment, probe)
    assert legacy_schema.value.code == "blocked-probe-evidence"

    payload = _probe_payload(db)
    payload["artifactSha256"] = "0" * 64
    _write_probe(db, probe, payload)
    with pytest.raises(MigrationPlanError) as digest:
        _build(db, models, config, environment, probe)
    assert digest.value.code == "blocked-probe-evidence"

    payload = _probe_payload(db)
    semantic_binding = "f" * 64
    payload["semanticBindingSha256"] = semantic_binding
    payload["providerEvidence"][0]["semanticBindingSha256"] = semantic_binding
    payload["artifactSha256"] = _artifact_digest(payload)
    _write_probe(db, probe, payload)
    with pytest.raises(MigrationPlanError) as source:
        _build(db, models, config, environment, probe)
    assert source.value.code == "blocked-probe-source-changed"


def test_plan_rejects_probe_after_credential_rotation(
    tmp_path: Path,
):
    db, models, config, environment, probe = _inputs(tmp_path)
    evidence = _probe_payload(db)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_AUTH_TOKEN"] = "marker-rotated-xuanju-secret"
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), XUANJU_SOURCE),
        )

    current = migration_planning_source(
        db,
        selected_source_ids={DY_SOURCE, XUANJU_SOURCE},
        approved_origin_source_ids={DY_SOURCE, XUANJU_SOURCE},
    )
    current_binding = migration._xuanju_binding_context(
        db, current, XUANJU_SOURCE
    )
    assert (
        current_binding["semanticBindingSha256"]
        != evidence["semanticBindingSha256"]
    )

    with pytest.raises(MigrationPlanError) as stale:
        _build(db, models, config, environment, probe)
    assert stale.value.code == "blocked-probe-source-changed"


def test_plan_rejects_env_duplicates_insecure_mode_and_symlink(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    environment.write_text(environment.read_text() + f"{DEEPSEEK_KEY}=duplicate\n")
    with pytest.raises(MigrationPlanError) as duplicate:
        _build(db, models, config, environment, probe)
    assert duplicate.value.code == "blocked-env-duplicate-key"

    environment.write_text(f"{DEEPSEEK_KEY}={OLD_SECRET}\n")
    environment.chmod(0o644)
    with pytest.raises(MigrationPlanError) as mode:
        _build(db, models, config, environment, probe)
    assert mode.value.code == "blocked-env-mode"

    if hasattr(os, "symlink"):
        replacement = tmp_path / "real.env"
        replacement.write_text(f"{DEEPSEEK_KEY}={OLD_SECRET}\n")
        environment.unlink()
        environment.symlink_to(replacement)
        with pytest.raises(MigrationPlanError) as linked:
            _build(db, models, config, environment, probe)
        assert linked.value.code == "blocked-environment-symlink"


def test_apply_updates_all_three_files_exactly_and_preserves_roles_and_fallbacks(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    before_config = yaml.safe_load(config.read_text())
    plan = _build(db, models, config, environment, probe)

    report = _apply(plan, db, models, config, environment)

    assert report["status"] == "applied"
    assert set(report["updatedFiles"]) == {"environment", "models", "config"}
    assert report["rollback"] == {"opaqueBackup": "deleted", "withinRun": True}
    _assert_no_secret(report)
    assert verify_provider_migration(plan, db, models, config, environment) is True
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in (models, config, environment))

    parsed_env = environment.read_text()
    assert f"{DEEPSEEK_KEY}=" not in parsed_env
    assert f"{XUANJU_KEY}=" in parsed_env
    parsed_models = yaml.safe_load(models.read_text().split("\n", 1)[1])
    assert set(parsed_models["providers"]) == {DY, XUANJU, "other"}
    assert parsed_models["providers"][XUANJU] == {
        "baseUrl": "https://hs32.com/v1",
        "apiKey": XUANJU_KEY,
        "api": "openai-responses",
        "models": [{"id": "GLM-5.3-flash"}],
    }
    assert parsed_models["providers"][DY] == {
        "baseUrl": f"https://{DY_HOST}/v1",
        "apiKey": "CC_SWITCH_DY_CODEX_KEY",
        "api": "openai-responses",
        "discovery": {"type": "openai-models-list", "injectV1": True},
    }

    parsed_config = yaml.safe_load(config.read_text())
    assert parsed_config["modelProviderOrder"] == ["openai-codex", DY, XUANJU]
    assert parsed_config["enabledModels"] == [
        f"{DY}/gpt-5.6",
        f"{DY}/gpt-5.6-sol",
        *NEW_SELECTORS,
    ]
    assert parsed_config["modelRoles"] == before_config["modelRoles"]
    assert parsed_config["retry"]["fallbackChains"] == before_config["retry"]["fallbackChains"]


def test_verify_rejects_wrong_migrated_credential(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)

    environment.write_text(
        environment.read_text().replace(
            f"{XUANJU_KEY}={XUANJU_SECRET}",
            f"{XUANJU_KEY}=marker-wrong-xuanju-secret",
        )
    )
    environment.chmod(0o600)

    assert verify_provider_migration(plan, db, models, config, environment) is False


def test_verify_rejects_noncanonical_special_credential_value(tmp_path: Path):
    special = "$(echo injected);$XUANJU_TOKEN"
    db, models, config, environment, probe = _inputs(tmp_path)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?",
            (XUANJU_SOURCE,),
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["env"]["ANTHROPIC_AUTH_TOKEN"] = special
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), XUANJU_SOURCE),
        )
    _write_probe(db, probe)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)

    canonical = f"{XUANJU_KEY}={shlex.quote(special)}"
    raw = environment.read_text()
    assert canonical in raw
    environment.write_text(raw.replace(canonical, f"{XUANJU_KEY}={special}"))
    environment.chmod(0o600)

    assert verify_provider_migration(plan, db, models, config, environment) is False


def test_verify_rejects_disabled_projected_provider_after_apply(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)

    parsed_config = yaml.safe_load(config.read_text())
    parsed_config["disabledProviders"] = [" CC-XUANJU "]
    config.write_text(yaml.safe_dump(parsed_config, sort_keys=False))
    config.chmod(0o600)

    assert verify_provider_migration(plan, db, models, config, environment) is False


def test_verify_rejects_environment_replacement_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    _apply(plan, db, models, config, environment)
    original_apply_source = migration._apply_source
    calls = 0

    def replace_environment_after_source_read(
        db_path: str | Path, source_ids: set[str]
    ):
        nonlocal calls
        result = original_apply_source(db_path, source_ids)
        calls += 1
        if calls == 2:
            replacement = tmp_path / "replacement.env"
            replacement.write_text(
                environment.read_text().replace(
                    f"{XUANJU_KEY}={XUANJU_SECRET}",
                    f"{XUANJU_KEY}=marker-raced-xuanju-secret",
                )
            )
            replacement.chmod(0o600)
            replacement.replace(environment)
        return result

    monkeypatch.setattr(
        migration, "_apply_source", replace_environment_after_source_read
    )
    assert verify_provider_migration(plan, db, models, config, environment) is False
    assert calls >= 2


def test_apply_reports_retained_recovery_set_after_unlink_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}
    unlink = migration._materializer._unlink_at
    failed = False

    def fail_once_for_environment_backup(directory_fd: int, name: str) -> bool:
        nonlocal failed
        if name == "environment.opaque" and not failed:
            failed = True
            return False
        return unlink(directory_fd, name)

    monkeypatch.setattr(
        migration._materializer, "_unlink_at", fail_once_for_environment_backup
    )
    report = _apply(plan, db, models, config, environment)

    assert failed is True
    assert report["status"] == "applied"
    assert report["rollback"] == {"opaqueBackup": "retained", "withinRun": True}
    assert verify_provider_migration(plan, db, models, config, environment) is True
    recovery_root = models.parent / ".omp-projection-rollback"
    runs = [path for path in recovery_root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    recovery = runs[0]
    assert {path.name for path in recovery.iterdir()} == {
        "models.opaque",
        "config.opaque",
        "environment.opaque",
        "manifest.json",
    }
    for kind, path in (
        ("models", models),
        ("config", config),
        ("environment", environment),
    ):
        payload = recovery / f"{kind}.opaque"
        assert stat.S_IMODE(payload.stat().st_mode) == 0o600
        assert payload.read_bytes() == before[path]
    _assert_no_secret(json.loads((recovery / "manifest.json").read_text()))


def test_apply_rejects_target_or_source_drift_without_partial_write(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}
    environment.write_text(environment.read_text() + "UNRELATED=value\n")

    with pytest.raises(MigrationPlanError) as target:
        _apply(plan, db, models, config, environment)
    assert target.value.code == "blocked-environment-changed"
    assert models.read_bytes() == before[models]
    assert config.read_bytes() == before[config]

    # A telemetry-only WAL write does not alter the selected semantic binding.
    telemetry_root = tmp_path / "telemetry"
    telemetry_root.mkdir()
    telemetry_db, telemetry_models, telemetry_config, telemetry_env, telemetry_probe = (
        _inputs(telemetry_root)
    )
    with sqlite3.connect(telemetry_db) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE migration_telemetry (value TEXT)")
        connection.commit()
        telemetry_plan = _build(
            telemetry_db,
            telemetry_models,
            telemetry_config,
            telemetry_env,
            telemetry_probe,
        )
        connection.execute(
            "INSERT INTO migration_telemetry (value) VALUES ('unrelated')"
        )
        connection.commit()
        assert _apply(
            telemetry_plan,
            telemetry_db,
            telemetry_models,
            telemetry_config,
            telemetry_env,
        )["status"] == "applied"

    # A selected provider semantic change remains a hard stop, even though the
    # source database identity and its WAL sidecars are unchanged.
    source_root = tmp_path / "source-drift"
    source_root.mkdir()
    source_db, source_models, source_config, source_env, source_probe = _inputs(
        source_root
    )
    with sqlite3.connect(source_db) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE migration_telemetry (value TEXT)")
        connection.commit()
        source_plan = _build(
            source_db, source_models, source_config, source_env, source_probe
        )
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ?", (DY_SOURCE,)
        ).fetchone()
        assert row is not None
        settings = json.loads(row[0])
        settings["models"][0]["name"] = "changed after plan"
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps(settings), DY_SOURCE),
        )
        connection.commit()
        with pytest.raises(MigrationPlanError) as source:
            _apply(
                source_plan,
                source_db,
                source_models,
                source_config,
                source_env,
            )
        assert source.value.code == "blocked-source-changed"

@pytest.mark.parametrize("drift", ("model", "blocker", "current"))
def test_apply_rejects_selected_source_drift_in_final_capture_window(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
):
    source_root = tmp_path / f"final-window-{drift}"
    source_root.mkdir()
    db, models, config, environment, probe = _inputs(source_root)
    with sqlite3.connect(db) as connection:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE migration_telemetry (value TEXT)")
        connection.commit()
        plan = _build(db, models, config, environment, probe)

    before = {path: path.read_bytes() for path in (models, config, environment)}
    original_apply_source = migration._apply_source
    mutated = False

    def mutate_selected_source_during_final_capture(
        db_path: str | Path, source_ids: set[str]
    ) -> dict[str, Any]:
        nonlocal mutated
        if source_ids == {DY_SOURCE, XUANJU_SOURCE} and not mutated:
            with sqlite3.connect(db_path) as connection:
                if drift in {"model", "blocker"}:
                    row = connection.execute(
                        "SELECT settings_config FROM providers WHERE id = ?",
                        (DY_SOURCE,),
                    ).fetchone()
                    assert row is not None
                    settings = json.loads(row[0])
                    if drift == "model":
                        settings["models"][0]["name"] = "raced source model"
                    else:
                        settings["models"] = []
                    connection.execute(
                        "UPDATE providers SET settings_config = ? WHERE id = ?",
                        (json.dumps(settings), DY_SOURCE),
                    )
                else:
                    connection.execute(
                        "UPDATE providers SET is_current = 0 WHERE id = ?",
                        (XUANJU_SOURCE,),
                    )
                connection.commit()
            mutated = True
        return original_apply_source(db_path, source_ids)

    monkeypatch.setattr(
        migration, "_apply_source", mutate_selected_source_during_final_capture
    )
    with pytest.raises(MigrationPlanError) as raised:
        _apply(plan, db, models, config, environment)

    assert mutated is True
    assert raised.value.code == "blocked-source-changed"
    assert {path: path.read_bytes() for path in (models, config, environment)} == before


def test_apply_rolls_back_the_three_file_transaction_on_post_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}

    monkeypatch.setattr(migration, "verify_provider_migration", lambda *args: False)
    with pytest.raises(MigrationPlanError) as raised:
        _apply(plan, db, models, config, environment)
    assert raised.value.code == "blocked-apply-verify"
    assert {path: path.read_bytes() for path in before} == before
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in before)



def test_apply_rolls_back_when_fsync_fails_after_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}
    original_fsync = migration.os.fsync
    failed = False

    def fail_once_after_models_replace(descriptor: int) -> None:
        nonlocal failed
        if not failed and os.fstat(descriptor).st_ino == models.stat().st_ino:
            failed = True
            raise OSError("synthetic post-replace fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(migration.os, "fsync", fail_once_after_models_replace)
    with pytest.raises(MigrationPlanError) as raised:
        _apply(plan, db, models, config, environment)

    assert failed is True
    assert raised.value.code == "blocked-target-write"
    assert {path: path.read_bytes() for path in before} == before


def test_apply_retains_0600_manifest_and_payloads_if_rollback_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}

    monkeypatch.setattr(migration, "verify_provider_migration", lambda *args: False)

    def fail_rollback(*_args: Any, **_kwargs: Any) -> None:
        raise MigrationPlanError("blocked-rollback")

    monkeypatch.setattr(migration, "_rollback", fail_rollback)
    with pytest.raises(MigrationPlanError) as raised:
        _apply(plan, db, models, config, environment)
    assert raised.value.code == "blocked-rollback"

    recovery_root = models.parent / ".omp-projection-rollback"
    runs = [path for path in recovery_root.iterdir() if path.is_dir()]
    assert len(runs) == 1
    recovery = runs[0]
    assert stat.S_IMODE(recovery.stat().st_mode) == 0o700
    assert {path.name for path in recovery.iterdir()} == {
        "models.opaque",
        "config.opaque",
        "environment.opaque",
        "manifest.json",
    }
    manifest = json.loads((recovery / "manifest.json").read_text())
    assert manifest["schemaVersion"] == 1
    assert manifest["planSha256"] == plan["planSha256"]
    _assert_no_secret(manifest)
    for kind, path in (
        ("models", models),
        ("config", config),
        ("environment", environment),
    ):
        payload = recovery / f"{kind}.opaque"
        assert stat.S_IMODE(payload.stat().st_mode) == 0o600
        assert payload.read_bytes() == before[path]
        assert manifest["files"][kind] == {
            "mode": "384",
            "sha256": hashlib.sha256(before[path]).hexdigest(),
            "size": str(len(before[path])),
        }

def test_apply_rejects_missing_consent_or_wrong_digest_before_mutation(tmp_path: Path):
    db, models, config, environment, probe = _inputs(tmp_path)
    plan = _build(db, models, config, environment, probe)
    before = {path: path.read_bytes() for path in (models, config, environment)}

    with pytest.raises(MigrationPlanError) as consent:
        apply_provider_migration_plan(
            plan,
            db,
            models,
            config,
            environment,
            accepted_plan_sha256=plan["planSha256"],
            allow_opaque_backup=False,
        )
    assert consent.value.code == "blocked-backup-consent"

    with pytest.raises(MigrationPlanError) as digest:
        apply_provider_migration_plan(
            plan,
            db,
            models,
            config,
            environment,
            accepted_plan_sha256="0" * 64,
            allow_opaque_backup=True,
        )
    assert digest.value.code == "blocked-accepted-plan-sha256"
    assert {path: path.read_bytes() for path in before} == before
