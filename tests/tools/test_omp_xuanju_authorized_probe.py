"""Focused, no-network contracts for the two-call GLM-5.2 evidence runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import importlib.util
import socket
import stat
import signal
import sqlite3
import subprocess
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest

import tools.omp_provider_migration as migration
import tools.omp_xuanju_authorized_probe as runner


TARGET_SOURCE = runner.SOURCE_ID
TARGET_MODEL = runner.MODEL


@pytest.fixture
def binding() -> runner._Binding:
    snapshot_sha256 = "a" * 64
    return runner._Binding(
        snapshot_sha256,
        {"realpath": "/fixture/cc-switch.db", "snapshotSha256": snapshot_sha256},
        "b" * 64,
        "c" * 64,
    )


def test_runner_accepts_exact_configured_model_without_name_alias() -> None:
    assert runner._exact_environment_models(
        {
            "environment_models": {
                "configured_model_values": [TARGET_MODEL],
                "name_to_configured_value": {},
            }
        }
    )
    assert not runner._exact_environment_models(
        {
            "environment_models": {
                "configured_model_values": ["GLM-5.3"],
                "name_to_configured_value": {},
            }
        }
    )


def _child_result(mode: str, phase: str = "live", **changes: Any) -> dict[str, Any]:
    value = runner._baseline(mode)
    if phase == "preflight":
        value.update(
            verification_kind="preflight-only",
            status="preflight-ok",
        )
    else:
        value.update(
            verification_kind=f"live-{mode}",
            status="usable",
            response_model_id=TARGET_MODEL,
            identity_status="observed-match",
            tool_result="smoke_ok" if mode == "tool" else None,
        )
    value.update(changes)
    assert runner._valid_result(value, mode, phase)
    return value


def _stable_reads(monkeypatch: pytest.MonkeyPatch, value: runner._Binding) -> None:
    monkeypatch.setattr(runner, "_read_binding", lambda _source_db: value)


def _no_socket(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("focused runner tests must not open a network socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)


def test_default_preflight_is_two_ordered_no_network_calls(
    monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    _no_socket(monkeypatch)
    calls: list[tuple[str, str]] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((mode, phase))
        return _child_result(mode, phase), 0

    monkeypatch.setattr(runner, "_invoke_child", child)
    summary = runner.run_preflight("/fixture/cc-switch.db")

    assert calls == [("text", "preflight"), ("tool", "preflight")]
    assert summary["status"] == "preflight-ok"
    assert summary["attemptedCalls"] == runner.MAX_API_CALLS
    assert summary["retryCount"] == 0
    assert summary["fallbackCount"] == 0


def test_execute_is_text_then_tool_exactly_once_and_writes_v2_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    _no_socket(monkeypatch)
    calls: list[tuple[str, str]] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((mode, phase))
        return _child_result(mode, phase), 0

    output = tmp_path / "evidence.json"
    summary = runner.run_authorized_probes(
        "/fixture/cc-switch.db", output, child=child
    )

    assert calls == [("text", "live"), ("tool", "live")]
    assert summary["status"] == "complete"
    assert summary["attemptedCalls"] == 2
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    parsed = migration._read_probe(output)
    assert parsed["schemaVersion"] == 2
    assert parsed["sourceSnapshotSha256"] == binding.snapshot
    assert parsed["providerEvidence"] == [
        {
            "sourceId": TARGET_SOURCE,
            "applyBindingSha256": binding.apply_binding_sha256,
        }
    ]
    assert [item["mode"] for item in parsed["observations"]] == ["text", "tool"]
    assert parsed["artifactSha256"] == hashlib.sha256(
        migration._canonical(
            {
                key: value
                for key, value in parsed.items()
                if key != "artifactSha256"
            },
            "blocked-test-artifact",
        )
    ).hexdigest()


def test_identity_mismatch_is_fail_closed_without_second_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        return (
            _child_result(
                mode,
                phase,
                response_model_id="GLM-5.3",
                identity_status="identity-mismatch",
                status="identity-mismatch",
            ),
            2,
        )

    output = tmp_path / "evidence.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert raised.value.status == "identity-mismatch"
    assert calls == ["text"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_unusable_result_never_falls_back_or_retries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        return _child_result(mode, phase, status="unusable-text-result"), 2

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", tmp_path / "evidence.json", child=child
        )
    assert raised.value.status == "unusable"
    assert calls == ["text"]


def test_source_or_credential_binding_rotation_after_child_blocks_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    rotated = runner._Binding(
        binding.snapshot,
        binding.precondition,
        binding.endpoint_sha256,
        "d" * 64,
    )
    reads = iter((binding, binding, binding, rotated))
    monkeypatch.setattr(runner, "_read_binding", lambda _source_db: next(reads))
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        return _child_result(mode, phase), 0

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", tmp_path / "evidence.json", child=child
        )
    assert raised.value.status == "blocked-source-changed"
    assert calls == ["text"]


def test_physical_precondition_drift_is_not_just_public_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    drifted = runner._Binding(
        binding.snapshot,
        {"realpath": "/fixture/cc-switch.db", "snapshotSha256": "e" * 64},
        binding.endpoint_sha256,
        binding.apply_binding_sha256,
    )
    reads = iter((binding, binding, binding, drifted))
    monkeypatch.setattr(runner, "_read_binding", lambda _source_db: next(reads))
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        return _child_result(mode, phase), 0

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", tmp_path / "evidence.json", child=child
        )
    assert raised.value.status == "blocked-source-changed"
    assert calls == ["text"]


def test_existing_output_is_refused_before_any_child_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    output.chmod(0o600)
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        return _child_result(mode, phase), 0

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert raised.value.status == "blocked-output-exists"
    assert calls == []
    assert output.read_text(encoding="utf-8") == "existing"


def test_timeout_is_terminal_and_reservation_is_0600(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[str] = []

    def child(mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append(mode)
        raise runner.RunnerFailure("temporary-timeout")

    output = tmp_path / "timeout.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert raised.value.status == "temporary-timeout"
    assert calls == ["text"]
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "in-progress"}


def test_direct_probe_child_command_has_fixed_model_mode_limits_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class FakeProcess:
        pid = 41001
        returncode = 0

        def __init__(self, command: list[str]) -> None:
            self.command = command
            self.communicate_timeout: float | None = None

        def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
            self.communicate_timeout = timeout
            mode = self.command[self.command.index("--mode") + 1]
            return (
                json.dumps(_child_result(mode, "preflight")).encode("utf-8"),
                b"",
            )

        def poll(self) -> int:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            return self.returncode

    process_holder: dict[str, FakeProcess] = {}

    def fake_popen(command: list[str], **kwargs: Any) -> FakeProcess:
        observed["command"] = command
        observed["kwargs"] = kwargs
        process = FakeProcess(command)
        process_holder["process"] = process
        return process

    monkeypatch.setattr(runner.subprocess, "Popen", fake_popen)
    result, returncode = runner._invoke_child("text", "preflight")

    assert returncode == 0
    assert result["status"] == "preflight-ok"
    command = observed["command"]
    assert command[command.index("--model") + 1] == "GLM-5.2"
    assert command[command.index("--mode") + 1] == "text"
    assert command[command.index("--timeout") + 1] == "60"
    assert command[command.index("--max-tokens") + 1] == "64"
    assert "--preflight-only" in command
    assert "--approve-current-origin" not in command
    assert observed["kwargs"]["start_new_session"] is True
    assert process_holder["process"].communicate_timeout == 75


def test_outer_timeout_kills_process_group_and_boundedly_reaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingProcess:
        pid = 41002
        returncode = None

        def __init__(self) -> None:
            self.communicate_calls = 0
            self.wait_calls = 0

        def communicate(self, *, timeout: float) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired("fixture", timeout)
            self.returncode = -signal.SIGKILL
            return b"", b""

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, *, timeout: float) -> int:
            self.wait_calls += 1
            self.returncode = -signal.SIGKILL
            return self.returncode

        def kill(self) -> None:
            self.returncode = -signal.SIGKILL

    process = HangingProcess()
    killed: list[tuple[int, int]] = []
    monkeypatch.setattr(
        runner.subprocess, "Popen", lambda _command, **_kwargs: process
    )
    monkeypatch.setattr(
        runner.os,
        "killpg",
        lambda pid, signum: killed.append((pid, signum)),
    )

    with pytest.raises(runner.RunnerFailure) as raised:
        runner._invoke_child("text", "live")

    assert raised.value.status == "temporary-timeout"
    assert killed == [(process.pid, signal.SIGKILL)]
    assert process.communicate_calls == 2


def test_tainted_or_open_child_payload_is_rejected_without_secret_output() -> None:
    secret = "credential-secret-must-not-escape"
    payload = _child_result("text", "live")
    payload["unexpected"] = secret
    completed = SimpleNamespace(
        stdout=json.dumps(payload).encode("utf-8"), stderr=b"", returncode=0
    )
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._decode_child_result(completed, "text", "live")
    assert raised.value.status == "unusable-worker-output"
    assert secret not in str(raised.value)


def test_artifact_builder_rejects_wrong_count_or_mode() -> None:
    observation = {
        "sourceId": TARGET_SOURCE,
        "environmentName": TARGET_MODEL,
        "configuredModelValue": TARGET_MODEL,
        "model": TARGET_MODEL,
        "observedModel": TARGET_MODEL,
        "mode": "text",
        "identityStatus": "observed-match",
        "usable": True,
    }
    with pytest.raises(runner.RunnerFailure) as count:
        runner.build_probe_artifact(
            source_snapshot_sha256="a" * 64,
            apply_binding_sha256="b" * 64,
            observations=[observation],
        )
    assert count.value.status == "blocked-artifact"
    with pytest.raises(runner.RunnerFailure) as mode:
        runner.build_probe_artifact(
            source_snapshot_sha256="a" * 64,
            apply_binding_sha256="b" * 64,
            observations=[observation, {**observation, "mode": "fallback"}],
        )
    assert mode.value.status == "blocked-artifact"


def test_binding_uses_private_apply_snapshot_and_never_returns_credential(
    tmp_path: Path,
) -> None:
    source_db = tmp_path / "cc-switch.db"
    environment = {
        "ANTHROPIC_BASE_URL": "https://xuanju.fixture/v1",
        "ANTHROPIC_AUTH_TOKEN": "fixture-private-credential",
        "ANTHROPIC_DEFAULT_GLM_MODEL": TARGET_MODEL,
        "ANTHROPIC_DEFAULT_GLM_MODEL_NAME": TARGET_MODEL,
    }
    settings = {"env": environment, "models": [{"id": TARGET_MODEL}]}
    with sqlite3.connect(source_db) as connection:
        connection.execute(
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
            )
            """
        )
        connection.execute(
            """
            INSERT INTO providers
                (id, app_type, name, settings_config, meta, is_current, sort_index)
            VALUES (?, 'claude', 'Xuanju fixture', ?, ?, 1, 1)
            """,
            (
                TARGET_SOURCE,
                json.dumps(settings),
                json.dumps({"apiFormat": "anthropic"}),
            ),
        )
    source_db.chmod(0o600)

    binding = runner._read_binding(source_db)

    assert runner._is_digest(binding.snapshot)
    assert runner._is_digest(binding.endpoint_sha256)
    assert runner._is_digest(binding.apply_binding_sha256)
    assert "fixture-private-credential" not in repr(binding)

    environment["ANTHROPIC_AUTH_TOKEN"] = "rotated-private-credential"
    with sqlite3.connect(source_db) as connection:
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ?",
            (json.dumps({"env": environment, "models": [{"id": TARGET_MODEL}]}), TARGET_SOURCE),
        )
    source_db.chmod(0o600)
    rotated = runner._read_binding(source_db)
    assert rotated.snapshot == binding.snapshot
    assert rotated.precondition != binding.precondition
    assert rotated.apply_binding_sha256 != binding.apply_binding_sha256

    with sqlite3.connect(source_db) as connection:
        connection.execute(
            "UPDATE providers SET is_current = 0 WHERE id = ?", (TARGET_SOURCE,)
        )
    source_db.chmod(0o600)
    with pytest.raises(runner.RunnerFailure) as not_current:
        runner._read_binding(source_db)
    assert not_current.value.status == "blocked-source-unavailable"


def test_direct_probe_uses_current_shm_manifest_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "fixture_cc_direct_probe", runner.DIRECT_PROBE
    )
    assert spec is not None and spec.loader is not None
    direct_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct_probe)

    db_manifest = {
        "device": 1,
        "fileId": 2,
        "owner": os.getuid(),
        "mode": 0o600,
        "size": 3,
        "mtimeNs": 4,
        "ctimeNs": 5,
        "sha256": "d" * 64,
    }
    shm_manifest = {
        "device": 6,
        "fileId": 7,
        "owner": os.getuid(),
        "mode": 0o600,
        "size": 8,
        "mtimeNs": 9,
        "ctimeNs": 10,
        "sha256": "e" * 64,
    }
    source_id = direct_probe.XUANJU_SOURCE_ID
    settings = {
        "env": {
            "ANTHROPIC_AUTH_TOKEN": "fixture-private-credential",
        }
    }
    rows = [
        (
            source_id,
            "claude",
            "fixture",
            settings,
            {},
            1,
            1,
        )
    ]

    class FakeConnection:
        def execute(self, *_args: Any, **_kwargs: Any) -> "FakeConnection":
            return self

        def fetchall(self) -> list[tuple[Any, ...]]:
            return rows

        def close(self) -> None:
            return None

    class FakeAdapter:
        _SOURCE_APP_TYPE = "claude"
        _PROVIDER_QUERY = "fixture"
        _MAX_PROVIDER_ROWS = 8
        _MAX_PROVIDER_FIELD_BYTES = 256 * 1024

        def __init__(self) -> None:
            self.shm_calls: list[tuple[Any, Any]] = []

        def _source_path(self, _path: Any) -> str:
            return "fixture.db"

        def _capture_source_snapshot(self, _path: str) -> Any:
            return SimpleNamespace(
                snapshot_path="fixture.snapshot",
                identity=object(),
                source_sha256=db_manifest["sha256"],
                wal_descriptor=-1,
                wal_identity=None,
                wal_sha256=None,
                shm_descriptor=-1,
                shm_identity="fixture-shm-identity",
                shm_sha256=shm_manifest["sha256"],
                descriptor=-1,
                temporary=SimpleNamespace(cleanup=lambda: None),
            )

        def _snapshot_uri(self, path: str) -> str:
            return path

        def _configure_readonly_connection(self, _connection: Any) -> None:
            return None

        def _validate_schema(self, _connection: Any) -> None:
            return None

        def _normalise_provider(self, _row: tuple[Any, ...]) -> dict[str, Any]:
            return {"source_id": source_id}

        def _source_snapshot(self, _normalized: Any) -> str:
            return "f" * 64

        def _provider_sort_key(self, _provider: Any) -> int:
            return 0

        def _assert_capture_stable(self, _path: str, _capture: Any) -> None:
            return None

        def _stable_realpath(self, _path: str, _identity: Any) -> str:
            return str(direct_probe.CC_SWITCH_DB)

        def _file_manifest(self, _identity: Any, _digest: str) -> dict[str, Any]:
            return db_manifest

        def _shm_manifest(self, identity: Any, digest: str) -> dict[str, Any]:
            self.shm_calls.append((identity, digest))
            return shm_manifest

        def _json_object(self, value: Any) -> dict[str, Any]:
            return value

        def _optional_object(
            self, value: dict[str, Any], key: str
        ) -> dict[str, Any]:
            return value[key]

        def _credential_reference(self, _environment: Any) -> str:
            return "ANTHROPIC_AUTH_TOKEN"

    adapter = FakeAdapter()
    monkeypatch.setattr(direct_probe, "_trusted_source_adapter", lambda: adapter)
    monkeypatch.setattr(direct_probe.sqlite3, "connect", lambda *_a, **_k: FakeConnection())

    precondition = {
        "realpath": str(direct_probe.CC_SWITCH_DB),
        **db_manifest,
        "snapshotSha256": "f" * 64,
        "wal": None,
        "shm": shm_manifest,
    }
    credential = direct_probe._source_credential(
        source_id,
        "ANTHROPIC_AUTH_TOKEN",
        precondition,
    )

    assert credential == "fixture-private-credential"
    assert adapter.shm_calls == [
        ("fixture-shm-identity", shm_manifest["sha256"])
    ]


def test_direct_admits_exact_configured_value_without_name_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location(
        "fixture_cc_direct_probe_configured_only", runner.DIRECT_PROBE
    )
    assert spec is not None and spec.loader is not None
    direct_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct_probe)
    snapshot = "f" * 64

    class FakeAdapter:
        _SOURCE_APP_TYPE = "claude"

        def inventory_source(
            self, _path: Any, *, provider_filter: str
        ) -> dict[str, Any]:
            return {
                "status": "inventory",
                "source_app_type": self._SOURCE_APP_TYPE,
                "source_snapshot_sha256": snapshot,
                "providers": [
                    {
                        "source_id": provider_filter,
                        "is_current": True,
                        "api_family": "anthropic",
                        "eligibility_status": "candidate",
                        "blocker_codes": [],
                        "environment_models": {
                            "configured_model_values": [TARGET_MODEL],
                            "name_to_configured_value": {},
                        },
                    }
                ],
            }

    adapter = FakeAdapter()
    monkeypatch.setattr(direct_probe, "_trusted_source_adapter", lambda: adapter)
    assert (
        direct_probe._public_exact_name_evidence(
            adapter,
            direct_probe.XUANJU_SOURCE_ID,
            "anthropic",
            TARGET_MODEL,
        )
        == snapshot
    )


@pytest.mark.parametrize("stage", ["inventory", "materialization"])
def test_direct_inactive_source_is_rejected_before_any_request(
    monkeypatch: pytest.MonkeyPatch, stage: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        f"fixture_cc_direct_probe_{stage}", runner.DIRECT_PROBE
    )
    assert spec is not None and spec.loader is not None
    direct_probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(direct_probe)
    snapshot = "f" * 64
    manifest = {
        "device": 1,
        "fileId": 2,
        "owner": os.getuid(),
        "mode": 0o600,
        "size": 3,
        "mtimeNs": 4,
        "ctimeNs": 5,
        "sha256": "d" * 64,
    }
    precondition = {
        "realpath": str(direct_probe.CC_SWITCH_DB),
        **manifest,
        "snapshotSha256": snapshot,
        "wal": None,
        "shm": None,
    }
    source_id = direct_probe.XUANJU_SOURCE_ID
    environment_models = {
        "configured_model_values": [TARGET_MODEL],
        "name_to_configured_value": {TARGET_MODEL: TARGET_MODEL},
    }
    inventory_provider = {
        "source_id": source_id,
        "is_current": stage != "inventory",
        "api_family": "anthropic",
        "eligibility_status": "candidate",
        "blocker_codes": [],
        "environment_models": environment_models,
    }
    materialized_provider = {
        "source_id": source_id,
        "is_current": False,
        "api_family": "anthropic",
        "blocker_codes": [],
        "credential_reference": "ANTHROPIC_AUTH_TOKEN",
        "declared_models": [TARGET_MODEL],
        "endpoint": "https://inactive.fixture/v1",
    }

    class FakeAdapter:
        _SOURCE_APP_TYPE = "claude"
        materialization_calls = 0

        def inventory_source(
            self, _path: Any, *, provider_filter: str
        ) -> dict[str, Any]:
            return {
                "status": "inventory",
                "source_app_type": self._SOURCE_APP_TYPE,
                "source_snapshot_sha256": snapshot,
                "providers": [
                    {
                        **inventory_provider,
                        "source_id": provider_filter,
                    }
                ],
            }

        def materialization_source(
            self, *_args: Any, **_kwargs: Any
        ) -> dict[str, Any]:
            self.materialization_calls += 1
            return {
                "source_app_type": self._SOURCE_APP_TYPE,
                "source_snapshot_sha256": snapshot,
                "precondition": precondition,
                "providers": [materialized_provider],
            }
    adapter = FakeAdapter()
    monkeypatch.setattr(direct_probe, "_trusted_source_adapter", lambda: adapter)
    monkeypatch.setattr(
        direct_probe,
        "_make_anthropic_client",
        lambda *_args, **_kwargs: pytest.fail(
            "inactive source reached network setup"
        ),
    )
    with pytest.raises(direct_probe.ProbeFailure) as raised:
        direct_probe._projection(TARGET_MODEL)

    assert raised.value.status == "blocked-source-unavailable"
