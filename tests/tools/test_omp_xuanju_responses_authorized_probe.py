"""Focused no-network contracts for the Xuanju OpenAI Responses runner."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import time
import socket
import stat
import subprocess
from types import SimpleNamespace
from typing import Any
import sqlite3
from copy import deepcopy

import pytest

import tools.omp_xuanju_responses_authorized_probe as runner


TARGET_SOURCE = runner.SOURCE_ID
TARGET_MODELS = runner.MODELS


@pytest.fixture
def binding() -> runner._Binding:
    snapshot = "a" * 64
    return runner._Binding(
        snapshot,
        {"realpath": "/fixture/cc-switch.db", "snapshotSha256": snapshot, "shm": None, "wal": None},
        "b" * 64,
        "c" * 64,
    )


def _create_real_source_db(
    tmp_path: Path, source_family: str = "openai_responses"
) -> Path:
    db_path = tmp_path / "cc-switch-current.db"
    settings = {
        "model": "qwen-3.8",
        "env": {
            "ANTHROPIC_BASE_URL": "https://hs32.com",
            "ANTHROPIC_AUTH_TOKEN": "fixture-private-credential",
        },
        "models": [],
    }
    with sqlite3.connect(db_path) as connection:
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
            VALUES (?, 'claude', ?, ?, ?, 1, 1)
            """,
            (
                TARGET_SOURCE,
                "Current Xuanju fixture",
                json.dumps(settings),
                json.dumps({"apiFormat": source_family}),
            ),
        )
    return db_path


def _semantic_source_views() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    precondition = {
        "realpath": "/fixture/cc-switch.db",
        "device": 1,
        "fileId": 2,
        "owner": os.getuid(),
        "mode": 0o600,
        "size": 10,
        "mtimeNs": 11,
        "ctimeNs": 12,
        "sha256": "a" * 64,
        "wal": {
            "device": 3,
            "fileId": 4,
            "owner": os.getuid(),
            "mode": 0o600,
            "size": 20,
            "mtimeNs": 21,
            "ctimeNs": 22,
            "sha256": "b" * 64,
        },
        "shm": {
            "device": 5,
            "fileId": 6,
            "owner": os.getuid(),
            "mode": 0o600,
            "size": 30,
            "mtimeNs": 31,
            "ctimeNs": 32,
            "sha256": "c" * 64,
        },
    }
    endpoint_sha256 = hashlib.sha256(runner.BUSINESS_URL.encode()).hexdigest()
    public_provider = {
        "source_id": TARGET_SOURCE,
        "app_type": "claude",
        "is_current": True,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint_https": True,
        "endpoint_sha256": endpoint_sha256,
        "apply_binding_sha256": "d" * 64,
        "credential_reference_present": True,
        "declared_models": list(TARGET_MODELS),
        "environment_models": {
            "configured_model_values": ["GLM-5.3"],
            "name_to_configured_value": {"GLM-5.3": "GLM-5.3"},
        },
        "blocker_codes": [],
    }
    private_provider = {
        "source_id": TARGET_SOURCE,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint": runner.BUSINESS_URL,
        "endpoint_sha256": endpoint_sha256,
        "apply_binding_sha256": "d" * 64,
        "credential_value": "fixture-private-credential",
    }
    materialized_provider = {
        "source_id": TARGET_SOURCE,
        "app_type": "claude",
        "is_current": True,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint": runner.BUSINESS_URL,
        "credential_reference": "ANTHROPIC_AUTH_TOKEN",
        "declared_models": list(TARGET_MODELS),
        "model_definitions": [{"id": model} for model in TARGET_MODELS],
        "blocker_codes": [],
    }
    return precondition, public_provider, private_provider, materialized_provider


def _child_result(model: str, mode: str, phase: str = "live", **changes: Any) -> dict[str, Any]:
    value = runner._baseline(model, mode)
    if phase == "preflight":
        value.update(verification_kind="preflight-only", status="preflight-ok")
    else:
        value.update(
            verification_kind=f"live-{mode}",
            status="usable",
            response_model_id=model,
            identity_status="observed-match",
            tool_result="smoke_ok" if mode == "tool" else None,
        )
    value.update(changes)
    assert runner._valid_result(value, model, mode, phase)
    return value


def _stable_reads(monkeypatch: pytest.MonkeyPatch, value: runner._Binding) -> None:
    monkeypatch.setattr(runner, "_read_binding", lambda _source_db: value)


def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("focused runner tests must not open a network socket")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(runner.http.client, "HTTPSConnection", forbidden)


def test_fixed_model_order_and_exact_text_then_tool_preflight(
    monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    _no_network(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode, phase))
        return _child_result(model, mode, phase), 0

    summary = runner.run_preflight("/fixture/cc-switch.db", child=child)

    assert calls == [(model, mode, "preflight") for model in TARGET_MODELS for mode in runner.MODES]
    assert summary["status"] == "preflight-ok"
    assert summary["attemptedCalls"] == runner.MAX_API_CALLS == 10
    assert summary["retryCount"] == summary["fallbackCount"] == 0
    assert summary["serial"] is True


@pytest.mark.parametrize("source_family", ["anthropic", "openai_responses"])
def test_real_cc_source_seam_preflight_is_ten_calls_and_network_free(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_family: str
) -> None:
    source_db = _create_real_source_db(tmp_path, source_family)
    _no_network(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode, phase))
        return _child_result(model, mode, phase), 0

    summary = runner.run_preflight(source_db, child=child)

    assert summary["status"] == "preflight-ok"
    assert summary["attemptedCalls"] == 10
    assert calls == [(model, mode, "preflight") for model in TARGET_MODELS for mode in runner.MODES]


def test_responses_request_shape_has_no_anthropic_route() -> None:
    text = runner.build_responses_request("GLM-5.3", "text")
    assert text == {
        "model": "GLM-5.3",
        "input": "Reply with a non-empty answer and no explanation.",
        "max_output_tokens": 64,
        "store": False,
    }
    tool = runner.build_responses_request("GLM-5.3", "tool")
    assert tool["model"] == "GLM-5.3"
    assert tool["max_output_tokens"] == 64
    assert tool["store"] is False
    assert tool["tool_choice"] == {"type": "function", "name": "smoke_ok"}
    assert tool["tools"] == [
        {
            "type": "function",
            "name": "smoke_ok",
            "description": "Inert fixed compatibility probe.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]
    assert runner.BUSINESS_URL == "https://hs32.com/v1/responses"
    assert runner.API_FAMILY == "openai_responses"
    assert runner.REQUEST_API_FAMILY == "openai-responses"


def test_mock_transport_exercises_request_serialization_and_text_parse() -> None:
    seen: dict[str, Any] = {}

    def transport(
        endpoint: str, headers: dict[str, str], body: bytes, timeout: float
    ) -> tuple[int, bytes]:
        seen.update(endpoint=endpoint, headers=headers, body=json.loads(body), timeout=timeout)
        return (
            200,
            json.dumps(
                {
                    "model": "GLM-5.3",
                    "status": "completed",
                    "output": [
                        {
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": "ok"}],
                        }
                    ],
                    "usage": {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
                }
            ).encode(),
        )

    result = runner._request_result("GLM-5.3", "text", "credential-secret", transport)

    assert result["status"] == "usable"
    assert result["response_model_id"] == "GLM-5.3"
    assert result["identity_status"] == "observed-match"
    assert result["token_usage"] == {"input_tokens": 7, "output_tokens": 2, "total_tokens": 9}
    assert seen["endpoint"] == runner.BUSINESS_URL
    assert seen["headers"] == {
        "Authorization": "Bearer credential-secret",
        "Content-Type": "application/json",
    }
    assert seen["body"] == runner.build_responses_request("GLM-5.3", "text")
    assert seen["timeout"] == runner.PER_CALL_TIMEOUT_SECONDS == 180

def test_mock_transport_observes_exact_function_call() -> None:
    def transport(*_args: Any) -> tuple[int, bytes]:
        return (
            200,
            json.dumps(
                {
                    "model": "Kimi-K3",
                    "status": "completed",
                    "output": [
                        {"type": "reasoning", "summary": []},
                        {
                            "type": "function_call",
                            "status": "completed",
                            "name": "smoke_ok",
                            "arguments": "{}",
                            "call_id": "call_fixture",
                        },
                    ],
                }
            ).encode(),
        )

    result = runner._request_result("Kimi-K3", "tool", "credential-secret", transport)

    assert result["status"] == "usable"
    assert result["tool_result"] == "smoke_ok"
    assert result["identity_status"] == "observed-match"

@pytest.mark.parametrize(
    ("response", "expected_status", "expected_identity"),
    [
        (
            {
                "model": "GLM-5.3-flash",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "ok"}],
                    }
                ],
            },
            "identity-mismatch",
            "identity-mismatch",
        ),
        (
            {
                "model": "GLM-5.3",
                "status": "completed",
                "output": [
                    {"type": "message", "role": "assistant", "status": "completed", "content": []}
                ],
            },
            "unusable-empty-text",
            "observed-match",
        ),
        (
            {
                "model": "GLM-5.3",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": ""}],
                    }
                ],
            },
            "unusable-empty-text",
            "observed-match",
        ),
    ],
)
def test_text_identity_and_empty_output_fail_closed(
    response: dict[str, Any], expected_status: str, expected_identity: str
) -> None:
    result = runner._request_result(
        "GLM-5.3",
        "text",
        "credential-secret",
        lambda *_args: (200, json.dumps(response).encode()),
    )
    assert result["status"] == expected_status
    assert result["identity_status"] == expected_identity


@pytest.mark.parametrize(
    "tainted_model",
    [
        "credential-secret",
        "https://10.0.0.1/v1/responses",
        {"credential": "credential-secret"},
    ],
)
def test_response_model_mismatch_is_redacted_from_result_and_artifact(
    tainted_model: Any,
) -> None:
    response = {
        "model": tainted_model,
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "text",
        "credential-secret",
        lambda *_args: (200, json.dumps(response).encode()),
    )
    assert result["response_model_id"] is None
    assert result["status"] in {"identity-mismatch", "identity-unverified"}
    assert result["identity_status"] == result["status"]
    serialized = json.dumps(result, sort_keys=True)
    assert "credential-secret" not in serialized
    assert "10.0.0.1" not in serialized



def test_tool_missing_expected_function_call_is_rejected() -> None:
    response = {
        "model": "GLM-5.3",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "no call"}],
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "tool",
        "credential-secret",
        lambda *_args: (200, json.dumps(response).encode()),
    )
    assert result["status"] == "unusable-tool-result"
    assert result["tool_result"] == "rejected"


def test_incomplete_and_non_responses_shapes_are_rejected() -> None:
    incomplete = {
        "model": "GLM-5.3",
        "status": "in_progress",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "text",
        "credential-secret",
        lambda *_args: (200, json.dumps(incomplete).encode()),
    )
    assert result["status"] == "unusable"

    missing_item_status = {
        "model": "GLM-5.3",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "ok"}],
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "text",
        "credential-secret",
        lambda *_args: (200, json.dumps(missing_item_status).encode()),
    )
    assert result["status"] == "unusable-text-result"

    text_item = {
        "model": "GLM-5.3",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "text", "text": "not output_text"}],
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "text",
        "credential-secret",
        lambda *_args: (200, json.dumps(text_item).encode()),
    )
    assert result["status"] == "unusable-text-result"

    missing_tool_status = {
        "model": "GLM-5.3",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "name": "smoke_ok",
                "arguments": "{}",
                "call_id": "call_fixture",
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "tool",
        "credential-secret",
        lambda *_args: (200, json.dumps(missing_tool_status).encode()),
    )
    assert result["status"] == "unusable-tool-result"

    missing_call_id = {
        "model": "GLM-5.3",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "status": "completed",
                "name": "smoke_ok",
                "arguments": "{}",
            }
        ],
    }
    result = runner._request_result(
        "GLM-5.3",
        "tool",
        "credential-secret",
        lambda *_args: (200, json.dumps(missing_call_id).encode()),
    )
    assert result["status"] == "unusable-tool-result"

def test_execute_is_exactly_ten_calls_and_writes_new_schema_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    _no_network(monkeypatch)
    calls: list[tuple[str, str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode, phase))
        return _child_result(model, mode, phase), 0

    output = tmp_path / "observation.json"
    summary = runner.run_authorized_probes(
        "/fixture/cc-switch.db", output, child=child
    )

    assert calls == [(model, mode, "live") for model in TARGET_MODELS for mode in runner.MODES]
    assert summary["status"] == "complete"
    assert summary["attemptedCalls"] == 10
    assert summary["outputMode"] == "0600"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["schemaVersion"] == runner.SCHEMA_VERSION == 2
    assert document["requestApi"] == "openai-responses"
    assert "sourceApiFamily" in document
    assert document["sourceApiFamily"] == "anthropic"
    assert document["semanticBindingSha256"] == binding.semantic_binding_sha256
    assert document["providerEvidence"] == [
        {
            "sourceId": TARGET_SOURCE,
            "semanticBindingSha256": binding.semantic_binding_sha256,
        }
    ]
    assert "sourceSnapshotSha256" not in document
    assert summary["semanticBindingSha256"] == binding.semantic_binding_sha256
    assert document["artifactKind"] == runner.ARTIFACT_KIND
    assert [item["model"] for item in document["observations"]] == [
        model for model in TARGET_MODELS for _mode in runner.MODES
    ]
    assert [item["mode"] for item in document["observations"]] == [
        mode for _model in TARGET_MODELS for mode in runner.MODES
    ]
    assert all(item["apiFamily"] == "openai-responses" for item in document["observations"])
    serialized = json.dumps(document)
    assert "endpoint" not in serialized.casefold()
    assert "hs32.com" not in serialized
    assert "credential-secret" not in serialized
    body = {key: value for key, value in document.items() if key != "artifactSha256"}
    assert document["artifactSha256"] == hashlib.sha256(runner._canonical(body)).hexdigest()


def test_post_publication_source_drift_invalidates_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    rotated = runner._Binding(
        binding.snapshot,
        binding.precondition,
        binding.endpoint_sha256,
        "d" * 64,
    )
    count = 0

    def read(_source_db: str) -> runner._Binding:
        nonlocal count
        count += 1
        return rotated if count == 24 else binding

    monkeypatch.setattr(runner, "_read_binding", read)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(model, mode, phase), 0

    output = tmp_path / "post-publication-drift.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", output, child=child
        )
    assert raised.value.status == "blocked-source-changed"
    assert len(calls) == runner.MAX_API_CALLS
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "blocked-source-changed"
    assert invalid["artifactStatus"] == "invalid"
    assert invalid["attemptedCalls"] == runner.MAX_API_CALLS
    assert invalid["calls"] == [
        {"model": model, "mode": mode}
        for model in TARGET_MODELS
        for mode in runner.MODES
    ]
    assert count == 24


def test_private_temp_is_not_visible_at_final_path_before_binding_confirmation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / "window.json"
    seen_temp: list[Path] = []
    real_write = runner._write_private_output

    def write_then_crash(temp: runner._PrivateOutput, payload: bytes) -> None:
        assert temp.path.parent == output.parent
        assert stat.S_IMODE(temp.path.stat().st_mode) == 0o600
        real_write(temp, payload)
        seen_temp.append(temp.path)
        assert json.loads(output.read_text(encoding="utf-8")) == {
            "status": "in-progress"
        }
        raise RuntimeError("write-complete fixture crash")

    monkeypatch.setattr(runner, "_write_private_output", write_then_crash)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db",
            output,
            child=lambda m, x, p: (_child_result(m, x, p), 0),
        )
    assert raised.value.status == "blocked-runner"
    assert not output.exists()
    assert seen_temp and not seen_temp[0].exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_binding_is_rechecked_after_temp_fsync_before_atomic_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    rotated = runner._Binding(
        binding.snapshot,
        binding.precondition,
        binding.endpoint_sha256,
        "d" * 64,
    )
    count = 0

    def read(_source_db: str) -> runner._Binding:
        nonlocal count
        count += 1
        return rotated if count == 23 else binding

    monkeypatch.setattr(runner, "_read_binding", read)
    output = tmp_path / "window-drift.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db",
            output,
            child=lambda m, x, p: (_child_result(m, x, p), 0),
        )
    assert raised.value.status == "blocked-source-changed"
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "blocked-source-changed"
    assert invalid["artifactStatus"] == "invalid"
    assert "artifactKind" not in invalid
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_exception_after_atomic_replace_removes_valid_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / "post-write-crash.json"
    real_publish = runner._publish_private_output

    def publish_then_crash(
        path: Path,
        reservation: runner._Reservation,
        temp: runner._PrivateOutput,
        payload: bytes,
    ) -> None:
        real_publish(path, reservation, temp, payload)
        raise RuntimeError("post-publish fixture crash")

    monkeypatch.setattr(runner, "_publish_private_output", publish_then_crash)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db",
            output,
            child=lambda m, x, p: (_child_result(m, x, p), 0),
        )
    assert raised.value.status == "blocked-runner"
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))



@pytest.mark.parametrize("fault_point", ["directory-fsync", "verify"])
def test_after_replace_fault_keeps_invalid_attempt_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
    fault_point: str,
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / f"after-replace-{fault_point}.json"

    if fault_point == "directory-fsync":
        real_sync = runner._sync_directory
        calls = 0

        def fail_after_replace(path: Path) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:  # reservation sync is first; this is post-replace.
                raise runner.RunnerFailure("blocked-output-write")
            real_sync(path)

        monkeypatch.setattr(runner, "_sync_directory", fail_after_replace)
    else:
        real_verify = runner._verify_output_payload
        calls = 0

        def fail_after_replace(
            path: Path, reservation: runner._Reservation, payload: bytes
        ) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise runner.RunnerFailure("blocked-output-write")
            real_verify(path, reservation, payload)

        monkeypatch.setattr(runner, "_verify_output_payload", fail_after_replace)

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db",
            output,
            child=lambda m, x, p: (_child_result(m, x, p), 0),
        )

    assert raised.value.status == "blocked-output-write"
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "blocked-output-write"
    assert invalid["artifactStatus"] == "invalid"
    assert invalid["attemptedCalls"] == runner.MAX_API_CALLS
    assert invalid["calls"] == [
        {"model": model, "mode": mode}
        for model in TARGET_MODELS
        for mode in runner.MODES
    ]
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))


def test_runner_failure_default_outcomes_do_not_erase_attempted_ledger(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / "default-outcomes.json"
    real_publish = runner._publish_private_output
    calls = 0

    def publish_then_fail(
        path: Path,
        reservation: runner._Reservation,
        temp: runner._PrivateOutput,
        payload: bytes,
    ) -> None:
        nonlocal calls
        real_publish(path, reservation, temp, payload)
        calls += 1
        if calls == 1:
            raise runner.RunnerFailure("blocked-output-write")
        # The second publication is the invalid failure summary.

    monkeypatch.setattr(runner, "_publish_private_output", publish_then_fail)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db",
            output,
            child=lambda m, x, p: (_child_result(m, x, p), 0),
        )

    assert raised.value.status == "blocked-output-write"
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "blocked-output-write"
    assert invalid["artifactStatus"] == "invalid"
    assert invalid["attemptedCalls"] == runner.MAX_API_CALLS
    assert len(invalid["calls"]) == runner.MAX_API_CALLS

def test_identity_mismatch_is_recorded_and_continues_without_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return (
            _child_result(
                model,
                mode,
                phase,
                response_model_id=None,
                identity_status="identity-mismatch",
                status="identity-mismatch",
            ),
            2,
        )

    output = tmp_path / "identity.json"
    summary = runner.run_authorized_probes(
        "/fixture/cc-switch.db", output, child=child
    )
    assert summary["status"] == "partial"
    assert summary["attemptedCalls"] == runner.MAX_API_CALLS == 10
    assert summary["retryCount"] == summary["fallbackCount"] == 0
    assert calls == [(model, mode) for model in TARGET_MODELS for mode in runner.MODES]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert [item["status"] for item in document["observations"]] == [
        "identity-mismatch"
    ] * runner.MAX_API_CALLS
    assert all(item["usable"] is False for item in document["observations"])
    assert all(item["observedModel"] is None for item in document["observations"])


def test_no_retry_or_fallback_on_unusable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []
    first = (TARGET_MODELS[0], "text")

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        if (model, mode) == first:
            return _child_result(model, mode, phase, status="unusable-empty-text"), 2
        return _child_result(model, mode, phase), 0

    output = tmp_path / "empty.json"
    summary = runner.run_authorized_probes(
        "/fixture/cc-switch.db", output, child=child
    )
    assert summary["status"] == "partial"
    assert summary["attemptedCalls"] == runner.MAX_API_CALLS == 10
    assert summary["retryCount"] == summary["fallbackCount"] == 0
    assert calls == [(model, mode) for model in TARGET_MODELS for mode in runner.MODES]
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["observations"][0]["status"] == "unusable-empty-text"
    assert document["observations"][0]["usable"] is False
    assert all(item["status"] == "usable" for item in document["observations"][1:])


def test_source_binding_drift_after_first_child_stops_without_more_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    count = 0
    rotated = runner._Binding(
        binding.snapshot,
        binding.precondition,
        binding.endpoint_sha256,
        "d" * 64,
    )

    def read(_source_db: str) -> runner._Binding:
        nonlocal count
        count += 1
        return rotated if count == 4 else binding

    monkeypatch.setattr(runner, "_read_binding", read)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(model, mode, phase), 0

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", tmp_path / "drift.json", child=child
        )
    assert raised.value.status == "blocked-source-changed"
    assert calls == [("GLM-5.3-flash", "text")]
    invalid = json.loads((tmp_path / "drift.json").read_text(encoding="utf-8"))
    assert invalid["status"] == "blocked-source-changed"
    assert invalid["artifactStatus"] == "invalid"
    assert invalid["attemptedCalls"] == 1
    assert invalid["calls"] == [{"model": "GLM-5.3-flash", "mode": "text"}]
    assert count == 4


def test_source_binding_drift_during_reservation_window_makes_zero_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    rotated = runner._Binding(
        binding.snapshot,
        binding.precondition,
        binding.endpoint_sha256,
        "d" * 64,
    )
    reads = iter((binding, rotated))
    monkeypatch.setattr(runner, "_read_binding", lambda _source_db: next(reads))
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(model, mode, phase), 0

    output = tmp_path / "reservation-window-drift.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", output, child=child
        )
    assert raised.value.status == "blocked-source-changed"
    assert calls == []
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "in-progress"}


def test_existing_output_is_refused_before_any_child_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    _stable_reads(monkeypatch, binding)
    output = tmp_path / "existing.json"
    output.write_text("existing", encoding="utf-8")
    output.chmod(0o600)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(model, mode, phase), 0

    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes(
            "/fixture/cc-switch.db", output, child=child
        )
    assert raised.value.status == "blocked-output-exists"
    assert calls == []
    assert output.read_text(encoding="utf-8") == "existing"


@pytest.mark.parametrize("failure_slot", [1, 5, 10])
def test_timeout_is_recorded_and_sequence_continues(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
    failure_slot: int,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        if len(calls) == failure_slot:
            raise runner.RunnerFailure("temporary-timeout")
        return _child_result(model, mode, phase), 0

    output = tmp_path / f"timeout-{failure_slot}.json"
    summary = runner.run_authorized_probes(
        "/fixture/cc-switch.db", output, child=child
    )
    assert summary["status"] == "partial"
    assert summary["attemptedCalls"] == runner.MAX_API_CALLS == 10
    assert calls == [(model, mode) for model in TARGET_MODELS for mode in runner.MODES]
    assert summary["outputMode"] == "0600"
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["observations"][failure_slot - 1]["status"] == "temporary-timeout"
    assert document["observations"][failure_slot - 1]["usable"] is False


@pytest.mark.parametrize(
    ("child_error", "expected_status"),
    [
        (RuntimeError("worker crash"), "unusable-worker-launch"),
        (runner.RunnerFailure("blocked-worker-reap"), "blocked-worker-reap"),
    ],
)
def test_worker_crash_or_reap_failure_stops_before_next_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
    child_error: BaseException,
    expected_status: str,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        raise child_error

    output = tmp_path / "worker-failure.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert raised.value.status == expected_status
    assert calls == [("GLM-5.3-flash", "text")]
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == expected_status
    assert invalid["attemptedCalls"] == 1
    assert invalid["artifactStatus"] == "invalid"


def test_crash_result_is_terminal_and_does_not_call_next_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(
            model,
            mode,
            phase,
            status="unusable-worker-crash",
            response_model_id=None,
            identity_status="not-observed",
            tool_result="not-run" if mode == "tool" else None,
        ), 2

    output = tmp_path / "crash-result.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)

    assert raised.value.status == "unusable-worker-crash"
    assert calls == [("GLM-5.3-flash", "text")]
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "unusable-worker-crash"
    assert invalid["attemptedCalls"] == 1
    assert invalid["artifactStatus"] == "invalid"


def test_decode_rejects_nonzero_success_as_worker_crash() -> None:
    payload = _child_result("GLM-5.3", "text")
    completed = SimpleNamespace(
        stdout=json.dumps(payload).encode("utf-8"),
        stderr=b"",
        returncode=2,
    )
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._decode_child_result(completed, "GLM-5.3", "text", "live")
    assert raised.value.status == "unusable-worker-crash"


def test_fork_child_unexpected_exception_emits_terminal_crash(
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    monkeypatch.setattr(
        runner.sys, "flags", SimpleNamespace(isolated=True, dont_write_bytecode=True)
    )

    def crash(_source_db: str | os.PathLike[str]) -> runner._Binding:
        raise RuntimeError("worker implementation crash")

    monkeypatch.setattr(runner, "_read_binding", crash)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._invoke_child("GLM-5.3", "text", "preflight", binding)
    assert raised.value.status == "unusable-worker-crash"





def test_phase_nonzero_success_result_is_worker_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        return _child_result(model, mode, phase), 2

    output = tmp_path / "nonzero-success.json"
    with pytest.raises(runner.RunnerFailure) as raised:
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert raised.value.status == "unusable-worker-crash"
    assert calls == [("GLM-5.3-flash", "text")]
    invalid = json.loads(output.read_text(encoding="utf-8"))
    assert invalid["status"] == "unusable-worker-crash"
    assert invalid["attemptedCalls"] == 1

def test_live_keyboard_interrupt_keeps_reserved_slot_and_exit_classification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        raise KeyboardInterrupt

    output = tmp_path / "interrupted.json"
    with pytest.raises(KeyboardInterrupt):
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert calls == [("GLM-5.3-flash", "text")]
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    assert summary["attemptedCalls"] == 1
    assert summary["artifactStatus"] == "invalid"


def test_live_system_exit_keeps_reserved_slot_and_safe_status(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding: runner._Binding,
) -> None:
    _stable_reads(monkeypatch, binding)
    calls: list[tuple[str, str]] = []

    def child(model: str, mode: str, phase: str) -> tuple[dict[str, Any], int]:
        calls.append((model, mode))
        raise SystemExit("private exit detail")

    output = tmp_path / "system-exit.json"
    with pytest.raises(SystemExit):
        runner.run_authorized_probes("/fixture/cc-switch.db", output, child=child)
    assert calls == [("GLM-5.3-flash", "text")]
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["status"] == "blocked-runner"
    assert summary["attemptedCalls"] == 1
    assert summary["artifactStatus"] == "invalid"
    assert "private exit detail" not in output.read_text(encoding="utf-8")

def test_failed_invalidation_removes_reserved_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "published.json"
    reservation = runner._reserve_output(output)
    runner._replace_output(output, reservation, b'{"valid":"artifact"}\n', close=False)

    def fail_replace(*_args: Any, **_kwargs: Any) -> None:
        raise runner.RunnerFailure("blocked-output-write")

    monkeypatch.setattr(runner, "_replace_output", fail_replace)
    runner._invalidate_published_output(output, reservation)
    assert not output.exists()
    assert reservation.descriptor == -1


def test_nonisolated_live_invocation_fails_before_any_request(
    monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    monkeypatch.setattr(
        runner.sys, "flags", SimpleNamespace(isolated=False, dont_write_bytecode=True)
    )
    calls: list[tuple[Any, ...]] = []

    def transport(*args: Any, **kwargs: Any) -> tuple[int, bytes]:
        calls.append((*args, *kwargs.values()))
        return 200, b"{}"

    monkeypatch.setattr(runner, "_https_transport", transport)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._invoke_child("GLM-5.3", "text", "live", binding)
    assert raised.value.status == "blocked-trusted-installation"
    assert calls == []


def test_main_nonisolated_entry_blocks_before_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner.sys, "flags", SimpleNamespace(isolated=False, dont_write_bytecode=False)
    )
    calls: list[str] = []
    emitted: list[dict[str, Any]] = []

    def forbidden(_source_db: str) -> dict[str, Any]:
        calls.append("preflight")
        raise AssertionError("preflight must not run")

    monkeypatch.setattr(runner, "run_preflight", forbidden)
    monkeypatch.setattr(runner, "_emit", emitted.append)
    assert runner.main(["--preflight-only"]) == 4
    assert calls == []
    assert emitted == [
        {
            **runner._safe_summary(
                status="blocked-trusted-installation", phase="preflight"
            )
        }
    ]


@pytest.mark.parametrize(
    ("error_kind", "expected_status", "expected_code"),
    [
        ("runner", "blocked-output-write", 4),
        ("interrupt", "interrupted", 3),
        ("exit", "blocked-runner", 4),
    ],
)
def test_main_safe_summary_preserves_reserved_call_ledger(
    monkeypatch: pytest.MonkeyPatch,
    error_kind: str,
    expected_status: str,
    expected_code: int,
) -> None:
    result = runner._failure_result("GLM-5.3-flash", "text", expected_status)
    outcome = runner._ChildOutcome(
        "GLM-5.3-flash", "text", expected_status, result, None
    )
    if error_kind == "runner":
        error: BaseException = runner.RunnerFailure(
            expected_status, outcomes=(outcome,), phase="live"
        )
    elif error_kind == "interrupt":
        error = KeyboardInterrupt()
        runner._attach_exception_context(error, (outcome,), "live")
    else:
        error = SystemExit("private exit detail")
        runner._attach_exception_context(error, (outcome,), "live")

    def fail(*_args: Any, **_kwargs: Any) -> None:
        raise error

    emitted: list[dict[str, Any]] = []
    monkeypatch.setattr(runner, "_require_trusted_runtime", lambda: None)
    monkeypatch.setattr(runner, "run_authorized_probes", fail)
    monkeypatch.setattr(runner, "_emit", emitted.append)
    assert runner.main(["--execute-authorized-probes"]) == expected_code
    assert emitted and emitted[0]["status"] == expected_status
    assert emitted[0]["attemptedCalls"] == 1
    assert emitted[0]["calls"] == [{"model": "GLM-5.3-flash", "mode": "text"}]
    assert emitted[0]["statusCounts"] == {expected_status: 1}
    if expected_code == 3:
        assert emitted[0]["status"] == "interrupted"



def test_sslkeylogfile_is_popped_and_fails_closed_before_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SSLKEYLOGFILE", "/tmp/should-never-be-used")
    calls: list[tuple[Any, ...]] = []

    def connection(*args: Any, **kwargs: Any) -> Any:
        calls.append((*args, *kwargs.values()))
        raise AssertionError("HTTPSConnection must not be created")

    monkeypatch.setattr(runner.http.client, "HTTPSConnection", connection)
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._https_transport(
            runner.BUSINESS_URL,
            {"Authorization": "Bearer secret"},
            b"{}",
            1,
        )
    assert raised.value.status == "blocked-trusted-installation"
    assert "SSLKEYLOGFILE" not in os.environ
    assert calls == []


def test_transport_context_explicitly_disables_tls_keylog(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SSLKEYLOGFILE", raising=False)
    contexts: list[Any] = []

    class Response:
        status = 200

        @staticmethod
        def read(_limit: int) -> bytes:
            return b"{}"

    class Connection:
        def __init__(self, *_args: Any, context: Any, **_kwargs: Any) -> None:
            contexts.append(context)

        @staticmethod
        def request(*_args: Any, **_kwargs: Any) -> None:
            return None

        @staticmethod
        def getresponse() -> Response:
            return Response()

        @staticmethod
        def close() -> None:
            return None

    monkeypatch.setattr(runner.http.client, "HTTPSConnection", Connection)
    status, body = runner._https_transport(
        runner.BUSINESS_URL,
        {"Authorization": "Bearer secret"},
        b"{}",
        1,
    )
    assert status == 200
    assert body == b"{}"
    assert len(contexts) == 1
    assert contexts[0].keylog_filename is None


def test_outer_timeout_kills_process_group_and_reaps_fork_child(
    monkeypatch: pytest.MonkeyPatch, binding: runner._Binding
) -> None:
    monkeypatch.setattr(
        runner.sys, "flags", SimpleNamespace(isolated=True, dont_write_bytecode=True)
    )
    monkeypatch.setattr(runner, "PROCESS_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(runner, "REAP_TIMEOUT_SECONDS", 0.2)

    def hanging_child(**_kwargs: Any) -> None:
        time.sleep(2)

    monkeypatch.setattr(runner, "_fork_child_main", hanging_child)
    sent: list[tuple[int, int]] = []
    real_kill = runner.os.kill

    def kill_pid(pid: int, signum: int) -> None:
        sent.append((pid, signum))
        real_kill(pid, signum)

    monkeypatch.setattr(runner.os, "kill", kill_pid)
    reaps: list[tuple[bool, int | None]] = []
    real_wait = runner._wait_pid

    def wait_pid(pid: int, timeout: float) -> tuple[bool, int | None]:
        result = real_wait(pid, timeout)
        reaps.append(result)
        return result

    monkeypatch.setattr(runner, "_wait_pid", wait_pid)
    killed: list[tuple[int, int]] = []
    real_killpg = runner.os.killpg

    def kill_group(pid: int, signum: int) -> None:
        killed.append((pid, signum))
        real_killpg(pid, signum)

    monkeypatch.setattr(runner.os, "killpg", kill_group)

    with pytest.raises(runner.RunnerFailure) as raised:
        runner._invoke_child("GLM-5.3", "text", "live", binding)

    assert raised.value.status == "temporary-timeout"
    assert killed and killed[0][1] == signal.SIGKILL
    assert sent and sent[0][1] == signal.SIGKILL
    assert reaps and reaps[-1][0] is True


def test_tainted_child_payload_is_rejected_without_secret_output() -> None:
    secret = "credential-secret-must-not-escape"
    payload = _child_result("GLM-5.3", "text")
    payload["unexpected"] = secret
    completed = SimpleNamespace(
        stdout=json.dumps(payload).encode(), stderr=b"", returncode=0
    )
    with pytest.raises(runner.RunnerFailure) as raised:
        runner._decode_child_result(completed, "GLM-5.3", "text", "live")
    assert raised.value.status == "unusable-worker-output"
    assert secret not in str(raised.value)


def test_artifact_builder_rejects_wrong_count_and_order(binding: runner._Binding) -> None:
    def observation(model: str, mode: str) -> dict[str, Any]:
        return {
            "sourceId": TARGET_SOURCE,
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

    complete = [observation(model, mode) for model, mode in runner.PROBES]
    with pytest.raises(runner.RunnerFailure) as count:
        runner.build_observation_artifact(
            semantic_binding_sha256=binding.semantic_binding_sha256,
            observations=complete[:-1],
        )
    assert count.value.status == "blocked-artifact"
    swapped = list(complete)
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(runner.RunnerFailure) as order:
        runner.build_observation_artifact(
            semantic_binding_sha256=binding.semantic_binding_sha256,
            observations=swapped,
        )
    assert runner._is_digest(binding.semantic_binding_sha256)
    assert order.value.status == "blocked-artifact"


def test_binding_seam_rejects_noncurrent_or_wrong_endpoint_and_never_returns_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precondition = {"realpath": "/fixture/cc-switch.db", "snapshotSha256": "a" * 64}
    public_provider = {
        "source_id": TARGET_SOURCE,
        "app_type": "claude",
        "is_current": True,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint_https": True,
        "endpoint_sha256": hashlib.sha256(runner.BUSINESS_URL.encode()).hexdigest(),
        "apply_binding_sha256": "c" * 64,
        "credential_reference_present": True,
        "declared_models": list(TARGET_MODELS),
        "environment_models": {"configured_model_values": [], "name_to_configured_value": {}},
        "blocker_codes": [],
    }
    private_provider = {
        "source_id": TARGET_SOURCE,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint": runner.BUSINESS_URL,
        "endpoint_sha256": public_provider["endpoint_sha256"],
        "apply_binding_sha256": "c" * 64,
        "credential_value": "fixture-private-credential",
    }
    materialized_provider = {
        "source_id": TARGET_SOURCE,
        "app_type": "claude",
        "is_current": True,
        "api_family": runner.SOURCE_API_FAMILY,
        "endpoint": runner.BUSINESS_URL,
        "credential_reference": "ANTHROPIC_AUTH_TOKEN",
        "declared_models": list(TARGET_MODELS),
        "model_definitions": [{"id": model} for model in TARGET_MODELS],
        "blocker_codes": [],
    }

    def public() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": dict(precondition),
            "providers": [dict(public_provider)],
        }

    def private() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": dict(precondition),
            "providers": [dict(private_provider)],
        }
    def materialized() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": dict(precondition),
            "providers": [dict(materialized_provider)],
        }

    monkeypatch.setattr(runner, "migration_planning_source", lambda *_a, **_k: public())
    monkeypatch.setattr(runner, "_migration_apply_source", lambda *_a, **_k: private())
    monkeypatch.setattr(runner, "materialization_source", lambda *_a, **_k: materialized())
    binding = runner._read_binding("/fixture/cc-switch.db")
    assert binding.apply_binding_sha256 == "c" * 64
    assert "fixture-private-credential" not in repr(binding)
    public_provider["apply_binding_sha256"] = "d" * 64
    private_provider["apply_binding_sha256"] = "d" * 64
    private_provider["credential_value"] = "rotated-private-credential"
    rotated = runner._read_binding("/fixture/cc-switch.db")
    assert rotated.apply_binding_sha256 == "d" * 64
    assert "rotated-private-credential" not in repr(rotated)


    public_provider["is_current"] = False
    with pytest.raises(runner.RunnerFailure) as not_current:
        runner._read_binding("/fixture/cc-switch.db")
    assert not_current.value.status == "blocked-source-unavailable"

    public_provider["is_current"] = True
    private_provider["endpoint"] = "https://wrong.example/v1/responses"
    with pytest.raises(runner.RunnerFailure) as wrong_endpoint:
        runner._read_binding("/fixture/cc-switch.db")
    assert wrong_endpoint.value.status == "blocked-source-unavailable"
def test_same_binding_ignores_unrelated_physical_noise_and_other_providers() -> None:
    precondition, _public, _private, _materialized = _semantic_source_views()
    left = runner._Binding(
        "a" * 64,
        precondition,
        "b" * 64,
        "c" * 64,
    )
    churned = deepcopy(precondition)
    churned["size"] = 101
    churned["mtimeNs"] = 102
    churned["ctimeNs"] = 103
    churned["sha256"] = "e" * 64
    assert isinstance(churned["wal"], dict)
    churned["wal"].update(size=201, mtimeNs=202, ctimeNs=203, sha256="f" * 64)
    assert isinstance(churned["shm"], dict)
    churned["shm"].update(size=301, mtimeNs=302, ctimeNs=303, sha256="0" * 64)
    right = runner._Binding(
        "9" * 64,
        churned,
        "b" * 64,
        "c" * 64,
    )
    assert left.semantic_binding_sha256 == right.semantic_binding_sha256
    assert runner._same_binding(left, right)


@pytest.mark.parametrize(
    "change",
    ["credential", "current", "api", "endpoint", "declared", "environment", "model_def"],
)
def test_target_semantic_changes_produce_a_new_binding(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    precondition, public_provider, private_provider, materialized_provider = (
        _semantic_source_views()
    )

    def public() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": deepcopy(precondition),
            "providers": [deepcopy(public_provider)],
        }

    def private() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": deepcopy(precondition),
            "providers": [deepcopy(private_provider)],
        }

    def materialized() -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": deepcopy(precondition),
            "providers": [deepcopy(materialized_provider)],
        }

    monkeypatch.setattr(runner, "migration_planning_source", lambda *_a, **_k: public())
    monkeypatch.setattr(runner, "_migration_apply_source", lambda *_a, **_k: private())
    monkeypatch.setattr(
        runner, "materialization_source", lambda *_a, **_k: materialized()
    )
    baseline = runner._read_binding("/fixture/cc-switch.db")

    if change == "credential":
        private_provider["credential_value"] = "rotated-private-credential"
        private_provider["apply_binding_sha256"] = "e" * 64
        public_provider["apply_binding_sha256"] = "e" * 64
    elif change == "current":
        public_provider["is_current"] = False
        materialized_provider["is_current"] = False
        with pytest.raises(runner.RunnerFailure) as raised:
            runner._read_binding("/fixture/cc-switch.db")
        assert raised.value.status == "blocked-source-unavailable"
        return
    elif change == "api":
        public_provider["api_family"] = "openai_responses"
        private_provider["api_family"] = "openai_responses"
        materialized_provider["api_family"] = "openai_responses"
        private_provider["apply_binding_sha256"] = "e" * 64
        public_provider["apply_binding_sha256"] = "e" * 64
    elif change == "endpoint":
        endpoint = "https://hs32.com/alternate"
        endpoint_sha256 = hashlib.sha256(endpoint.encode()).hexdigest()
        public_provider["endpoint_sha256"] = endpoint_sha256
        private_provider["endpoint"] = endpoint
        private_provider["endpoint_sha256"] = endpoint_sha256
        materialized_provider["endpoint"] = endpoint
        private_provider["apply_binding_sha256"] = "e" * 64
        public_provider["apply_binding_sha256"] = "e" * 64
    elif change == "declared":
        new_model = "new-model"
        public_provider["declared_models"].append(new_model)
        materialized_provider["declared_models"].append(new_model)
        materialized_provider["model_definitions"].append({"id": new_model})
    elif change == "environment":
        public_provider["environment_models"]["configured_model_values"].append(
            "new-model"
        )
        public_provider["environment_models"]["name_to_configured_value"][
            "new-model"
        ] = "new-model"
    else:
        materialized_provider["model_definitions"][0]["name"] = "rotated-definition"

    changed = runner._read_binding("/fixture/cc-switch.db")
    assert changed.semantic_binding_sha256 != baseline.semantic_binding_sha256
    assert not runner._same_binding(baseline, changed)


def test_binding_group_ignores_wal_and_shm_manifest_churn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    precondition, public_provider, private_provider, materialized_provider = (
        _semantic_source_views()
    )
    calls = {"planning": 0, "apply": 0, "materialization": 0}
    capture_index = 0

    def noisy_precondition() -> dict[str, Any]:
        nonlocal capture_index
        capture_index += 1
        current = deepcopy(precondition)
        current["size"] = int(current["size"]) + capture_index
        current["mtimeNs"] = int(current["mtimeNs"]) + capture_index
        current["ctimeNs"] = int(current["ctimeNs"]) + capture_index
        current["sha256"] = f"{capture_index:x}" * 64
        for sidecar, offset in (("wal", 1), ("shm", 2)):
            manifest = current[sidecar]
            assert isinstance(manifest, dict)
            manifest["size"] = int(manifest["size"]) + capture_index
            manifest["mtimeNs"] = int(manifest["mtimeNs"]) + capture_index
            manifest["ctimeNs"] = int(manifest["ctimeNs"]) + capture_index
            manifest["sha256"] = f"{(capture_index + offset):x}" * 64
        return current

    def view(provider: dict[str, Any]) -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": noisy_precondition(),
            "providers": [deepcopy(provider)],
        }

    def public() -> dict[str, Any]:
        calls["planning"] += 1
        return view(public_provider)

    def private() -> dict[str, Any]:
        calls["apply"] += 1
        return view(private_provider)

    def materialized() -> dict[str, Any]:
        calls["materialization"] += 1
        return view(materialized_provider)

    monkeypatch.setattr(runner, "migration_planning_source", lambda *_a, **_k: public())
    monkeypatch.setattr(runner, "_migration_apply_source", lambda *_a, **_k: private())
    monkeypatch.setattr(
        runner, "materialization_source", lambda *_a, **_k: materialized()
    )

    binding = runner._read_binding("/fixture/cc-switch.db")

    assert calls == {"planning": 1, "apply": 2, "materialization": 1}
    assert binding.semantic_binding_sha256 is not None


@pytest.mark.parametrize(
    "change",
    ["credential", "endpoint", "api", "current", "declared", "model_def"],
)
def test_binding_group_retries_and_rejects_semantic_clamp_changes(
    monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    precondition, public_provider, private_provider, materialized_provider = (
        _semantic_source_views()
    )
    after_private = deepcopy(private_provider)
    if change == "credential":
        after_private["credential_value"] = "rotated-private-credential"
        after_private["apply_binding_sha256"] = "e" * 64
    elif change == "endpoint":
        endpoint = "https://hs32.com/alternate"
        after_private["endpoint"] = endpoint
        after_private["endpoint_sha256"] = hashlib.sha256(
            endpoint.encode()
        ).hexdigest()
    elif change == "api":
        after_private["api_family"] = "openai_responses"
    elif change == "current":
        materialized_provider["is_current"] = False
    elif change == "declared":
        materialized_provider["declared_models"].append("new-model")
        materialized_provider["model_definitions"].append({"id": "new-model"})
    else:
        public_provider["model_definitions"] = deepcopy(
            materialized_provider["model_definitions"]
        )
        materialized_provider["model_definitions"][0]["name"] = "rotated-definition"

    calls = {"planning": 0, "apply": 0, "materialization": 0}

    def view(
        provider: dict[str, Any], *, current: dict[str, Any] = precondition
    ) -> dict[str, Any]:
        return {
            "source_app_type": "claude",
            "source_snapshot_sha256": "a" * 64,
            "precondition": deepcopy(current),
            "providers": [deepcopy(provider)],
        }

    def public() -> dict[str, Any]:
        calls["planning"] += 1
        return view(public_provider)

    def private() -> dict[str, Any]:
        calls["apply"] += 1
        return view(
            private_provider
            if calls["apply"] % 2
            else after_private
        )

    def materialized() -> dict[str, Any]:
        calls["materialization"] += 1
        return view(materialized_provider)

    monkeypatch.setattr(runner, "migration_planning_source", lambda *_a, **_k: public())
    monkeypatch.setattr(runner, "_migration_apply_source", lambda *_a, **_k: private())
    monkeypatch.setattr(
        runner, "materialization_source", lambda *_a, **_k: materialized()
    )

    with pytest.raises(runner.RunnerFailure) as raised:
        runner._read_binding("/fixture/cc-switch.db")
    assert raised.value.status in {
        "blocked-source-changed",
        "blocked-source-unavailable",
    }
    if change == "current":
        assert calls == {"planning": 1, "apply": 2, "materialization": 1}
    else:
        assert calls == {
            "planning": runner.MAX_BINDING_CAPTURE_ATTEMPTS,
            "apply": runner.MAX_BINDING_CAPTURE_ATTEMPTS * 2,
            "materialization": runner.MAX_BINDING_CAPTURE_ATTEMPTS,
        }
