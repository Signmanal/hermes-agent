"""Authorized, fail-closed two-call GLM-5.2 evidence runner.

The runner has two deliberately separate phases:

* preflight validates the exact Xuanju source/model seam and asks the existing
  direct probe to validate the text and tool routes without making a network
  request;
* ``--execute-authorized-probes`` reserves one output path and invokes the
  direct probe exactly once for ``text`` and once for ``tool``.

Only the direct probe owns provider network mechanics and credential extraction.
This module only carries redacted source facts in the parent, and accepts a
closed, redacted result from each child.  It never accepts endpoint, key, or
header arguments.  Failure messages intentionally contain status codes only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import signal
import subprocess
import sys
from typing import Any, Callable, Final, Mapping, NoReturn, Sequence


# ``python -I -B tools/omp_xuanju_authorized_probe.py`` excludes the checkout
# from import search.  Restore only this file's own parent, never cwd/PYTHONPATH.
if __package__ in {None, ""}:  # pragma: no cover - exercised by CLI process.
    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from tools import omp_provider_migration as _migration
from tools.cc_switch_omp_source import (
    SourceInventoryError,
    _migration_apply_source,
    migration_planning_source,
)
__all__ = [
    "RunnerFailure",
    "build_probe_artifact",
    "main",
    "run_authorized_probes",
    "run_preflight",
]


# These are installation-bound defaults.  Tests replace them with temporary
# fixtures and a mock child runner; callers cannot supply an endpoint/key.
SOURCE_DB: Final = Path("/Users/chiyuchou/.cc-switch/cc-switch.db")
DIRECT_PROBE: Final = Path(
    "/Users/chiyuchou/.omp/agent/routing-health/omp_cc_direct_probe.py"
)
PROBE_PYTHON: Final = Path("/Users/chiyuchou/.hermes/hermes-agent/venv/bin/python")
DEFAULT_OUTPUT: Final = Path(
    "/Users/chiyuchou/.omp/agent/routing-health/xuanju-glm-5.2-evidence-v2.json"
)

SOURCE_ID: Final = "616ccdfd-8de6-4c82-a4cc-24f71edf0c49"
PROVIDER: Final = "cc-xuanju"
MODEL: Final = "GLM-5.2"
API_FAMILY: Final = "anthropic"
MODES: Final = ("text", "tool")
SCHEMA_VERSION: Final = 2
MAX_API_CALLS: Final = 2
PER_CALL_TIMEOUT_SECONDS: Final = 60
PROCESS_TIMEOUT_SECONDS: Final = 75
REAP_TIMEOUT_SECONDS: Final = 1.0
MAX_OUTPUT_TOKENS: Final = 64
MAX_CHILD_OUTPUT_BYTES: Final = 16_384
MAX_ARTIFACT_BYTES: Final = 64 * 1024

# Compatibility-shaped manifest facts make the fixed two-call contract easy to
# audit without exposing any provider data.
PER_PROBE_TIMEOUT_SECONDS: Final = PER_CALL_TIMEOUT_SECONDS
MODEL_ROUTES: Final = ((PROVIDER, MODEL),)
PROBES: Final = tuple(
    (provider, model, mode)
    for mode in MODES
    for provider, model in MODEL_ROUTES
)

_RESULT_KEYS: Final = frozenset(
    {
        "configured_selector",
        "request_model_id",
        "response_model_id",
        "identity_status",
        "verification_kind",
        "status",
        "tool_result",
        "token_usage",
        "latency_ms",
    }
)
_SAFE_STATUSES: Final = frozenset({
    "blocked",
    "blocked-api-mismatch",
    "blocked-credential-reference-missing",
    "blocked-credential-unavailable",
    "blocked-invalid-arguments",
    "blocked-model-not-listed",
    "blocked-no-follow-unavailable",
    "blocked-origin-approval-required",
    "blocked-source-changed",
    "blocked-source-unavailable",
    "blocked-transport-policy",
    "blocked-trusted-installation",
    "preflight-ok",
    "usable",
    "identity-mismatch",
    "identity-unverified",
    "interrupted",
    "temporary",
    "temporary-timeout",
    "unusable",
    "unusable-empty-text",
    "unusable-text-result",
    "unusable-tool-result",
    "unusable-worker-launch",
    "unusable-worker-output",
    "usable-nonexact-text",
})
_IDENTITY_STATUSES: Final = frozenset(
    {"not-observed", "identity-unverified", "identity-mismatch", "observed-match"}
)
_DIGEST_HEX: Final = frozenset("0123456789abcdef")


class RunnerFailure(ValueError):
    """Safe status only; private exception text is never rendered."""

    def __init__(self, status: str) -> None:
        if not isinstance(status, str) or not status or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in status
        ):
            status = "blocked-runner"
        self.status = status
        super().__init__(status)


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        raise RunnerFailure("blocked-invalid-arguments")


@dataclass(frozen=True)
class _Binding:
    """Redacted facts binding one stable source/credential snapshot."""

    snapshot: str
    precondition: dict[str, Any]
    endpoint_sha256: str
    apply_binding_sha256: str


@dataclass(frozen=True)
class _ChildOutcome:
    mode: str
    status: str
    result: dict[str, Any]
    process_exit_code: int | None

@dataclass
class _Reservation:
    descriptor: int
    device: int
    inode: int


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _DIGEST_HEX for character in value)
    )


def _safe_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and len(value) <= 128
        and all(character.isalnum() or character in "._:-/" for character in value)
    )


def _baseline(mode: str) -> dict[str, Any]:
    return {
        "configured_selector": f"{PROVIDER}/{MODEL}",
        "request_model_id": MODEL,
        "response_model_id": None,
        "identity_status": "not-observed",
        "verification_kind": "not-run",
        "status": "blocked",
        "tool_result": "not-run" if mode == "tool" else None,
        "token_usage": None,
        "latency_ms": 0,
    }


def _valid_token_usage(value: Any) -> bool:
    return value is None or (
        isinstance(value, dict)
        and bool(value)
        and set(value).issubset({"input_tokens", "output_tokens", "total_tokens"})
        and all(
            isinstance(tokens, int)
            and not isinstance(tokens, bool)
            and 0 <= tokens <= 1_000_000_000
            for tokens in value.values()
        )
    )


def _valid_result(value: Any, mode: str, phase: str) -> bool:
    """Validate the direct-probe envelope without retaining untrusted fields."""
    if not isinstance(value, dict) or set(value) != _RESULT_KEYS:
        return False
    if (
        value["configured_selector"] != f"{PROVIDER}/{MODEL}"
        or value["request_model_id"] != MODEL
        or value["status"] not in _SAFE_STATUSES
        or value["identity_status"] not in _IDENTITY_STATUSES
        or not _valid_token_usage(value["token_usage"])
        or not isinstance(value["latency_ms"], int)
        or isinstance(value["latency_ms"], bool)
        or not 0 <= value["latency_ms"] <= PROCESS_TIMEOUT_SECONDS * 1000
    ):
        return False

    response_model = value["response_model_id"]
    if response_model is not None and not _safe_identifier(response_model):
        return False
    if phase == "preflight":
        return (
            value["verification_kind"] == "preflight-only"
            and value["status"] == "preflight-ok"
            and response_model is None
            and value["identity_status"] == "not-observed"
            and value["token_usage"] is None
            and value["tool_result"] == ("not-run" if mode == "tool" else None)
        )
    if phase != "live" or value["verification_kind"] != f"live-{mode}":
        return False
    if value["identity_status"] == "observed-match":
        if response_model != MODEL:
            return False
    elif value["identity_status"] == "identity-mismatch":
        if response_model is None or response_model == MODEL:
            return False
    elif value["identity_status"] == "identity-unverified":
        if response_model is not None:
            return False
    else:
        return False
    if mode == "tool":
        return value["tool_result"] in {"rejected", "smoke_ok"}
    return value["tool_result"] is None


def _successful_live_result(value: Mapping[str, Any], mode: str) -> bool:
    """Require exact identity and usable semantics for artifact admission."""
    return (
        value.get("verification_kind") == f"live-{mode}"
        and value.get("status") == "usable"
        and value.get("identity_status") == "observed-match"
        and value.get("response_model_id") == MODEL
        and value.get("tool_result") == ("smoke_ok" if mode == "tool" else None)
    )


def _exact_environment_models(provider: Mapping[str, Any]) -> bool:
    environment_models = provider.get("environment_models")
    if not isinstance(environment_models, dict) or set(environment_models) != {
        "configured_model_values",
        "name_to_configured_value",
    }:
        return False
    configured = environment_models["configured_model_values"]
    named = environment_models["name_to_configured_value"]
    return (
        isinstance(configured, list)
        and len(configured) == len(set(configured))
        and all(isinstance(item, str) and item for item in configured)
        and isinstance(named, dict)
        and all(
            isinstance(name, str)
            and bool(name)
            and isinstance(value, str)
            and bool(value)
            for name, value in named.items()
        )
        and MODEL in configured
    )


def _read_binding(source_db: str | os.PathLike[str]) -> _Binding:
    """Read public facts plus the private apply seam from one stable capture.

    The public planning result proves model availability; the apply-only result
    proves that the endpoint and credential value were read from that same
    physical DB/WAL/SHM precondition.  The private record is cleared before
    returning and is never sent to a child or serialized.
    """
    source: Any = None
    private_source: Any = None
    try:
        try:
            source = migration_planning_source(
                source_db,
                selected_source_ids={SOURCE_ID},
                approved_origin_source_ids={SOURCE_ID},
            )
            private_source = _migration_apply_source(
                source_db,
                selected_source_ids={SOURCE_ID},
                approved_origin_source_ids={SOURCE_ID},
            )
        except SourceInventoryError:
            raise RunnerFailure("blocked-source-unavailable") from None
        except (OSError, TypeError, ValueError, UnicodeError):
            raise RunnerFailure("blocked-source-unavailable") from None

        if not isinstance(source, dict) or set(source) != {
            "source_app_type",
            "source_snapshot_sha256",
            "precondition",
            "providers",
        }:
            raise RunnerFailure("blocked-source-unavailable")
        if not isinstance(private_source, dict) or set(private_source) != {
            "source_app_type",
            "source_snapshot_sha256",
            "precondition",
            "providers",
        }:
            raise RunnerFailure("blocked-source-unavailable")
        snapshot = source["source_snapshot_sha256"]
        if (
            source["source_app_type"] != "claude"
            or private_source["source_app_type"] != "claude"
            or not _is_digest(snapshot)
            or private_source["source_snapshot_sha256"] != snapshot
        ):
            raise RunnerFailure("blocked-source-unavailable")
        precondition = source["precondition"]
        private_precondition = private_source["precondition"]
        if (
            not isinstance(precondition, dict)
            or private_precondition != precondition
        ):
            raise RunnerFailure("blocked-source-changed")
        providers = source["providers"]
        private_providers = private_source["providers"]
        if (
            not isinstance(providers, list)
            or len(providers) != 1
            or not isinstance(private_providers, list)
            or len(private_providers) != 1
        ):
            raise RunnerFailure("blocked-source-unavailable")
        provider = providers[0]
        private_provider = private_providers[0]
        expected_keys = {
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
        }
        private_keys = {
            "source_id",
            "api_family",
            "endpoint",
            "endpoint_sha256",
            "apply_binding_sha256",
            "credential_value",
        }
        if (
            not isinstance(provider, dict)
            or set(provider) != expected_keys
            or not isinstance(private_provider, dict)
            or set(private_provider) != private_keys
        ):
            raise RunnerFailure("blocked-source-unavailable")
        endpoint = private_provider["endpoint"]
        credential = private_provider["credential_value"]
        if (
            provider["source_id"] != SOURCE_ID
            or provider["app_type"] != "claude"
            or provider["is_current"] is not True
            or provider["api_family"] != API_FAMILY
            or provider["endpoint_https"] is not True
            or not _is_digest(provider["endpoint_sha256"])
            or not _is_digest(provider["apply_binding_sha256"])
            or provider["credential_reference_present"] is not True
            or provider["blocker_codes"] != []
            or not isinstance(provider["declared_models"], list)
            or not _exact_environment_models(provider)
            or private_provider["source_id"] != SOURCE_ID
            or private_provider["api_family"] != API_FAMILY
            or not isinstance(endpoint, str)
            or not endpoint
            or not isinstance(credential, str)
            or not credential
            or "\x00" in credential
            or private_provider["endpoint_sha256"] != provider["endpoint_sha256"]
            or private_provider["apply_binding_sha256"]
            != provider["apply_binding_sha256"]
            or hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            != provider["endpoint_sha256"]
        ):
            raise RunnerFailure("blocked-source-unavailable")
        return _Binding(
            snapshot=snapshot,
            precondition=json.loads(json.dumps(precondition, sort_keys=True)),
            endpoint_sha256=provider["endpoint_sha256"],
            apply_binding_sha256=provider["apply_binding_sha256"],
        )
    finally:
        if isinstance(private_source, dict):
            private_providers = private_source.get("providers")
            if isinstance(private_providers, list):
                for private_provider in private_providers:
                    if isinstance(private_provider, dict):
                        private_provider.clear()
                private_providers.clear()
            private_source.clear()
        if isinstance(source, dict):
            source.clear()


def _same_binding(left: _Binding, right: _Binding) -> bool:
    return (
        left.snapshot == right.snapshot
        and left.precondition == right.precondition
        and left.endpoint_sha256 == right.endpoint_sha256
        and left.apply_binding_sha256 == right.apply_binding_sha256
    )


def _assert_child_source_path(
    source_db: str | os.PathLike[str],
    child: Callable[[str, str], tuple[dict[str, Any], int | None]],
) -> None:
    """Do not pair an arbitrary DB with the installation-bound direct probe."""
    if (
        getattr(child, "__module__", None) != __name__
        or getattr(child, "__name__", None) != "_invoke_child"
    ):
        return
    try:
        supplied = os.path.abspath(os.fspath(source_db))
        expected = os.path.abspath(os.fspath(SOURCE_DB))
    except (OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-source-path") from None
    if supplied != expected:
        raise RunnerFailure("blocked-source-path")


def _direct_probe_command(mode: str, phase: str) -> list[str]:
    if mode not in MODES or phase not in {"preflight", "live"}:
        raise RunnerFailure("blocked-invalid-arguments")
    command = [
        str(PROBE_PYTHON),
        "-I",
        "-B",
        str(DIRECT_PROBE),
        "--model",
        MODEL,
        "--mode",
        mode,
        "--timeout",
        str(PER_CALL_TIMEOUT_SECONDS),
        "--max-tokens",
        str(MAX_OUTPUT_TOKENS),
    ]
    command.append("--preflight-only" if phase == "preflight" else "--approve-current-origin")
    return command


def _decode_child_result(
    completed: Any, mode: str, phase: str
) -> tuple[dict[str, Any], int | None]:
    if not isinstance(completed, subprocess.CompletedProcess):
        # Mock runners may return a small compatible object.  It must still
        # provide bytes-only stdout/stderr and an integer return code.
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        returncode = getattr(completed, "returncode", None)
    else:
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
    if (
        not isinstance(stdout, bytes)
        or not isinstance(stderr, bytes)
        or stderr
        or len(stdout) > MAX_CHILD_OUTPUT_BYTES
        or not isinstance(returncode, int)
        or isinstance(returncode, bool)
    ):
        raise RunnerFailure("unusable-worker-output")
    try:
        parsed = json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
        raise RunnerFailure("unusable-worker-output") from None
    if not _valid_result(parsed, mode, phase):
        raise RunnerFailure("unusable-worker-output")
    # Copy only the known fields; no raw child object crosses the boundary.
    return {key: parsed[key] for key in _RESULT_KEYS}, returncode

def _kill_process_group(process: Any) -> None:
    """Kill a new session and boundedly reap its leader and forked workers."""
    pid = getattr(process, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except OSError:
        try:
            process.kill()
        except (OSError, ProcessLookupError, AttributeError):
            pass


def _bounded_reap(process: Any) -> None:
    try:
        process.communicate(timeout=REAP_TIMEOUT_SECONDS)
        return
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
    except (OSError, ValueError, AttributeError):
        return
    try:
        process.wait(timeout=REAP_TIMEOUT_SECONDS)
    except (OSError, ValueError, AttributeError, subprocess.TimeoutExpired):
        pass


def _invoke_child(mode: str, phase: str) -> tuple[dict[str, Any], int | None]:
    command = _direct_probe_command(mode, phase)
    process: Any = None
    try:
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            close_fds=True,
            start_new_session=True,
        )
    except (OSError, UnicodeError, ValueError):
        raise RunnerFailure("unusable-worker-launch") from None
    try:
        stdout, stderr = process.communicate(timeout=PROCESS_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        _kill_process_group(process)
        _bounded_reap(process)
        raise RunnerFailure("temporary-timeout") from None
    except (OSError, UnicodeError, ValueError):
        _kill_process_group(process)
        _bounded_reap(process)
        raise RunnerFailure("unusable-worker-output") from None
    finally:
        if process is not None:
            try:
                still_running = process.poll() is None
            except (OSError, ValueError, AttributeError):
                still_running = True
            if still_running:
                _kill_process_group(process)
                _bounded_reap(process)
    completed = subprocess.CompletedProcess(
        command,
        process.returncode,
        stdout=stdout,
        stderr=stderr,
    )
    return _decode_child_result(completed, mode, phase)


def _canonical_artifact_body(
    snapshot: str,
    binding: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not _is_digest(snapshot) or not _is_digest(binding):
        raise RunnerFailure("blocked-artifact")
    if len(observations) != MAX_API_CALLS:
        raise RunnerFailure("blocked-artifact")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for observation in observations:
        if not isinstance(observation, Mapping) or set(observation) != {
            "sourceId",
            "environmentName",
            "configuredModelValue",
            "model",
            "observedModel",
            "mode",
            "identityStatus",
            "usable",
        }:
            raise RunnerFailure("blocked-artifact")
        mode = observation["mode"]
        if mode not in MODES or mode in seen:
            raise RunnerFailure("blocked-artifact")
        seen.add(mode)
        if (
            observation["sourceId"] != SOURCE_ID
            or observation["environmentName"] != MODEL
            or observation["configuredModelValue"] != MODEL
            or observation["model"] != MODEL
            or observation["observedModel"] != MODEL
            or observation["identityStatus"] != "observed-match"
            or observation["usable"] is not True
        ):
            raise RunnerFailure("blocked-artifact")
        normalized.append(
            {
                "sourceId": SOURCE_ID,
                "environmentName": MODEL,
                "configuredModelValue": MODEL,
                "model": MODEL,
                "observedModel": MODEL,
                "mode": mode,
                "identityStatus": "observed-match",
                "usable": True,
            }
        )
    if set(seen) != set(MODES):
        raise RunnerFailure("blocked-artifact")
    normalized.sort(key=lambda item: item["mode"])
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sourceSnapshotSha256": snapshot,
        "providerEvidence": [
            {"sourceId": SOURCE_ID, "applyBindingSha256": binding}
        ],
        "observations": normalized,
    }


def build_probe_artifact(
    *,
    source_snapshot_sha256: str,
    apply_binding_sha256: str,
    observations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and self-hash a closed canonical v2 evidence document."""
    body = _canonical_artifact_body(
        source_snapshot_sha256, apply_binding_sha256, observations
    )
    digest = hashlib.sha256(
        _migration._canonical(body, "blocked-artifact")
    ).hexdigest()
    return {**body, "artifactSha256": digest}


def _artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    if not isinstance(artifact, Mapping) or set(artifact) != {
        "schemaVersion",
        "sourceSnapshotSha256",
        "providerEvidence",
        "observations",
        "artifactSha256",
    }:
        raise RunnerFailure("blocked-artifact")
    document = dict(artifact)
    digest = document["artifactSha256"]
    body = {key: value for key, value in document.items() if key != "artifactSha256"}
    if (
        not _is_digest(digest)
        or hashlib.sha256(_migration._canonical(body, "blocked-artifact")).hexdigest()
        != digest
    ):
        raise RunnerFailure("blocked-artifact")
    encoded = _migration._canonical(document, "blocked-artifact") + b"\n"
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise RunnerFailure("blocked-artifact")
    return encoded


def read_probe_artifact(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read back a written v2 probe artifact under this module's own contract.

    ``omp_provider_migration._validate_probe`` validates a different document:
    its migration probe carries ``_XUANJU_PROBE_MODELS`` x ``_XUANJU_PROBE_MODES``
    (five models, ten observations) with migration observation fields. This
    runner's artifact is a single-model, two-observation record, so a completed
    artifact is not — and cannot be — readable by that validator. The writer
    used to call it anyway, which made every write fail with blocked-artifact.
    """
    try:
        raw = Path(path).read_bytes()
        parsed = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError):
        raise RunnerFailure("blocked-artifact") from None
    if not isinstance(parsed, dict):
        raise RunnerFailure("blocked-artifact")
    # Re-validating through the writer's own contract proves the round trip.
    if _artifact_bytes(parsed) != raw:
        raise RunnerFailure("blocked-artifact")
    return parsed


def _secure_parent(path: Path) -> None:
    try:
        details = os.lstat(path)
    except OSError:
        raise RunnerFailure("blocked-output-directory") from None
    if (
        path.is_symlink()
        or not stat.S_ISDIR(details.st_mode)
        or details.st_uid != os.getuid()
        or stat.S_IMODE(details.st_mode) != 0o700
    ):
        raise RunnerFailure("blocked-output-directory")


def _reserve_output(path: Path) -> _Reservation:
    """Exclusively reserve output before the first authorized child starts."""
    if not path.is_absolute() or any(
        part in {"", ".", ".."} for part in path.parts[1:]
    ):
        raise RunnerFailure("blocked-output-path")
    _secure_parent(path.parent)
    descriptor = -1
    reserved = False
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW")
            | getattr(os, "O_CLOEXEC"),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        details = os.fstat(descriptor)
        if (
            not stat.S_ISREG(details.st_mode)
            or details.st_uid != os.getuid()
            or stat.S_IMODE(details.st_mode) != 0o600
            or details.st_nlink != 1
        ):
            raise RunnerFailure("blocked-output-write")
        marker = b'{"status":"in-progress"}\n'
        view = memoryview(marker)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise RunnerFailure("blocked-output-write")
            view = view[written:]
        os.fsync(descriptor)
        reserved = True
        return _Reservation(descriptor, details.st_dev, details.st_ino)
    except FileExistsError:
        raise RunnerFailure("blocked-output-exists") from None
    except (AttributeError, OSError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        if descriptor != -1 and not reserved:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _close_reservation(reservation: _Reservation | None) -> None:
    if reservation is None or reservation.descriptor == -1:
        return
    try:
        os.close(reservation.descriptor)
    except OSError:
        pass
    reservation.descriptor = -1


def _replace_output(path: Path, reservation: _Reservation, payload: bytes) -> None:
    """Commit through the reserved descriptor and reject path replacement."""
    try:
        _secure_parent(path.parent)
        opened = os.fstat(reservation.descriptor)
        current = os.lstat(path)
        if (
            (opened.st_dev, opened.st_ino) != (reservation.device, reservation.inode)
            or (current.st_dev, current.st_ino) != (reservation.device, reservation.inode)
            or stat.S_IMODE(opened.st_mode) != 0o600
            or stat.S_IMODE(current.st_mode) != 0o600
            or opened.st_nlink != 1
            or current.st_nlink != 1
        ):
            raise RunnerFailure("blocked-output-write")
        os.ftruncate(reservation.descriptor, 0)
        os.lseek(reservation.descriptor, 0, os.SEEK_SET)
        view = memoryview(payload)
        while view:
            written = os.write(reservation.descriptor, view)
            if written <= 0:
                raise RunnerFailure("blocked-output-write")
            view = view[written:]
        os.fsync(reservation.descriptor)
        final = os.fstat(reservation.descriptor)
        visible = os.lstat(path)
        if (
            (final.st_dev, final.st_ino) != (reservation.device, reservation.inode)
            or (visible.st_dev, visible.st_ino) != (reservation.device, reservation.inode)
            or stat.S_IMODE(final.st_mode) != 0o600
            or stat.S_IMODE(visible.st_mode) != 0o600
            or final.st_nlink != 1
            or visible.st_nlink != 1
        ):
            raise RunnerFailure("blocked-output-write")
    except RunnerFailure:
        raise
    except (OSError, ValueError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        _close_reservation(reservation)


def _safe_summary(
    *, status: str, phase: str, outcomes: Sequence[_ChildOutcome] = ()
) -> dict[str, Any]:
    return {
        "status": status,
        "phase": phase,
        "sourceId": SOURCE_ID,
        "model": MODEL,
        "modes": list(MODES),
        "maximumApiCalls": MAX_API_CALLS,
        "attemptedCalls": len(outcomes),
        "callModes": [outcome.mode for outcome in outcomes],
        "statusCounts": dict(
            sorted(Counter(outcome.status for outcome in outcomes).items())
        ),
        "retryCount": 0,
        "fallbackCount": 0,
        "serial": True,
    }


def _run_phase(
    source_db: str | os.PathLike[str],
    *,
    phase: str,
    child: Callable[[str, str], tuple[dict[str, Any], int | None]] | None = None,
) -> tuple[dict[str, Any], list[_ChildOutcome], _Binding]:
    if phase not in {"preflight", "live"}:
        raise RunnerFailure("blocked-invalid-arguments")
    child_runner = _invoke_child if child is None else child
    _assert_child_source_path(source_db, child_runner)
    binding = _read_binding(source_db)
    outcomes: list[_ChildOutcome] = []
    for mode in MODES:
        # A credential rotation or endpoint/source edit between calls is a hard
        # stop.  There is never a retry, alternate route, or fallback model.
        current = _read_binding(source_db)
        if not _same_binding(binding, current):
            raise RunnerFailure("blocked-source-changed")
        try:
            result, returncode = child_runner(mode, phase)
        except RunnerFailure:
            raise
        except (OSError, TypeError, ValueError):
            raise RunnerFailure("unusable-worker-launch") from None
        except Exception:
            raise RunnerFailure("unusable-worker-launch") from None
        status = str(result.get("status"))
        outcomes.append(_ChildOutcome(mode, status, result, returncode))
        if phase == "live" and not _successful_live_result(result, mode):
            if result.get("identity_status") == "identity-mismatch":
                raise RunnerFailure("identity-mismatch")
            raise RunnerFailure("unusable")
        if phase == "preflight" and result.get("status") != "preflight-ok":
            raise RunnerFailure("blocked-preflight")
        after = _read_binding(source_db)
        if not _same_binding(binding, after):
            raise RunnerFailure("blocked-source-changed")
    return _safe_summary(status="preflight-ok" if phase == "preflight" else "complete", phase=phase, outcomes=outcomes), outcomes, binding


def run_preflight(
    source_db: str | os.PathLike[str] = SOURCE_DB,
    *,
    child: Callable[[str, str], tuple[dict[str, Any], int | None]] | None = None,
) -> dict[str, Any]:
    """Validate the two exact routes without reserving output or doing I/O."""
    summary, _outcomes, _binding = _run_phase(
        source_db, phase="preflight", child=child
    )
    return summary


def run_authorized_probes(
    source_db: str | os.PathLike[str] = SOURCE_DB,
    output: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    child: Callable[[str, str], tuple[dict[str, Any], int | None]] | None = None,
) -> dict[str, Any]:
    """Reserve once, run text then tool, and publish one canonical v2 artifact."""
    output_path = Path(output)
    # Reserve before any child starts.  The reservation itself is the no-retry
    # marker if a child times out or the process is interrupted.
    reservation: _Reservation | None = None
    outcomes: list[_ChildOutcome] = []
    try:
        initial = _read_binding(source_db)
        reservation = _reserve_output(output_path)
        summary, outcomes, binding = _run_phase(source_db, phase="live", child=child)
        if not _same_binding(initial, binding):
            raise RunnerFailure("blocked-source-changed")
        observations = [
            {
                "sourceId": SOURCE_ID,
                "environmentName": MODEL,
                "configuredModelValue": MODEL,
                "model": MODEL,
                "observedModel": MODEL,
                "mode": outcome.mode,
                "identityStatus": "observed-match",
                "usable": True,
            }
            for outcome in outcomes
        ]
        artifact = build_probe_artifact(
            source_snapshot_sha256=binding.snapshot,
            apply_binding_sha256=binding.apply_binding_sha256,
            observations=observations,
        )
        _replace_output(output_path, reservation, _artifact_bytes(artifact))
        return {
            **summary,
            "artifactSha256": artifact["artifactSha256"],
            "outputMode": "0600",
        }
    except KeyboardInterrupt:
        raise RunnerFailure("interrupted") from None
    finally:
        _close_reservation(reservation)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source-db", default=str(SOURCE_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument(
        "--preflight-only", action="store_false", dest="execute", help=argparse.SUPPRESS
    )
    actions.add_argument(
        "--execute-authorized-probes",
        action="store_true",
        dest="execute",
        help=argparse.SUPPRESS,
    )
    parser.set_defaults(execute=False)
    return parser


def _emit(value: Mapping[str, Any]) -> None:
    try:
        sys.stdout.write(
            json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
            + "\n"
        )
        sys.stdout.flush()
    except (OSError, TypeError, ValueError):
        pass


def main(argv: list[str] | None = None) -> int:
    argv_values = list(sys.argv[1:] if argv is None else argv)
    phase = "live" if "--execute-authorized-probes" in argv_values else "preflight"
    try:
        args = _parser().parse_args(argv_values)
        if args.execute:
            summary = run_authorized_probes(args.source_db, args.output)
        else:
            summary = run_preflight(args.source_db)
        _emit(summary)
        return 0
    except RunnerFailure as failure:
        _emit(_safe_summary(status=failure.status, phase=phase))
        return 4
    except KeyboardInterrupt:
        _emit(_safe_summary(status="interrupted", phase=phase))
        return 3
    except (OSError, TypeError, ValueError, UnicodeError):
        _emit(_safe_summary(status="blocked-runner", phase=phase))
        return 4
    except Exception:
        _emit(_safe_summary(status="blocked-runner", phase=phase))
        return 4


if __name__ == "__main__":  # pragma: no cover - exercised by the CLI process.
    raise SystemExit(main())
