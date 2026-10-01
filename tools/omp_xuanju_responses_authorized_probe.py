"""Fail-closed, no-fallback Xuanju OpenAI Responses observation runner.

This runner is deliberately independent from the legacy GLM-5.2 Anthropic
runner.  It executes the fixed five-model, text-then-tool matrix (ten calls)
only after binding the current CC Switch source and its apply-only credential
snapshot.  The parent owns ordering, source drift checks, output reservation,
and process-group timeouts.  The watched child is this same module in a
capability-gated child mode; it reads the credential only inside that child and
emits a closed result envelope.

Preflight never opens a network connection.  The live transport is a tiny
standard-library HTTPS client so the exact ``/v1/responses`` request shape is
visible and testable without relying on an ambient SDK route.  No endpoint,
credential, header, raw body, or provider exception is ever serialized.
"""

from __future__ import annotations

import argparse
import select
from collections import Counter
from dataclasses import dataclass
import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import signal
import ssl
import stat
import tempfile
import subprocess
import sys
import time
from typing import Any, Callable, Final, Mapping, NoReturn, Sequence
from urllib.parse import urlsplit


# ``python -I -B path/to/this.py`` excludes the checkout from import search.
# Restore only this file's project root; never cwd/PYTHONPATH.
if __package__ in {None, ""}:  # pragma: no cover - exercised by child process.
    _PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)

from tools.cc_switch_omp_source import (  # noqa: E402
    SourceInventoryError,
    _migration_apply_source,
    materialization_source,
    migration_planning_source,
)

__all__ = [
    "API_FAMILY",
    "BUSINESS_URL",
    "MODELS",
    "MODES",
    "PROBES",
    "RunnerFailure",
    "build_observation_artifact",
    "build_responses_request",
    "main",
    "run_authorized_probes",
    "run_preflight",
]


# Installation-bound defaults.  Tests replace the source seam and child
# callback; production callers cannot supply an endpoint or credential.
SOURCE_DB: Final = Path("/Users/chiyuchou/.cc-switch/cc-switch.db")
PROBE_PYTHON: Final = Path("/Users/chiyuchou/.hermes/hermes-agent/venv/bin/python")
DEFAULT_OUTPUT: Final = Path(
    "/Users/chiyuchou/.omp/agent/routing-health/"
    "xuanju-openai-responses-5models.json"
)
SOURCE_ID: Final = "616ccdfd-8de6-4c82-a4cc-24f71edf0c49"
PROVIDER: Final = "cc-xuanju"
# The source family is bound as observed, not migrated or relabelled.  Current
# installations may expose either the historical Anthropic family or the
# OpenAI Responses family; both are accepted only for this fixed source.  The
# request contract below is independently and explicitly Responses-only.
SOURCE_API_FAMILY: Final = "anthropic"
SOURCE_API_FAMILIES: Final = frozenset({"anthropic", "openai_responses"})
API_FAMILY: Final = "openai_responses"
REQUEST_API_FAMILY: Final = "openai-responses"
BUSINESS_URL: Final = "https://hs32.com/v1/responses"
MODELS: Final = (
    "GLM-5.3-flash",
    "Kimi-K3",
    "deepseek-v4-pro-0813",
    "deepseek-v4.1-flash",
    "GLM-5.3",
)
MODES: Final = ("text", "tool")
PROBES: Final = tuple((model, mode) for model in MODELS for mode in MODES)
SCHEMA_VERSION: Final = 2
ARTIFACT_KIND: Final = "xuanju-openai-responses-observation"
MAX_API_CALLS: Final = len(PROBES)
PER_CALL_TIMEOUT_SECONDS: Final = 180
PROCESS_TIMEOUT_SECONDS: Final = 180
REAP_TIMEOUT_SECONDS: Final = 1.0
MAX_OUTPUT_TOKENS: Final = 64
MAX_CHILD_OUTPUT_BYTES: Final = 16_384
MAX_RESPONSE_BYTES: Final = 1_048_576
MAX_ARTIFACT_BYTES: Final = 128 * 1024
CAPABILITY_BYTES: Final = 32
MAX_BINDING_CAPTURE_ATTEMPTS: Final = 3

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
_SAFE_STATUSES: Final = frozenset(
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
        "preflight-ok",
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
        "usable",
    }
)

_TERMINAL_LIVE_FAILURES: Final = frozenset(
    {
        "blocked-artifact",
        "blocked-child-capability",
        "blocked-output-directory",
        "blocked-output-exists",
        "blocked-output-path",
        "blocked-output-write",
        "blocked-origin-approval-required",
        "blocked-source-changed",
        "blocked-source-path",
        "blocked-source-unavailable",
        "blocked-transport-policy",
        "blocked-trusted-installation",
        "blocked-worker-reap",
        "unusable-worker-launch",
        "unusable-worker-output",
        "unusable-worker-crash",
    }
)

_PARTIAL_LIVE_FAILURES: Final = frozenset(
    {
        "identity-mismatch",
        "identity-unverified",
        "temporary",
        "temporary-timeout",
        "unusable",
        "unusable-empty-text",
        "unusable-text-result",
        "unusable-tool-result",
        "unusable-worker-launch",
        "unusable-worker-output",
    }
)
_IDENTITY_STATUSES: Final = frozenset(
    {"not-observed", "identity-unverified", "identity-mismatch", "observed-match"}
)
_DIGEST_HEX: Final = frozenset("0123456789abcdef")
_SAFE_IDENTIFIER_CHARS: Final = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:/-[]"
)


class RunnerFailure(ValueError):
    """A safe status-only failure; private exception text is never rendered."""

    def __init__(
        self,
        status: str,
        *,
        outcomes: Sequence[Any] = (),
        phase: str | None = None,
    ) -> None:
        if not isinstance(status, str) or not status or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789-_"
            for character in status
        ):
            status = "blocked-runner"
        self.status = status
        self.outcomes = tuple(outcomes)
        self.phase = phase
        super().__init__(status)


def _require_trusted_runtime() -> None:
    """Reject live execution outside the isolated, no-bytecode launcher."""
    flags = sys.flags
    if not bool(
        getattr(flags, "isolated", False)
        and getattr(flags, "dont_write_bytecode", False)
    ):
        raise RunnerFailure("blocked-trusted-installation")


def _clear_ssl_keylog_env() -> None:
    """Never allow ambient TLS key logging to coexist with bearer traffic."""
    if "SSLKEYLOGFILE" in os.environ:
        os.environ.pop("SSLKEYLOGFILE", None)
        raise RunnerFailure("blocked-trusted-installation")


class _JsonArgumentParser(argparse.ArgumentParser):
    def error(self, _message: str) -> NoReturn:
        raise RunnerFailure("blocked-invalid-arguments")


def _semantic_digest(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class _Binding:
    """Redacted target semantics and the safe identity of one source capture.

    ``snapshot`` and ``precondition`` are retained as the source seam's
    per-capture evidence.  They are deliberately *not* used for continuity:
    the source seam has already proved that each individual planning/apply
    read was stable, while the probe only carries the target provider's
    semantic digest between calls.
    """

    snapshot: str
    precondition: dict[str, Any]
    endpoint_sha256: str
    apply_binding_sha256: str
    source_api_family: str = SOURCE_API_FAMILY
    semantic_binding_sha256: str | None = None

    def __post_init__(self) -> None:
        # Keep the small fixture-facing constructor useful while ensuring
        # every binding has one explicit target semantic digest.  Production
        # reads pass the digest built from the complete target projection
        # below; this fallback intentionally excludes all physical evidence.
        if self.semantic_binding_sha256 is None:
            payload = {
                "sourceId": SOURCE_ID,
                "appType": "claude",
                "isCurrent": True,
                "apiFamily": self.source_api_family,
                "endpointOriginSha256": self.endpoint_sha256,
                "applyBindingSha256": self.apply_binding_sha256,
                "declaredModels": [],
                "environmentModels": {
                    "configured_model_values": [],
                    "name_to_configured_value": {},
                },
                "modelDefinitions": [],
            }
            object.__setattr__(
                self,
                "semantic_binding_sha256",
                _semantic_digest(payload),
            )

@dataclass(frozen=True)
class _ChildOutcome:
    model: str
    mode: str
    status: str
    result: dict[str, Any]
    process_exit_code: int | None


@dataclass
class _Reservation:
    descriptor: int
    device: int
    inode: int
    current_device: int | None = None
    current_inode: int | None = None

    def __post_init__(self) -> None:
        if self.current_device is None:
            self.current_device = self.device
        if self.current_inode is None:
            self.current_inode = self.inode


@dataclass
class _PrivateOutput:
    descriptor: int
    path: Path
    device: int
    inode: int
    published: bool = False

Transport = Callable[
    [str, Mapping[str, str], bytes, float], tuple[int, bytes]
]


def _canonical(value: Any) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError):
        raise RunnerFailure("blocked-artifact") from None


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in _DIGEST_HEX for character in value)
    )


def _safe_identifier(value: Any) -> bool:
    return (
        isinstance(value, str)
        and 0 < len(value) <= 128
        and all(character in _SAFE_IDENTIFIER_CHARS for character in value)
    )

_MODEL_DEFINITION_KEYS: Final = frozenset(
    {
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
    }
)


def _safe_semantic_json(value: Any, *, depth: int = 0) -> Any:
    """Copy bounded JSON semantics without retaining source-owned objects."""
    if depth > 16:
        raise RunnerFailure("blocked-source-unavailable")
    if value is None or isinstance(value, (bool, int, str)):
        if isinstance(value, str) and (
            not value or len(value) > 512 or "\x00" in value
        ):
            raise RunnerFailure("blocked-source-unavailable")
        return value
    if isinstance(value, float):
        # ``allow_nan=False`` in ``_semantic_digest`` rejects non-finite values.
        return value
    if isinstance(value, list):
        if len(value) > 256:
            raise RunnerFailure("blocked-source-unavailable")
        return [
            _safe_semantic_json(item, depth=depth + 1)
            for item in value
        ]
    if isinstance(value, dict):
        if len(value) > 64:
            raise RunnerFailure("blocked-source-unavailable")
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key or len(key) > 128:
                raise RunnerFailure("blocked-source-unavailable")
            normalized[key] = _safe_semantic_json(item, depth=depth + 1)
        return normalized
    raise RunnerFailure("blocked-source-unavailable")


def _normalised_environment_models(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "configured_model_values",
        "name_to_configured_value",
    }:
        raise RunnerFailure("blocked-source-unavailable")
    configured = value["configured_model_values"]
    names = value["name_to_configured_value"]
    if (
        not isinstance(configured, list)
        or not isinstance(names, dict)
        or any(not _safe_identifier(item) for item in configured)
        or len(configured) != len(set(configured))
        or any(
            not _safe_identifier(key) or not _safe_identifier(item)
            for key, item in names.items()
        )
    ):
        raise RunnerFailure("blocked-source-unavailable")
    return {
        "configured_model_values": sorted(configured),
        "name_to_configured_value": {
            key: names[key] for key in sorted(names)
        },
    }


def _normalised_model_definitions(
    provider: Mapping[str, Any], declared_models: Sequence[str]
) -> list[dict[str, Any]]:
    if "model_definitions" not in provider:
        # Older source-seam payloads expose only declarations.  The
        # declaration itself is still a stable model-definition summary.
        return [{"id": model} for model in declared_models]
    raw = provider["model_definitions"]
    if not isinstance(raw, list) or len(raw) != len(declared_models):
        raise RunnerFailure("blocked-source-unavailable")
    normalized: list[dict[str, Any]] = []
    for expected_model, definition in zip(declared_models, raw):
        if (
            not isinstance(definition, dict)
            or set(definition) - _MODEL_DEFINITION_KEYS
            or definition.get("id") != expected_model
        ):
            raise RunnerFailure("blocked-source-unavailable")
        copied = _safe_semantic_json(definition)
        if not isinstance(copied, dict):
            raise RunnerFailure("blocked-source-unavailable")
        normalized.append(copied)
    return normalized


def _normalised_endpoint(endpoint: str) -> str:
    parsed = urlsplit(endpoint)
    try:
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        raise RunnerFailure("blocked-source-unavailable") from None
    if not isinstance(hostname, str) or not hostname:
        raise RunnerFailure("blocked-source-unavailable")
    host = hostname.casefold()
    normalized_port = "" if port in (None, 443) else f":{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return f"{parsed.scheme.casefold()}://{host}{normalized_port}{path}"


def _target_semantic_digest(
    provider: Mapping[str, Any],
    private_provider: Mapping[str, Any],
    endpoint: str,
) -> str:
    declared_models = provider["declared_models"]
    if not isinstance(declared_models, list) or not all(
        _safe_identifier(model) for model in declared_models
    ):
        raise RunnerFailure("blocked-source-unavailable")
    environment_models = _normalised_environment_models(
        provider["environment_models"]
    )
    blocker_codes = provider["blocker_codes"]
    if (
        not isinstance(blocker_codes, list)
        or len(blocker_codes) != len(set(blocker_codes))
        or any(not _safe_identifier(code) for code in blocker_codes)
    ):
        raise RunnerFailure("blocked-source-unavailable")
    model_definitions = _normalised_model_definitions(provider, declared_models)
    endpoint_origin_sha256 = hashlib.sha256(
        _normalised_endpoint(endpoint).encode("utf-8")
    ).hexdigest()
    try:
        return _semantic_digest(
            {
                "sourceId": provider["source_id"],
                "appType": provider["app_type"],
                "isCurrent": provider["is_current"],
                "apiFamily": provider["api_family"],
                "endpointOriginSha256": endpoint_origin_sha256,
                "applyBindingSha256": private_provider["apply_binding_sha256"],
                "credentialReferencePresent": provider["credential_reference_present"],
                "declaredModels": list(declared_models),
                "environmentModels": environment_models,
                "modelDefinitions": model_definitions,
                "blockerCodes": list(blocker_codes),
            }
        )
    except (TypeError, ValueError, UnicodeError):
        raise RunnerFailure("blocked-source-unavailable") from None


def _safe_file_identity(precondition: Any) -> tuple[Any, ...] | None:
    """Return only replacement/security identity, excluding physical churn."""
    if not isinstance(precondition, dict):
        return None
    realpath = precondition.get("realpath")
    if not isinstance(realpath, str) or not realpath or "\x00" in realpath:
        return None
    fields = ("device", "fileId", "owner", "mode")
    present = [field in precondition for field in fields]
    if not any(present):
        # Small test seams from before the identity fields existed are still
        # safe to compare by their already-resolved path.
        return ("main", realpath)
    if not all(present) or any(
        type(precondition[field]) is not int or precondition[field] < 0
        for field in fields
    ):
        return None
    return ("main", realpath, *(precondition[field] for field in fields))


def _baseline(model: str, mode: str) -> dict[str, Any]:
    return {
        "configured_selector": f"{PROVIDER}/{model}",
        "request_model_id": model,
        "response_model_id": None,
        "identity_status": "not-observed",
        "verification_kind": "not-run",
        "status": "blocked",
        "tool_result": "not-run" if mode == "tool" else None,
        "token_usage": None,
        "latency_ms": 0,
    }

def _failure_result(model: str, mode: str, status: str) -> dict[str, Any]:
    """Build a closed, status-only row for a failed live invocation."""
    safe_status = status if status in _SAFE_STATUSES else "unusable-worker-launch"
    result = _baseline(model, mode)
    result.update(
        verification_kind=f"live-{mode}",
        status=safe_status,
        latency_ms=(
            PER_CALL_TIMEOUT_SECONDS * 1000
            if safe_status == "temporary-timeout"
            else 0
        ),
    )
    return result


def _terminal_live_failure(status: str) -> bool:
    return (
        status in _TERMINAL_LIVE_FAILURES
        or status == "blocked"
        or status.startswith("blocked-")
    )


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


def _valid_result(value: Any, model: str, mode: str, phase: str) -> bool:
    """Validate a closed child envelope without retaining untrusted fields."""
    if not isinstance(value, dict) or set(value) != _RESULT_KEYS:
        return False
    if (
        value["configured_selector"] != f"{PROVIDER}/{model}"
        or value["request_model_id"] != model
        or value["status"] not in _SAFE_STATUSES
        or value["identity_status"] not in _IDENTITY_STATUSES
        or not _valid_token_usage(value["token_usage"])
        or not isinstance(value["latency_ms"], int)
        or isinstance(value["latency_ms"], bool)
        or not 0 <= value["latency_ms"] <= PROCESS_TIMEOUT_SECONDS * 1000
    ):
        return False
    response_model = value["response_model_id"]
    # A provider's model field is tainted.  The only value that may cross the
    # child boundary is the exact requested model; mismatches are represented
    # by a status with a redacted ``None`` value.
    if response_model is not None and response_model != model:
        return False
    if phase == "preflight":
        return (
            value["verification_kind"] == "preflight-only"
            and value["status"] in {"preflight-ok", "unusable-worker-crash"}
            and response_model is None
            and value["identity_status"] == "not-observed"
            and value["token_usage"] is None
            and value["tool_result"] == ("not-run" if mode == "tool" else None)
        )
    if phase != "live" or value["verification_kind"] != f"live-{mode}":
        return False
    if value["identity_status"] == "observed-match":
        if response_model != model:
            return False
    elif value["identity_status"] in {"identity-mismatch", "identity-unverified"}:
        if response_model is not None:
            return False
    elif value["identity_status"] == "not-observed":
        if response_model is not None:
            return False
    else:
        return False
    if mode == "tool":
        return value["tool_result"] in {"not-run", "rejected", "smoke_ok"}
    return value["tool_result"] is None


def _successful_live_result(value: Mapping[str, Any], model: str, mode: str) -> bool:
    return (
        value.get("verification_kind") == f"live-{mode}"
        and value.get("status") == "usable"
        and value.get("identity_status") == "observed-match"
        and value.get("response_model_id") == model
        and value.get("tool_result") == ("smoke_ok" if mode == "tool" else None)
    )


def _valid_declared_models(provider: Mapping[str, Any]) -> bool:
    """Validate only the source seam's closed model-list shape.

    The selected five values come from the separately verified Responses
    ``/v1/models`` surface; CC environment aliases (including ``[1M]``) are
    deliberately not used as an allowlist here.
    """
    models = provider.get("declared_models")
    if not isinstance(models, list) or not models:
        return False
    if not all(isinstance(model, str) and model for model in models):
        return False
    return len(models) == len(set(models))


def _valid_source_endpoint(endpoint: Any) -> bool:
    """Bind the current source to the documented hs32.com origin only."""
    if not isinstance(endpoint, str) or not endpoint or endpoint != endpoint.strip():
        return False
    try:
        parsed = urlsplit(endpoint)
    except (TypeError, ValueError):
        return False
    return (
        parsed.scheme.casefold() == "https"
        and isinstance(parsed.hostname, str)
        and parsed.hostname.casefold() == "hs32.com"
        and parsed.username is None
        and parsed.password is None
        and parsed.query == ""
        and parsed.fragment == ""
        and (parsed.port is None or parsed.port == 443)
    )


def _binding_public_keys() -> tuple[set[str], set[str]]:
    base = {
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
    return base, base | {"model_definitions"}


def _binding_from_sources(
    source_db: str | os.PathLike[str], *, retain_credential: bool
) -> tuple[_Binding, str | None]:
    """Read one semantically stable source group.

    Each source seam call owns its own immutable, no-follow stability proof.
    The probe therefore must not compare the complete precondition dictionaries
    across calls: SQLite WAL/SHM telemetry can legitimately churn between
    otherwise equivalent captures.  Apply is read both before and after the
    public planning/materialization views, and its private binding summary
    brackets those views.  Only semantic disagreements (or a replacement of
    the source identity) are retried.
    """

    expected_source_keys = {
        "source_app_type",
        "source_snapshot_sha256",
        "precondition",
        "providers",
    }
    public_keys, public_keys_with_model_definitions = _binding_public_keys()
    private_keys = {
        "source_id",
        "api_family",
        "endpoint",
        "endpoint_sha256",
        "apply_binding_sha256",
        "credential_value",
    }
    materialized_keys = {
        "source_id",
        "app_type",
        "is_current",
        "api_family",
        "endpoint",
        "credential_reference",
        "declared_models",
        "model_definitions",
        "blocker_codes",
    }

    def clear_source(value: Any) -> None:
        if not isinstance(value, dict):
            return
        providers = value.get("providers")
        if isinstance(providers, list):
            for item in providers:
                if isinstance(item, dict):
                    item.clear()
            providers.clear()
        value.clear()

    def source_parts(value: Any) -> tuple[str, dict[str, Any], dict[str, Any]]:
        if (
            not isinstance(value, dict)
            or set(value) != expected_source_keys
            or value["source_app_type"] != "claude"
            or not _is_digest(value["source_snapshot_sha256"])
            or not isinstance(value["precondition"], dict)
            or not isinstance(value["providers"], list)
            or len(value["providers"]) != 1
            or not isinstance(value["providers"][0], dict)
        ):
            raise RunnerFailure("blocked-source-unavailable")
        return (
            value["source_snapshot_sha256"],
            value["precondition"],
            value["providers"][0],
        )

    def validate_apply_provider(provider: dict[str, Any]) -> tuple[str, str]:
        if set(provider) != private_keys:
            raise RunnerFailure("blocked-source-unavailable")
        endpoint = provider["endpoint"]
        credential = provider["credential_value"]
        endpoint_digest = (
            hashlib.sha256(endpoint.encode("utf-8")).hexdigest()
            if isinstance(endpoint, str)
            else ""
        )
        if (
            provider["source_id"] != SOURCE_ID
            or provider["api_family"] not in SOURCE_API_FAMILIES
            or not _valid_source_endpoint(endpoint)
            or not _is_digest(provider["endpoint_sha256"])
            or provider["endpoint_sha256"] != endpoint_digest
            or not _is_digest(provider["apply_binding_sha256"])
            or not isinstance(credential, str)
            or not credential
            or not credential.strip()
            or "\x00" in credential
        ):
            raise RunnerFailure("blocked-source-unavailable")
        return endpoint, credential

    def validate_planning_provider(
        provider: dict[str, Any],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        if set(provider) not in (
            public_keys,
            public_keys_with_model_definitions,
        ):
            raise RunnerFailure("blocked-source-unavailable")
        if (
            provider["source_id"] != SOURCE_ID
            or provider["app_type"] != "claude"
            or provider["is_current"] is not True
            or provider["api_family"] not in SOURCE_API_FAMILIES
            or provider["endpoint_https"] is not True
            or not _is_digest(provider["endpoint_sha256"])
            or not _is_digest(provider["apply_binding_sha256"])
            or provider["credential_reference_present"] is not True
            or provider["blocker_codes"] != []
            or not _valid_declared_models(provider)
        ):
            raise RunnerFailure("blocked-source-unavailable")
        environment_models = _normalised_environment_models(
            provider["environment_models"]
        )
        declared_models = provider["declared_models"]
        model_definitions = _normalised_model_definitions(provider, declared_models)
        return environment_models, model_definitions

    def validate_materialized_provider(
        provider: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if set(provider) != materialized_keys:
            raise RunnerFailure("blocked-source-unavailable")
        if (
            provider["source_id"] != SOURCE_ID
            or provider["app_type"] != "claude"
            or provider["is_current"] is not True
            or provider["api_family"] not in SOURCE_API_FAMILIES
            or not _valid_source_endpoint(provider["endpoint"])
            or not isinstance(provider["credential_reference"], str)
            or not _safe_identifier(provider["credential_reference"])
            or provider["blocker_codes"] != []
            or not _valid_declared_models(provider)
        ):
            raise RunnerFailure("blocked-source-unavailable")
        return _normalised_model_definitions(
            provider, provider["declared_models"]
        )

    def capture_once() -> tuple[_Binding, str | None]:
        apply_before_source: Any = None
        planning_source: Any = None
        materialized_source: Any = None
        apply_after_source: Any = None
        try:
            try:
                # Keep this order: private apply facts bracket the two public
                # views, without exposing the credential to either one.
                apply_before_source = _migration_apply_source(
                    source_db,
                    selected_source_ids={SOURCE_ID},
                    approved_origin_source_ids={SOURCE_ID},
                )
                planning_source = migration_planning_source(
                    source_db,
                    selected_source_ids={SOURCE_ID},
                    approved_origin_source_ids={SOURCE_ID},
                )
                materialized_source = materialization_source(
                    source_db,
                    selected_source_ids={SOURCE_ID},
                    approved_origin_source_ids={SOURCE_ID},
                )
                apply_after_source = _migration_apply_source(
                    source_db,
                    selected_source_ids={SOURCE_ID},
                    approved_origin_source_ids={SOURCE_ID},
                )
            except SourceInventoryError:
                # The source seam already fails closed on an unstable capture.
                # Retry the whole semantic clamp so telemetry WAL churn between
                # independent views does not become a permanent false negative.
                raise RunnerFailure("blocked-source-changed") from None
            except (OSError, TypeError, ValueError, UnicodeError):
                raise RunnerFailure("blocked-source-unavailable") from None

            apply_before_snapshot, apply_before_precondition, apply_before_provider = (
                source_parts(apply_before_source)
            )
            planning_snapshot, planning_precondition, planning_provider = source_parts(
                planning_source
            )
            materialized_snapshot, materialized_precondition, materialized_provider = (
                source_parts(materialized_source)
            )
            apply_after_snapshot, apply_after_precondition, apply_after_provider = (
                source_parts(apply_after_source)
            )

            # A source snapshot is a logical provider projection, not a file
            # manifest.  It is useful as a semantic guard across all views.
            snapshots = {
                apply_before_snapshot,
                planning_snapshot,
                materialized_snapshot,
                apply_after_snapshot,
            }
            if len(snapshots) != 1:
                raise RunnerFailure("blocked-source-changed")

            # Retain only replacement/security identity from the preconditions.
            # Size, timestamps, hashes, WAL, and SHM are intentionally ignored.
            preconditions = (
                apply_before_precondition,
                planning_precondition,
                materialized_precondition,
                apply_after_precondition,
            )
            identities = tuple(_safe_file_identity(item) for item in preconditions)
            if any(identity is None for identity in identities):
                raise RunnerFailure("blocked-source-unavailable")
            if len(set(identities)) != 1:
                raise RunnerFailure("blocked-source-changed")

            apply_before_endpoint, apply_before_credential = validate_apply_provider(
                apply_before_provider
            )
            apply_after_endpoint, _apply_after_credential = validate_apply_provider(
                apply_after_provider
            )
            planning_environment_models, planning_model_definitions = (
                validate_planning_provider(planning_provider)
            )
            materialized_model_definitions = validate_materialized_provider(
                materialized_provider
            )

            # Apply is the private clamp.  Compare only semantic facts; neither
            # apply precondition may be equal to a neighboring full manifest.
            if (
                apply_before_provider["source_id"]
                != apply_after_provider["source_id"]
                or apply_before_provider["api_family"]
                != apply_after_provider["api_family"]
                or apply_before_endpoint != apply_after_endpoint
                or apply_before_provider["endpoint_sha256"]
                != apply_after_provider["endpoint_sha256"]
                or apply_before_provider["apply_binding_sha256"]
                != apply_after_provider["apply_binding_sha256"]
            ):
                raise RunnerFailure("blocked-source-changed")

            # Bind public planning and materialization to the same private
            # apply facts and to one target provider identity.
            if (
                planning_provider["source_id"] != materialized_provider["source_id"]
                or planning_provider["app_type"] != materialized_provider["app_type"]
                or planning_provider["is_current"]
                != materialized_provider["is_current"]
                or planning_provider["api_family"] != materialized_provider["api_family"]
                or planning_provider["endpoint_sha256"]
                != apply_before_provider["endpoint_sha256"]
                or materialized_provider["endpoint"]
                != apply_before_endpoint
                or hashlib.sha256(
                    materialized_provider["endpoint"].encode("utf-8")
                ).hexdigest()
                != planning_provider["endpoint_sha256"]
                or planning_provider["apply_binding_sha256"]
                != apply_before_provider["apply_binding_sha256"]
                or planning_provider["declared_models"]
                != materialized_provider["declared_models"]
                or planning_provider["blocker_codes"]
                != materialized_provider["blocker_codes"]
            ):
                raise RunnerFailure("blocked-source-changed")

            # Planning normally omits model definitions, so its declaration is
            # the fallback definition.  When a planning seam supplies explicit
            # definitions, require their full bounded semantics to match
            # materialization.  In either shape both views then hash the same
            # target model projection, including environment aliases.
            if (
                "model_definitions" in planning_provider
                and planning_model_definitions != materialized_model_definitions
            ):
                raise RunnerFailure("blocked-source-changed")
            planning_model_digest = _semantic_digest(
                {
                    "declaredModels": list(planning_provider["declared_models"]),
                    "environmentModels": planning_environment_models,
                    "modelDefinitions": (
                        materialized_model_definitions
                        if "model_definitions" not in planning_provider
                        else planning_model_definitions
                    ),
                }
            )
            materialized_model_digest = _semantic_digest(
                {
                    "declaredModels": list(materialized_provider["declared_models"]),
                    "environmentModels": planning_environment_models,
                    "modelDefinitions": materialized_model_definitions,
                }
            )
            if planning_model_digest != materialized_model_digest:
                raise RunnerFailure("blocked-source-changed")

            semantic_provider = dict(planning_provider)
            semantic_provider["environment_models"] = planning_environment_models
            semantic_provider["model_definitions"] = materialized_model_definitions
            semantic_binding_sha256 = _target_semantic_digest(
                semantic_provider,
                apply_before_provider,
                apply_before_endpoint,
            )
            try:
                retained_precondition = json.loads(
                    json.dumps(planning_precondition, sort_keys=True)
                )
            except (TypeError, ValueError, UnicodeError):
                raise RunnerFailure("blocked-source-unavailable") from None
            binding = _Binding(
                snapshot=planning_snapshot,
                precondition=retained_precondition,
                endpoint_sha256=planning_provider["endpoint_sha256"],
                apply_binding_sha256=planning_provider["apply_binding_sha256"],
                source_api_family=planning_provider["api_family"],
                semantic_binding_sha256=semantic_binding_sha256,
            )
            return (
                binding,
                apply_before_credential if retain_credential else None,
            )
        finally:
            clear_source(apply_after_source)
            clear_source(materialized_source)
            clear_source(planning_source)
            clear_source(apply_before_source)

    for attempt in range(MAX_BINDING_CAPTURE_ATTEMPTS):
        try:
            return capture_once()
        except RunnerFailure as failure:
            if (
                failure.status != "blocked-source-changed"
                or attempt + 1 >= MAX_BINDING_CAPTURE_ATTEMPTS
            ):
                raise

    raise AssertionError("unreachable")


def _read_binding(source_db: str | os.PathLike[str]) -> _Binding:
    return _binding_from_sources(source_db, retain_credential=False)[0]


def _read_binding_with_credential(
    source_db: str | os.PathLike[str],
) -> tuple[_Binding, str]:
    binding, credential = _binding_from_sources(source_db, retain_credential=True)
    if not isinstance(credential, str) or not credential:
        raise RunnerFailure("blocked-credential-unavailable")
    return binding, credential
def _same_binding(left: _Binding, right: _Binding) -> bool:
    left_identity = _safe_file_identity(left.precondition)
    right_identity = _safe_file_identity(right.precondition)
    if left_identity is None or right_identity is None:
        return False
    return (
        left.semantic_binding_sha256 == right.semantic_binding_sha256
        and left_identity == right_identity
    )


def _assert_child_source_path(
    source_db: str | os.PathLike[str],
    child: Callable[[str, str, str], tuple[dict[str, Any], int | None]],
) -> None:
    # Test callbacks may use fixture paths.  The production callback is this
    # module's _invoke_child and is installation-bound to the configured DB.
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


def build_responses_request(model: str, mode: str) -> dict[str, Any]:
    """Build the exact OpenAI Responses body for one fixed probe."""
    if model not in MODELS or mode not in MODES:
        raise RunnerFailure("blocked-invalid-arguments")
    prompt = (
        "Reply with a non-empty answer and no explanation."
        if mode == "text"
        else "Call smoke_ok exactly once with {} and no text."
    )
    request: dict[str, Any] = {
        "model": model,
        "input": prompt,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "store": False,
    }
    if mode == "tool":
        request["tools"] = [
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
        request["tool_choice"] = {"type": "function", "name": "smoke_ok"}
    return request


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)




def _usage(response: Any) -> dict[str, int] | None:
    usage = _field(response, "usage")
    if usage is None:
        return None
    result: dict[str, int] = {}
    for target, source_names in (
        ("input_tokens", ("input_tokens", "prompt_tokens")),
        ("output_tokens", ("output_tokens", "completion_tokens")),
        ("total_tokens", ("total_tokens",)),
    ):
        for source in source_names:
            value = _field(usage, source)
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
                result[target] = value
                break
    return result or None


def _text_response(response: Any) -> str | None:
    output = _field(response, "output")
    if not isinstance(output, list):
        return None
    parts: list[str] = []
    for item in output:
        item_type = _field(item, "type")
        if item_type == "reasoning":
            continue
        if (
            item_type != "message"
            or _field(item, "role") != "assistant"
            or _field(item, "status") != "completed"
        ):
            return None
        content = _field(item, "content")
        if not isinstance(content, list):
            return None
        for part in content:
            if _field(part, "type") != "output_text":
                return None
            text = _field(part, "text")
            if not isinstance(text, str):
                return None
            if text.strip():
                parts.append(text.strip())
    return "\n".join(parts).strip()


def _only_smoke_ok(response: Any) -> bool:
    output = _field(response, "output")
    if not isinstance(output, list):
        return False
    calls: list[Any] = []
    for item in output:
        item_type = _field(item, "type")
        if item_type == "reasoning":
            continue
        if (
            item_type != "function_call"
            or _field(item, "status") != "completed"
        ):
            return False
        calls.append(item)
    if (
        len(calls) != 1
        or _field(calls[0], "name") != "smoke_ok"
        or not _safe_identifier(_field(calls[0], "call_id"))
    ):
        return False
    arguments = _field(calls[0], "arguments")
    if not isinstance(arguments, str):
        return False
    try:
        return json.loads(arguments) == {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _safe_response_model(response: Any, requested_model: str) -> str | None:
    """Retain ``response.model`` only when it exactly binds to the request."""
    model = _field(response, "model")
    return (
        requested_model
        if isinstance(model, str) and model == requested_model
        else None
    )


def _temporary_exception(exc: BaseException) -> bool:
    code = getattr(exc, "status_code", None)
    return bool(
        code == 429
        or (isinstance(code, int) and code >= 500)
        or isinstance(exc, (TimeoutError, ConnectionError))
        or exc.__class__.__name__ in {"ReadTimeout", "ConnectTimeout"}
    )


def _request_result(
    model: str,
    mode: str,
    credential: str,
    transport: Transport,
) -> dict[str, Any]:
    result = _baseline(model, mode)
    result["verification_kind"] = f"live-{mode}"
    started = time.monotonic()
    try:
        body = _canonical(build_responses_request(model, mode))
        headers = {
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        }
        status_code, raw = transport(
            BUSINESS_URL, headers, body, PER_CALL_TIMEOUT_SECONDS
        )
        if not isinstance(status_code, int) or isinstance(status_code, bool):
            raise RunnerFailure("unusable")
        if not isinstance(raw, bytes) or len(raw) > MAX_RESPONSE_BYTES:
            raise RunnerFailure("unusable")
        if status_code < 200 or status_code >= 300:
            result["status"] = (
                "temporary" if status_code == 429 or status_code >= 500 else "unusable"
            )
            return result
        try:
            response = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError, json.JSONDecodeError):
            result["status"] = "unusable"
            return result
        if not isinstance(response, dict):
            result["status"] = "unusable"
            return result
        if response.get("status") != "completed":
            result["status"] = "unusable"
            return result
        result["token_usage"] = _usage(response)
        response_model = _safe_response_model(response, model)
        result["response_model_id"] = response_model
        if response_model is None:
            raw_model = _field(response, "model")
            result["identity_status"] = (
                "identity-mismatch"
                if isinstance(raw_model, str)
                else "identity-unverified"
            )
            result["status"] = result["identity_status"]
            return result
        result["identity_status"] = "observed-match"
        if mode == "tool":
            if not _only_smoke_ok(response):
                result["status"] = "unusable-tool-result"
                result["tool_result"] = "rejected"
                return result
            result["tool_result"] = "smoke_ok"
        else:
            text = _text_response(response)
            if text is None:
                result["status"] = "unusable-text-result"
                return result
            if not text:
                result["status"] = "unusable-empty-text"
                return result
        result["status"] = "usable"
        return result
    except RunnerFailure as failure:
        result["status"] = failure.status
        return result
    except Exception as exc:
        temporary = _temporary_exception(exc)
        del exc
        result["status"] = "temporary" if temporary else "unusable"
        return result
    finally:
        result["latency_ms"] = max(
            0,
            min(
                PROCESS_TIMEOUT_SECONDS * 1000,
                round((time.monotonic() - started) * 1000),
            ),
        )

def _https_transport(
    endpoint: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    """POST once to the fixed HTTPS Responses endpoint, with no redirect/retry."""
    _clear_ssl_keylog_env()
    if endpoint != BUSINESS_URL:
        raise RunnerFailure("blocked-transport-policy")
    parsed = urlsplit(endpoint)
    if (
        parsed.scheme != "https"
        or parsed.hostname != "hs32.com"
        or parsed.path != "/v1/responses"
    ):
        raise RunnerFailure("blocked-transport-policy")
    try:
        context = ssl.create_default_context()
        context.keylog_filename = None
        if getattr(context, "keylog_filename", None) is not None:
            raise RunnerFailure("blocked-transport-policy")
    except RunnerFailure:
        raise
    except (AttributeError, OSError, ValueError):
        raise RunnerFailure("blocked-transport-policy") from None
    connection: http.client.HTTPSConnection | None = None
    try:
        connection = http.client.HTTPSConnection(
            parsed.hostname,
            parsed.port or 443,
            timeout=timeout,
            context=context,
        )
        connection.request("POST", parsed.path, body=body, headers=dict(headers))
        response = connection.getresponse()
        raw = response.read(MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise RunnerFailure("unusable")
        return int(response.status), raw
    finally:
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass


def _decode_child_result(
    completed: Any, model: str, mode: str, phase: str
) -> tuple[dict[str, Any], int | None]:
    stdout = getattr(completed, "stdout", None)
    stderr = getattr(completed, "stderr", None)
    returncode = getattr(completed, "returncode", None)
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
    if not _valid_result(parsed, model, mode, phase):
        raise RunnerFailure("unusable-worker-output")
    status = parsed["status"]
    # HTTP/API/response-parse failures intentionally use a positive nonzero
    # child exit while carrying a closed status row, so they remain eligible
    # for the next probe.  A signal exit, crash marker, or successful row with
    # a nonzero exit is a worker protocol failure and must stop the phase.
    if returncode < 0 or status == "unusable-worker-crash" or (
        returncode != 0 and status in {"usable", "preflight-ok"}
    ):
        raise RunnerFailure("unusable-worker-crash")
    return {key: parsed[key] for key in _RESULT_KEYS}, returncode

def _kill_process_group(process_or_pid: Any) -> None:
    pid = process_or_pid if isinstance(process_or_pid, int) else getattr(process_or_pid, "pid", None)
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
        return
    except (ProcessLookupError, OSError):
        pass
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except OSError:
        if isinstance(process_or_pid, int):
            return
        try:
            process_or_pid.kill()
        except (OSError, ProcessLookupError, AttributeError):
            pass


def _write_fd(fd: int, payload: bytes) -> None:
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        if written <= 0:
            raise OSError("short pipe write")
        view = view[written:]


def _wait_pid(pid: int, timeout: float) -> tuple[bool, int | None]:
    deadline = time.monotonic() + timeout
    while True:
        try:
            waited, status = os.waitpid(pid, os.WNOHANG)
        except ChildProcessError:
            return True, None
        except OSError:
            return False, None
        if waited == pid:
            return True, status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False, None
        time.sleep(min(0.01, remaining))


def _prepare_child_fds(capability_read_fd: int, result_write_fd: int) -> tuple[int, int]:
    """Drop inherited parent descriptors before any source/credential read."""
    cap_keep = os.dup(capability_read_fd)
    result_keep = os.dup(result_write_fd)
    os.dup2(cap_keep, 3)
    os.dup2(result_keep, 4)
    for descriptor in (cap_keep, result_keep):
        if descriptor not in {3, 4}:
            os.close(descriptor)
    try:
        maximum = int(os.sysconf("SC_OPEN_MAX"))
    except (AttributeError, OSError, ValueError):
        maximum = 4096
    os.closerange(5, max(5, min(maximum, 1_048_576)))
    return 3, 4


def _fork_child_main(
    *,
    model: str,
    mode: str,
    phase: str,
    source_db: str | os.PathLike[str],
    binding: _Binding,
    capability: bytes,
    capability_read_fd: int,
    result_write_fd: int,
) -> NoReturn:
    """Run one watched child from fork-inherited memory and exit safely."""
    result = _baseline(model, mode)
    exit_code = 2
    try:
        try:
            os.setsid()
        except OSError:
            raise RunnerFailure("blocked-child-capability") from None
        token = os.read(capability_read_fd, CAPABILITY_BYTES + 1)
        if token != capability:
            raise RunnerFailure("blocked-child-capability")
        if phase == "preflight":
            current = _read_binding(source_db)
            if not _same_binding(binding, current):
                raise RunnerFailure("blocked-source-changed")
            result.update(verification_kind="preflight-only", status="preflight-ok")
            exit_code = 0
        else:
            _clear_ssl_keylog_env()
            _require_trusted_runtime()
            current, credential = _read_binding_with_credential(source_db)
            if not _same_binding(binding, current):
                del credential
                raise RunnerFailure("blocked-source-changed")
            try:
                result = _request_result(model, mode, credential, _https_transport)
            finally:
                del credential
            exit_code = 0 if result.get("status") == "usable" else 2
    except RunnerFailure as failure:
        result["verification_kind"] = (
            "preflight-only" if phase == "preflight" else f"live-{mode}"
        )
        result["status"] = failure.status
    except BaseException:
        # Never turn an unexpected exception into the ordinary ``unusable``
        # response path: the parent may continue after API/parse failures, but
        # a worker crash is a terminal protocol failure.
        result["verification_kind"] = (
            "preflight-only" if phase == "preflight" else f"live-{mode}"
        )
        result["status"] = "unusable-worker-crash"
    finally:
        try:
            os.close(capability_read_fd)
        except OSError:
            pass
        try:
            payload = json.dumps(
                result, ensure_ascii=True, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
            if len(payload) <= MAX_CHILD_OUTPUT_BYTES:
                _write_fd(result_write_fd, payload)
        except (OSError, TypeError, ValueError, UnicodeError):
            pass
        try:
            os.close(result_write_fd)
        except OSError:
            pass
        os._exit(exit_code)


def _invoke_child(
    model: str,
    mode: str,
    phase: str,
    binding: _Binding | None = None,
    source_db: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, Any], int | None]:
    """Fork one capability-bound child and enforce the fixed hard wall."""
    if source_db is None:
        source_db = SOURCE_DB
    if model not in MODELS or mode not in MODES or phase not in {"preflight", "live"}:
        raise RunnerFailure("blocked-invalid-arguments")
    if phase == "live":
        _clear_ssl_keylog_env()
        _require_trusted_runtime()
    if binding is None:
        binding = _read_binding(source_db)
    capability = secrets.token_bytes(CAPABILITY_BYTES)
    capability_read_fd, capability_write_fd = os.pipe()
    result_read_fd, result_write_fd = os.pipe()
    try:
        pid = os.fork()
    except (OSError, AttributeError):
        for descriptor in (
            capability_read_fd,
            capability_write_fd,
            result_read_fd,
            result_write_fd,
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise RunnerFailure("unusable-worker-launch") from None
    if pid == 0:
        try:
            os.close(capability_write_fd)
            os.close(result_read_fd)
            child_capability_fd, child_result_fd = _prepare_child_fds(
                capability_read_fd, result_write_fd
            )
            _fork_child_main(
                model=model,
                mode=mode,
                phase=phase,
                source_db=source_db,
                binding=binding,
                capability=capability,
                capability_read_fd=child_capability_fd,
                result_write_fd=child_result_fd,
            )
        except BaseException:
            os._exit(2)
    reaped = False
    try:
        os.close(capability_read_fd)
        capability_read_fd = -1
        os.close(result_write_fd)
        result_write_fd = -1
        _write_fd(capability_write_fd, capability)
        os.close(capability_write_fd)
        capability_write_fd = -1
        del capability
        ready, _unused_writable, _unused_exceptional = select.select(
            [result_read_fd], [], [], PROCESS_TIMEOUT_SECONDS
        )
        if not ready:
            raise RunnerFailure("temporary-timeout")
        stdout = os.read(result_read_fd, MAX_CHILD_OUTPUT_BYTES + 1)
        os.close(result_read_fd)
        result_read_fd = -1
        reaped, wait_status = _wait_pid(pid, REAP_TIMEOUT_SECONDS)
        if not reaped:
            raise RunnerFailure("temporary-timeout")
        try:
            returncode = (
                os.waitstatus_to_exitcode(wait_status)
                if wait_status is not None
                else 2
            )
        except (AttributeError, ValueError):
            returncode = 2
        completed = subprocess.CompletedProcess(
            [], returncode, stdout=stdout, stderr=b""
        )
        return _decode_child_result(completed, model, mode, phase)
    except KeyboardInterrupt:
        raise
    except RunnerFailure:
        raise
    except (OSError, UnicodeError, ValueError):
        raise RunnerFailure("unusable-worker-output") from None
    finally:
        for descriptor in (capability_read_fd, capability_write_fd, result_write_fd, result_read_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if not reaped:
            try:
                _kill_process_group(pid)
                reaped_after_kill, _unused_status = _wait_pid(
                    pid, REAP_TIMEOUT_SECONDS
                )
            except KeyboardInterrupt:
                # Preserve SIGINT classification; never relabel a user
                # interrupt as a reap failure.
                raise
            except BaseException:
                raise RunnerFailure("blocked-worker-reap") from None
            if not reaped_after_kill:
                raise RunnerFailure("blocked-worker-reap")


def _canonical_observation(observation: Mapping[str, Any], model: str, mode: str) -> dict[str, Any]:
    expected_keys = {
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
    }
    if not isinstance(observation, Mapping) or set(observation) != expected_keys:
        raise RunnerFailure("blocked-artifact")
    source_id = observation["sourceId"]
    api_family = observation["apiFamily"]
    configured_model = observation["configuredModelValue"]
    observed_model = observation["observedModel"]
    identity_status = observation["identityStatus"]
    status = observation["status"]
    tool_result = observation["toolResult"]
    usable = observation["usable"]
    if (
        source_id != SOURCE_ID
        or api_family != REQUEST_API_FAMILY
        or configured_model != model
        or observation["model"] != model
        or observation["mode"] != mode
        or identity_status not in _IDENTITY_STATUSES
        or status not in _SAFE_STATUSES
        or not isinstance(usable, bool)
        or not _valid_token_usage(observation["tokenUsage"])
        or not isinstance(observation["latencyMs"], int)
        or isinstance(observation["latencyMs"], bool)
        or not 0 <= observation["latencyMs"] <= PROCESS_TIMEOUT_SECONDS * 1000
        or (
            observed_model is not None
            and observed_model != model
        )
    ):
        raise RunnerFailure("blocked-artifact")
    if mode == "text":
        if tool_result is not None:
            raise RunnerFailure("blocked-artifact")
    elif mode == "tool":
        if tool_result not in {"not-run", "rejected", "smoke_ok"}:
            raise RunnerFailure("blocked-artifact")
    else:
        raise RunnerFailure("blocked-artifact")
    if identity_status == "observed-match":
        if observed_model != model:
            raise RunnerFailure("blocked-artifact")
    elif identity_status in {"identity-mismatch", "identity-unverified", "not-observed"}:
        if observed_model is not None:
            raise RunnerFailure("blocked-artifact")
    else:
        raise RunnerFailure("blocked-artifact")
    if status == "usable":
        if (
            usable is not True
            or identity_status != "observed-match"
            or tool_result != ("smoke_ok" if mode == "tool" else None)
        ):
            raise RunnerFailure("blocked-artifact")
    else:
        if status not in _PARTIAL_LIVE_FAILURES or usable is not False:
            raise RunnerFailure("blocked-artifact")
    return {
        "sourceId": SOURCE_ID,
        "apiFamily": REQUEST_API_FAMILY,
        "configuredModelValue": model,
        "model": model,
        "observedModel": observed_model,
        "mode": mode,
        "identityStatus": identity_status,
        "status": status,
        "toolResult": tool_result,
        "tokenUsage": observation["tokenUsage"],
        "latencyMs": observation["latencyMs"],
        "usable": usable,
    }


def build_observation_artifact(
    *,
    semantic_binding_sha256: str,
    observations: Sequence[Mapping[str, Any]],
    source_api_family: str = SOURCE_API_FAMILY,
) -> dict[str, Any]:
    if source_api_family not in SOURCE_API_FAMILIES:
        raise RunnerFailure("blocked-artifact")
    if not _is_digest(semantic_binding_sha256):
        raise RunnerFailure("blocked-artifact")
    if len(observations) != MAX_API_CALLS:
        raise RunnerFailure("blocked-artifact")
    normalized: list[dict[str, Any]] = []
    for index, (model, mode) in enumerate(PROBES):
        try:
            normalized.append(_canonical_observation(observations[index], model, mode))
        except (IndexError, KeyError, TypeError, ValueError):
            raise RunnerFailure("blocked-artifact") from None
    body = {
        "schemaVersion": SCHEMA_VERSION,
        "artifactKind": ARTIFACT_KIND,
        "requestApi": REQUEST_API_FAMILY,
        "sourceApiFamily": source_api_family,
        "semanticBindingSha256": semantic_binding_sha256,
        "providerEvidence": [
            {
                "sourceId": SOURCE_ID,
                "semanticBindingSha256": semantic_binding_sha256,
            }
        ],
        "observations": normalized,
    }
    digest = hashlib.sha256(_canonical(body)).hexdigest()
    return {**body, "artifactSha256": digest}


def _artifact_bytes(artifact: Mapping[str, Any]) -> bytes:
    required = {
        "schemaVersion",
        "artifactKind",
        "requestApi",
        "sourceApiFamily",
        "semanticBindingSha256",
        "providerEvidence",
        "observations",
        "artifactSha256",
    }
    if not isinstance(artifact, Mapping) or set(artifact) != required:
        raise RunnerFailure("blocked-artifact")
    digest = artifact["artifactSha256"]
    body = {key: value for key, value in artifact.items() if key != "artifactSha256"}
    evidence = body.get("providerEvidence")
    semantic_binding_sha256 = body.get("semanticBindingSha256")
    observations = body.get("observations")
    evidence_valid = (
        isinstance(evidence, list)
        and len(evidence) == 1
        and isinstance(evidence[0], dict)
        and set(evidence[0]) == {"sourceId", "semanticBindingSha256"}
        and evidence[0].get("sourceId") == SOURCE_ID
        and evidence[0].get("semanticBindingSha256") == semantic_binding_sha256
        and _is_digest(semantic_binding_sha256)
    )
    if (
        not _is_digest(digest)
        or hashlib.sha256(_canonical(body)).hexdigest() != digest
        or body.get("schemaVersion") != SCHEMA_VERSION
        or body.get("artifactKind") != ARTIFACT_KIND
        or body.get("requestApi") != REQUEST_API_FAMILY
        or body.get("sourceApiFamily") not in SOURCE_API_FAMILIES
        or not evidence_valid
        or not isinstance(observations, list)
        or len(observations) != MAX_API_CALLS
    ):
        raise RunnerFailure("blocked-artifact")
    for index, (model, mode) in enumerate(PROBES):
        _canonical_observation(observations[index], model, mode)
    encoded = _canonical(dict(artifact)) + bytes((10,))
    if len(encoded) > MAX_ARTIFACT_BYTES:
        raise RunnerFailure("blocked-artifact")
    return encoded



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


def _sync_directory(path: Path) -> None:
    """Durably persist a same-directory rename or unlink."""
    descriptor = -1
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        os.fsync(descriptor)
    except (AttributeError, OSError, ValueError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        if descriptor != -1:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _reservation_identity(reservation: _Reservation) -> tuple[int, int]:
    return (
        reservation.current_device
        if reservation.current_device is not None
        else reservation.device,
        reservation.current_inode
        if reservation.current_inode is not None
        else reservation.inode,
    )


def _regular_owned(details: os.stat_result, mode: int = 0o600) -> bool:
    return (
        stat.S_ISREG(details.st_mode)
        and details.st_uid == os.getuid()
        and stat.S_IMODE(details.st_mode) == mode
        and details.st_nlink == 1
    )


def _reserve_output(path: Path) -> _Reservation:
    if not path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts[1:]):
        raise RunnerFailure("blocked-output-path")
    _secure_parent(path.parent)
    descriptor = -1
    reserved = False
    details: os.stat_result | None = None
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        os.fchmod(descriptor, 0o600)
        details = os.fstat(descriptor)
        if not _regular_owned(details):
            raise RunnerFailure("blocked-output-write")
        marker = b'{"status":"in-progress"}\n'
        _write_fd(descriptor, marker)
        os.fsync(descriptor)
        _sync_directory(path.parent)
        reserved = True
        return _Reservation(descriptor, details.st_dev, details.st_ino)
    except FileExistsError:
        raise RunnerFailure("blocked-output-exists") from None
    except RunnerFailure:
        raise
    except (AttributeError, OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        if descriptor != -1 and not reserved:
            try:
                os.close(descriptor)
            except OSError:
                pass
            if details is not None:
                try:
                    current = os.lstat(path)
                    if (
                        (current.st_dev, current.st_ino) == (details.st_dev, details.st_ino)
                        and _regular_owned(current)
                    ):
                        os.unlink(path)
                        try:
                            _sync_directory(path.parent)
                        except RunnerFailure:
                            pass
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


def _create_private_output(path: Path) -> _PrivateOutput:
    """Create a 0600 temp file beside the reserved final path."""
    _secure_parent(path.parent)
    descriptor = -1
    temp_path: Path | None = None
    created = False
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        temp_path = Path(name)
        os.fchmod(descriptor, 0o600)
        details = os.fstat(descriptor)
        visible = os.lstat(temp_path)
        if (
            not _regular_owned(details)
            or (visible.st_dev, visible.st_ino) != (details.st_dev, details.st_ino)
            or not _regular_owned(visible)
        ):
            raise RunnerFailure("blocked-output-write")
        created = True
        return _PrivateOutput(
            descriptor=descriptor,
            path=temp_path,
            device=details.st_dev,
            inode=details.st_ino,
        )
    except RunnerFailure:
        raise
    except (AttributeError, OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        if not created:
            if descriptor != -1:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except OSError:
                    pass


def _close_private_output(temp: _PrivateOutput | None, *, unlink: bool = True) -> None:
    if temp is None:
        return
    if temp.descriptor != -1:
        try:
            os.close(temp.descriptor)
        except OSError:
            pass
        temp.descriptor = -1
    if unlink and not temp.published:
        try:
            visible = os.lstat(temp.path)
            if (
                (visible.st_dev, visible.st_ino) == (temp.device, temp.inode)
                and _regular_owned(visible)
            ):
                os.unlink(temp.path)
                try:
                    _sync_directory(temp.path.parent)
                except RunnerFailure:
                    pass
        except OSError:
            pass


def _write_private_output(temp: _PrivateOutput, payload: bytes) -> None:
    try:
        if not isinstance(payload, bytes):
            raise RunnerFailure("blocked-output-write")
        opened = os.fstat(temp.descriptor)
        visible = os.lstat(temp.path)
        if (
            (opened.st_dev, opened.st_ino) != (temp.device, temp.inode)
            or (visible.st_dev, visible.st_ino) != (temp.device, temp.inode)
            or not _regular_owned(opened)
            or not _regular_owned(visible)
        ):
            raise RunnerFailure("blocked-output-write")
        os.ftruncate(temp.descriptor, 0)
        os.lseek(temp.descriptor, 0, os.SEEK_SET)
        _write_fd(temp.descriptor, payload)
        os.fsync(temp.descriptor)
        final = os.fstat(temp.descriptor)
        visible = os.lstat(temp.path)
        if (
            (final.st_dev, final.st_ino) != (temp.device, temp.inode)
            or (visible.st_dev, visible.st_ino) != (temp.device, temp.inode)
            or not _regular_owned(final)
            or not _regular_owned(visible)
        ):
            raise RunnerFailure("blocked-output-write")
    except RunnerFailure:
        raise
    except (OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-output-write") from None


def _publish_private_output(
    path: Path,
    reservation: _Reservation,
    temp: _PrivateOutput,
    payload: bytes,
) -> None:
    """Atomically publish a fsynced sibling temp over our reserved inode."""
    try:
        _secure_parent(path.parent)
        current = os.lstat(path)
        temp_details = os.fstat(temp.descriptor)
        temp_visible = os.lstat(temp.path)
        expected = _reservation_identity(reservation)
        if (
            (current.st_dev, current.st_ino) != expected
            or not _regular_owned(current)
            or (temp_details.st_dev, temp_details.st_ino) != (temp.device, temp.inode)
            or (temp_visible.st_dev, temp_visible.st_ino) != (temp.device, temp.inode)
            or not _regular_owned(temp_details)
            or not _regular_owned(temp_visible)
        ):
            raise RunnerFailure("blocked-output-write")
        # Before the first publication, the open reservation descriptor proves
        # that the path was not swapped between lstat and replace.  On later
        # replacement of an invalid marker, current identity is the proof.
        if (
            (reservation.current_device, reservation.current_inode)
            == (reservation.device, reservation.inode)
            and reservation.descriptor != -1
        ):
            opened = os.fstat(reservation.descriptor)
            if (
                (opened.st_dev, opened.st_ino) != (reservation.device, reservation.inode)
                or not _regular_owned(opened)
            ):
                raise RunnerFailure("blocked-output-write")
        os.replace(temp.path, path)
        temp.published = True
        reservation.current_device = temp.device
        reservation.current_inode = temp.inode
        _sync_directory(path.parent)
        _verify_output_payload(path, reservation, payload)
    except RunnerFailure:
        raise
    except (OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-output-write") from None


def _replace_output(
    path: Path, reservation: _Reservation, payload: bytes, *, close: bool = True
) -> None:
    """Write privately, then atomically replace the currently owned output."""
    temp: _PrivateOutput | None = None
    try:
        temp = _create_private_output(path)
        _write_private_output(temp, payload)
        _publish_private_output(path, reservation, temp, payload)
    finally:
        _close_private_output(temp, unlink=True)
        if close:
            _close_reservation(reservation)


def _verify_output_payload(
    path: Path, reservation: _Reservation, payload: bytes
) -> None:
    """Prove the current output inode contains exactly the supplied bytes."""
    read_fd = -1
    try:
        _secure_parent(path.parent)
        visible = os.lstat(path)
        if (
            (visible.st_dev, visible.st_ino) != _reservation_identity(reservation)
            or not _regular_owned(visible)
        ):
            raise RunnerFailure("blocked-output-write")
        read_fd = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        read_details = os.fstat(read_fd)
        if (
            (read_details.st_dev, read_details.st_ino)
            != _reservation_identity(reservation)
            or not _regular_owned(read_details)
            or os.read(read_fd, len(payload) + 1) != payload
        ):
            raise RunnerFailure("blocked-output-write")
        final_read = os.fstat(read_fd)
        final_visible = os.lstat(path)
        if (
            (final_read.st_dev, final_read.st_ino)
            != _reservation_identity(reservation)
            or (final_visible.st_dev, final_visible.st_ino)
            != _reservation_identity(reservation)
            or not _regular_owned(final_read)
            or not _regular_owned(final_visible)
        ):
            raise RunnerFailure("blocked-output-write")
    except RunnerFailure:
        raise
    except (OSError, TypeError, ValueError):
        raise RunnerFailure("blocked-output-write") from None
    finally:
        if read_fd != -1:
            try:
                os.close(read_fd)
            except OSError:
                pass


def _unlink_reserved_output(path: Path, reservation: _Reservation) -> None:
    """Unlink only the inode still owned by this reservation."""
    try:
        _secure_parent(path.parent)
        visible = os.lstat(path)
        if (
            (visible.st_dev, visible.st_ino) != _reservation_identity(reservation)
            or not _regular_owned(visible)
        ):
            raise RunnerFailure("blocked-output-write")
        os.unlink(path)
        _sync_directory(path.parent)
        try:
            os.lstat(path)
        except FileNotFoundError:
            return
        raise RunnerFailure("blocked-output-write")
    except RunnerFailure:
        raise
    except (OSError, ValueError):
        raise RunnerFailure("blocked-output-write") from None


def _invalidate_published_output(path: Path, reservation: _Reservation) -> None:
    """Remove our output after a publication or invalidation failure."""
    try:
        _unlink_reserved_output(path, reservation)
    except RunnerFailure:
        try:
            if path.exists():
                raise
        except OSError:
            raise RunnerFailure("blocked-output-write") from None
    finally:
        _close_reservation(reservation)

def _safe_summary(
    *,
    status: str,
    phase: str,
    outcomes: Sequence[_ChildOutcome] = (),
    source_api_family: str = SOURCE_API_FAMILY,
    semantic_binding_sha256: str | None = None,
    artifact_status: str | None = None,
) -> dict[str, Any]:
    summary = {
        "status": status,
        "phase": phase,
        "sourceId": SOURCE_ID,
        "provider": PROVIDER,
        "apiFamily": REQUEST_API_FAMILY,
        "requestApi": REQUEST_API_FAMILY,
        "sourceApiFamily": source_api_family,
        "maximumApiCalls": MAX_API_CALLS,
        "attemptedCalls": len(outcomes),
        "calls": [
            {"model": outcome.model, "mode": outcome.mode}
            for outcome in outcomes
        ],
        "statusCounts": dict(
            sorted(Counter(outcome.status for outcome in outcomes).items())
        ),
        "retryCount": 0,
        "fallbackCount": 0,
        "serial": True,
        "perCallTimeoutSeconds": PER_CALL_TIMEOUT_SECONDS,
        "processTimeoutSeconds": PROCESS_TIMEOUT_SECONDS,
        "maxOutputTokens": MAX_OUTPUT_TOKENS,
    }
    if semantic_binding_sha256 is not None:
        if not _is_digest(semantic_binding_sha256):
            raise RunnerFailure("blocked-artifact")
        summary["semanticBindingSha256"] = semantic_binding_sha256
    if artifact_status is not None:
        summary["artifactStatus"] = artifact_status
    return summary


def _failure_summary_payload(
    failure: RunnerFailure,
    source_api_family: str,
    semantic_binding_sha256: str | None = None,
) -> bytes | None:
    outcomes = tuple(
        outcome for outcome in failure.outcomes if isinstance(outcome, _ChildOutcome)
    )
    if not outcomes:
        return None
    summary = _safe_summary(
        status=failure.status,
        phase=failure.phase or "live",
        outcomes=outcomes,
        source_api_family=source_api_family,
        semantic_binding_sha256=semantic_binding_sha256,
        artifact_status="invalid",
    )
    return _canonical(summary) + b"\n"



def _attach_exception_context(
    error: BaseException, outcomes: Sequence[_ChildOutcome], phase: str
) -> None:
    try:
        setattr(error, "_omp_probe_outcomes", tuple(outcomes))
        setattr(error, "_omp_probe_phase", phase)
    except BaseException:
        pass


def _exception_outcomes(
    error: BaseException, fallback: Sequence[_ChildOutcome] = ()
) -> tuple[_ChildOutcome, ...]:
    values = getattr(error, "_omp_probe_outcomes", None)
    if values is None and isinstance(error, RunnerFailure):
        values = error.outcomes
    if not isinstance(values, (tuple, list)):
        return tuple(fallback)
    captured = tuple(value for value in values if isinstance(value, _ChildOutcome))
    # RunnerFailure defaults to an empty outcome tuple.  That default is not
    # evidence that no calls were attempted: preserve the caller's local
    # reservation ledger when the exception carries no actual rows.
    return captured if captured else tuple(fallback)


def _exception_phase(error: BaseException, fallback: str) -> str:
    value = getattr(error, "_omp_probe_phase", fallback)
    return value if value in {"preflight", "live"} else fallback


def _run_phase(
    source_db: str | os.PathLike[str],
    *,
    phase: str,
    child: Callable[[str, str, str], tuple[dict[str, Any], int | None]] | None = None,
    expected_binding: _Binding | None = None,
) -> tuple[dict[str, Any], list[_ChildOutcome], _Binding]:
    if phase not in {"preflight", "live"}:
        raise RunnerFailure("blocked-invalid-arguments")
    child_runner = _invoke_child if child is None else child
    _assert_child_source_path(source_db, child_runner)
    binding = _read_binding(source_db)
    if expected_binding is not None and not _same_binding(expected_binding, binding):
        raise RunnerFailure("blocked-source-changed")
    outcomes: list[_ChildOutcome] = []

    def stop(status: str) -> NoReturn:
        raise RunnerFailure(status, outcomes=tuple(outcomes), phase=phase)

    def checked_binding() -> _Binding:
        try:
            return _read_binding(source_db)
        except RunnerFailure as failure:
            stop(failure.status)
        except (OSError, TypeError, ValueError, UnicodeError):
            stop("blocked-source-unavailable")
        raise AssertionError("unreachable")


    for model, mode in PROBES:
        current = checked_binding()
        if not _same_binding(binding, current):
            stop("blocked-source-changed")

        # Reserve the call slot before invoking the child.  If launch itself
        # fails, the attempted call still has a closed status-only row.
        attempt_index = len(outcomes)
        placeholder = _failure_result(model, mode, "unusable-worker-launch")
        outcomes.append(
            _ChildOutcome(model, mode, "unusable-worker-launch", placeholder, None)
        )
        result: dict[str, Any] | None = None
        returncode: int | None = None
        failure_status: str | None = None
        try:
            if child is None:
                result, returncode = child_runner(
                    model, mode, phase, binding, source_db
                )  # type: ignore[misc]
            else:
                result, returncode = child_runner(model, mode, phase)
        except RunnerFailure as failure:
            failure_status = (
                failure.status
                if failure.status in _SAFE_STATUSES
                else "unusable-worker-launch"
            )
        except KeyboardInterrupt as interruption:
            failed = _failure_result(model, mode, "interrupted")
            outcomes[attempt_index] = _ChildOutcome(
                model, mode, "interrupted", failed, returncode
            )
            _attach_exception_context(interruption, outcomes, phase)
            raise
        except SystemExit as _exit_signal:
            failed = _failure_result(model, mode, "blocked-runner")
            outcomes[attempt_index] = _ChildOutcome(
                model, mode, "blocked-runner", failed, returncode
            )
            _attach_exception_context(_exit_signal, outcomes, phase)
            raise
        except (OSError, TypeError, ValueError):
            failure_status = "unusable-worker-launch"
        except Exception:
            failure_status = "unusable-worker-launch"

        if failure_status is not None:
            if phase == "live":
                failed = _failure_result(model, mode, failure_status)
            else:
                failed = _baseline(model, mode)
                failed.update(
                    verification_kind="preflight-only",
                    status=failure_status,
                )
            outcomes[attempt_index] = _ChildOutcome(
                model, mode, failure_status, failed, returncode
            )
            if phase != "live" or _terminal_live_failure(failure_status):
                stop("blocked-preflight" if phase == "preflight" else failure_status)
            after = checked_binding()
            if not _same_binding(binding, after):
                stop("blocked-source-changed")
            continue

        if result is None or not _valid_result(result, model, mode, phase):
            failed = _failure_result(model, mode, "unusable-worker-output")
            outcomes[attempt_index] = _ChildOutcome(
                model, mode, "unusable-worker-output", failed, returncode
            )
            if phase != "live":
                stop("blocked-preflight")
            after = checked_binding()
            if not _same_binding(binding, after):
                stop("blocked-source-changed")
            continue

        status = str(result["status"])
        nonzero_exit = (
            isinstance(returncode, int)
            and not isinstance(returncode, bool)
            and returncode != 0
        )
        worker_crash_exit = (
            isinstance(returncode, int)
            and not isinstance(returncode, bool)
            and returncode < 0
        )
        if status == "unusable-worker-crash" or worker_crash_exit or (
            nonzero_exit and status in {"usable", "preflight-ok"}
        ):
            if phase == "live":
                failed = _failure_result(model, mode, "unusable-worker-crash")
            else:
                failed = _baseline(model, mode)
                failed.update(
                    verification_kind="preflight-only",
                    status="unusable-worker-crash",
                )
            outcomes[attempt_index] = _ChildOutcome(
                model, mode, "unusable-worker-crash", failed, returncode
            )
            stop("unusable-worker-crash")
        outcome = _ChildOutcome(model, mode, status, result, returncode)
        outcomes[attempt_index] = outcome
        if phase == "preflight" and status != "preflight-ok":
            stop("unusable-worker-crash" if status == "unusable-worker-crash" else "blocked-preflight")
        if (
            phase == "live"
            and not _successful_live_result(result, model, mode)
            and _terminal_live_failure(str(result["status"]))
        ):
            stop(str(result["status"]))
        after = checked_binding()
        if not _same_binding(binding, after):
            stop("blocked-source-changed")
    phase_status = (
        "preflight-ok"
        if phase == "preflight"
        else (
            "complete"
            if all(
                _successful_live_result(outcome.result, outcome.model, outcome.mode)
                for outcome in outcomes
            )
            else "partial"
        )
    )
    return (
        _safe_summary(
            status=phase_status,
            phase=phase,
            outcomes=outcomes,
            source_api_family=binding.source_api_family,
            semantic_binding_sha256=binding.semantic_binding_sha256,
        ),
        outcomes,
        binding,
    )


def run_preflight(
    source_db: str | os.PathLike[str] = SOURCE_DB,
    *,
    child: Callable[[str, str, str], tuple[dict[str, Any], int | None]] | None = None,
) -> dict[str, Any]:
    """Validate all ten exact routes without a network request or output file."""
    summary, _outcomes, _binding = _run_phase(source_db, phase="preflight", child=child)
    return summary


def run_authorized_probes(
    source_db: str | os.PathLike[str] = SOURCE_DB,
    output: str | os.PathLike[str] = DEFAULT_OUTPUT,
    *,
    child: Callable[[str, str, str], tuple[dict[str, Any], int | None]] | None = None,
) -> dict[str, Any]:
    """Reserve once, run text then tool for each model, and publish one artifact."""
    output_path = Path(output)
    reservation: _Reservation | None = None
    private_output: _PrivateOutput | None = None
    outcomes: list[_ChildOutcome] = []
    initial: _Binding | None = None

    def adopt_context(error: BaseException) -> tuple[_ChildOutcome, ...]:
        captured = _exception_outcomes(error, outcomes)
        outcomes[:] = list(captured)
        if isinstance(error, RunnerFailure):
            error.outcomes = captured
            error.phase = error.phase or "live"
        return captured

    def publish_failure(failure: RunnerFailure) -> None:
        if reservation is None or failure.status in {
            "blocked-output-directory",
            "blocked-output-exists",
            "blocked-output-path",
        }:
            return
        source_api_family = (
            initial.source_api_family
            if isinstance(initial, _Binding)
            else SOURCE_API_FAMILY
        )
        semantic_binding_sha256 = (
            initial.semantic_binding_sha256
            if isinstance(initial, _Binding)
            else None
        )
        payload = _failure_summary_payload(
            failure,
            source_api_family,
            semantic_binding_sha256,
        )
        if payload is None:
            return
        _replace_output(output_path, reservation, payload, close=False)
        _verify_output_payload(output_path, reservation, payload)

    def cleanup_failed_publication() -> None:
        if reservation is None:
            return
        try:
            _invalidate_published_output(output_path, reservation)
        except BaseException:
            pass

    try:
        initial = _read_binding(source_db)
        reservation = _reserve_output(output_path)
        summary, outcomes, binding = _run_phase(
            source_db,
            phase="live",
            child=child,
            expected_binding=initial,
        )
        observations = [
            {
                "sourceId": SOURCE_ID,
                "apiFamily": REQUEST_API_FAMILY,
                "configuredModelValue": outcome.model,
                "model": outcome.model,
                "observedModel": outcome.result["response_model_id"],
                "mode": outcome.mode,
                "identityStatus": outcome.result["identity_status"],
                "status": outcome.result["status"],
                "toolResult": outcome.result["tool_result"],
                "tokenUsage": outcome.result["token_usage"],
                "latencyMs": outcome.result["latency_ms"],
                "usable": _successful_live_result(
                    outcome.result, outcome.model, outcome.mode
                ),
            }
            for outcome in outcomes
        ]
        artifact = build_observation_artifact(
            semantic_binding_sha256=binding.semantic_binding_sha256,
            observations=observations,
            source_api_family=binding.source_api_family,
        )
        payload = _artifact_bytes(artifact)
        # The final path remains the reservation marker until this private
        # sibling has been fully written and fsynced.
        private_output = _create_private_output(output_path)
        _write_private_output(private_output, payload)
        try:
            final_binding = _read_binding(source_db)
        except RunnerFailure as error:
            raise RunnerFailure(
                error.status, outcomes=tuple(outcomes), phase="live"
            ) from None
        except BaseException:
            raise RunnerFailure(
                "blocked-source-changed", outcomes=tuple(outcomes), phase="live"
            ) from None
        if (
            not _same_binding(initial, binding)
            or not _same_binding(binding, final_binding)
        ):
            raise RunnerFailure(
                "blocked-source-changed", outcomes=tuple(outcomes), phase="live"
            )
        binding = final_binding
        _publish_private_output(output_path, reservation, private_output, payload)
        try:
            published_binding = _read_binding(source_db)
        except RunnerFailure as error:
            raise RunnerFailure(
                error.status, outcomes=tuple(outcomes), phase="live"
            ) from None
        except BaseException:
            raise RunnerFailure(
                "blocked-source-changed", outcomes=tuple(outcomes), phase="live"
            ) from None
        if not _same_binding(binding, published_binding):
            raise RunnerFailure(
                "blocked-source-changed", outcomes=tuple(outcomes), phase="live"
            )
        _close_reservation(reservation)
        reservation = None
        return {
            **summary,
            "artifactSha256": artifact["artifactSha256"],
            "outputMode": "0600",
        }
    except KeyboardInterrupt as interruption:
        captured = adopt_context(interruption)
        failure = RunnerFailure(
            "interrupted", outcomes=captured, phase="live"
        )
        try:
            publish_failure(failure)
        except BaseException:
            cleanup_failed_publication()
        _attach_exception_context(interruption, captured, "live")
        raise
    except SystemExit as exit_signal:
        captured = adopt_context(exit_signal)
        failure = RunnerFailure(
            "blocked-runner", outcomes=captured, phase="live"
        )
        try:
            publish_failure(failure)
        except BaseException:
            cleanup_failed_publication()
        _attach_exception_context(exit_signal, captured, "live")
        raise
    except RunnerFailure as failure:
        captured = adopt_context(failure)
        try:
            publish_failure(failure)
        except BaseException:
            cleanup_failed_publication()
            raise RunnerFailure(
                "blocked-output-write", outcomes=captured, phase="live"
            ) from None
        raise
    except BaseException as error:
        captured = adopt_context(error)
        failure = RunnerFailure(
            "blocked-runner", outcomes=captured, phase="live"
        )
        try:
            publish_failure(failure)
        except BaseException:
            cleanup_failed_publication()
        raise failure from None
    finally:
        _close_private_output(private_output, unlink=True)
        _close_reservation(reservation)


def _parser() -> argparse.ArgumentParser:
    parser = _JsonArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--source-db", default=str(SOURCE_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--preflight-only", action="store_true", dest="preflight_only")
    actions.add_argument(
        "--execute-authorized-probes", action="store_true", dest="execute"
    )
    parser.set_defaults(preflight_only=False, execute=False)
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
        _require_trusted_runtime()
        if args.execute:
            summary = run_authorized_probes(args.source_db, args.output)
        else:
            summary = run_preflight(args.source_db)
        _emit(summary)
        return 0
    except RunnerFailure as failure:
        outcomes = _exception_outcomes(failure, failure.outcomes)
        failure_phase = _exception_phase(failure, failure.phase or phase)
        _emit(
            _safe_summary(
                status=failure.status,
                phase=failure_phase,
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 3 if failure.status == "interrupted" else 4
    except KeyboardInterrupt as interruption:
        outcomes = _exception_outcomes(interruption)
        _emit(
            _safe_summary(
                status="interrupted",
                phase=_exception_phase(interruption, phase),
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 3
    except SystemExit as _exit_signal:
        outcomes = _exception_outcomes(_exit_signal)
        _emit(
            _safe_summary(
                status="blocked-runner",
                phase=_exception_phase(_exit_signal, phase),
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 4
    except (OSError, TypeError, ValueError, UnicodeError) as error:
        outcomes = _exception_outcomes(error)
        _emit(
            _safe_summary(
                status="blocked-runner",
                phase=_exception_phase(error, phase),
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 4
    except Exception as error:
        outcomes = _exception_outcomes(error)
        _emit(
            _safe_summary(
                status="blocked-runner",
                phase=_exception_phase(error, phase),
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 4
    except BaseException as error:
        outcomes = _exception_outcomes(error)
        _emit(
            _safe_summary(
                status="blocked-runner",
                phase=_exception_phase(error, phase),
                outcomes=outcomes,
                artifact_status="invalid" if outcomes else None,
            )
        )
        return 4
if __name__ == "__main__":  # pragma: no cover - exercised by CLI process.
    raise SystemExit(main())
