"""Strict OMP 18.2.0 model-document validation for the CC Switch projection."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit, urlunsplit


__all__ = ["ProjectionPlanError"]


_MAX_JSON_DEPTH = 32
_MAX_IDENTIFIER_LENGTH = 128

_OMP_APIS = frozenset({
    "anthropic-messages",
    "azure-openai-responses",
    "bedrock-converse-stream",
    "google-gemini-cli",
    "google-generative-ai",
    "google-vertex",
    "openai-codex-responses",
    "openai-completions",
    "openai-responses",
})
_OMP_MODEL_FIELDS = frozenset({
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
    "remoteCompaction",
    "supportsTools",
    "thinking",
    "tokenizer",
})
_OMP_MODEL_OVERRIDE_FIELDS = _OMP_MODEL_FIELDS - frozenset({"api", "baseUrl", "id"})
_OMP_PROVIDER_FIELDS = frozenset({
    "api",
    "apiKey",
    "auth",
    "authHeader",
    "baseUrl",
    "compat",
    "disableStrictTools",
    "discovery",
    "guardrailIdentifier",
    "guardrailTrace",
    "guardrailVersion",
    "headers",
    "modelOverrides",
    "models",
    "remoteCompaction",
    "requestMetadata",
    "transport",
})
_COMPAT_BOOLEAN_FIELDS = frozenset({
    "allowAnthropicHeaderOverrides",
    "allowsSyntheticReasoningContentForToolCalls",
    "alwaysSendMaxTokens",
    "disableReasoningOnForcedToolChoice",
    "disableReasoningOnToolChoice",
    "qwenTemplateReasoningEffort",
    "replayUnsignedThinking",
    "requiresAssistantAfterToolResult",
    "requiresAssistantContentForToolCalls",
    "requiresMistralToolIds",
    "requiresReasoningContentForToolCalls",
    "requiresThinkingAsText",
    "requiresToolResultId",
    "requiresToolResultName",
    "strictResponsesPairing",
    "stripImageInput",
    "supportsContextManagement",
    "supportsDeveloperRole",
    "supportsEagerToolInputStreaming",
    "supportsForcedToolChoice",
    "supportsImageDetailOriginal",
    "supportsLongPromptCacheRetention",
    "supportsMultipleSystemMessages",
    "supportsReasoningEffort",
    "supportsReasoningParams",
    "supportsReasoningSummary",
    "supportsStore",
    "supportsStrictMode",
    "supportsToolChoice",
    "supportsUsageInStreaming",
})
_COMPAT_FIELDS = _COMPAT_BOOLEAN_FIELDS | frozenset({
    "cacheControlFormat",
    "extraBody",
    "maxTokensField",
    "openRouterRouting",
    "promptCacheMaximumCheckpoints",
    "promptCacheMinimumTokens",
    "promptCacheMode",
    "reasoningContentField",
    "reasoningEffortMap",
    "streamIdleTimeoutMs",
    "streamMarkupHealingPattern",
    "thinkingFormat",
    "toolStrictMode",
    "vercelGatewayRouting",
    "whenThinking",
})
_COMPAT_WHEN_THINKING_FIELDS = _COMPAT_FIELDS - frozenset({
    "promptCacheMaximumCheckpoints",
    "promptCacheMinimumTokens",
    "promptCacheMode",
    "whenThinking",
})
_REMOTE_COMPACTION_FIELDS = frozenset({
    "api",
    "enabled",
    "endpoint",
    "model",
    "streamingEndpoint",
    "v2Endpoint",
    "v2StreamingEnabled",
})
_TOKENIZERS = frozenset({
    "claude-v3",
    "claude-v47",
    "claude-v5",
    "claude-v5-sonnet",
    "deepseek-v3",
    "glm5",
    "kimi-k2",
    "qwen3",
})
_GLOB_SELECTOR_CHARACTERS = frozenset("*?[]{}!\\")
_THINKING_MODES = frozenset({
    "effort",
    "budget",
    "google-level",
    "anthropic-adaptive",
    "anthropic-budget-effort",
})
_THINKING_EFFORTS = frozenset({"minimal", "low", "medium", "high", "xhigh", "max"})
_EFFORT_ORDER = ("minimal", "low", "medium", "high", "xhigh", "max")
_THINKING_FIELDS = frozenset({
    "mode",
    "efforts",
    "defaultLevel",
    "effortMap",
    "supportsDisplay",
    "requiresEffort",
    "minLevel",
    "maxLevel",
    "levels",
})


class ProjectionPlanError(ValueError):
    """A projection input or transaction failed a safety gate."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _reject(code: str) -> None:
    raise ProjectionPlanError(code)


def require_exact_keys(value: Any, keys: set[str], code: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        _reject(code)


def require_exact_keys_subset(
    value: Any, keys: set[str] | frozenset[str], code: str
) -> None:
    if not isinstance(value, dict) or not set(value).issubset(keys):
        _reject(code)


def canonical_https_url(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(ord(character) < 32 for character in value)
    ):
        _reject(code)
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except (TypeError, ValueError):
        _reject(code)
    before_fragment = value.split("#", 1)[0]
    if (
        parsed.scheme.casefold() != "https"
        or not parsed.netloc
        or "@" in parsed.netloc
        or "?" in before_fragment
        or "#" in value
        or parsed.hostname is None
        or (port is not None and port <= 0)
    ):
        _reject(code)
    hostname = parsed.hostname.casefold()
    host = f"[{hostname}]" if ":" in hostname else hostname
    if port is not None and port != 443:
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") if parsed.path != "/" else ""
    return urlunsplit(("https", host, path, "", ""))


def require_provider_id(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(
            character in _GLOB_SELECTOR_CHARACTERS or character.isspace()
            for character in value
        )
        or value[0]
        not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:-"
            for character in value
        )
    ):
        _reject(code)
    return value


def require_model_id(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or any(
            character in _GLOB_SELECTOR_CHARACTERS or character.isspace()
            for character in value
        )
        or value[0]
        not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        or any(
            character
            not in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._:/-"
            for character in value
        )
        or value.endswith("/")
        or any(segment in {"", ".", ".."} for segment in value.split("/"))
    ):
        _reject(code)
    return value


def require_env_reference(value: Any, code: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_IDENTIFIER_LENGTH
        or value[0] not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        or any(
            character not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_"
            for character in value
        )
    ):
        _reject(code)
    return value


def require_nonempty_string(value: Any, code: str) -> str:
    if not isinstance(value, str) or not value:
        _reject(code)
    return value


def validate_id_list(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _reject(code)
    items = [require_provider_id(item, code) for item in value]
    if items != sorted(items) or len(set(items)) != len(items):
        _reject(code)
    return items


def validate_model_id_list(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _reject(code)
    items = [require_model_id(item, code) for item in value]
    if items != sorted(items) or len(set(items)) != len(items):
        _reject(code)
    return items


def validate_selected_models(value: Any, code: str) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        _reject(code)
    items = [require_model_id(item, code) for item in value]
    if not items or len(set(items)) != len(items):
        _reject(code)
    return sorted(items)


def validate_distinct_model_id_list(value: Any, code: str) -> list[str]:
    if not isinstance(value, list):
        _reject(code)
    items = [require_model_id(item, code) for item in value]
    if len(set(items)) != len(items):
        _reject(code)
    return items


def require_finite_number(value: Any, code: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _reject(code)
    if isinstance(value, float) and not math.isfinite(value):
        _reject(code)


def _require_nonnegative_number(value: Any, code: str) -> None:
    require_finite_number(value, code)
    if value < 0:
        _reject(code)


def _require_positive_number(value: Any, code: str) -> None:
    require_finite_number(value, code)
    if value <= 0:
        _reject(code)


def is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def ensure_json_value(
    value: Any, code: str, depth: int = 0, seen: set[int] | None = None
) -> None:
    if depth > _MAX_JSON_DEPTH:
        _reject(code)
    if value is None or isinstance(value, (str, bool, int)):
        if isinstance(value, str) and len(value) > 65536:
            _reject(code)
        return
    if isinstance(value, float):
        require_finite_number(value, code)
        return
    if seen is None:
        seen = set()
    if isinstance(value, (list, dict)):
        identity = id(value)
        if identity in seen:
            _reject(code)
        seen.add(identity)
        try:
            if isinstance(value, list):
                for item in value:
                    ensure_json_value(item, code, depth + 1, seen)
            else:
                for key, item in value.items():
                    if not isinstance(key, str) or len(key) > 8192:
                        _reject(code)
                    ensure_json_value(item, code, depth + 1, seen)
        finally:
            seen.remove(identity)
        return
    _reject(code)


def canonical_json(value: Any, code: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError, UnicodeError):
        _reject(code)


def json_clone(value: Any, code: str) -> dict[str, Any]:
    ensure_json_value(value, code)
    try:
        cloned = json.loads(canonical_json(value, code).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        _reject(code)
    if not isinstance(cloned, dict):
        _reject(code)
    return cloned


def document_sha256(document: Mapping[str, Any]) -> str:
    return json_sha256(document)


def json_sha256(value: Any, code: str = "blocked-target-parse") -> str:
    return hashlib.sha256(canonical_json(value, code)).hexdigest()


def validate_document(document: Any, code: str) -> dict[str, Any]:
    require_exact_keys(document, {"providers"}, code)
    providers = document["providers"]
    if not isinstance(providers, dict):
        _reject(code)
    for provider_name, provider in providers.items():
        require_provider_id(provider_name, code)
        validate_provider(provider, code)
    return document


def validate_provider(provider: Any, code: str) -> dict[str, Any]:
    if not isinstance(provider, dict) or not set(provider).issubset(
        _OMP_PROVIDER_FIELDS
    ):
        _reject(code)
    for key, value in provider.items():
        validator = _PROVIDER_FIELD_VALIDATORS.get(key)
        if validator is None:
            _reject(code)
        validator(value, code)
    _validate_provider_configuration(provider, code)
    return provider


def _validate_provider_configuration(provider: Mapping[str, Any], code: str) -> None:
    models = provider.get("models", [])
    if not isinstance(models, list):
        _reject(code)
    overrides = provider.get("modelOverrides", {})
    if not isinstance(overrides, dict):
        _reject(code)
    if not models:
        has_configuration = (
            "baseUrl" in provider
            or "headers" in provider
            or "compat" in provider
            or "apiKey" in provider
            or provider.get("auth") == "none"
            or provider.get("disableStrictTools") is True
            or "guardrailIdentifier" in provider
            or "requestMetadata" in provider
            or "remoteCompaction" in provider
            or bool(overrides)
            or "discovery" in provider
        )
        if not has_configuration:
            _reject(code)
    else:
        if "baseUrl" not in provider:
            _reject(code)
        auth = provider.get("auth", "apiKey")
        if "apiKey" not in provider and auth not in {"none", "oauth"}:
            _reject(code)
        for model in models:
            if "api" not in provider and "api" not in model:
                _reject(code)
    discovery = provider.get("discovery")
    if (
        isinstance(discovery, dict)
        and "api" not in provider
        and discovery.get("type") != "proxy"
    ):
        _reject(code)


def effective_model_index(
    provider: Mapping[str, Any], code: str
) -> dict[str, dict[str, Any]]:
    models = model_index(provider.get("models", []), code)
    overrides = provider.get("modelOverrides", {})
    if not isinstance(overrides, dict):
        _reject(code)
    effective: dict[str, dict[str, Any]] = {}
    for model_id, model in models.items():
        override = overrides.get(model_id)
        effective[model_id] = (
            merge_model_override(model, override, code)
            if override is not None
            else dict(model)
        )
    return effective


def merge_model_override(
    model: Mapping[str, Any], override: Mapping[str, Any], code: str
) -> dict[str, Any]:
    validate_model_override(override, code)
    merged = dict(model)
    for key, value in override.items():
        merger = _OVERRIDE_MERGERS.get(key)
        merged[key] = merger(merged.get(key), value, code) if merger else value
    return merged


def _merge_shallow_mapping(base: Any, override: Any, code: str) -> dict[str, Any]:
    if base is None:
        base = {}
    if not isinstance(base, dict) or not isinstance(override, dict):
        _reject(code)
    return {**base, **override}


def _merge_compat(base: Any, override: Any, code: str) -> dict[str, Any]:
    if base is None:
        if not isinstance(override, dict):
            _reject(code)
        return dict(override)
    if not isinstance(base, dict) or not isinstance(override, dict):
        _reject(code)
    return _merge_nested_mappings(base, override)


def _merge_nested_mappings(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        previous = base.get(key)
        merged[key] = (
            _merge_nested_mappings(previous, value)
            if isinstance(previous, dict) and isinstance(value, dict)
            else value
        )
    return merged


def model_index(models: Any, code: str) -> dict[str, dict[str, Any]]:
    if not isinstance(models, list):
        _reject(code)
    indexed: dict[str, dict[str, Any]] = {}
    for model in models:
        validated = validate_model(model, code)
        model_id = validated["id"]
        if model_id in indexed:
            _reject(code)
        indexed[model_id] = validated
    return indexed


def validate_model(value: Any, code: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not set(value).issubset(_OMP_MODEL_FIELDS)
        or "id" not in value
    ):
        _reject(code)
    require_model_id(value["id"], code)
    _validate_model_fields(value, code, override=False)
    return value


def validate_model_override(value: Any, code: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not set(value).issubset(
        _OMP_MODEL_OVERRIDE_FIELDS
    ):
        _reject(code)
    _validate_model_fields(value, code, override=True)
    return value


def _validate_model_fields(
    value: Mapping[str, Any], code: str, *, override: bool
) -> None:
    validators = (
        _MODEL_OVERRIDE_FIELD_VALIDATORS if override else _MODEL_FIELD_VALIDATORS
    )
    for key, item in value.items():
        if key == "id":
            continue
        validator = validators.get(key)
        if validator is None:
            _reject(code)
        validator(item, code)


def _validate_cost(value: Any, code: str, *, partial: bool = False) -> None:
    keys = {"cacheRead", "cacheWrite", "input", "output"}
    if partial:
        require_exact_keys_subset(value, keys, code)
    else:
        require_exact_keys(value, keys, code)
    for item in value.values():
        require_finite_number(item, code)


def _validate_partial_cost(value: Any, code: str) -> None:
    _validate_cost(value, code, partial=True)


def _validate_remote_compaction(value: Any, code: str) -> None:
    require_exact_keys_subset(value, _REMOTE_COMPACTION_FIELDS, code)
    for key, item in value.items():
        validator = _REMOTE_COMPACTION_VALIDATORS.get(key)
        if validator is None:
            _reject(code)
        validator(item, code)


def _validate_thinking(value: Any, code: str) -> None:
    require_exact_keys_subset(value, _THINKING_FIELDS, code)
    if value.get("mode") not in _THINKING_MODES:
        _reject(code)
    if not (
        "efforts" in value
        or "levels" in value
        or ("minLevel" in value and "maxLevel" in value)
    ):
        _reject(code)
    for key, item in value.items():
        if key == "mode":
            continue
        validator = _THINKING_FIELD_VALIDATORS.get(key)
        if validator is None:
            _reject(code)
        validator(item, code)


def _validate_efforts(value: Any, code: str) -> None:
    if not isinstance(value, list) or any(
        item not in _THINKING_EFFORTS for item in value
    ):
        _reject(code)


def _validate_effort(value: Any, code: str) -> None:
    if value not in _THINKING_EFFORTS:
        _reject(code)


def _validate_effort_map(value: Any, code: str) -> None:
    require_exact_keys_subset(value, _THINKING_EFFORTS, code)
    if any(not isinstance(item, str) for item in value.values()):
        _reject(code)


def _validate_discovery(value: Any, code: str) -> None:
    require_exact_keys_subset(value, {"injectV1", "timeoutMs", "type"}, code)
    discovery_type = value.get("type")
    if discovery_type not in {
        "litellm",
        "llama.cpp",
        "lm-studio",
        "ollama",
        "openai-models-list",
        "proxy",
    }:
        _reject(code)
    if "injectV1" in value and (
        type(value["injectV1"]) is not bool or discovery_type != "openai-models-list"
    ):
        _reject(code)
    if "timeoutMs" in value:
        _require_positive_number(value["timeoutMs"], code)


def _validate_compat(value: Any, code: str, *, when_thinking: bool = False) -> None:
    allowed = _COMPAT_WHEN_THINKING_FIELDS if when_thinking else _COMPAT_FIELDS
    require_exact_keys_subset(value, allowed, code)
    for key, item in value.items():
        validator = _COMPAT_VALUE_VALIDATORS.get(key)
        if validator is None:
            _reject(code)
        validator(item, code)


def _validate_when_thinking(value: Any, code: str) -> None:
    _validate_compat(value, code, when_thinking=True)


def _validate_extra_body(value: Any, code: str) -> None:
    if not isinstance(value, dict):
        _reject(code)
    ensure_json_value(value, code)


def _validate_routing(value: Any, code: str) -> None:
    require_exact_keys_subset(value, {"only", "order"}, code)
    for item in value.values():
        if not isinstance(item, list) or any(
            not isinstance(entry, str) for entry in item
        ):
            _reject(code)


def _validate_string_mapping(value: Any, code: str) -> None:
    if not isinstance(value, dict):
        _reject(code)
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        _reject(code)


def _validate_api(value: Any, code: str) -> None:
    if value not in _OMP_APIS:
        _reject(code)


def _validate_bool(value: Any, code: str) -> None:
    if type(value) is not bool:
        _reject(code)


def _validate_input(value: Any, code: str) -> None:
    if not isinstance(value, list) or any(
        entry not in {"text", "image"} for entry in value
    ):
        _reject(code)


def _validate_tokenizer(value: Any, code: str) -> None:
    if value not in _TOKENIZERS:
        _reject(code)


def _validate_image_decoder(value: Any, code: str) -> None:
    if value != "stb":
        _reject(code)


def _validate_provider_auth(value: Any, code: str) -> None:
    if value not in {"apiKey", "none", "oauth"}:
        _reject(code)


def _validate_guardrail_trace(value: Any, code: str) -> None:
    if value not in {"disabled", "enabled", "enabled_full"}:
        _reject(code)


def _validate_transport(value: Any, code: str) -> None:
    if value != "pi-native":
        _reject(code)


def _validate_cache_control_format(value: Any, code: str) -> None:
    if value != "anthropic":
        _reject(code)


def _validate_max_tokens_field(value: Any, code: str) -> None:
    if value not in {"max_completion_tokens", "max_tokens"}:
        _reject(code)


def _validate_prompt_cache_mode(value: Any, code: str) -> None:
    if value not in {"automatic", "explicit", "none"}:
        _reject(code)


def _validate_reasoning_content_field(value: Any, code: str) -> None:
    if value not in {"reasoning", "reasoning_content", "reasoning_text"}:
        _reject(code)


def _validate_stream_markup_pattern(value: Any, code: str) -> None:
    if value not in {"dsml", "kimi", "qwen", "thinking"}:
        _reject(code)


def _validate_thinking_format(value: Any, code: str) -> None:
    if value not in {"openai", "openrouter", "qwen", "qwen-chat-template", "zai"}:
        _reject(code)


def _validate_tool_strict_mode(value: Any, code: str) -> None:
    if value not in {"all_strict", "none"}:
        _reject(code)


def _validate_model_overrides(value: Any, code: str) -> None:
    if not isinstance(value, dict):
        _reject(code)
    for model_id, override in value.items():
        require_model_id(model_id, code)
        validate_model_override(override, code)


def _validate_models(value: Any, code: str) -> None:
    model_index(value, code)


Validator = Callable[[Any, str], None]


_REMOTE_COMPACTION_VALIDATORS: dict[str, Validator] = {
    "api": _validate_api,
    "enabled": _validate_bool,
    "endpoint": require_nonempty_string,
    "model": require_nonempty_string,
    "streamingEndpoint": require_nonempty_string,
    "v2Endpoint": require_nonempty_string,
    "v2StreamingEnabled": _validate_bool,
}
_THINKING_FIELD_VALIDATORS: dict[str, Validator] = {
    "defaultLevel": _validate_effort,
    "effortMap": _validate_effort_map,
    "efforts": _validate_efforts,
    "levels": _validate_efforts,
    "maxLevel": _validate_effort,
    "minLevel": _validate_effort,
    "requiresEffort": _validate_bool,
    "supportsDisplay": _validate_bool,
}
_COMPAT_VALUE_VALIDATORS: dict[str, Validator] = {
    **{name: _validate_bool for name in _COMPAT_BOOLEAN_FIELDS},
    "cacheControlFormat": _validate_cache_control_format,
    "extraBody": _validate_extra_body,
    "maxTokensField": _validate_max_tokens_field,
    "openRouterRouting": _validate_routing,
    "promptCacheMaximumCheckpoints": _require_nonnegative_number,
    "promptCacheMinimumTokens": _require_nonnegative_number,
    "promptCacheMode": _validate_prompt_cache_mode,
    "reasoningContentField": _validate_reasoning_content_field,
    "reasoningEffortMap": _validate_effort_map,
    "streamIdleTimeoutMs": _require_nonnegative_number,
    "streamMarkupHealingPattern": _validate_stream_markup_pattern,
    "thinkingFormat": _validate_thinking_format,
    "toolStrictMode": _validate_tool_strict_mode,
    "vercelGatewayRouting": _validate_routing,
    "whenThinking": _validate_when_thinking,
}
_MODEL_FIELD_VALIDATORS: dict[str, Validator] = {
    "api": _validate_api,
    "baseUrl": require_nonempty_string,
    "compactionModel": require_nonempty_string,
    "compat": _validate_compat,
    "contextPromotionTarget": require_nonempty_string,
    "contextWindow": _require_positive_number,
    "cost": _validate_cost,
    "headers": _validate_string_mapping,
    "imageInputDecoder": _validate_image_decoder,
    "input": _validate_input,
    "maxTokens": _require_positive_number,
    "name": require_nonempty_string,
    "omitMaxOutputTokens": _validate_bool,
    "preferWebsockets": _validate_bool,
    "premiumMultiplier": require_finite_number,
    "reasoning": _validate_bool,
    "remoteCompaction": _validate_remote_compaction,
    "supportsTools": _validate_bool,
    "thinking": _validate_thinking,
    "tokenizer": _validate_tokenizer,
}
_MODEL_OVERRIDE_FIELD_VALIDATORS: dict[str, Validator] = {
    **_MODEL_FIELD_VALIDATORS,
    "contextWindow": require_finite_number,
    "cost": _validate_partial_cost,
    "maxTokens": require_finite_number,
}
_PROVIDER_FIELD_VALIDATORS: dict[str, Validator] = {
    "api": _validate_api,
    "apiKey": require_nonempty_string,
    "auth": _validate_provider_auth,
    "authHeader": _validate_bool,
    "baseUrl": require_nonempty_string,
    "compat": _validate_compat,
    "disableStrictTools": _validate_bool,
    "discovery": _validate_discovery,
    "guardrailIdentifier": require_nonempty_string,
    "guardrailTrace": _validate_guardrail_trace,
    "guardrailVersion": require_nonempty_string,
    "headers": _validate_string_mapping,
    "modelOverrides": _validate_model_overrides,
    "models": _validate_models,
    "remoteCompaction": _validate_remote_compaction,
    "requestMetadata": _validate_string_mapping,
    "transport": _validate_transport,
}
_OVERRIDE_MERGERS: dict[str, Callable[[Any, Any, str], dict[str, Any]]] = {
    "compat": _merge_compat,
    "cost": _merge_shallow_mapping,
    "headers": _merge_shallow_mapping,
    "remoteCompaction": _merge_shallow_mapping,
}


def normalized_thinking(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    if "efforts" in value:
        efforts = value["efforts"]
    elif "levels" in value:
        efforts = value["levels"]
    else:
        minimum = _EFFORT_ORDER.index(value["minLevel"])
        maximum = _EFFORT_ORDER.index(value["maxLevel"])
        efforts = list(_EFFORT_ORDER[minimum : max(minimum, maximum) + 1])
    normalized = {"mode": value["mode"], "efforts": efforts}
    for field in {"defaultLevel", "effortMap", "requiresEffort", "supportsDisplay"}:
        if field in value:
            normalized[field] = value[field]
    return normalized


def source_model_matches(source: Mapping[str, Any], target: Mapping[str, Any]) -> bool:
    """A source may declare only ``id``; every declared model field must match."""
    for field, value in source.items():
        normalizer = _MODEL_FIELD_NORMALIZERS.get(field)
        if normalizer:
            if normalizer(target.get(field)) != normalizer(value):
                return False
        elif target.get(field) != value:
            return False
    return True


def assert_selected_provider_transport(provider: Mapping[str, Any], code: str) -> None:
    if (
        provider.get("headers")
        or provider.get("transport") is not None
        or provider.get("authHeader") is True
        or "remoteCompaction" in provider
    ):
        _reject(code)
    compat = provider.get("compat")
    if compat is not None and not harmless_empty_compat(compat):
        _reject(code)


def assert_selected_model_transport(
    source_model: Mapping[str, Any],
    direct_target_model: Mapping[str, Any],
    override: Any,
    effective_target_model: Mapping[str, Any],
    target_provider: Mapping[str, Any],
    expected_api: str,
    source_endpoint: str,
) -> None:
    if override is not None and not isinstance(override, dict):
        _reject("blocked-model-metadata")
    if (
        "remoteCompaction" in direct_target_model
        or "remoteCompaction" in effective_target_model
        or (isinstance(override, dict) and "remoteCompaction" in override)
    ):
        _reject("blocked-model-metadata")
    if (
        source_model.get("headers")
        or direct_target_model.get("headers")
        or effective_target_model.get("headers")
        or (isinstance(override, dict) and override.get("headers"))
    ):
        _reject("blocked-model-metadata")
    for field in {"api", "baseUrl"}:
        if field in source_model:
            if effective_target_model.get(field) != source_model[field]:
                _reject("blocked-model-metadata")
        elif field in direct_target_model or (
            isinstance(override, dict) and field in override
        ):
            _reject("blocked-model-metadata")
    if "compat" in source_model:
        if normalized_compat(effective_target_model.get("compat")) != normalized_compat(
            source_model["compat"]
        ):
            _reject("blocked-model-metadata")
    elif "compat" in direct_target_model or (
        isinstance(override, dict) and "compat" in override
    ):
        if not harmless_empty_compat(effective_target_model.get("compat")):
            _reject("blocked-model-metadata")
    if effective_target_model.get("api", target_provider["api"]) != expected_api:
        _reject("blocked-api-family-mismatch")
    if (
        canonical_https_url(
            effective_target_model.get("baseUrl", target_provider["baseUrl"]),
            "blocked-target-url",
        )
        != source_endpoint
    ):
        _reject("blocked-url-mismatch")


def normalized_compat(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    normalized: dict[str, Any] = {}
    for key, item in value.items():
        if key == "extraBody" and item == {}:
            continue
        normalized[key] = (
            normalized_compat(item)
            if key == "whenThinking" and isinstance(item, dict)
            else item
        )
    return normalized


_MODEL_FIELD_NORMALIZERS: dict[str, Callable[[Any], Any]] = {
    "compat": normalized_compat,
    "thinking": normalized_thinking,
}


def harmless_empty_compat(value: Any) -> bool:
    return normalized_compat(value) == {}


def assert_source_model_secret_free(model: Mapping[str, Any], code: str) -> None:
    if model.get("headers"):
        _reject(code)
    compat = model.get("compat")
    if isinstance(compat, dict) and _contains_nonempty_extra_body(compat):
        _reject(code)


def _contains_nonempty_extra_body(value: Mapping[str, Any]) -> bool:
    if "extraBody" in value and value["extraBody"] != {}:
        return True
    nested = value.get("whenThinking")
    return isinstance(nested, dict) and _contains_nonempty_extra_body(nested)
