"""Contract tests for the read-only, redacted OMP config handoff seam."""

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools import omp_config_handoff
from tools.omp_config_handoff import (
    ConfigHandoffError,
    build_config_handoff,
    build_provider_migration_handoff,
)


DY = "cc-dy-codex"
ACTIVE = "cc-deepseek"
DEEPSEEK = "deepseek"
SOL = f"{DY}/gpt-5.6-sol"
GPT = f"{DY}/gpt-5.6"


def _config() -> dict[str, Any]:
    return {
        "modelProviderOrder": [DEEPSEEK, DY, ACTIVE],
        "enabledModels": ["deepseek/deepseek-chat", SOL, ACTIVE + "/deepseek-chat"],
        "retry": {
            "fallbackChains": {
                "default": [
                    DEEPSEEK + "/deepseek-chat",
                    SOL,
                    ACTIVE + "/deepseek-chat",
                ]
            }
        },
        "modelRoles": {"default": "openai-codex/gpt-5.6-sol", "fallback": SOL},
        "telemetry": {"token": "other-sensitive-value"},
    }


def test_provider_migration_inserts_xuanju_after_dy_without_touching_other_routes(
    tmp_path: Path,
):
    config = _config()
    config["modelProviderOrder"] = ["openai-codex", DY, "tail-provider"]
    config["enabledModels"].insert(1, GPT)
    path, _ = _write_config(tmp_path, config)

    result = build_provider_migration_handoff(path)

    assert result["changes"]["addEnabledModels"] == ["cc-xuanju/GLM-5.3-flash"]
    assert result["changes"]["modelProviderOrder"] == {
        "operation": "insert-after",
        "providers": ["cc-xuanju"],
        "after": DY,
        "result": ["openai-codex", DY, "cc-xuanju", "tail-provider"],
    }
    assert result["changes"]["enabledModels"]["result"] == [
        "deepseek/deepseek-chat",
        GPT,
        SOL,
        ACTIVE + "/deepseek-chat",
        "cc-xuanju/GLM-5.3-flash",
    ]
    assert result["changes"]["preserveFallbackChains"] is True
    assert result["changes"]["preserveModelRoles"] is True


def _write_config(
    tmp_path: Path, config: dict[str, Any] | None = None
) -> tuple[Path, bytes]:
    path = tmp_path / "config.yml"
    raw = yaml.safe_dump(
        config if config is not None else _config(), sort_keys=False
    ).encode()
    path.write_bytes(raw)
    return path, raw


def _write_raw(tmp_path: Path, raw: bytes) -> Path:
    path = tmp_path / "config.yml"
    path.write_bytes(raw)
    return path


def _plan(*selectors: str, remove_active: bool = True) -> dict[str, Any]:
    return {
        "configHandoff": {
            "addModelSelectors": [
                {
                    "provider": selector.split("/", 1)[0],
                    "model": selector.split("/", 1)[1],
                }
                for selector in selectors
            ]
        },
        "removeProviders": [ACTIVE] if remove_active else [],
    }


def _handoff_digest(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "planSha256"}
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _replace_scalar(
    document: dict[str, Any], path: tuple[str | int, ...]
) -> dict[str, Any]:
    tampered = copy.deepcopy(document)
    parent: Any = tampered
    for part in path[:-1]:
        parent = parent[part]
    leaf = parent[path[-1]]
    if isinstance(leaf, str):
        parent[path[-1]] = leaf + "-tampered"
    elif isinstance(leaf, bool):
        parent[path[-1]] = not leaf
    elif isinstance(leaf, int):
        parent[path[-1]] = leaf + 1
    else:
        raise AssertionError(f"unexpected handoff scalar {leaf!r}")
    return tampered


def _scalar_paths(value: Any, prefix: tuple[str | int, ...] = ()):
    if isinstance(value, dict):
        for key, nested in value.items():
            if key != "planSha256":
                yield from _scalar_paths(nested, prefix + (key,))
    elif isinstance(value, list):
        for index, nested in enumerate(value):
            yield from _scalar_paths(nested, prefix + (index,))
    else:
        yield prefix


def test_handoff_appends_only_verified_models_to_existing_enabled_models(
    tmp_path: Path,
):
    path, _ = _write_config(tmp_path)
    result = build_config_handoff(_plan(SOL, GPT), path, verified_selectors={SOL, GPT})
    changes = result["changes"]
    assert changes["addEnabledModels"] == [SOL, GPT]
    assert changes["enabledModels"] == {
        "operation": "append",
        "selectors": [SOL, GPT],
        "result": ["deepseek/deepseek-chat", SOL, GPT],
    }
    assert changes["modelProviderOrder"] == [DEEPSEEK, DY]
    assert changes["fallbackChains"] == {"default": ["deepseek/deepseek-chat", SOL]}
    assert changes["modelRoles"] == {
        "default": "openai-codex/gpt-5.6-sol",
        "fallback": SOL,
    }


@pytest.mark.parametrize("value", [pytest.param("absent", id="absent"), None, []])
def test_handoff_preserves_unrestricted_enabled_models(
    tmp_path: Path, value: str | list[Any] | None
):
    config = _config()
    if value == "absent":
        del config["enabledModels"]
    else:
        config["enabledModels"] = value
    path, raw = _write_config(tmp_path, config)

    result = build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    assert result["changes"]["enabledModels"] == {"operation": "preserve-unrestricted"}
    assert result["changes"]["addEnabledModels"] == []
    assert path.read_bytes() == raw


def test_handoff_rejects_projection_handoff_extensions(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    plan = _plan(SOL)
    plan["configHandoff"]["restrictEnabledModels"] = {
        "authorized": True,
        "selectors": [{"provider": DY, "model": "gpt-5.6-sol"}],
    }

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(plan, path, verified_selectors={SOL})
    assert raised.value.code == "blocked-invalid-plan"


def test_unverified_selector_is_blocked(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors=set())
    assert raised.value.code == "blocked-unverified-model"


@pytest.mark.parametrize(
    "selector",
    [
        f"{DY}[beta]/gpt-5.6",
        f"{DY}/gpt-*",
        f"{DY}/gpt-?",
        f"{DY}/gpt-[5.6",
        f"{DY}/gpt-5.6]",
        f"{DY}/gpt-\\*",
        f"{DY}/gpt/[5].6",
        f"{DY}/family//gpt-5.6",
        f"{DY}/family/../gpt-5.6",
        f"{DY}/gpt-\n5.6",
    ],
)
def test_matcher_metacharacters_or_escape_sequences_are_blocked(
    tmp_path: Path, selector: str
):
    path, _ = _write_config(tmp_path)
    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(selector), path, verified_selectors={selector})
    assert raised.value.code == "blocked-nonliteral-model-selector"


def test_safe_slash_bearing_model_id_is_preserved(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    selector = f"{DY}/catalog/gpt-5.6"

    result = build_config_handoff(_plan(selector), path, verified_selectors={selector})

    assert result["changes"]["enabledModels"]["selectors"] == [selector]
    assert result["changes"]["enabledModels"]["result"][-1] == selector


def test_active_provider_is_removed_from_all_routing_and_role_fields(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    result = build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
    changes = result["changes"]
    assert ACTIVE not in changes["modelProviderOrder"]
    assert all(
        not value.startswith(ACTIVE + "/")
        for value in changes["enabledModels"]["result"]
    )
    assert all(
        not value.startswith(ACTIVE + "/")
        for values in changes["fallbackChains"].values()
        for value in values
    )
    assert changes["modelRoles"]["fallback"] == SOL


def test_active_model_role_is_blocked_even_when_selector_is_verified(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    config = _config()
    config["modelRoles"]["primary"] = ACTIVE + "/deepseek-chat"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
    assert raised.value.code == "blocked-active-role-removal"


def test_removed_provider_cannot_be_requested_again(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    selector = ACTIVE + "/deepseek-chat"

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(selector), path, verified_selectors={selector})
    assert raised.value.code == "blocked-active-provider-selector"


def test_retained_provider_is_not_cleaned_or_rejected_from_roles(tmp_path: Path):
    config = _config()
    config["modelRoles"]["primary"] = ACTIVE + "/deepseek-chat"
    path, _ = _write_config(tmp_path, config)

    result = build_config_handoff(
        _plan(SOL, remove_active=False), path, verified_selectors={SOL}
    )

    changes = result["changes"]
    assert changes["modelProviderOrder"] == [DEEPSEEK, DY, ACTIVE]
    assert changes["enabledModels"]["result"] == [
        "deepseek/deepseek-chat",
        SOL,
        ACTIVE + "/deepseek-chat",
    ]
    assert changes["fallbackChains"]["default"] == [
        "deepseek/deepseek-chat",
        SOL,
        ACTIVE + "/deepseek-chat",
    ]
    assert changes["modelRoles"]["primary"] == ACTIVE + "/deepseek-chat"

@pytest.mark.parametrize(
    ("location", "selector"),
    [
        ("role", "cc-xuanju/GLM-5.3"),
        ("fallback", "cc-xuanju/GLM-5.3"),
        ("role", "CC-XUANJU/Kimi-K3"),
        ("fallback", "CC-XUANJU/Kimi-K3"),
        ("enabled", "CC-XUANJU/Kimi-K3"),
        ("role", "pi/slow"),
        ("fallback", "PI/SLOW"),
    ],
)
def test_provider_migration_rejects_nonflash_xuanju_routes(
    tmp_path: Path, location: str, selector: str
):
    config = _config()
    if location == "role":
        config["modelRoles"]["primary"] = selector
    elif location == "fallback":
        config["retry"]["fallbackChains"]["default"].append(selector)
    else:
        config["enabledModels"].append(selector)
    path, _ = _write_config(tmp_path, config)

    with pytest.raises(ConfigHandoffError) as raised:
        build_provider_migration_handoff(path)
    assert raised.value.code == "blocked-config-migration-state"


@pytest.mark.parametrize(
    ("provider", "expected_code"),
    [
        ("CC-XUANJU", "blocked-config-migration-state"),
        ("CC-DY-CODEX", "blocked-unsafe-config-value"),
    ],
)
def test_provider_migration_rejects_noncanonical_target_provider_order(
    tmp_path: Path, provider: str, expected_code: str
):
    config = _config()
    config["modelProviderOrder"].append(provider)
    path, _ = _write_config(tmp_path, config)

    with pytest.raises(ConfigHandoffError) as raised:
        build_provider_migration_handoff(path)
    assert raised.value.code == expected_code


@pytest.mark.parametrize(
    "disabled_provider",
    ["cc-xuanju", " CC-XUANJU ", "cc-dy-codex", " CC-DY-CODEX "],
)
def test_provider_migration_rejects_disabled_projected_provider(
    tmp_path: Path, disabled_provider: str
):
    config = _config()
    config["disabledProviders"] = ["other", disabled_provider]
    path, _ = _write_config(tmp_path, config)

    with pytest.raises(ConfigHandoffError) as raised:
        build_provider_migration_handoff(path)
    assert raised.value.code == "blocked-config-migration-state"


def test_provider_migration_rejects_malformed_disabled_providers(
    tmp_path: Path,
):
    config = _config()
    config["disabledProviders"] = {"cc-xuanju": True}
    path, _ = _write_config(tmp_path, config)

    with pytest.raises(ConfigHandoffError) as raised:
        build_provider_migration_handoff(path)
    assert raised.value.code == "blocked-unsafe-config-value"

def test_provider_migration_preserves_existing_flash_routes(
    tmp_path: Path,
):
    config = _config()
    config["enabledModels"].insert(0, GPT)
    flash = "cc-xuanju/GLM-5.3-flash"
    config["modelRoles"]["primary"] = flash
    config["retry"]["fallbackChains"]["default"].append(flash)
    path, raw = _write_config(tmp_path, config)

    result = build_config_handoff(
        _plan(SOL), path, verified_selectors={SOL}
    )
    changes = result["changes"]
    assert changes["modelRoles"]["primary"] == flash
    assert changes["fallbackChains"]["default"][-1] == flash

    rendered = omp_config_handoff.render_provider_migration_config(
        raw, build_provider_migration_handoff(path)["changes"]
    )
    rendered_config = yaml.safe_load(rendered)
    assert rendered_config["modelRoles"]["primary"] == flash
    assert rendered_config["retry"]["fallbackChains"]["default"][-1] == flash


def test_provider_migration_accepts_omp_casefolded_flash_routes(
    tmp_path: Path,
):
    config = _config()
    config["enabledModels"].insert(0, GPT)
    flash = "CC-XUANJU/glm-5.3-FLASH"
    config["enabledModels"].append(flash)
    config["modelRoles"]["primary"] = flash
    config["retry"]["fallbackChains"]["default"].append(flash)
    path, _ = _write_config(tmp_path, config)
    result = build_provider_migration_handoff(path)

    assert result["changes"]["addEnabledModels"] == []
    assert result["changes"]["enabledModels"]["result"][-1] == flash

    generic = build_config_handoff(
        _plan(SOL), path, verified_selectors={SOL}
    )
    assert generic["changes"]["modelRoles"]["primary"] == flash
    assert generic["changes"]["fallbackChains"]["default"][-1] == flash

@pytest.mark.parametrize(
    ("version", "target_kind", "code"),
    [
        ("18.1.21", "file", "blocked-omp-version"),
        ("18.1.20", "file", "blocked-omp-version"),
        ("18.2.0", "symlink", "blocked-target-symlink"),
    ],
)
def test_fixed_omp_version_and_real_target_are_required(
    tmp_path: Path, version: str, target_kind: str, code: str
):
    path, _ = _write_config(tmp_path)
    if target_kind == "symlink":
        linked = tmp_path / "linked.yml"
        linked.symlink_to(path)
        path = linked
    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(
            _plan(SOL), path, verified_selectors={SOL}, omp_version=version
        )
    assert raised.value.code == code


def test_handoff_digest_covers_every_precondition_and_change(tmp_path: Path):
    path, raw = _write_config(tmp_path)
    before = hashlib.sha256(raw).hexdigest()
    result = build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    target = result["preconditions"]["target"]
    details = path.stat()
    assert target["realpath"] == str(path.resolve())
    assert target["sha256"] == before
    assert target["device"] == details.st_dev
    assert target["fileId"] == details.st_ino
    assert target["mode"] == details.st_mode & 0o777
    assert target["ctimeNs"] == details.st_ctime_ns
    assert target["size"] == len(raw)
    assert set(result["changes"]) == {
        "addEnabledModels",
        "enabledModels",
        "modelProviderOrder",
        "fallbackChains",
        "modelRoles",
    }

    for scalar_path in _scalar_paths(result):
        tampered = _replace_scalar(result, scalar_path)
        assert _handoff_digest(tampered) != result["planSha256"]

    added_field = copy.deepcopy(result)
    added_field["changes"]["unapproved"] = True
    assert _handoff_digest(added_field) != result["planSha256"]
    assert path.read_bytes() == raw

    serialized = json.dumps(result, sort_keys=True)
    assert "native-secret" not in serialized
    assert "other-sensitive-value" not in serialized


@pytest.mark.parametrize(
    "remove_providers",
    [None, ["other"], [ACTIVE, ACTIVE], [ACTIVE, "other"]],
)
def test_handoff_requires_an_exact_remove_provider_fragment(
    tmp_path: Path, remove_providers: Any
):
    path, _ = _write_config(tmp_path)
    plan = _plan(SOL)
    plan["removeProviders"] = remove_providers

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(plan, path, verified_selectors={SOL})
    assert raised.value.code == "blocked-invalid-plan"


def test_handoff_requires_a_remove_provider_fragment(tmp_path: Path):
    path, _ = _write_config(tmp_path)
    plan = _plan(SOL)
    del plan["removeProviders"]

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(plan, path, verified_selectors={SOL})
    assert raised.value.code == "blocked-invalid-plan"


@pytest.mark.parametrize(
    "raw",
    [
        b"telemetry: !secret native-secret\n",
        b"enabledModels: &models [cc-dy-codex/gpt-5.6-sol]\n",
        (
            b"modelRoles:\n"
            b"  default: cc-dy-codex/gpt-5.6-sol\n"
            b"  default: native-secret\n"
        ),
    ],
)
def test_handoff_rejects_unsafe_yaml_without_leaking_values(tmp_path: Path, raw: bytes):
    path = _write_raw(tmp_path, raw)

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
    assert raised.value.code == "blocked-invalid-yaml"
    assert "native-secret" not in str(raised.value)


def test_handoff_bounds_yaml_depth_nodes_and_scalars(tmp_path: Path):
    too_deep = (
        "".join(
            f"{'  ' * level}next:\n"
            for level in range(omp_config_handoff._MAX_YAML_DEPTH + 1)
        )
        + f"{'  ' * (omp_config_handoff._MAX_YAML_DEPTH + 1)}value: x\n"
    ).encode()
    too_many_nodes = b"items:\n" + b"".join(
        b"  - x\n" for _ in range(omp_config_handoff._MAX_YAML_NODES)
    )
    too_long_scalar = b"telemetry: " + b"x" * (
        omp_config_handoff._MAX_YAML_SCALAR_CHARS + 1
    )

    for raw in (too_deep, too_many_nodes, too_long_scalar):
        path = _write_raw(tmp_path, raw)
        with pytest.raises(ConfigHandoffError) as raised:
            build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
        assert raised.value.code == "blocked-invalid-yaml"


def test_handoff_rejects_oversized_config_before_parsing(tmp_path: Path):
    path = _write_raw(tmp_path, b"#" * (omp_config_handoff._MAX_CONFIG_BYTES + 1))

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
    assert raised.value.code == "blocked-target-size"


def test_handoff_refuses_fifo_config_without_waiting_for_a_writer(tmp_path: Path):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFOs are unavailable on this platform")

    path = tmp_path / "config.yml"
    os.mkfifo(path)

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    assert raised.value.code == "blocked-target-not-file"


@pytest.mark.parametrize("mode", [0o600, 0o644])
def test_handoff_accepts_current_user_readonly_config_modes(tmp_path: Path, mode: int):
    path, _ = _write_config(tmp_path)
    path.chmod(mode)

    result = build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    assert result["preconditions"]["target"]["mode"] == mode


@pytest.mark.parametrize("mode", [0o620, 0o606])
def test_handoff_rejects_group_or_world_writable_config_modes(
    tmp_path: Path, mode: int
):
    path, _ = _write_config(tmp_path)
    path.chmod(mode)

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    assert raised.value.code == "blocked-target-permissions"


def test_handoff_rejects_missing_nonblocking_open_support(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path, _ = _write_config(tmp_path)
    monkeypatch.delattr(omp_config_handoff.os, "O_NONBLOCK", raising=False)

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    assert raised.value.code == "blocked-nonblocking"


def test_handoff_rejects_path_replaced_after_fd_snapshot(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    path, raw = _write_config(tmp_path)
    original_read = omp_config_handoff._read_descriptor

    def replace_path_after_read(descriptor: int, maximum: int) -> bytes:
        snapshot = original_read(descriptor, maximum)
        replacement = tmp_path / "replacement.yml"
        replacement.write_bytes(raw)
        replacement.replace(path)
        return snapshot

    monkeypatch.setattr(omp_config_handoff, "_read_descriptor", replace_path_after_read)

    with pytest.raises(ConfigHandoffError) as raised:
        build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
    assert raised.value.code == "blocked-target-changed"


def test_handoff_rejects_nonpublic_routing_values_without_serializing_them(
    tmp_path: Path,
):
    invalid_configs: list[dict[str, Any]] = []

    invalid = _config()
    invalid["enabledModels"] = [{"authorization": "native-secret"}]
    invalid_configs.append(invalid)

    invalid = _config()
    invalid["modelProviderOrder"] = [DY + "\nnative-secret"]
    invalid_configs.append(invalid)

    invalid = _config()
    invalid["modelRoles"] = {"default": {"headers": {"Authorization": "native-secret"}}}
    invalid_configs.append(invalid)

    invalid = _config()
    invalid["retry"]["fallbackChains"] = {
        "default": {"extraBody": {"token": "native-secret"}}
    }
    invalid_configs.append(invalid)

    for config in invalid_configs:
        path, _ = _write_config(tmp_path, config)
        with pytest.raises(ConfigHandoffError) as raised:
            build_config_handoff(_plan(SOL), path, verified_selectors={SOL})
        assert raised.value.code == "blocked-unsafe-config-value"
        assert "native-secret" not in str(raised.value)


def test_handoff_redacts_unrelated_nested_credentials(tmp_path: Path):
    config = _config()
    config["providers"] = {
        DY: {
            "apiKey": "native-secret",
            "headers": {"Authorization": "Bearer native-secret"},
            "extraBody": {"nested": {"token": "native-secret"}},
        }
    }
    path, _ = _write_config(tmp_path, config)

    result = build_config_handoff(_plan(SOL), path, verified_selectors={SOL})

    serialized = json.dumps(result, sort_keys=True)
    assert "native-secret" not in serialized
    assert "Authorization" not in serialized
    assert "extraBody" not in serialized
