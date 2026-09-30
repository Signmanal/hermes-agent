"""Contract tests for ``hermes model-sync cc-switch``.

These target the public parser/handler seam and replace source, projection, and
config dependencies with fakes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


class _BlockedError(ValueError):
    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _materializer(**overrides):
    values = {
        "load_projection_plan": lambda path: {"plan": path},
        "validate_projection_plan": lambda plan: plan,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.fixture
def model_sync(monkeypatch):
    from hermes_cli.subcommands import model_sync

    monkeypatch.setattr(model_sync, "_secure_materializer_supported", lambda: True)
    return model_sync


def _parser(model_sync, sink):
    parser = argparse.ArgumentParser(prog="hermes")
    sub = parser.add_subparsers(dest="command")
    model_sync.build_model_sync_parser(
        sub, cmd_model_sync=lambda args: sink.append(args) or 0
    )
    return parser


def _apply_args(**overrides):
    values = {
        "model_sync_target": "cc-switch",
        "model_sync_action": "apply",
        "plan": "plan.json",
        "db": "db",
        "target": "target",
        "yes": True,
        "accepted_plan_sha256": "a" * 64,
        "allow_opaque_backup": False,
        "json": True,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_parser_registers_cc_switch_inventory_and_all_flags(model_sync):
    seen = []
    parser = _parser(model_sync, seen)
    args = parser.parse_args([
        "model-sync",
        "cc-switch",
        "inventory",
        "--db",
        "/tmp/source.db",
        "--provider",
        "anthropic",
        "--json",
    ])
    assert args.command == "model-sync"
    assert args.model_sync_target == "cc-switch"
    assert args.model_sync_action == "inventory"
    assert args.db == "/tmp/source.db"
    assert args.provider == "anthropic"
    assert args.json is True
    assert args.func(args) == 0
    assert seen == [args]


def test_parser_plan_captures_contract_and_defaults_to_dry_run(model_sync):
    parser = _parser(model_sync, [])
    args = parser.parse_args([
        "model-sync",
        "cc-switch",
        "plan",
        "--db",
        "source.db",
        "--target",
        "models.yml",
        "--provider-map",
        '{"anthropic":"cc-anthropic"}',
        "--selected-models",
        '{"anthropic":["model"]}',
        "--exclude",
        "a",
        "--exclude",
        "b",
        "--approve-origin",
        "team",
        "--omp-version",
        "18.2.0",
        "--output",
        "plan.json",
    ])
    assert args.db == "source.db"
    assert args.target == "models.yml"
    assert args.provider_map == '{"anthropic":"cc-anthropic"}'
    assert args.selected_models == '{"anthropic":["model"]}'
    assert args.exclude == ["a", "b"]
    assert args.approve_origin == ["team"]
    assert args.omp_version == "18.2.0"
    assert args.output == "plan.json"
    assert not getattr(args, "apply", False)


def test_parser_apply_requires_independent_digest_and_explicit_backup_consent(
    model_sync,
):
    parser = _parser(model_sync, [])
    common = [
        "model-sync",
        "cc-switch",
        "apply",
        "--plan",
        "plan.json",
        "--db",
        "source.db",
        "--target",
        "models.yml",
        "--yes",
    ]

    with pytest.raises(SystemExit):
        parser.parse_args(common)
    with pytest.raises(SystemExit):
        parser.parse_args(common + ["--accepted-plan-sha256", "not-a-digest"])

    digest = "A" * 64
    args = parser.parse_args(common + ["--accepted-plan-sha256", digest])
    assert args.accepted_plan_sha256 == digest.lower()
    assert args.allow_opaque_backup is False

    consented = parser.parse_args(
        common + ["--accepted-plan-sha256", digest, "--allow-opaque-backup"]
    )
    assert consented.allow_opaque_backup is True


def test_inventory_dispatches_safe_source_inventory(model_sync, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        model_sync,
        "_load_source",
        lambda: SimpleNamespace(
            inventory_source=lambda db, provider_filter=None: (
                calls.append((db, provider_filter))
                or {"status": "inventory", "provider": provider_filter}
            )
        ),
    )
    args = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="inventory",
        db="db.sqlite",
        provider="openai",
        json=True,
    )
    assert model_sync.cmd_model_sync(args) == 0
    assert calls == [("db.sqlite", "openai")]
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "inventory", "provider": "openai"}
    assert "secret" not in json.dumps(payload)


def test_inventory_serializes_environment_model_evidence(model_sync, monkeypatch, capsys):
    inventory = {
        "status": "inventory",
        "source_app_type": "claude",
        "providers": [
            {
                "declared_models": ["gpt-5.6-sol"],
                "environment_models": {
                    "configured_model_values": ["gpt-6-astra[1M]"],
                    "name_to_configured_value": {
                        "gpt-6-astra": "gpt-6-astra[1M]"
                    },
                },
            }
        ],
    }
    monkeypatch.setattr(
        model_sync,
        "_load_source",
        lambda: SimpleNamespace(
            inventory_source=lambda _db, provider_filter=None: inventory
        ),
    )
    args = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="inventory",
        db="db.sqlite",
        provider=None,
        json=True,
    )

    assert model_sync.cmd_model_sync(args) == 0
    assert json.loads(capsys.readouterr().out) == inventory


def test_plan_passes_all_parameters_and_never_applies(model_sync, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            build_projection_plan=lambda **kwargs: (
                calls.append(kwargs) or {"status": "planned"}
            )
        ),
    )
    args = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="plan",
        db="db",
        target="target",
        provider_map='{"source":"cc-source"}',
        selected_models='{"source":["model"]}',
        exclude=["x"],
        approve_origin=["source"],
        omp_version="18.2.0",
        output=None,
        json=True,
        apply=False,
    )
    assert model_sync.cmd_model_sync(args) == 0
    assert calls == [
        {
            "db_path": "db",
            "target_path": "target",
            "provider_map": {"source": "cc-source"},
            "selected_models": {"source": ["model"]},
            "excluded_source_ids": {"x"},
            "approved_current_origin_source_ids": {"source"},
            "omp_version": "18.2.0",
        }
    ]
    assert "secret" not in capsys.readouterr().out


def test_apply_requires_yes_before_loading_or_applying(model_sync, monkeypatch, capsys):
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("must not load"),
    )
    assert model_sync.cmd_model_sync(_apply_args(yes=False)) != 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["code"] == "confirmation-required"
    assert "secret" not in json.dumps(payload)


def test_apply_rejects_missing_digest_before_loading_or_applying(
    model_sync, monkeypatch, capsys
):
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("must not load"),
    )
    assert model_sync.cmd_model_sync(_apply_args(accepted_plan_sha256=None)) != 0
    assert json.loads(capsys.readouterr().out)["code"] == "blocked-accepted-plan-sha256"


def test_apply_delegates_wrong_digest_without_starting_a_write(
    model_sync, monkeypatch, capsys
):
    plan = {"planSha256": "a" * 64}
    calls = []

    def reject_wrong_digest(value, db, target, **kwargs):
        calls.append((value, db, target, kwargs))
        raise _BlockedError("blocked-accepted-plan-sha256")

    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=lambda path: plan,
            apply_projection_plan=reject_wrong_digest,
        ),
    )
    assert model_sync.cmd_model_sync(_apply_args(accepted_plan_sha256="b" * 64)) != 0
    assert calls == [
        (
            plan,
            "db",
            "target",
            {
                "accepted_plan_sha256": "b" * 64,
                "allow_opaque_backup": False,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out)["code"] == "blocked-accepted-plan-sha256"


@pytest.mark.parametrize("allow_opaque_backup", [False, True])
def test_apply_passes_independent_digest_and_backup_consent(
    model_sync, monkeypatch, capsys, allow_opaque_backup
):
    plan = {"planSha256": "a" * 64}
    calls = []
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=lambda path: plan,
            apply_projection_plan=lambda value, db, target, **kwargs: (
                calls.append((value, db, target, kwargs)) or {"status": "applied"}
            ),
        ),
    )
    assert (
        model_sync.cmd_model_sync(_apply_args(allow_opaque_backup=allow_opaque_backup))
        == 0
    )
    assert calls == [
        (
            plan,
            "db",
            "target",
            {
                "accepted_plan_sha256": "a" * 64,
                "allow_opaque_backup": allow_opaque_backup,
            },
        )
    ]
    assert json.loads(capsys.readouterr().out) == {"status": "applied"}


def test_apply_delegates_no_follow_plan_loading(model_sync, monkeypatch, capsys):
    def reject_plan(path):
        raise _BlockedError("blocked-plan-load")

    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=reject_plan,
            apply_projection_plan=lambda *args, **kwargs: pytest.fail("must not apply"),
        ),
    )
    assert model_sync.cmd_model_sync(_apply_args(plan="malformed.json")) != 0
    assert json.loads(capsys.readouterr().out)["code"] == "blocked-plan-load"


def test_apply_delegates_closed_schema_validation(model_sync, monkeypatch, capsys):
    def reject_schema(plan):
        raise _BlockedError("blocked-plan-schema")

    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=lambda path: {"invalid": True},
            validate_projection_plan=reject_schema,
            apply_projection_plan=lambda *args, **kwargs: pytest.fail("must not apply"),
        ),
    )
    assert model_sync.cmd_model_sync(_apply_args(plan="malformed.json")) != 0
    assert json.loads(capsys.readouterr().out)["code"] == "blocked-plan-schema"


def test_verify_and_handoff_dispatch_public_seams(model_sync, monkeypatch):
    calls = []
    plan = {"schemaVersion": 1}
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=lambda path: plan,
            verify_projection=lambda value, db, target: (
                calls.append(("verify", value, db, target)) or True
            ),
        ),
    )
    monkeypatch.setattr(
        model_sync,
        "_load_handoff",
        lambda: SimpleNamespace(
            build_config_handoff=lambda value, config, verified_selectors: (
                calls.append(("handoff", value, config, verified_selectors))
                or {"ok": True}
            )
        ),
    )
    verify = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="verify",
        plan="p",
        db="d",
        target="t",
        json=True,
    )
    handoff = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="handoff-plan",
        plan="p",
        config="c",
        repeat_verified_selector=["claude/gpt"],
        output=None,
        json=True,
    )
    assert model_sync.cmd_model_sync(verify) == 0
    assert model_sync.cmd_model_sync(handoff) == 0
    assert calls == [
        ("verify", plan, "d", "t"),
        ("handoff", plan, "c", ["claude/gpt"]),
    ]


def test_missing_materializer_leaves_main_and_unrelated_help_available(tmp_path):
    """A selected model-sync action blocks if its Unix-only import is unavailable."""
    repo = Path(__file__).resolve().parents[2]
    script = textwrap.dedent(
        """
        import builtins
        import importlib
        import sys

        target = "tools.omp_model_materializer"
        original_import = builtins.__import__
        original_import_module = importlib.import_module

        def unavailable_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == target:
                raise ModuleNotFoundError("simulated unavailable materializer")
            return original_import(name, globals, locals, fromlist, level)

        def unavailable_import_module(name, package=None):
            if name == target:
                raise ModuleNotFoundError("simulated unavailable materializer")
            return original_import_module(name, package)

        builtins.__import__ = unavailable_import
        importlib.import_module = unavailable_import_module

        main = importlib.import_module("hermes_cli.main")
        parser, _ = main._build_cli_parser()
        assert target not in sys.modules
        try:
            parser.parse_args(["status", "--help"])
        except SystemExit as error:
            assert error.code == 0
        else:
            raise AssertionError("unrelated help must exit successfully")

        args = parser.parse_args([
            "model-sync", "cc-switch", "plan", "--db", "source.db",
            "--target", "models.yml", "--provider-map", "{}",
            "--selected-models", "{}",
        ])
        assert args.func(args) == 1
        print("MODEL_SYNC_IMPORT_BLOCKED")
        """
    )
    environment = os.environ | {
        "HOME": str(tmp_path),
        "HERMES_HOME": str(tmp_path / "hermes-home"),
        "PYTHONPATH": str(repo),
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.rstrip().endswith("MODEL_SYNC_IMPORT_BLOCKED")
    assert (
        '{"status":"blocked","code":"blocked-unsupported-platform"}' in completed.stdout
    )


def test_parser_exposes_explicit_read_only_provider_migration_plan(model_sync):
    parser = _parser(model_sync, [])

    args = parser.parse_args([
        "model-sync",
        "cc-switch",
        "plan",
        "--provider-migration",
        "--enable-xuanju",
        "--db",
        "source.db",
        "--target",
        "models.yml",
        "--config",
        "config.yml",
        "--env",
        ".env",
        "--provider-map",
        '{"dy":"cc-dy-codex","xuanju":"cc-xuanju"}',
        "--selected-models",
        '{"dy":["gpt-5.6"],"xuanju":["GLM-5.3-flash"]}',
        "--probe-evidence",
        "probe.json",
    ])
    assert args.provider_migration is True
    assert args.enable_xuanju is True
    assert args.config == "config.yml"
    assert args.env == ".env"
    assert args.probe_evidence == ["probe.json"]
    assert not getattr(args, "yes", False)


def test_provider_migration_plan_apply_and_verify_dispatch_public_seams(
    model_sync, monkeypatch, capsys
):
    calls = []
    plan = {"planSha256": "a" * 64}
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            build_provider_migration_plan=lambda **kwargs: (
                calls.append(("plan", kwargs)) or {"status": "planned"}
            ),
            load_provider_migration_plan=lambda path: plan,
            apply_provider_migration_plan=lambda value, db, models, config, env, **kwargs: (
                calls.append(("apply", value, db, models, config, env, kwargs))
                or {"status": "applied"}
            ),
            verify_provider_migration=lambda value, db, models, config, env: (
                calls.append(("verify", value, db, models, config, env)) or True
            ),
        ),
    )
    common = {
        "model_sync_target": "cc-switch",
        "provider_migration": True,
        "db": "db",
        "target": "models",
        "config": "config",
        "env": "env",
        "json": True,
        "output": None,
    }
    plan_args = argparse.Namespace(
        **common,
        model_sync_action="plan",
        provider_map='{"dy":"cc-dy-codex","xuanju":"cc-xuanju"}',
        selected_models='{"dy":["gpt-5.6"],"xuanju":["GLM-5.3-flash"]}',
        probe_evidence=["probe.json"],
        enable_xuanju=True,
        exclude=[],
        approve_origin=["dy", "xuanju"],
        omp_version="18.2.0",
    )
    apply_args = argparse.Namespace(
        **common,
        model_sync_action="apply",
        plan="plan.json",
        yes=True,
        accepted_plan_sha256="a" * 64,
        allow_opaque_backup=True,
    )
    verify_args = argparse.Namespace(
        **common, model_sync_action="verify", plan="plan.json"
    )

    assert model_sync.cmd_model_sync(plan_args) == 0
    assert model_sync.cmd_model_sync(apply_args) == 0
    assert model_sync.cmd_model_sync(verify_args) == 0
    assert calls == [
        (
            "plan",
            {
                "db_path": "db",
                "models_path": "models",
                "config_path": "config",
                "environment_path": "env",
                "provider_map": {"dy": "cc-dy-codex", "xuanju": "cc-xuanju"},
                "selected_models": {"dy": ["gpt-5.6"], "xuanju": ["GLM-5.3-flash"]},
                "probe_evidence_path": ["probe.json"],
                "approved_current_origin_source_ids": {"dy", "xuanju"},
                "enable_xuanju": True,
                "omp_version": "18.2.0",
            },
        ),
        (
            "apply",
            plan,
            "db",
            "models",
            "config",
            "env",
            {
                "accepted_plan_sha256": "a" * 64,
                "allow_opaque_backup": True,
            },
        ),
        ("verify", plan, "db", "models", "config", "env"),
    ]
    assert "secret" not in capsys.readouterr().out


@pytest.mark.parametrize(
    ("action", "protected_name"),
    [
        ("plan", "db"),
        ("plan", "target"),
        ("plan", "config"),
        ("plan", "env"),
        ("plan", "probe"),
        ("apply", "plan"),
        ("apply", "db"),
        ("apply", "target"),
        ("apply", "config"),
        ("apply", "env"),
        ("verify", "plan"),
        ("verify", "db"),
        ("verify", "target"),
        ("verify", "config"),
        ("verify", "env"),
        ("handoff-plan", "plan"),
        ("handoff-plan", "config"),
    ],
)
def test_provider_migration_output_cannot_alias_any_input(
    model_sync,
    monkeypatch,
    capsys,
    tmp_path: Path,
    action: str,
    protected_name: str,
):
    paths = {
        name: tmp_path / name
        for name in ("db", "target", "config", "env", "probe", "plan")
    }
    for path in paths.values():
        path.write_text("input")

    common = {
        "model_sync_target": "cc-switch",
        "model_sync_action": action,
        "db": str(paths["db"]),
        "target": str(paths["target"]),
        "output": str(paths[protected_name]),
        "json": True,
    }
    if action == "plan":
        args = argparse.Namespace(
            **common,
            provider_migration=True,
            config=str(paths["config"]),
            env=str(paths["env"]),
            probe_evidence=[str(paths["probe"])],
            provider_map="{}",
            selected_models="{}",
            exclude=[],
            approve_origin=[],
            omp_version="18.2.0",
            enable_xuanju=True,
        )
    elif action == "apply":
        args = argparse.Namespace(
            **common,
            provider_migration=True,
            plan=str(paths["plan"]),
            config=str(paths["config"]),
            env=str(paths["env"]),
            yes=True,
            accepted_plan_sha256="a" * 64,
            allow_opaque_backup=False,
        )
    elif action == "verify":
        args = argparse.Namespace(
            **common,
            provider_migration=True,
            plan=str(paths["plan"]),
            config=str(paths["config"]),
            env=str(paths["env"]),
        )
    else:
        args = argparse.Namespace(
            **common,
            plan=str(paths["plan"]),
            config=str(paths["config"]),
            repeat_verified_selector=[],
        )

    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("input collision must be rejected before loading"),
    )
    monkeypatch.setattr(
        model_sync,
        "_load_handoff",
        lambda: pytest.fail("input collision must be rejected before loading"),
    )
    assert model_sync.cmd_model_sync(args) != 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "code": "blocked-output",
    }


def test_output_alias_hardlink_is_rejected_before_loading(
    model_sync, monkeypatch, capsys, tmp_path: Path
):
    db = tmp_path / "db"
    output = tmp_path / "report"
    db.write_text("database")
    os.link(db, output)
    args = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="plan",
        provider_migration=False,
        db=str(db),
        target=str(tmp_path / "models"),
        provider_map="{}",
        selected_models="{}",
        exclude=[],
        approve_origin=[],
        omp_version="18.2.0",
        output=str(output),
        json=True,
    )
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("hardlink collision must be rejected before loading"),
    )

    assert model_sync.cmd_model_sync(args) != 0
    assert json.loads(capsys.readouterr().out)["code"] == "blocked-output"
    assert output.read_text() == "database"


def test_output_refuses_existing_unrelated_destination_without_overwrite(
    model_sync, monkeypatch, capsys, tmp_path: Path
):
    output = tmp_path / "report.json"
    output.write_text("keep this report")
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("existing output must be rejected before loading"),
    )
    args = argparse.Namespace(
        model_sync_target="cc-switch",
        model_sync_action="plan",
        provider_migration=False,
        db=str(tmp_path / "db"),
        target=str(tmp_path / "models"),
        provider_map="{}",
        selected_models="{}",
        exclude=[],
        approve_origin=[],
        omp_version="18.2.0",
        output=str(output),
        json=True,
    )

    assert model_sync.cmd_model_sync(args) != 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "code": "blocked-output",
    }
    assert output.read_text() == "keep this report"


def test_apply_existing_output_is_rejected_before_materializer(
    model_sync, monkeypatch, capsys, tmp_path: Path
):
    output = tmp_path / "report.json"
    output.write_text("keep this apply report")
    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: pytest.fail("existing output must be rejected before apply"),
    )

    assert (
        model_sync.cmd_model_sync(
            _apply_args(
                plan=str(tmp_path / "plan.json"),
                db=str(tmp_path / "db"),
                target=str(tmp_path / "models"),
                output=str(output),
            )
        )
        != 0
    )
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "code": "blocked-output",
    }
    assert output.read_text() == "keep this apply report"


def test_apply_output_reservation_detects_path_replacement_without_overwrite(
    model_sync, monkeypatch, capsys, tmp_path: Path
):
    output = tmp_path / "report.json"
    attacker = tmp_path / "attacker.json"
    attacker_payload = "attacker-owned report"

    def apply(value, db, target, **kwargs):
        attacker.write_text(attacker_payload)
        attacker.replace(output)
        return {"status": "applied"}

    monkeypatch.setattr(
        model_sync,
        "_load_materializer",
        lambda: _materializer(
            load_projection_plan=lambda path: {"plan": path},
            apply_projection_plan=apply,
        ),
    )
    args = _apply_args(
        plan=str(tmp_path / "plan.json"),
        db=str(tmp_path / "db"),
        target=str(tmp_path / "models"),
        output=str(output),
    )

    assert model_sync.cmd_model_sync(args) != 0
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "code": "blocked-output",
    }
    assert output.read_text() == attacker_payload
