"""Contract tests for digest-authorized secure OMP projection application."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sqlite3
from pathlib import Path

import pytest
import yaml
import tools.omp_model_materializer as materializer
from tests.tools.test_omp_model_materializer import (
    _DEEPSEEK_TARGET,
    _DY_SOURCE_ID,
    _DY_TARGET,
    _PRIVATE_HOST,
    _SECRET_MARKER,
    _build,
    _write_canonical_db,
    _write_generated_models,
)
from tools.omp_model_materializer import (
    ProjectionPlanError,
    apply_projection_plan,
    load_projection_plan,
    verify_projection,
)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rehash(plan: dict) -> dict:
    plan["planSha256"] = hashlib.sha256(
        json.dumps(
            {key: value for key, value in plan.items() if key != "planSha256"},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return plan


def _apply(plan: dict, db: Path, target: Path, **kwargs):
    arguments = {
        "accepted_plan_sha256": plan["planSha256"],
        "allow_opaque_backup": True,
    }
    arguments.update(kwargs)
    return apply_projection_plan(plan, db, target, **arguments)


def _block_style_provider_entry(name: str, provider: dict) -> str:
    lines = yaml.safe_dump({name: provider}, sort_keys=False).splitlines(keepends=True)
    return f"  {lines[0]}    # exact comment inside {name}\n" + "".join(
        f"  {line}" for line in lines[1:]
    )


def test_plan_contains_full_preconditions_result_and_digest(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)

    assert set(plan["preconditions"]) == {"source", "target"}
    source_precondition = plan["preconditions"]["source"]
    details = db.stat()
    assert source_precondition == {
        "realpath": str(db.resolve()),
        "device": details.st_dev,
        "fileId": details.st_ino,
        "owner": details.st_uid,
        "sha256": _hash(db),
        "mode": details.st_mode & 0o777,
        "size": details.st_size,
        "mtimeNs": details.st_mtime_ns,
        "ctimeNs": details.st_ctime_ns,
        "snapshotSha256": plan["source"]["snapshotSha256"],
        "wal": None,
        "shm": None,
    }
    parent = target.parent.stat()
    target_precondition = plan["preconditions"]["target"]
    assert target_precondition == {
        "realpath": str(target.resolve()),
        "device": target.stat().st_dev,
        "fileId": target.stat().st_ino,
        "sha256": _hash(target),
        "mode": target.stat().st_mode & 0o777,
        "ctimeNs": target.stat().st_ctime_ns,
        "parent": {
            "device": parent.st_dev,
            "fileId": parent.st_ino,
            "ctimeNs": parent.st_ctime_ns,
        },
    }
    assert set(plan["result"]) == {"targetBytesSha256", "targetDocumentSha256"}
    assert all(len(plan["result"][name]) == 64 for name in plan["result"])
    assert plan["policy"] == {"allowOpaqueBackup": False, "requireTargetHeader": True}


def test_apply_accepts_live_wal_without_source_change(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    connection = sqlite3.connect(db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        plan = _build(db, target)

        assert _apply(plan, db, target)["status"] == "applied"
        assert verify_projection(plan, db, target)
    finally:
        connection.close()


def test_apply_rejects_live_wal_after_unrelated_source_write(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    connection = sqlite3.connect(db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        plan = _build(db, target)
        before = target.read_bytes()

        connection.execute("INSERT INTO source_activity (value) VALUES ('new')")
        connection.commit()

        with pytest.raises(ProjectionPlanError) as raised:
            _apply(plan, db, target)
        assert raised.value.code == "blocked-source-changed"
        assert target.read_bytes() == before
    finally:
        connection.close()


def test_apply_rejects_live_wal_after_provider_change(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    connection = sqlite3.connect(db)
    try:
        assert connection.execute("PRAGMA journal_mode = WAL").fetchone() == ("wal",)
        connection.execute("PRAGMA wal_autocheckpoint = 0")
        connection.execute("CREATE TABLE source_activity (value TEXT NOT NULL)")
        connection.commit()
        plan = _build(db, target)
        before = target.read_bytes()

        connection.execute(
            "UPDATE providers SET name = ? WHERE id = ? AND app_type = 'claude'",
            ("Changed DY source", _DY_SOURCE_ID),
        )
        connection.commit()

        with pytest.raises(ProjectionPlanError) as raised:
            _apply(plan, db, target)
        assert raised.value.code == "blocked-source-changed"
        assert target.read_bytes() == before
    finally:
        connection.close()


def test_apply_requires_independently_supplied_accepted_digest(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    before = target.read_bytes()

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target, accepted_plan_sha256="0" * 64)

    assert raised.value.code == "blocked-accepted-plan-sha256"
    assert target.read_bytes() == before


def test_apply_requires_explicit_opaque_backup_consent(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target, allow_opaque_backup=False)

    assert raised.value.code == "blocked-backup-consent"


def test_apply_success_removes_only_deepseek_and_cleans_secure_run(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    before_mode = target.stat().st_mode & 0o777
    plan = _build(db, target)

    report = _apply(plan, db, target)
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    rollback_root = tmp_path / ".omp-projection-rollback"

    assert "cc-deepseek" not in document["providers"]
    assert _DY_TARGET in document["providers"]
    assert target.read_text(encoding="utf-8").startswith(
        "# CC Switch source projection; generated for OMP.\n"
    )
    assert target.stat().st_mode & 0o777 == before_mode
    assert report["withinRun"] == {"deleted": ["cc-deepseek"], "unavailable": []}
    assert report["rollback"] == {"retained": False, "withinRun": True}
    assert report["cleanup"] == {
        "backup": "deleted",
        "runDirectory": "deleted",
        "temporary": "published",
    }
    assert rollback_root.stat().st_mode & 0o777 == 0o700
    assert list(rollback_root.iterdir()) == []
    assert not list(tmp_path.glob(".models.yml.omp-projection-*.tmp"))
    serialized = json.dumps(report, sort_keys=True)
    assert _SECRET_MARKER not in serialized
    assert _PRIVATE_HOST not in serialized


def test_apply_preserves_unselected_provider_remote_compaction(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    remote_compaction = {
        "api": "openai-responses",
        "enabled": False,
        "endpoint": "https://unselected-secondary.invalid/v1",
        "model": "unselected-secondary-model",
    }
    document["providers"]["unselected-remote"] = {
        "api": "openai-responses",
        "apiKey": "UNSELECTED_REMOTE_KEY",
        "baseUrl": "https://unselected.invalid/v1",
        "models": [],
        "remoteCompaction": remote_compaction,
    }
    target.write_text(
        "# CC Switch source projection; generated for OMP.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    plan = _build(db, target)

    _apply(plan, db, target)

    assert (
        yaml.safe_load(target.read_text(encoding="utf-8"))["providers"][
            "unselected-remote"
        ]["remoteCompaction"]
        == remote_compaction
    )


def test_apply_verify_failure_restores_original_bytes_and_mode_atomically(
    tmp_path: Path, monkeypatch
):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    before_bytes, before_mode = target.read_bytes(), target.stat().st_mode & 0o777
    monkeypatch.setattr(
        materializer, "verify_projection", lambda *args, **kwargs: False
    )

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target)

    assert raised.value.code == "blocked-apply-verify"
    assert target.read_bytes() == before_bytes
    assert target.stat().st_mode & 0o777 == before_mode
    assert list((tmp_path / ".omp-projection-rollback").iterdir()) == []


def test_apply_rejects_tampered_selected_model_after_planning(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    document["providers"][_DY_TARGET]["models"][0]["maxTokens"] = 1
    target.write_text(
        "# CC Switch source projection; generated for OMP.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    assert not verify_projection(plan, db, target)
    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target)
    assert raised.value.code == "blocked-target-changed"


def test_apply_rejects_target_hash_mode_and_identity_staleness(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    original = target.read_bytes()
    original_mode = target.stat().st_mode & 0o777

    target.write_bytes(original + b"\n# stale content\n")
    with pytest.raises(ProjectionPlanError) as changed:
        _apply(plan, db, target)
    assert changed.value.code == "blocked-target-changed"

    target.write_bytes(original)
    os.chmod(target, original_mode ^ 0o100)
    with pytest.raises(ProjectionPlanError) as mode_changed:
        _apply(plan, db, target)
    assert mode_changed.value.code == "blocked-target-changed"

    os.chmod(target, original_mode)
    replacement = tmp_path / "replacement.yml"
    replacement.write_bytes(original)
    os.chmod(replacement, original_mode)
    os.replace(replacement, target)
    with pytest.raises(ProjectionPlanError) as identity_changed:
        _apply(plan, db, target)
    assert identity_changed.value.code == "blocked-target-changed"


@pytest.mark.parametrize("unsafe_bits", [0o020, 0o002])
def test_apply_rejects_target_becoming_group_or_world_writable_before_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unsafe_bits: int
):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    original_bytes = target.read_bytes()
    original_mode = target.stat().st_mode & 0o777
    original_write = materializer._write_target_temp

    def write_then_make_target_unsafe(*args, **kwargs):
        replacement = original_write(*args, **kwargs)
        os.chmod(target, (original_mode & ~0o022) | unsafe_bits)
        return replacement

    monkeypatch.setattr(
        materializer, "_write_target_temp", write_then_make_target_unsafe
    )
    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target)

    assert raised.value.code == "blocked-target-path"
    assert target.read_bytes() == original_bytes
    assert target.stat().st_mode & 0o022 == unsafe_bits


def test_apply_rejects_source_hash_and_identity_staleness(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    with sqlite3.connect(db) as connection:
        row = connection.execute(
            "SELECT settings_config FROM providers WHERE id = ? AND app_type = 'claude'",
            (_DY_SOURCE_ID,),
        ).fetchone()
        settings = json.loads(row[0])
        settings["models"][0]["name"] = "changed after plan"
        connection.execute(
            "UPDATE providers SET settings_config = ? WHERE id = ? AND app_type = 'claude'",
            (json.dumps(settings), _DY_SOURCE_ID),
        )
    with pytest.raises(ProjectionPlanError) as hash_changed:
        _apply(plan, db, target)
    assert hash_changed.value.code == "blocked-source-changed"
    assert not verify_projection(plan, db, target)

    db = _write_canonical_db(tmp_path / "identity")
    target = _write_generated_models(tmp_path / "identity")
    plan = _build(db, target)
    replacement = db.with_name("replacement.db")
    replacement.write_bytes(db.read_bytes())
    os.chmod(replacement, db.stat().st_mode & 0o777)
    os.replace(replacement, db)
    with pytest.raises(ProjectionPlanError) as identity_changed:
        _apply(plan, db, target)
    assert identity_changed.value.code == "blocked-source-changed"


def test_apply_recomputes_operations_and_rejects_rehashed_plan_attacks(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    before = target.read_bytes()

    tampered = copy.deepcopy(plan)
    tampered["removeProviders"] = [_DY_TARGET]
    _rehash(tampered)
    with pytest.raises(ProjectionPlanError) as unauthorized_removal:
        _apply(tampered, db, target)
    assert unauthorized_removal.value.code == "blocked-plan-action"

    tampered = copy.deepcopy(plan)
    tampered["providerActions"][0]["models"] = ["gpt-5.6"]
    _rehash(tampered)
    with pytest.raises(ProjectionPlanError) as changed_selection:
        _apply(tampered, db, target)
    assert changed_selection.value.code == "blocked-plan-action"
    assert target.read_bytes() == before


def test_apply_rejects_tampered_plan_schema_even_when_rehashed(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    tampered = copy.deepcopy(plan)
    tampered["unexpected"] = "not in the closed schema"
    _rehash(tampered)

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(tampered, db, target)

    assert raised.value.code == "blocked-plan-schema"


def test_apply_rejects_rehashed_plan_with_unsafe_target_mode(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    before = target.read_bytes()
    tampered = copy.deepcopy(plan)
    tampered["preconditions"]["target"]["mode"] |= 0o022
    _rehash(tampered)

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(tampered, db, target)

    assert raised.value.code == "blocked-plan-schema"
    assert target.read_bytes() == before


def test_apply_cas_rejects_a_concurrent_target_change(tmp_path: Path, monkeypatch):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    original_write = materializer._write_target_temp
    raced = target.read_bytes() + b"\n# racing writer\n"

    def write_then_race(*args, **kwargs):
        name = original_write(*args, **kwargs)
        target.write_bytes(raced)
        return name

    monkeypatch.setattr(materializer, "_write_target_temp", write_then_race)
    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target)

    assert raised.value.code == "blocked-target-changed"
    assert target.read_bytes() == raced


@pytest.mark.require_symlinks
def test_apply_refuses_target_replaced_by_symlink_after_plan(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    saved = tmp_path / "saved-models.yml"
    os.replace(target, saved)
    target.symlink_to(saved)

    with pytest.raises(ProjectionPlanError) as raised:
        _apply(plan, db, target)

    assert raised.value.code == "blocked-target-symlink"
    assert (
        "cc-deepseek" in yaml.safe_load(saved.read_text(encoding="utf-8"))["providers"]
    )


def test_strict_plan_loader_refuses_noncanonical_plans(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    canonical = json.dumps(
        plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical + b"\n")

    assert load_projection_plan(plan_path) == plan

    plan_path.write_text(json.dumps(plan, sort_keys=True), encoding="utf-8")
    with pytest.raises(ProjectionPlanError) as noncanonical:
        load_projection_plan(plan_path)
    assert noncanonical.value.code == "blocked-plan-canonical"


@pytest.mark.require_symlinks
def test_strict_plan_loader_refuses_symlinked_plan(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(
        json.dumps(
            plan, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    linked = tmp_path / "plan-link.json"
    linked.symlink_to(plan_path)

    with pytest.raises(ProjectionPlanError) as symlink:
        load_projection_plan(linked)
    assert symlink.value.code == "blocked-plan-load"


def test_apply_semantic_noop_keeps_target_byte_identical(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    document = yaml.safe_load(target.read_text(encoding="utf-8"))
    document["providers"].pop("cc-deepseek")
    target.write_text(
        "# CC Switch source projection; generated for OMP.\n"
        + yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    plan = _build(db, target)
    before = target.read_bytes()

    report = _apply(plan, db, target)

    assert report == {
        "status": "unchanged",
        "target": str(target.resolve()),
        "withinRun": {"deleted": [], "unavailable": []},
        "rollback": {"retained": False, "withinRun": False},
        "cleanup": {
            "backup": "not-created",
            "runDirectory": "not-created",
            "temporary": "not-created",
        },
    }
    assert target.read_bytes() == before
    assert not (tmp_path / ".omp-projection-rollback").exists()


def test_apply_splices_only_deepseek_and_preserves_unrelated_yaml_bytes(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    original = target.read_text(encoding="utf-8")
    original = original.replace(
        f"  {_DY_TARGET}:\n",
        f"  # retained provider comment and exact formatting\n  {_DY_TARGET}:\n",
    ).replace(
        "  cc-deepseek:\n",
        "  # deepseek comment remains as an inert adjacent comment\n  cc-deepseek:\n",
    )
    target.write_text(original, encoding="utf-8")
    expected_untouched_prefix = original.split("  cc-deepseek:\n", 1)[0]
    plan = _build(db, target)

    _apply(plan, db, target)

    rendered = target.read_text(encoding="utf-8")
    assert rendered.startswith(expected_untouched_prefix)
    assert "cc-deepseek:" not in rendered
    assert "retained provider comment and exact formatting" in rendered
    assert (
        yaml.safe_load(rendered)["providers"][_DY_TARGET]["api"] == "openai-responses"
    )


@pytest.mark.parametrize(
    "order",
    [
        (_DEEPSEEK_TARGET, _DY_TARGET, "preserved-provider"),
        (_DY_TARGET, _DEEPSEEK_TARGET, "preserved-provider"),
        (_DY_TARGET, "preserved-provider", _DEEPSEEK_TARGET),
    ],
)
def test_apply_splices_block_style_deepseek_at_any_provider_position_exactly(
    tmp_path: Path, order: tuple[str, str, str]
):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    providers = yaml.safe_load(target.read_text(encoding="utf-8"))["providers"]
    providers["preserved-provider"] = {
        "api": "openai-responses",
        "apiKey": "PRESERVED_PROVIDER_KEY",
        "baseUrl": "https://preserved.invalid/v1",
        "models": [],
    }
    entries = {
        name: _block_style_provider_entry(name, providers[name]) for name in order
    }
    entries[_DEEPSEEK_TARGET] = (
        entries[_DEEPSEEK_TARGET].removesuffix("\n")
        + "  # exact inline comment for removed provider\n"
    )
    body = "providers:\n"
    expected = "providers:\n"
    for index, name in enumerate(order):
        if index:
            body += "\n"
            expected += "\n"
        leading_comment = f"  # exact leading comment for {name}\n"
        body += leading_comment + entries[name]
        expected += leading_comment
        if name != _DEEPSEEK_TARGET:
            expected += entries[name]
    if order[-1] == _DEEPSEEK_TARGET:
        trailing_comments = "# exact trailing file comment\n\n"
        body += trailing_comments
        expected += trailing_comments
    target.write_text(
        "# CC Switch source projection; generated for OMP.\n" + body,
        encoding="utf-8",
    )
    plan = _build(db, target)

    report = _apply(plan, db, target)

    rendered = target.read_text(encoding="utf-8")
    assert rendered == "# CC Switch source projection; generated for OMP.\n" + expected
    assert report["withinRun"]["deleted"] == [_DEEPSEEK_TARGET]
    assert list(yaml.safe_load(rendered)["providers"]) == [
        name for name in order if name != _DEEPSEEK_TARGET
    ]


def test_apply_splice_consumes_multiline_flow_collection_closer(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    providers = yaml.safe_load(target.read_text(encoding="utf-8"))["providers"]
    retained = _block_style_provider_entry(_DY_TARGET, providers[_DY_TARGET])
    removed = (
        "  cc-deepseek:\n"
        "    auth: apiKey\n"
        "    baseUrl: https://deepseek.invalid/v1\n"
        "    apiKey: CC_SWITCH_DEEPSEEK_KEY\n"
        "    api: anthropic-messages\n"
        "    models: [\n"
        "      {id: deepseek-chat}\n"
        "    ]\n"
    )
    leading_comment = "  # retained provider after multiline flow\n"
    header = "# CC Switch source projection; generated for OMP.\nproviders:\n"
    target.write_text(header + removed + leading_comment + retained, encoding="utf-8")
    plan = _build(db, target)

    _apply(plan, db, target)

    assert target.read_text(encoding="utf-8") == header + leading_comment + retained


def test_apply_rejects_target_ctime_but_allows_unrelated_parent_entries(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    original_mode = target.stat().st_mode & 0o777
    os.chmod(target, original_mode ^ 0o100)
    os.chmod(target, original_mode)

    with pytest.raises(ProjectionPlanError) as ctime_changed:
        _apply(plan, db, target)
    assert ctime_changed.value.code == "blocked-target-changed"

    parent = tmp_path / "parent-identity"
    db = _write_canonical_db(parent)
    target = _write_generated_models(parent)
    plan = _build(db, target)
    (parent / "unrelated-parent-entry").write_text(
        "changes parent ctime", encoding="utf-8"
    )

    report = _apply(plan, db, target)
    assert report["withinRun"]["deleted"] == ["cc-deepseek"]


def test_apply_locks_old_and_replacement_target_inodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    plan = _build(db, target)
    original_inode = target.stat().st_ino
    locked_inodes: list[int] = []
    original_lock = materializer._acquire_file_lock

    def record_lock(descriptor: int, code: str) -> None:
        locked_inodes.append(os.fstat(descriptor).st_ino)
        original_lock(descriptor, code)

    monkeypatch.setattr(materializer, "_acquire_file_lock", record_lock)
    _apply(plan, db, target)

    assert original_inode in locked_inodes
    assert target.stat().st_ino in locked_inodes
    assert len(locked_inodes) >= 2


def test_build_refuses_hardlinked_target(tmp_path: Path):
    db = _write_canonical_db(tmp_path)
    target = _write_generated_models(tmp_path)
    sibling = tmp_path / "models-hardlink.yml"
    os.link(target, sibling)

    with pytest.raises(ProjectionPlanError) as raised:
        _build(db, target)

    assert raised.value.code == "blocked-target-path"
