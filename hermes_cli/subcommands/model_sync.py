"""``hermes model-sync cc-switch`` command surface.

This command only inventories and builds explicit plans unless ``apply --yes`` is
selected.  Its machine-readable output deliberately contains no exception text.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import stat
from pathlib import Path
from typing import Any, Callable


_BLOCKED_INVALID_JSON = "blocked-invalid-json"
_BLOCKED_INVALID_COMMAND = "blocked-invalid-command"
_BLOCKED_OUTPUT = "blocked-output"
_BLOCKED_UNSUPPORTED_PLATFORM = "blocked-unsupported-platform"
_CONFIRMATION_REQUIRED = "confirmation-required"
_OMP_PACKAGE_PATH_FALLBACK = Path(
    "/Users/chiyuchou/.npm-global/lib/node_modules/@oh-my-pi/pi-coding-agent/package.json"
)


class _ParseError(ValueError):
    """A CLI JSON input failed its public, non-sensitive validation boundary."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError
        value[key] = item
    return value


def _reject_json_constant(_constant: str) -> None:
    raise ValueError


def _json_mapping(raw: str) -> dict[str, Any]:
    """Parse one command-line JSON object without accepting ambiguous JSON."""
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError, TypeError, ValueError):
        raise _ParseError(_BLOCKED_INVALID_JSON) from None
    if not isinstance(value, dict):
        raise _ParseError(_BLOCKED_INVALID_JSON)
    return value


def _load_materializer():
    """Import the Unix-only transaction implementation only for a selected action."""
    try:
        return importlib.import_module("tools.omp_model_materializer")
    except (ImportError, AttributeError):
        raise _DependencyUnavailable from None


def _load_source():
    """Import the CC Switch reader only for the inventory action."""
    try:
        return importlib.import_module("tools.cc_switch_omp_source")
    except (ImportError, AttributeError):
        raise _DependencyUnavailable from None


def _load_handoff():
    """Import the OMP handoff implementation only for the handoff action."""
    try:
        return importlib.import_module("tools.omp_config_handoff")
    except (ImportError, AttributeError):
        raise _DependencyUnavailable from None


def _migration_omp_package_path() -> str | os.PathLike[str]:
    """Return the reviewed installed OMP manifest path for collision guards."""
    try:
        migration = importlib.import_module("tools.omp_provider_migration")
        value = getattr(migration, "_OMP_PACKAGE_PATH", None)
    except (ImportError, AttributeError):
        value = None
    return value if isinstance(value, (str, os.PathLike)) else _OMP_PACKAGE_PATH_FALLBACK


def _secure_materializer_supported() -> bool:
    """Whether this host exposes the primitives required for safe mutation."""
    no_follow = getattr(os, "O_NOFOLLOW", None)
    if not isinstance(no_follow, int) or no_follow == 0:
        return False
    try:
        fcntl = importlib.import_module("fcntl")
    except ImportError:
        return False
    return callable(getattr(fcntl, "flock", None))


class _DependencyUnavailable(RuntimeError):
    """A selected model-sync implementation cannot run on this host."""


def _load_plan(path: str | os.PathLike[str], materializer: Any) -> dict[str, Any]:
    """Delegate strict no-follow loading and schema validation to the materializer."""
    return materializer.validate_projection_plan(
        materializer.load_projection_plan(path)
    )


def _accepted_plan_sha256(value: Any) -> str:
    """Require an independently supplied canonical plan digest before apply."""
    if not isinstance(value, str) or len(value) != 64:
        raise _ParseError("blocked-accepted-plan-sha256")
    try:
        int(value, 16)
    except ValueError:
        raise _ParseError("blocked-accepted-plan-sha256") from None
    return value.lower()


def _parse_accepted_plan_sha256(value: str) -> str:
    """Argparse boundary for a non-sensitive independent plan digest."""
    try:
        return _accepted_plan_sha256(value)
    except _ParseError as error:
        raise argparse.ArgumentTypeError(error.code) from None


def _render(value: Any) -> bytes:
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError, RecursionError):
        raise _ParseError(_BLOCKED_OUTPUT) from None


class _OutputReservation:
    """A create-once 0600 report inode held until the command completes."""

    def __init__(self, destination: Path, descriptor: int) -> None:
        self.destination = destination
        self.descriptor = descriptor
        details = os.fstat(descriptor)
        self.identity = (details.st_dev, details.st_ino)
        self.finalized = False

    def _same_file(self) -> bool:
        if self.descriptor == -1:
            return False
        try:
            path_details = os.lstat(self.destination)
            fd_details = os.fstat(self.descriptor)
        except (OSError, ValueError):
            return False
        return (
            stat.S_ISREG(path_details.st_mode)
            and stat.S_ISREG(fd_details.st_mode)
            and path_details.st_dev == self.identity[0]
            and path_details.st_ino == self.identity[1]
            and fd_details.st_dev == self.identity[0]
            and fd_details.st_ino == self.identity[1]
        )

    def write(self, payload: bytes) -> None:
        if self.finalized or self.descriptor == -1:
            raise _ParseError(_BLOCKED_OUTPUT)
        try:
            if not self._same_file():
                raise OSError
            os.lseek(self.descriptor, 0, os.SEEK_SET)
            os.ftruncate(self.descriptor, 0)
            remaining = memoryview(payload)
            while remaining:
                written = os.write(self.descriptor, remaining)
                if written <= 0:
                    raise OSError
                remaining = remaining[written:]
            os.fsync(self.descriptor)
            if not self._same_file():
                raise OSError
            os.close(self.descriptor)
            self.descriptor = -1
            self.finalized = True
        except (OSError, TypeError, ValueError):
            self.abort()
            raise _ParseError(_BLOCKED_OUTPUT) from None

    def abort(self) -> None:
        if self.finalized:
            return
        same_file = self._same_file()
        descriptor = self.descriptor
        self.descriptor = -1
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        if same_file:
            with contextlib.suppress(OSError):
                os.unlink(self.destination)


def _reserve_output(path: str | os.PathLike[str]) -> _OutputReservation:
    descriptor = -1
    destination: Path | None = None
    created_identity: tuple[int, int] | None = None
    try:
        destination = Path(path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        if isinstance(no_follow, int):
            flags |= no_follow
        descriptor = os.open(destination, flags, 0o600)
        details = os.fstat(descriptor)
        created_identity = (details.st_dev, details.st_ino)
        os.fchmod(descriptor, 0o600)
        return _OutputReservation(destination, descriptor)
    except (OSError, TypeError, ValueError):
        if (
            descriptor != -1
            and destination is not None
            and created_identity is not None
        ):
            with contextlib.suppress(OSError):
                current = os.lstat(destination)
                if (current.st_dev, current.st_ino) == created_identity:
                    os.unlink(destination)
        if descriptor != -1:
            with contextlib.suppress(OSError):
                os.close(descriptor)
        raise _ParseError(_BLOCKED_OUTPUT) from None

def _write_output(path: str | os.PathLike[str], payload: bytes) -> None:
    """Create and fill a private JSON document without replacing a file."""
    reservation = _reserve_output(path)
    try:
        reservation.write(payload)
    except BaseException:
        reservation.abort()
        raise


def _guard_output_path(
    output: str | os.PathLike[str] | None,
    protected_paths: list[str | os.PathLike[str] | None],
) -> None:
    """Reject output paths that alias an input, including links and hardlinks."""
    if output is None:
        return
    try:
        raw_output = os.fspath(output)
        if isinstance(raw_output, bytes):
            raw_output = os.fsdecode(raw_output)
        if not isinstance(raw_output, str) or not raw_output:
            raise ValueError
        destination = Path(raw_output)
        destination_real = destination.resolve(strict=False)
        for protected in protected_paths:
            if protected is None:
                continue
            raw_protected = os.fspath(protected)
            if isinstance(raw_protected, bytes):
                raw_protected = os.fsdecode(raw_protected)
            if not isinstance(raw_protected, str) or not raw_protected:
                raise ValueError
            protected_path = Path(raw_protected)
            if destination_real == protected_path.resolve(strict=False):
                raise _ParseError(_BLOCKED_OUTPUT)
            try:
                if destination.exists() and protected_path.exists() and os.path.samefile(
                    destination, protected_path
                ):
                    raise _ParseError(_BLOCKED_OUTPUT)
            except FileNotFoundError:
                pass
    except _ParseError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError):
        raise _ParseError(_BLOCKED_OUTPUT) from None


def _prepare_output(
    output: str | os.PathLike[str] | None,
    protected_paths: list[str | os.PathLike[str] | None],
) -> _OutputReservation | None:
    if output is None:
        return None
    _guard_output_path(output, protected_paths)
    return _reserve_output(output)


@contextlib.contextmanager
def _output_slot(
    output: str | os.PathLike[str] | None,
    protected_paths: list[str | os.PathLike[str] | None],
):
    reservation = _prepare_output(output, protected_paths)
    try:
        yield reservation
    except BaseException:
        if reservation is not None:
            reservation.abort()
        raise


def _run_with_output(
    output: str | os.PathLike[str] | None,
    protected_paths: list[str | os.PathLike[str] | None],
    operation: Callable[[], Any],
) -> Any:
    with _output_slot(output, protected_paths) as reservation:
        result = operation()
        _emit(result, output, reservation=reservation)
        return result


def _emit(
    value: Any,
    output: str | os.PathLike[str] | None = None,
    *,
    reservation: _OutputReservation | None = None,
) -> None:
    """Emit product-owned JSON to stdout or a reserved mode-0600 output file."""
    payload = _render(value)
    if reservation is not None:
        reservation.write(payload)
    elif output is None:
        print(payload.decode("utf-8"), end="")
    else:
        _write_output(output, payload)


def _blocked(code: str) -> int:
    print(json.dumps({"status": "blocked", "code": code}, separators=(",", ":")))
    return 1


def _inventory(args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)

    def operation() -> Any:
        source = _load_source()
        return source.inventory_source(args.db, provider_filter=args.provider)

    _run_with_output(output, [getattr(args, "db", None)], operation)
    return 0


def _plan(args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)
    provider_migration = getattr(args, "provider_migration", False)
    config: str | None = None
    environment: str | None = None
    probe_evidence_paths: list[str] = []
    if provider_migration:
        config = getattr(args, "config", None)
        environment = getattr(args, "env", None)
        probe_evidence = getattr(args, "probe_evidence", None)
        if isinstance(probe_evidence, str):
            probe_evidence_paths = [probe_evidence]
        elif isinstance(probe_evidence, list) and all(
            isinstance(value, str) and value for value in probe_evidence
        ):
            probe_evidence_paths = list(probe_evidence)
        if (
            not isinstance(config, str)
            or not config
            or not isinstance(environment, str)
            or not environment
            or not probe_evidence_paths
        ):
            raise _ParseError("blocked-migration-arguments")
        protected_paths = [
            getattr(args, "db", None),
            getattr(args, "target", None),
            config,
            environment,
            *probe_evidence_paths,
        ]
        protected_paths.append(_migration_omp_package_path())
    else:
        protected_paths = [
            getattr(args, "db", None),
            getattr(args, "target", None),
        ]

    def operation() -> Any:
        materializer = _load_materializer()
        if provider_migration:
            return materializer.build_provider_migration_plan(
                db_path=args.db,
                models_path=args.target,
                config_path=config,
                environment_path=environment,
                provider_map=_json_mapping(args.provider_map),
                selected_models=_json_mapping(args.selected_models),
                probe_evidence_path=probe_evidence_paths,
                approved_current_origin_source_ids=set(args.approve_origin),
                enable_xuanju=getattr(args, "enable_xuanju", False),
                omp_version=args.omp_version,
            )
        return materializer.build_projection_plan(
            db_path=args.db,
            target_path=args.target,
            provider_map=_json_mapping(args.provider_map),
            selected_models=_json_mapping(args.selected_models),
            excluded_source_ids=set(args.exclude),
            approved_current_origin_source_ids=set(args.approve_origin),
            omp_version=args.omp_version,
        )

    _run_with_output(output, protected_paths, operation)
    return 0


def _apply(args: argparse.Namespace) -> int:
    if not args.yes:
        return _blocked(_CONFIRMATION_REQUIRED)
    accepted_plan_sha256 = _accepted_plan_sha256(
        getattr(args, "accepted_plan_sha256", None)
    )
    output = getattr(args, "output", None)
    provider_migration = getattr(args, "provider_migration", False)
    config: str | None = None
    environment: str | None = None
    if provider_migration:
        config = getattr(args, "config", None)
        environment = getattr(args, "env", None)
        if not all(isinstance(value, str) and value for value in (config, environment)):
            raise _ParseError("blocked-migration-arguments")
        protected_paths = [
            getattr(args, "plan", None),
            getattr(args, "db", None),
            getattr(args, "target", None),
            config,
            environment,
        ]
        protected_paths.append(_migration_omp_package_path())
    else:
        protected_paths = [
            getattr(args, "plan", None),
            getattr(args, "db", None),
            getattr(args, "target", None),
        ]

    def operation() -> Any:
        materializer = _load_materializer()
        if provider_migration:
            return materializer.apply_provider_migration_plan(
                materializer.load_provider_migration_plan(args.plan),
                args.db,
                args.target,
                config,
                environment,
                accepted_plan_sha256=accepted_plan_sha256,
                allow_opaque_backup=getattr(args, "allow_opaque_backup", False),
            )
        return materializer.apply_projection_plan(
            _load_plan(args.plan, materializer),
            args.db,
            args.target,
            accepted_plan_sha256=accepted_plan_sha256,
            allow_opaque_backup=getattr(args, "allow_opaque_backup", False),
        )

    _run_with_output(output, protected_paths, operation)
    return 0


def _verify(args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)
    provider_migration = getattr(args, "provider_migration", False)
    config: str | None = None
    environment: str | None = None
    if provider_migration:
        config = getattr(args, "config", None)
        environment = getattr(args, "env", None)
        if not all(isinstance(value, str) and value for value in (config, environment)):
            raise _ParseError("blocked-migration-arguments")
        protected_paths = [
            getattr(args, "plan", None),
            getattr(args, "db", None),
            getattr(args, "target", None),
            config,
            environment,
        ]
        protected_paths.append(_migration_omp_package_path())
    else:
        protected_paths = [
            getattr(args, "plan", None),
            getattr(args, "db", None),
            getattr(args, "target", None),
        ]

    def operation() -> Any:
        materializer = _load_materializer()
        if provider_migration:
            return materializer.verify_provider_migration(
                materializer.load_provider_migration_plan(args.plan),
                args.db,
                args.target,
                config,
                environment,
            )
        return materializer.verify_projection(
            _load_plan(args.plan, materializer), args.db, args.target
        )

    verified = _run_with_output(
        output,
        protected_paths,
        lambda: {
            "status": "verified" if operation() else "blocked"
        },
    )
    return 0 if verified["status"] == "verified" else 1


def _handoff_plan(args: argparse.Namespace) -> int:
    output = getattr(args, "output", None)

    def operation() -> Any:
        materializer = _load_materializer()
        handoff = _load_handoff()
        return handoff.build_config_handoff(
            _load_plan(args.plan, materializer),
            args.config,
            args.repeat_verified_selector,
        )

    _run_with_output(
        output,
        [getattr(args, "plan", None), getattr(args, "config", None)],
        operation,
    )
    return 0


_ACTION_HANDLERS: dict[str, Callable[[argparse.Namespace], int]] = {
    "inventory": _inventory,
    "plan": _plan,
    "apply": _apply,
    "verify": _verify,
    "handoff-plan": _handoff_plan,
}


def _operation_error_code(error: Exception) -> str:
    """Keep implementation errors machine-readable without exposing details."""
    code = getattr(error, "code", None)
    return (
        code
        if isinstance(code, str) and code.startswith("blocked-")
        else _BLOCKED_INVALID_JSON
    )


def cmd_model_sync(args: argparse.Namespace) -> int:
    """Dispatch the explicit CC Switch model-projection workflow."""
    try:
        handler = _ACTION_HANDLERS.get(args.model_sync_action)
        if handler is None:
            return _blocked(_BLOCKED_INVALID_COMMAND)
        if not _secure_materializer_supported():
            return _blocked(_BLOCKED_UNSUPPORTED_PLATFORM)
        return handler(args)
    except _DependencyUnavailable:
        return _blocked(_BLOCKED_UNSUPPORTED_PLATFORM)
    except _ParseError as error:
        return _blocked(error.code)
    except Exception as error:
        return _blocked(_operation_error_code(error))


def build_model_sync_parser(
    subparsers, *, cmd_model_sync: Callable = cmd_model_sync
) -> None:
    """Attach the standalone, deliberately narrow ``model-sync cc-switch`` tree."""
    parser = subparsers.add_parser(
        "model-sync",
        help="Safely inventory and project CC Switch Claude models for OMP",
    )
    targets = parser.add_subparsers(dest="model_sync_target", required=True)
    cc_switch = targets.add_parser(
        "cc-switch", help="Operate on a CC Switch provider database"
    )
    actions = cc_switch.add_subparsers(dest="model_sync_action", required=True)

    inventory = actions.add_parser(
        "inventory", help="Read a redacted provider inventory"
    )
    inventory.add_argument("--db", required=True)
    inventory.add_argument("--provider")
    inventory.add_argument("--output")
    inventory.add_argument("--json", action="store_true")

    plan = actions.add_parser(
        "plan",
        help="Build a read-only projection plan; provider migration is explicit",
    )
    plan.add_argument("--db", required=True)
    plan.add_argument("--target", required=True)
    plan.add_argument("--provider-map", required=True)
    plan.add_argument("--selected-models", required=True)
    plan.add_argument("--exclude", action="append", default=[])
    plan.add_argument("--approve-origin", action="append", default=[])
    plan.add_argument("--omp-version", default="18.2.0")
    plan.add_argument(
        "--provider-migration",
        action="store_true",
        help="Use the opt-in cc-xuanju three-file migration contract",
    )
    plan.add_argument(
        "--enable-xuanju",
        action="store_true",
        help="Explicitly opt in to creating cc-xuanju",
    )
    plan.add_argument("--config")
    plan.add_argument("--env")
    plan.add_argument("--probe-evidence", action="append", default=[])
    plan.add_argument("--output")
    plan.add_argument("--json", action="store_true")

    apply = actions.add_parser(
        "apply", help="Apply a previously generated projection plan"
    )
    apply.add_argument("--plan", required=True)
    apply.add_argument("--db", required=True)
    apply.add_argument("--target", required=True)
    apply.add_argument(
        "--provider-migration",
        action="store_true",
        help="Apply only the separately accepted cc-xuanju migration plan",
    )
    apply.add_argument("--config")
    apply.add_argument("--env")
    apply.add_argument("--yes", action="store_true")
    apply.add_argument(
        "--accepted-plan-sha256",
        required=True,
        type=_parse_accepted_plan_sha256,
        help="Independently accepted SHA-256 digest of the exact plan",
    )
    apply.add_argument(
        "--allow-opaque-backup",
        action="store_true",
        help="Explicitly permit an opaque rollback backup for this apply only",
    )
    apply.add_argument("--output")
    apply.add_argument("--json", action="store_true")

    verify = actions.add_parser("verify", help="Verify a projection against its plan")
    verify.add_argument("--plan", required=True)
    verify.add_argument("--db", required=True)
    verify.add_argument("--target", required=True)
    verify.add_argument(
        "--provider-migration",
        action="store_true",
        help="Verify the cc-xuanju three-file migration contract",
    )
    verify.add_argument("--config")
    verify.add_argument("--env")
    verify.add_argument("--output")
    verify.add_argument("--json", action="store_true")

    handoff = actions.add_parser(
        "handoff-plan", help="Build a read-only OMP config handoff"
    )
    handoff.add_argument("--plan", required=True)
    handoff.add_argument("--config", required=True)
    handoff.add_argument("--repeat-verified-selector", action="append", default=[])
    handoff.add_argument("--output")
    handoff.add_argument("--json", action="store_true")

    parser.set_defaults(func=cmd_model_sync)
