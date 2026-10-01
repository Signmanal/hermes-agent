"""Focused no-network contracts for the temporary CC Switch model runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
import socket
import sqlite3
import ssl
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

import httpx

import tools.cc_switch_model_discovery as discovery
from tools.cc_switch_omp_source import _migration_apply_source


DY_SECRET = "marker-dy-discovery-token"
XUANJU_SECRET = "marker-xuanju-discovery-token"
DY_ENDPOINT = "https://dy.discovery.test/v1"
XUANJU_ENDPOINT = "https://xuanju.discovery.test/explicit/base"
# The discovery runner opens each ancestor of an input path one level at a time
# with O_NOFOLLOW and refuses a group/world-writable directory. So the fixture
# root must (a) sit OUTSIDE the Hermes home (tests/home_io_guard.py rejects I/O
# against the real home, and the checkout lives at ~/.hermes/hermes-agent), and
# (b) have an all-user-owned, non-world-writable ancestry with no symlinks.
# ~/.cache is user-owned 0755; the world-writable /private/var/tmp and the
# symlinked /var, /tmp are both rejected.
_TEST_TEMP_ROOT = Path.home() / ".cache"
_TEST_TEMP_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)


class _BytesStream(httpx.AsyncByteStream):
    def __init__(self, body: bytes) -> None:
        self._body = body

    async def __aiter__(self):
        yield self._body

    async def aclose(self) -> None:
        return None


def _response(
    status: int,
    *,
    value: Any | None = None,
    content: bytes | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    body = content
    if body is None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
    return httpx.Response(status, headers=headers, stream=_BytesStream(body))
def _write_source_db(tmp_path: Path) -> Path:
    path = tmp_path / "cc-switch.db"
    rows = [
        (
            discovery._DY_SOURCE_ID,
            "DY synthetic",
            DY_ENDPOINT,
            DY_SECRET,
            1,
        ),
        (
            discovery._XUANJU_SOURCE_ID,
            "Xuanju synthetic",
            XUANJU_ENDPOINT,
            XUANJU_SECRET,
            0,
        ),
    ]
    with sqlite3.connect(path) as connection:
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
            ) VALUES (?, 'claude', ?, ?, ?, ?, ?)
            """,
            [
                (
                    source_id,
                    name,
                    json.dumps(
                        {
                            "env": {
                                "ANTHROPIC_BASE_URL": endpoint,
                                "ANTHROPIC_AUTH_TOKEN": credential,
                            },
                            "models": [{"id": "fixture-model"}],
                        }
                    ),
                    json.dumps({"apiFormat": "openai_responses"}),
                    current,
                    index,
                )
                for index, (source_id, name, endpoint, credential, current) in enumerate(
                    rows, start=1
                )
            ],
        )
    path.chmod(0o600)
    return path


def _private_source(path: Path) -> dict[str, Any]:
    return _migration_apply_source(
        path,
        selected_source_ids=discovery._SOURCE_IDS,
        approved_origin_source_ids=discovery._SOURCE_IDS,
    )


def _authorization_from(source: dict[str, Any]) -> dict[str, Any]:
    precondition = source["precondition"]
    by_source_id = {record["source_id"]: record for record in source["providers"]}
    return {
        "schemaVersion": 1,
        "sourceSnapshotSha256": source["source_snapshot_sha256"],
        "sourcePrecondition": {
            "realpath": precondition["realpath"],
            "database": {
                key: precondition[key] for key in discovery._FILE_MANIFEST_KEYS
            },
            "wal": precondition["wal"],
            # Test databases use rollback journal mode, so there is no SHM
            # sidecar. A real authorization must include its full manifest.
            "shm": precondition["shm"],
        },
        "providers": [
            {
                "provider": label,
                "sourceId": source_id,
                "sourceApiFamily": by_source_id[source_id]["api_family"],
                "endpointSha256": by_source_id[source_id]["endpoint_sha256"],
                "applyBindingSha256": by_source_id[source_id][
                    "apply_binding_sha256"
                ],
            }
            for label, source_id in discovery._PROVIDERS
        ],
    }


def _write_authorization(tmp_path: Path, source: dict[str, Any]) -> tuple[Path, str]:
    payload = json.dumps(
        _authorization_from(source), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    path = tmp_path / "accepted-discovery.json"
    path.write_bytes(payload)
    path.chmod(0o600)
    return path, hashlib.sha256(payload).hexdigest()


def _inputs(tmp_path: Path) -> tuple[Path, Path, str]:
    db_path = _write_source_db(tmp_path)
    authorization_path, accepted_digest = _write_authorization(
        tmp_path, _private_source(db_path)
    )
    return db_path, authorization_path, accepted_digest


def _blocked(case: unittest.TestCase, result: dict[str, Any]) -> None:
    case.assertEqual(
        result,
        {
            label: {
                "statusCategory": "source_blocked",
                "httpStatus": None,
                "modelCount": 0,
                "models": [],
            }
            for label, _source_id in discovery._PROVIDERS
        },
    )


class TestCcSwitchModelDiscovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._temporary = tempfile.TemporaryDirectory(dir=_TEST_TEMP_ROOT)
        self.addCleanup(self._temporary.cleanup)
        self.tmp_path = Path(self._temporary.name)
        self.db_path, self.authorization_path, self.accepted_digest = _inputs(
            self.tmp_path
        )
        # In-process calls exercise the sealed API, not the isolated CLI launch.
        # Keep SSLKEYLOGFILE enforcement active while supplying only the runtime
        # property that `python -I -B` would provide in production.
        self.enterContext(
            patch.object(discovery, "_is_isolated_runtime", return_value=True)
        )

        def _network_forbidden(*_args: Any, **_kwargs: Any) -> None:
            self.fail("focused discovery tests must not open a real network socket")

        self.enterContext(patch.object(socket.socket, "connect", _network_forbidden))

    async def _run(
        self, transport: httpx.AsyncBaseTransport | None = None
    ) -> dict[str, Any]:
        async def _invoke() -> dict[str, Any]:
            return await discovery.run_model_discovery(
                self.db_path,
                self.authorization_path,
                accepted_authorization_sha256=self.accepted_digest,
            )

        if transport is None:
            return await _invoke()
        with patch.object(discovery, "_new_transport", return_value=transport):
            return await _invoke()

    async def test_one_ordered_request_per_source_and_closed_schema(self) -> None:
        source_calls: list[object] = []
        original_source = discovery._migration_apply_source

        def _extract_once(*args: Any, **kwargs: Any) -> dict[str, Any]:
            source_calls.append(None)
            return original_source(*args, **kwargs)

        requests: list[tuple[str, str, str]] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            host = request.url.host
            requests.append((host, request.method, request.url.path))
            self.assertEqual(request.headers["accept-encoding"], "identity")
            self.assertTrue(request.headers["authorization"].startswith("Bearer "))
            if host == "dy.discovery.test":
                return _response(
                    200,
                    value={
                        "data": [
                            {
                                "id": "dy-model",
                                "supported_endpoint_types": ["openai-responses"],
                            },
                            {"id": "dy-unknown"},
                        ]
                    },
                )
            self.assertEqual(host, "xuanju.discovery.test")
            return _response(
                200,
                value={
                    "data": [
                        {
                            "id": "xuanju-model",
                            "supported_endpoint_types": ["anthropic-messages"],
                        }
                    ]
                },
            )

        with patch.object(discovery, "_migration_apply_source", _extract_once):
            result = await self._run(httpx.MockTransport(handler))

        self.assertEqual(source_calls, [None])
        self.assertEqual(
            requests,
            [
                ("dy.discovery.test", "GET", "/v1/models"),
                ("xuanju.discovery.test", "GET", "/explicit/base/v1/models"),
            ],
        )
        self.assertEqual(
            result,
            {
                "dy": {
                    "statusCategory": "ok",
                    "httpStatus": 200,
                    "modelCount": 2,
                    "models": [
                        {
                            "id": "dy-model",
                            "supportedEndpointTypesPresent": True,
                            "derivedApiFamily": "openai-responses",
                        },
                        {
                            "id": "dy-unknown",
                            "supportedEndpointTypesPresent": False,
                            "derivedApiFamily": "unknown",
                        },
                    ],
                },
                "xuanju": {
                    "statusCategory": "ok",
                    "httpStatus": 200,
                    "modelCount": 1,
                    "models": [
                        {
                            "id": "xuanju-model",
                            "supportedEndpointTypesPresent": True,
                            "derivedApiFamily": "anthropic-messages",
                        }
                    ],
                },
            },
        )

    async def test_transport_forbids_proxy_redirect_retry_http2_and_tls_downgrade(
        self,
    ) -> None:
        observed_transports: list[dict[str, Any]] = []
        requests: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.host)
            if request.url.host == "dy.discovery.test":
                return _response(
                    307,
                    headers={
                        "location": "https://redirected.discovery.test/v1/models"
                    },
                )
            return _response(200, value={"data": []})

        def _transport_factory(**kwargs: Any) -> httpx.AsyncBaseTransport:
            observed_transports.append(kwargs)
            return httpx.MockTransport(handler)

        with (
            patch.dict(os.environ, {"HTTPS_PROXY": "http://proxy.discovery.test:8080"}),
            patch.object(discovery.httpx, "AsyncHTTPTransport", _transport_factory),
            patch.object(discovery.certifi, "where", wraps=discovery.certifi.where) as ca_bundle,
        ):
            result = await self._run()

        self.assertEqual(requests, ["dy.discovery.test", "xuanju.discovery.test"])
        self.assertEqual(result["dy"]["statusCategory"], "redirect_blocked")
        self.assertEqual(result["dy"]["httpStatus"], 307)
        self.assertEqual(result["xuanju"]["statusCategory"], "ok")
        self.assertEqual(len(observed_transports), len(discovery._PROVIDERS))
        for observed_transport in observed_transports:
            self.assertEqual(observed_transport["retries"], 0)
            self.assertIs(observed_transport["trust_env"], False)
            self.assertIs(observed_transport["http1"], True)
            self.assertIs(observed_transport["http2"], False)
            context = observed_transport["verify"]
            self.assertIsInstance(context, ssl.SSLContext)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            self.assertTrue(context.check_hostname)
        self.assertEqual(ca_bundle.call_count, len(discovery._PROVIDERS))

    async def test_total_timeout_is_not_a_retry_and_second_source_stops_at_deadline(
        self,
    ) -> None:
        requests: list[str] = []

        async def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request.url.host)
            if request.url.host == "dy.discovery.test":
                await asyncio.sleep(0.05)
            return _response(200, value={"data": []})

        with patch.object(discovery, "_TOTAL_TIMEOUT_SECONDS", 0.01):
            result = await self._run(httpx.MockTransport(handler))

        self.assertEqual(requests, ["dy.discovery.test"])
        self.assertEqual(result["dy"]["statusCategory"], "timeout")
        self.assertEqual(result["xuanju"]["statusCategory"], "timeout")

    async def test_streamed_body_cap_is_enforced_without_content_length(self) -> None:
        class _OversizedStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b"x" * 65

            async def aclose(self) -> None:
                return None

        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "dy.discovery.test":
                return httpx.Response(200, stream=_OversizedStream())
            return _response(200, value={"data": []})

        with patch.object(discovery, "_MAX_BODY_BYTES", 64):
            result = await self._run(httpx.MockTransport(handler))

        self.assertEqual(result["dy"]["statusCategory"], "response_too_large")
        self.assertEqual(result["dy"]["models"], [])
        self.assertEqual(result["xuanju"]["statusCategory"], "ok")

    async def test_strict_json_and_unknown_endpoint_types_fail_closed_or_stay_unknown(
        self,
    ) -> None:
        self.assertEqual(
            discovery._normalise_models_url(
                "https://xuanju.discovery.test/explicit/base"
            ),
            "https://xuanju.discovery.test/explicit/base/v1/models",
        )
        self.assertEqual(
            discovery._normalise_models_url(
                "https://xuanju.discovery.test/explicit/v1/"
            ),
            "https://xuanju.discovery.test/explicit/v1/models",
        )
        for endpoint in (
            "https://xuanju.discovery.test/explicit/../base",
            "https://xuanju.discovery.test/explicit/%2e%2e/base",
            "https://xuanju.discovery.test/explicit//base",
            "https://xuanju.discovery.test/explicit/base?target=other",
            "https://xuanju.discovery.test/explicit/base#fragment",
            "https://user@xuanju.discovery.test/explicit/base",
        ):
            with self.subTest(endpoint=endpoint):
                with self.assertRaises(discovery._DiscoveryBlocked):
                    discovery._normalise_models_url(endpoint)
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "dy.discovery.test":
                return _response(200, content=b'{"data":[],"data":[]}')
            return _response(
                200,
                value={
                    "data": [
                        {
                            "id": "xuanju-new-api",
                            "supported_endpoint_types": ["unrecognized-api"],
                        }
                    ]
                },
            )

        result = await self._run(httpx.MockTransport(handler))

        self.assertEqual(result["dy"]["statusCategory"], "response_invalid")
        self.assertEqual(result["dy"]["httpStatus"], 200)
        xuanju = result["xuanju"]
        self.assertEqual(xuanju["statusCategory"], "ok")
        self.assertEqual(
            xuanju["models"],
            [
                {
                    "id": "xuanju-new-api",
                    "supportedEndpointTypesPresent": True,
                    "derivedApiFamily": "unknown",
                }
            ],
        )

    async def test_secret_taint_redacts_an_entire_provider_result(self) -> None:
        async def handler(request: httpx.Request) -> httpx.Response:
            if request.url.host == "dy.discovery.test":
                return _response(200, value={"data": [{"id": DY_SECRET}]})
            return _response(
                200,
                value={"data": [{"id": "XUANJU.DISCOVERY.TEST"}]},
            )

        result = await self._run(httpx.MockTransport(handler))

        self.assertEqual(
            result["dy"],
            {
                "statusCategory": "redaction_blocked",
                "httpStatus": None,
                "modelCount": 0,
                "models": [],
            },
        )
        self.assertEqual(result["xuanju"]["statusCategory"], "redaction_blocked")
        serialized = json.dumps(result, sort_keys=True)
        self.assertNotIn(DY_SECRET, serialized)
        self.assertNotIn(DY_ENDPOINT, serialized)
        self.assertNotIn(XUANJU_SECRET, serialized)
        self.assertNotIn(XUANJU_ENDPOINT, serialized)

    async def test_exact_two_private_source_records_are_required_before_network(
        self,
    ) -> None:
        original_source = discovery._migration_apply_source

        def _extra_record(*args: Any, **kwargs: Any) -> dict[str, Any]:
            source = original_source(*args, **kwargs)
            source["providers"].append(dict(source["providers"][0]))
            return source

        async def handler(_request: httpx.Request) -> httpx.Response:
            self.fail("source validation must finish before any request")

        with patch.object(discovery, "_migration_apply_source", _extra_record):
            result = await self._run(httpx.MockTransport(handler))

        _blocked(self, result)

    async def test_runtime_and_source_preconditions_block_before_network(
        self,
    ) -> None:
        async def handler(_request: httpx.Request) -> httpx.Response:
            self.fail("filesystem precondition failure must make zero requests")

        for mutation in (
            "group_readable",
            "symlink",
            "hardlink",
            "snapshot_mutated",
            "shm_precondition",
            "in_place_race",
        ):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(
                dir=_TEST_TEMP_ROOT
            ) as temporary:
                db_path, authorization_path, accepted_digest = _inputs(Path(temporary))
                supplied_path = db_path
                if mutation == "group_readable":
                    db_path.chmod(0o640)
                elif mutation == "symlink":
                    supplied_path = Path(temporary) / "db-link"
                    supplied_path.symlink_to(db_path)
                elif mutation == "hardlink":
                    supplied_path = Path(temporary) / "db-hardlink"
                    os.link(db_path, supplied_path)
                elif mutation == "snapshot_mutated":
                    with db_path.open("ab") as database:
                        database.write(b"\x00")
                elif mutation == "shm_precondition":
                    shm_path = db_path.with_name(db_path.name + "-shm")
                    shm_path.write_bytes(b"fixture-shm")
                    shm_path.chmod(0o600)
                with patch.object(
                    discovery,
                    "_new_transport",
                    return_value=httpx.MockTransport(handler),
                ):
                    if mutation == "in_place_race":
                        original_apply = discovery._migration_apply_source

                        def _mutate_after_apply(*args: Any, **kwargs: Any) -> dict[str, Any]:
                            source = original_apply(*args, **kwargs)
                            before = db_path.stat()
                            content = db_path.read_bytes()
                            marker = b"DY synthetic"
                            replacement = b"DY rewrite!"
                            self.assertEqual(len(marker), len(replacement))
                            self.assertIn(marker, content)
                            db_path.write_bytes(content.replace(marker, replacement, 1))
                            after = db_path.stat()
                            self.assertEqual(after.st_ino, before.st_ino)
                            self.assertEqual(after.st_size, before.st_size)
                            self.assertEqual(after.st_mode & 0o777, before.st_mode & 0o777)
                            return source

                        with patch.object(
                            discovery,
                            "_migration_apply_source",
                            _mutate_after_apply,
                        ):
                            result = await discovery.run_model_discovery(
                                supplied_path,
                                authorization_path,
                                accepted_authorization_sha256=accepted_digest,
                            )
                    else:
                        result = await discovery.run_model_discovery(
                            supplied_path,
                            authorization_path,
                            accepted_authorization_sha256=accepted_digest,
                        )
                _blocked(self, result)

        with patch.dict(os.environ, {"SSLKEYLOGFILE": "test-keylog"}):
            result = await self._run(httpx.MockTransport(handler))
        _blocked(self, result)

    def test_isolated_cli_invalid_input_emits_only_the_closed_blocked_schema(self) -> None:
        process = subprocess.run(
            [sys.executable, "-I", "-B", str(Path(discovery.__file__))],
            cwd=Path(discovery.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )

        self.assertEqual(process.returncode, 0)
        self.assertEqual(process.stderr, "")
        _blocked(self, json.loads(process.stdout))

        manifest_path = self.tmp_path / "generated-discovery.json"
        builder = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(discovery.__file__)),
                "--build-authorization",
                "--source-db",
                str(self.db_path),
                "--output",
                str(manifest_path),
            ],
            cwd=Path(discovery.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(builder.returncode, 0)
        self.assertEqual(builder.stderr, "")
        summary = json.loads(builder.stdout)
        self.assertEqual(
            set(summary),
            {"status", "path", "authorizationSha256", "providerCount"},
        )
        self.assertEqual(summary["status"], "authorization_written")
        self.assertEqual(summary["path"], str(manifest_path.resolve()))
        self.assertEqual(summary["providerCount"], len(discovery._PROVIDERS))
        payload = manifest_path.read_bytes()
        self.assertEqual(manifest_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(
            summary["authorizationSha256"], hashlib.sha256(payload).hexdigest()
        )
        serialized = payload.decode("utf-8")
        self.assertNotIn(DY_SECRET, serialized)
        self.assertNotIn(XUANJU_SECRET, serialized)
        self.assertNotIn(DY_ENDPOINT, serialized)
        self.assertNotIn(XUANJU_ENDPOINT, serialized)

        second = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(discovery.__file__)),
                "--build-authorization",
                "--source-db",
                str(self.db_path),
                "--output",
                str(manifest_path),
            ],
            cwd=Path(discovery.__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(second.returncode, 0)
        self.assertEqual(second.stderr, "")
        _blocked(self, json.loads(second.stdout))
        self.assertEqual(manifest_path.read_bytes(), payload)

        watchdog_temp_root = Path(tempfile.gettempdir())
        roots_before = set(watchdog_temp_root.glob(".hermes-cc-watchdog-*"))
        injected_environment = os.environ.copy()
        injected_environment[discovery._WATCHDOG_ENV] = "external-token"
        injected = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(Path(discovery.__file__)),
                "--__cc-switch-watchdog=external-token",
            ],
            cwd=Path(discovery.__file__).resolve().parents[1],
            env=injected_environment,
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        self.assertEqual(injected.returncode, 0)
        self.assertEqual(injected.stderr, "")
        _blocked(self, json.loads(injected.stdout))

        read_fd, write_fd = os.pipe()
        try:
            os.write(write_fd, b"x")
            with self.assertRaises(discovery._DiscoveryBlocked):
                discovery._read_watchdog_capability(
                    read_fd,
                    b"\x00" * discovery._WATCHDOG_TOKEN_BYTES,
                    discovery.time.monotonic() + 0.1,
                )
        finally:
            os.close(write_fd)
        def _slow_snapshot_payload(_argv: list[str]) -> dict[str, Any]:
            raw_copy = Path(tempfile.tempdir or tempfile.gettempdir()) / "raw-source-snapshot"
            raw_copy.write_bytes(b"x" * (1024 * 1024))
            discovery.time.sleep(1.0)
            return {}

        with (
            patch.object(discovery, "_run_cli_payload", _slow_snapshot_payload),
            patch.object(discovery, "_TOTAL_TIMEOUT_SECONDS", 1.0),
            patch.object(discovery, "_WATCHDOG_CLEANUP_RESERVE_SECONDS", 0.5),
        ):
            with self.assertRaises(discovery._DiscoveryBlocked):
                discovery._run_watchdog([])

        roots_after = set(watchdog_temp_root.glob(".hermes-cc-watchdog-*"))
        self.assertEqual(roots_after - roots_before, set())


if __name__ == "__main__":  # pragma: no cover - direct focused-test entry point.
    unittest.main()
