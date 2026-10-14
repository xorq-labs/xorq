from __future__ import annotations

import base64
import hashlib
import io
import json
import pickle
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest
from attr import define, field, frozen
from git import Repo
from git.cmd import Git

import xorq.api as xo
from xorq.catalog.backend import _FETCH_BATCH_SIZE, GitPointerBackend
from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.constants import CONTENT_STORE_YAML
from xorq.catalog.content_store import (
    _CONTROL_RETRY_ATTEMPTS,
    _CONTROL_RETRY_MAX_DELAY_SECONDS,
    _PRESIGNED_BATCH_SIZE,
    ContentCache,
    ContentSpec,
    ContentStoreConfig,
    PresignedContentStore,
    PresignedContentStoreConfig,
    compute_content_key,
    parse_pointer,
    write_pointer,
)
from xorq.catalog.exceptions import (
    CatalogConfigurationError,
    CatalogServiceHTTPError,
    ContentIntegrityError,
    ContentStoreCapabilityError,
    ContentStoreError,
)


CATALOG_ID = "11111111-1111-4111-8111-111111111111"
TOKEN_ENV = "XORQ_CATALOG_TOKEN"
TOKEN_SERVICE_ENV = "XORQ_CATALOG_TOKEN_SERVICE_URL"


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _upload_headers(sha256: str, size: int) -> dict[str, str]:
    checksum = base64.b64encode(bytes.fromhex(sha256)).decode("ascii")
    return {
        "content-length": str(size),
        "x-amz-checksum-sha256": checksum,
        "x-xorq-signed-test": f"upload-{sha256[:8]}",
    }


def _download_headers(sha256: str) -> dict[str, str]:
    return {"x-xorq-signed-test": f"download-{sha256[:8]}"}


@define
class _ServiceState:
    service_url: str = ""
    uploads: dict[str, tuple[str, int]] = field(factory=dict)
    objects: dict[str, bytes] = field(factory=dict)
    already_verified: set[str] = field(factory=set)
    corrupt_downloads: set[str] = field(factory=set)
    control_requests: list[tuple[str, str | None, dict[str, Any]]] = field(factory=list)
    object_requests: list[tuple[str, str, dict[str, str]]] = field(factory=list)
    completed: list[str] = field(factory=list)
    reverse_uploads: bool = False
    reverse_downloads: bool = False
    next_control_error: int | None = None
    control_error_repeats: int = 1
    control_error_retry_after: str | None = None
    rejected_authorizations: dict[str, tuple[int, str]] = field(factory=dict)
    redirect_download: bool = False
    redirect_was_followed: bool = False
    max_batch_size: int | None = None
    expire_next_upload_batch: bool = False
    expire_next_download_batch: bool = False
    download_expires_at: str | None = None
    upload_request_headers: dict[str, str] | None = None
    completion_response_overrides: dict[str, Any] = field(factory=dict)
    redirect_upload: bool = False
    next_upload_id: int = 1

    def upload_id(self, sha256: str, size: int) -> str:
        value = f"00000000-0000-4000-8000-{self.next_upload_id:012d}"
        self.next_upload_id += 1
        self.uploads[value] = (sha256, size)
        return value


class _HostedBlobHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self) -> _ServiceState:
        return self.server.state  # type: ignore[attr-defined, no-any-return]

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _body(self) -> bytes:
        length = int(self.headers.get("content-length", "0"))
        return self.rfile.read(length)

    def _json_body(self) -> dict[str, Any]:
        body = self._body()
        return json.loads(body or b"{}")

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _record_control(self, body: dict[str, Any]) -> None:
        self.state.control_requests.append(
            (self.path, self.headers.get("authorization"), body)
        )

    def _maybe_control_error(self) -> bool:
        status = self.state.next_control_error
        if status is None:
            return False
        self.state.control_error_repeats -= 1
        if self.state.control_error_repeats <= 0:
            self.state.next_control_error = None
        body = json.dumps({"error": "service_unavailable"}).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if self.state.control_error_retry_after is not None:
            self.send_header("retry-after", self.state.control_error_retry_after)
        self.end_headers()
        self.wfile.write(body)
        return True

    def do_POST(self) -> None:  # noqa: N802
        body = self._json_body()
        self._record_control(body)
        rejection = self.state.rejected_authorizations.get(
            self.headers.get("authorization", "")
        )
        if rejection is not None:
            status, error = rejection
            self._send_json(status, {"error": error})
            return
        if self._maybe_control_error():
            return

        prefix = f"/v1/catalogs/{CATALOG_ID}/blobs/"
        if self.path == f"{prefix}uploads:batch":
            if (
                self.state.max_batch_size is not None
                and len(body["objects"]) > self.state.max_batch_size
            ):
                self._send_json(400, {"error": "invalid_request"})
                return
            expires_at = (
                "2000-01-01T00:00:00Z"
                if self.state.expire_next_upload_batch
                else "2099-01-01T00:00:00Z"
            )
            self.state.expire_next_upload_batch = False
            uploads = []
            for spec in body["objects"]:
                sha256, size = spec["sha256"], spec["size"]
                upload_id = self.state.upload_id(sha256, size)
                if sha256 in self.state.already_verified:
                    upload = {
                        "upload_id": upload_id,
                        "sha256": sha256,
                        "size": size,
                        "status": "already_verified",
                    }
                else:
                    upload = {
                        "upload_id": upload_id,
                        "sha256": sha256,
                        "size": size,
                        "status": "upload_required",
                        "request": {
                            "url": f"{self.state.service_url}objects/{sha256}?signed=put",
                            "headers": (
                                self.state.upload_request_headers
                                if self.state.upload_request_headers is not None
                                else _upload_headers(sha256, size)
                            ),
                            "expires_at": expires_at,
                        },
                    }
                uploads.append(upload)
            if self.state.reverse_uploads:
                uploads.reverse()
            self._send_json(200, {"uploads": uploads})
            return

        if self.path.startswith(f"{prefix}uploads/") and self.path.endswith(
            "/complete"
        ):
            upload_id = self.path.removeprefix(f"{prefix}uploads/").removesuffix(
                "/complete"
            )
            sha256, size = self.state.uploads[upload_id]
            content = self.state.objects.get(sha256)
            if content is None or len(content) != size or _digest(content) != sha256:
                self._send_json(409, {"error": "blob_unavailable"})
                return
            self.state.completed.append(upload_id)
            self.state.already_verified.add(sha256)
            response = {
                "upload_id": upload_id,
                "sha256": sha256,
                "size": size,
                "status": "verified",
            }
            response.update(self.state.completion_response_overrides)
            self._send_json(200, response)
            return

        if self.path == f"{prefix}downloads:batch":
            if (
                self.state.max_batch_size is not None
                and len(body["objects"]) > self.state.max_batch_size
            ):
                self._send_json(400, {"error": "invalid_request"})
                return
            expires_at = (
                "2000-01-01T00:00:00Z"
                if self.state.expire_next_download_batch
                else self.state.download_expires_at or "2099-01-01T00:00:00Z"
            )
            self.state.expire_next_download_batch = False
            downloads = [
                {
                    "sha256": spec["sha256"],
                    "size": spec["size"],
                    "request": {
                        "url": (
                            f"{self.state.service_url}objects/{spec['sha256']}"
                            "?signed=get"
                        ),
                        "headers": _download_headers(spec["sha256"]),
                        "expires_at": expires_at,
                    },
                }
                for spec in body["objects"]
            ]
            if self.state.reverse_downloads:
                downloads.reverse()
            self._send_json(200, {"downloads": downloads})
            return

        self._send_json(404, {"error": "not_found"})

    def do_PUT(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/redirect-target":
            self._body()
            self.state.redirect_was_followed = True
            self._send_bytes(200, b"redirect must not be followed")
            return
        if not parsed.path.startswith("/objects/"):
            self._body()
            self._send_json(404, {"error": "not_found"})
            return
        sha256 = parsed.path.removeprefix("/objects/")
        content = self._body()
        headers = {name.lower(): value for name, value in self.headers.items()}
        self.state.object_requests.append(("PUT", sha256, headers))
        if self.state.redirect_upload:
            self.send_response(302)
            self.send_header("location", f"{self.state.service_url}redirect-target")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        expected_headers = _upload_headers(sha256, len(content))
        if any(headers.get(name) != value for name, value in expected_headers.items()):
            self._send_json(400, {"error": "signed_headers_changed"})
            return
        self.state.objects[sha256] = content
        self._send_bytes(200, b"")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlsplit(self.path)
        if parsed.path == "/redirect-target":
            self.state.redirect_was_followed = True
            self._send_bytes(200, b"redirect must not be followed")
            return
        if not parsed.path.startswith("/objects/"):
            self._send_json(404, {"error": "not_found"})
            return
        sha256 = parsed.path.removeprefix("/objects/")
        headers = {name.lower(): value for name, value in self.headers.items()}
        self.state.object_requests.append(("GET", sha256, headers))
        if self.state.redirect_download:
            self.send_response(302)
            self.send_header("location", f"{self.state.service_url}redirect-target")
            self.send_header("content-length", "0")
            self.end_headers()
            return
        expected_headers = _download_headers(sha256)
        if any(headers.get(name) != value for name, value in expected_headers.items()):
            self._send_json(400, {"error": "signed_headers_changed"})
            return
        content = self.state.objects[sha256]
        if sha256 in self.state.corrupt_downloads:
            content = bytes([content[0] ^ 0xFF]) + content[1:]
        self._send_bytes(200, content)


@frozen
class _ServiceFixture:
    url: str
    state: _ServiceState


@pytest.fixture
def hosted_blob_service() -> Iterator[_ServiceFixture]:
    state = _ServiceState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HostedBlobHandler)
    server.state = state  # type: ignore[attr-defined]
    host, port = server.server_address
    state.service_url = f"http://{host}:{port}/"
    thread = threading.Thread(
        target=server.serve_forever,
        kwargs={"poll_interval": 0.01},
        daemon=True,
    )
    thread.start()
    try:
        yield _ServiceFixture(state.service_url, state)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _store(service: _ServiceFixture) -> PresignedContentStore:
    return PresignedContentStore(
        catalog_id=CATALOG_ID,
        service_url=service.url,
        remote_url=f"{service.url}alice/demo.git",
    )


def _set_token(
    monkeypatch: pytest.MonkeyPatch,
    service: _ServiceFixture,
    token: str,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, token)
    monkeypatch.setenv(TOKEN_SERVICE_ENV, service.url)


def _backend(
    tmp_path: Path,
    service: _ServiceFixture,
    *,
    repo_name: str = "repo",
    remote_url: str | None = None,
) -> GitPointerBackend:
    repo = Repo.init(tmp_path / repo_name, initial_branch="main")
    repo.create_remote("origin", remote_url or f"{service.url}alice/demo.git")
    PresignedContentStoreConfig(
        catalog_id=CATALOG_ID, service_url=service.url
    ).write_yaml(Path(repo.working_dir) / CONTENT_STORE_YAML)
    cache = ContentCache(
        cache_dir=tmp_path / f"{repo_name}-cache", max_bytes=1024 * 1024
    )
    return GitPointerBackend.from_repo(repo, cache=cache)


def _prepare_downloads(
    backend: GitPointerBackend,
    service: _ServiceFixture,
    contents: tuple[bytes, ...],
) -> list[tuple[Path, str, str]]:
    entries = Path(backend.repo.working_dir) / "entries"
    prepared = []
    for index, content in enumerate(contents):
        sha256 = _digest(content)
        key = compute_content_key(CATALOG_ID, sha256)
        service.state.objects[sha256] = content
        service.state.already_verified.add(sha256)
        target = entries / f"entry-{index}.zip"
        write_pointer(backend._pointer_path(target), sha256, len(content))
        prepared.append((target, key, sha256))
    return prepared


def test_ensure_present_announces_puts_and_completes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    content = b"an archive sent through the signed object endpoint"
    sha256 = _digest(content)
    key = compute_content_key(CATALOG_ID, sha256)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    assert _store(hosted_blob_service).ensure_present(key, source, sha256=sha256)

    state = hosted_blob_service.state
    assert [
        urlsplit(path).path.rsplit("/", 1)[-1] for path, _, _ in state.control_requests
    ] == [
        "uploads:batch",
        "complete",
    ]
    assert all(auth == "Bearer write-token" for _, auth, _ in state.control_requests)
    assert state.control_requests[0][2] == {
        "objects": [{"sha256": sha256, "size": len(content)}]
    }
    assert state.objects[sha256] == content
    assert len(state.completed) == 1
    method, object_sha, headers = state.object_requests[0]
    assert (method, object_sha) == ("PUT", sha256)
    assert "authorization" not in headers


def test_ensure_present_many_correlates_results_and_skips_verified_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    state = hosted_blob_service.state
    state.reverse_uploads = True
    first, second = b"already stored", b"must be uploaded"
    first_sha, second_sha = _digest(first), _digest(second)
    state.already_verified.add(first_sha)
    state.objects[first_sha] = first
    first_path, second_path = tmp_path / "first.zip", tmp_path / "second.zip"
    first_path.write_bytes(first)
    second_path.write_bytes(second)
    first_key = compute_content_key(CATALOG_ID, first_sha)
    second_key = compute_content_key(CATALOG_ID, second_sha)

    uploaded = _store(hosted_blob_service).ensure_present_many(
        [
            (ContentSpec(first_key, first_sha, len(first)), first_path),
            (ContentSpec(second_key, second_sha, len(second)), second_path),
        ]
    )

    assert uploaded == {second_key}
    assert state.objects[second_sha] == second
    assert [method for method, _, _ in state.object_requests] == ["PUT"]
    assert (
        len(
            [
                path
                for path, _, _ in state.control_requests
                if path.endswith("uploads:batch")
            ]
        )
        == 1
    )


def test_upload_remints_an_expired_signed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    hosted_blob_service.state.expire_next_upload_batch = True
    content = b"upload after an expired first mint"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    assert _store(hosted_blob_service).ensure_present(
        compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
    )

    control_paths = [path for path, _, _ in hosted_blob_service.state.control_requests]
    assert sum(path.endswith("uploads:batch") for path in control_paths) == 2
    assert control_paths[-1].endswith("complete")
    assert hosted_blob_service.state.objects[sha256] == content


@pytest.mark.parametrize(
    ("header", "value", "match"),
    (
        ("content-length", None, "invalid content-length"),
        ("x-amz-checksum-sha256", "wrong", "invalid SHA-256 header"),
    ),
)
def test_upload_rejects_missing_or_wrong_integrity_headers_before_transfer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    header: str,
    value: str | None,
    match: str,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    content = b"upload header validation"
    sha256 = _digest(content)
    headers = _upload_headers(sha256, len(content))
    if value is None:
        headers.pop(header)
    else:
        headers[header] = value
    hosted_blob_service.state.upload_request_headers = headers
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError, match=match):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert hosted_blob_service.state.object_requests == []
    assert hosted_blob_service.state.completed == []


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("sha256", "0" * 64, "completed a different object"),
        (
            "upload_id",
            "00000000-0000-4000-8000-999999999999",
            "completed a different upload",
        ),
        ("status", "pending", "did not verify the upload"),
    ),
)
def test_upload_rejects_mismatched_completion_responses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    field: str,
    value: str,
    match: str,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    hosted_blob_service.state.completion_response_overrides[field] = value
    content = b"completion response validation"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError, match=match):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert len(hosted_blob_service.state.object_requests) == 1
    assert len(hosted_blob_service.state.completed) == 1


def test_signed_upload_redirect_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    hosted_blob_service.state.redirect_upload = True
    content = b"upload must stay on its signed URL"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert not hosted_blob_service.state.redirect_was_followed
    assert sha256 not in hosted_blob_service.state.objects
    assert hosted_blob_service.state.completed == []


def test_get_many_uses_one_mint_and_correlates_reordered_downloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    state = hosted_blob_service.state
    state.reverse_downloads = True
    first, second = b"first archive", b"second archive"
    first_sha, second_sha = _digest(first), _digest(second)
    state.objects.update({first_sha: first, second_sha: second})
    first_key = compute_content_key(CATALOG_ID, first_sha)
    second_key = compute_content_key(CATALOG_ID, second_sha)
    first_dest, second_dest = tmp_path / "first.zip", tmp_path / "second.zip"

    _store(hosted_blob_service).get_many(
        [
            (ContentSpec(first_key, first_sha, len(first)), first_dest),
            (ContentSpec(second_key, second_sha, len(second)), second_dest),
        ]
    )

    assert first_dest.read_bytes() == first
    assert second_dest.read_bytes() == second
    assert len(state.control_requests) == 1
    path, authorization, body = state.control_requests[0]
    assert path.endswith("downloads:batch")
    assert authorization == "Bearer read-token"
    assert body == {
        "objects": [
            {"sha256": first_sha, "size": len(first)},
            {"sha256": second_sha, "size": len(second)},
        ]
    }
    assert {sha for method, sha, _ in state.object_requests if method == "GET"} == {
        first_sha,
        second_sha,
    }
    for method, _sha256, headers in state.object_requests:
        assert method == "GET"
        assert "authorization" not in headers


def test_get_many_splits_to_the_service_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    hosted_blob_service.state.max_batch_size = 2
    objects = []
    expected = {}
    for index in range(5):
        content = f"archive-{index}".encode()
        sha256 = _digest(content)
        hosted_blob_service.state.objects[sha256] = content
        destination = tmp_path / f"{index}.zip"
        objects.append(
            (
                ContentSpec(
                    compute_content_key(CATALOG_ID, sha256),
                    sha256,
                    len(content),
                ),
                destination,
            )
        )
        expected[destination] = content

    _store(hosted_blob_service).get_many(objects)

    assert {path: path.read_bytes() for path in expected} == expected


def test_download_remints_an_expired_signed_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    hosted_blob_service.state.expire_next_download_batch = True
    content = b"download after an expired first mint"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    destination = tmp_path / "download.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content
    download_mints = [
        path
        for path, _, _ in hosted_blob_service.state.control_requests
        if path.endswith("downloads:batch")
    ]
    assert len(download_mints) == 2
    assert [method for method, _, _ in hosted_blob_service.state.object_requests] == [
        "GET"
    ]


def test_download_integrity_failure_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    content = b"same-size content corruption"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    hosted_blob_service.state.corrupt_downloads.add(sha256)
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"keep this existing archive")

    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch"):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            destination,
            sha256=sha256,
            size=len(content),
        )

    assert destination.read_bytes() == b"keep this existing archive"
    assert sorted(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("size_delta", (-1, 1), ids=("truncated", "oversized"))
def test_download_size_failure_preserves_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    size_delta: int,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    expected = b"content with an exact expected length"
    sha256 = _digest(expected)
    hosted_blob_service.state.objects[sha256] = (
        expected[:-1] if size_delta < 0 else expected + b"!"
    )
    destination = tmp_path / "existing.zip"
    destination.write_bytes(b"original")

    with pytest.raises(ContentIntegrityError, match="Size mismatch after download"):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            destination,
            sha256=sha256,
            size=len(expected),
        )

    assert destination.read_bytes() == b"original"
    assert sorted(tmp_path.iterdir()) == [destination]


def test_token_is_resolved_at_request_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    content = b"runtime token"
    sha256 = _digest(content)
    path = tmp_path / "object.zip"
    path.write_bytes(content)
    store = _store(hosted_blob_service)
    _set_token(monkeypatch, hosted_blob_service, "first-runtime-token")
    assert store.ensure_present(
        compute_content_key(CATALOG_ID, sha256), path, sha256=sha256
    )
    monkeypatch.setenv(TOKEN_ENV, "second-runtime-token")
    destination = tmp_path / "download.zip"
    store.get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    authorizations = [
        authorization
        for _, authorization, _ in hosted_blob_service.state.control_requests
    ]
    assert authorizations == [
        "Bearer first-runtime-token",
        "Bearer first-runtime-token",
        "Bearer second-runtime-token",
    ]


def test_public_download_does_not_disclose_a_foreign_service_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "foreign-service-secret")
    monkeypatch.setenv(TOKEN_SERVICE_ENV, "https://other.example/")
    content = b"public content"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    destination = tmp_path / "public.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content
    assert hosted_blob_service.state.control_requests[0][1] is None


@pytest.mark.parametrize(
    ("status", "error_code"),
    (
        pytest.param(401, "unauthorized", id="unauthorized"),
        pytest.param(403, "forbidden", id="forbidden"),
    ),
)
def test_public_download_retries_anonymously_after_a_rejected_scoped_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    status: int,
    error_code: str,
) -> None:
    token = f"rejected-{error_code}-token"
    _set_token(monkeypatch, hosted_blob_service, token)
    hosted_blob_service.state.rejected_authorizations[f"Bearer {token}"] = (
        status,
        error_code,
    )
    content = b"public content despite a rejected token"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    destination = tmp_path / "public.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content
    assert [
        authorization
        for _, authorization, _ in hosted_blob_service.state.control_requests
    ] == [f"Bearer {token}", None]


@pytest.mark.parametrize(
    ("status", "error_code"),
    ((401, "invalid_token"), (403, "unauthorized"), (429, "rate_limited")),
)
def test_public_download_does_not_retry_other_auth_failures_anonymously(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    status: int,
    error_code: str,
) -> None:
    token = "scoped-token"
    _set_token(monkeypatch, hosted_blob_service, token)
    monkeypatch.setattr("xorq.catalog.content_store.time.sleep", lambda _: None)
    hosted_blob_service.state.rejected_authorizations[f"Bearer {token}"] = (
        status,
        error_code,
    )
    content = b"must not trigger an anonymous retry"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content

    with pytest.raises(ContentStoreError):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "download.zip",
            sha256=sha256,
            size=len(content),
        )

    # A rate limit is retried, but always with the same credentials.
    attempts = _CONTROL_RETRY_ATTEMPTS if status == 429 else 1
    assert [
        authorization
        for _, authorization, _ in hosted_blob_service.state.control_requests
    ] == [f"Bearer {token}"] * attempts
    assert hosted_blob_service.state.object_requests == []


def test_upload_requires_an_explicit_token_service_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "unscoped-secret")
    content = b"must not leave the process"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError, match=TOKEN_SERVICE_ENV):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert hosted_blob_service.state.control_requests == []


def test_service_error_redacts_token_and_skips_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    secret = "token-must-not-appear-in-errors"
    _set_token(monkeypatch, hosted_blob_service, secret)
    monkeypatch.setattr("xorq.catalog.content_store.time.sleep", lambda _: None)
    hosted_blob_service.state.next_control_error = 503
    hosted_blob_service.state.control_error_repeats = _CONTROL_RETRY_ATTEMPTS
    content = b"service error"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError) as exc_info:
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert secret not in str(exc_info.value)
    assert not hosted_blob_service.state.object_requests


def test_signed_download_redirect_is_refused_and_destination_is_atomic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    content = b"redirected content"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    hosted_blob_service.state.redirect_download = True
    destination = tmp_path / "destination.zip"
    destination.write_bytes(b"original")

    with pytest.raises(ContentStoreError):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            destination,
            sha256=sha256,
            size=len(content),
        )

    assert not hosted_blob_service.state.redirect_was_followed
    assert destination.read_bytes() == b"original"
    assert sorted(tmp_path.iterdir()) == [destination]


def test_presigned_object_url_requires_https_away_from_loopback() -> None:
    request = {
        "url": "http://objects.example/archive.zip",
        "headers": {},
        "expires_at": "2099-01-01T00:00:00Z",
    }

    with pytest.raises(ContentStoreError, match="HTTPS is required"):
        PresignedContentStore._presigned_request(request, allow_loopback_http=True)

    request["url"] = "http://127.0.0.1:8080/archive.zip"
    url, headers, _expiry = PresignedContentStore._presigned_request(
        request, allow_loopback_http=True
    )
    assert url == request["url"]
    assert headers == {}


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("url", "https://user:secret@objects.example/a", "invalid presigned URL"),
        ("url", "https://objects.example/a#fragment", "invalid presigned URL"),
        ("url", "https://objects.example:bad/a", "invalid presigned URL"),
        ("headers", {"bad header": "value"}, "invalid signed headers"),
        ("headers", {"x-signed": "value\r\ninjected: yes"}, "invalid signed headers"),
        ("expires_at", "tomorrow", "invalid presigned expiry"),
        (
            "expires_at",
            "2099-01-01T00:00:00",
            "timezone-naive presigned expiry",
        ),
    ),
)
def test_presigned_request_rejects_unsafe_transport_metadata(
    field: str,
    value: object,
    match: str,
) -> None:
    request = {
        "url": "https://objects.example/archive.zip?signed=get",
        "headers": {"x-signed": "value"},
        "expires_at": "2099-01-01T00:00:00Z",
    }
    request[field] = value

    with pytest.raises(ContentStoreError, match=match):
        PresignedContentStore._presigned_request(request, allow_loopback_http=True)


def test_loopback_control_and_object_requests_bypass_environment_proxy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    for variable in ("NO_PROXY", "no_proxy"):
        monkeypatch.delenv(variable, raising=False)
    for variable in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy"):
        monkeypatch.setenv(variable, "http://127.0.0.1:9")

    content = b"loopback traffic stays direct"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    destination = tmp_path / "download.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content


def test_presigned_config_is_strict_and_client_lifecycle_apis_are_unavailable(
    hosted_blob_service: _ServiceFixture,
) -> None:
    parsed = ContentStoreConfig.from_dict(
        {
            "type": "presigned",
            "catalog_id": CATALOG_ID,
            "service_url": hosted_blob_service.url,
        }
    )
    assert isinstance(parsed, PresignedContentStoreConfig)

    with pytest.raises(ValueError, match="unknown fields"):
        ContentStoreConfig.from_dict(
            {
                "type": "presigned",
                "catalog_id": CATALOG_ID,
                "service_url": hosted_blob_service.url,
                "token": "must-never-be-committed",
            }
        )
    with pytest.raises((TypeError, ValueError)):
        PresignedContentStoreConfig(
            catalog_id="not-a-uuid", service_url=hosted_blob_service.url
        )
    with pytest.raises((TypeError, ValueError)):
        PresignedContentStoreConfig(
            catalog_id=CATALOG_ID, service_url="http://catalog.example"
        )

    store = _store(hosted_blob_service)
    with pytest.raises(ContentStoreCapabilityError, match="hosted catalog service"):
        store.delete("unused")
    with pytest.raises(ContentStoreCapabilityError, match="hosted catalog service"):
        list(store.list_keys())


def test_pointer_backend_binds_presigned_store_to_its_git_remote(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)

    store = backend.content_store
    assert isinstance(store, PresignedContentStore)
    assert store.catalog_id == CATALOG_ID
    assert store.service_url == hosted_blob_service.url
    assert store.remote_url == f"{hosted_blob_service.url}alice/demo.git"

    untrusted = _backend(
        tmp_path,
        hosted_blob_service,
        repo_name="untrusted-repo",
        remote_url="https://other.example/alice/demo.git",
    )
    with pytest.raises(ValueError, match="does not match"):
        _ = untrusted.content_store


def test_pointer_backend_revalidates_a_mutated_git_remote(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    original_url = f"{hosted_blob_service.url}alice/demo.git"
    original_store = backend.content_store

    backend.repo.remotes.origin.set_url("https://other.example/alice/demo.git")
    with pytest.raises(ValueError, match="does not match"):
        _ = backend.content_store

    backend.repo.remotes.origin.set_url(original_url)
    assert backend.content_store is original_store


def test_catalog_rejects_an_untrusted_remote_before_replacing_the_origin(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)
    original_url = f"{hosted_blob_service.url}alice/demo.git"

    with pytest.raises(ValueError, match="does not match"):
        catalog.set_remote(
            "origin",
            "https://other.example/alice/demo.git",
            force=True,
        )

    assert tuple(backend.repo.remotes.origin.urls) == (original_url,)


def test_hosted_git_operations_revalidate_fetch_and_push_urls(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)
    original_url = f"{hosted_blob_service.url}alice/demo.git"
    remote = backend.repo.remotes.origin

    remote.set_url("https://other.example/alice/demo.git")
    with pytest.raises(CatalogConfigurationError, match="does not match"):
        catalog.fetch()

    remote.set_url(original_url)
    backend.repo.git.config(
        "remote.origin.pushurl",
        "https://other.example/alice/demo.git",
    )
    with pytest.raises(
        CatalogConfigurationError, match="fetch and push URLs must match"
    ):
        catalog.push()


def test_hosted_git_operations_reject_multiple_urls_on_one_remote(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)
    backend.repo.remotes.origin.add_url(f"{hosted_blob_service.url}alice/other.git")

    with pytest.raises(CatalogConfigurationError, match="exactly one fetch URL"):
        catalog.fetch()


def test_hosted_write_preflight_leaves_no_partial_catalog_files(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)
    backend.repo.delete_remote(backend.repo.remotes.origin)
    assert not backend.repo.is_dirty(untracked_files=True)

    with pytest.raises(ValueError, match="exactly one Git remote"):
        catalog.add(xo.memtable({"value": [1, 2, 3]}), sync=False)

    assert not backend.repo.is_dirty(untracked_files=True)
    assert not (Path(backend.repo.working_dir) / "entries").exists()
    assert not (Path(backend.repo.working_dir) / "metadata").exists()

    source = tmp_path / "source.zip"
    source.write_bytes(b"content")
    target = Path(backend.repo.working_dir) / "entries" / "entry.zip"
    with pytest.raises(ValueError, match="exactly one Git remote"):
        backend.stage_content(source, target)
    assert not target.exists()


@pytest.mark.parametrize(
    "pointer_bytes",
    (
        pytest.param(
            f"xorq-pointer v1\nsha256 {'a' * 64}\nsize 1".encode(),
            id="missing-trailing-newline",
        ),
        pytest.param(
            f"xorq-pointer v1\r\nsha256 {'a' * 64}\nsize 10\r\n".encode(),
            id="mixed-line-endings",
        ),
        pytest.param(
            f"xorq-pointer v1\nsha256 {'a' * 64}\nsize 00\n".encode(),
            id="leading-zero-size",
        ),
        pytest.param(
            f"xorq-pointer v1\nsha256 {'a' * 64}\nsize +1\n".encode(),
            id="signed-size",
        ),
        pytest.param(
            f"xorq-pointer v1\nsha256 {'A' * 64}\nsize 10\n".encode(),
            id="uppercase-digest",
        ),
        pytest.param(
            f"xorq-pointer v1\nsha256 {'a' * 64}\nsize 5000000001\n".encode(),
            id="size-over-the-blob-limit",
        ),
    ),
)
def test_hosted_backend_rejects_noncanonical_pointer_bytes(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
    pointer_bytes: bytes,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    target = Path(backend.repo.working_dir) / "entries" / "entry.zip"
    pointer = backend._pointer_path(target)
    pointer.parent.mkdir(parents=True)
    pointer.write_bytes(pointer_bytes)

    with pytest.raises(ContentIntegrityError, match="corrupt pointer"):
        backend.fetch_content(target)

    assert hosted_blob_service.state.control_requests == []


def test_hosted_catalog_rejects_names_the_service_hook_cannot_accept(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)
    entry = CatalogEntry("safe-entry", catalog, require_exists=False)

    for invalid in ("-leading", "has.dot", "café", "a" * 129):
        with pytest.raises(ValueError, match="hosted alias must be 1-128 ASCII"):
            CatalogAlias(invalid, entry)
        with pytest.raises(ValueError, match="hosted entry name must be 1-128 ASCII"):
            CatalogEntry(invalid, catalog, require_exists=False)


def test_hosted_stage_and_unlink_use_only_the_server_blob_lifecycle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    backend = _backend(tmp_path, hosted_blob_service)
    source = tmp_path / "source.zip"
    content = b"archive staged by the real pointer backend"
    source.write_bytes(content)
    sha256 = _digest(content)
    target = Path(backend.repo.working_dir) / "entries" / "entry.zip"

    # PresignedContentStore.exists/delete are intentionally unsupported, so
    # success also proves stage_content used ensure_present instead of either.
    backend.stage_content(source, target)

    pointer = backend._pointer_path(target)
    assert parse_pointer(pointer) == (sha256, len(content))
    assert target.read_bytes() == content
    assert hosted_blob_service.state.objects[sha256] == content
    assert [
        path.rsplit("/", 1)[-1]
        for path, _, _ in hosted_blob_service.state.control_requests
    ] == ["uploads:batch", "complete"]

    hosted_blob_service.state.control_requests.clear()
    hosted_blob_service.state.object_requests.clear()
    backend.repo.delete_remote(backend.repo.remotes.origin)
    backend = GitPointerBackend.from_repo(backend.repo, cache=backend.cache)
    backend.stage_unlink(target)

    assert not target.exists()
    assert not pointer.exists()
    assert hosted_blob_service.state.objects[sha256] == content
    assert hosted_blob_service.state.control_requests == []
    assert hosted_blob_service.state.object_requests == []

    with pytest.raises(ContentStoreCapabilityError, match="is server-owned"):
        backend.gc_content_store()


def test_pointer_backend_batches_two_presigned_cache_misses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    hosted_blob_service.state.reverse_downloads = True
    contents = (b"first backend archive", b"second backend archive")
    prepared = _prepare_downloads(backend, hosted_blob_service, contents)

    backend.fetch_content(*(target for target, _, _ in prepared))

    assert [target.read_bytes() for target, _, _ in prepared] == list(contents)
    assert all(backend.cache.contains(key) for _, key, _ in prepared)
    downloads = [
        request
        for request in hosted_blob_service.state.control_requests
        if request[0].endswith("downloads:batch")
    ]
    assert len(downloads) == 1
    assert {
        sha256
        for method, sha256, _ in hosted_blob_service.state.object_requests
        if method == "GET"
    } == {sha256 for _, _, sha256 in prepared}


def test_pointer_backend_combines_cache_hits_with_duplicate_download_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    cached, missing = b"cached archive", b"downloaded archive"
    cached_sha, missing_sha = _digest(cached), _digest(missing)
    cached_key = compute_content_key(CATALOG_ID, cached_sha)
    missing_key = compute_content_key(CATALOG_ID, missing_sha)
    cached_source = tmp_path / "cached-source.zip"
    cached_source.write_bytes(cached)
    backend.cache.put(cached_key, cached_source)
    hosted_blob_service.state.objects[missing_sha] = missing

    entries = Path(backend.repo.working_dir) / "entries"
    cached_target = entries / "cached.zip"
    missing_targets = (entries / "missing-a.zip", entries / "missing-b.zip")
    write_pointer(backend._pointer_path(cached_target), cached_sha, len(cached))
    for target in missing_targets:
        write_pointer(backend._pointer_path(target), missing_sha, len(missing))

    backend.fetch_content(cached_target, *missing_targets)

    assert cached_target.read_bytes() == cached
    assert [target.read_bytes() for target in missing_targets] == [missing, missing]
    assert backend.cache.contains(missing_key)
    assert [
        sha256
        for method, sha256, _ in hosted_blob_service.state.object_requests
        if method == "GET"
    ] == [missing_sha]


def test_pointer_backend_batch_failure_leaves_targets_and_cache_untouched(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    prepared = _prepare_downloads(
        backend,
        hosted_blob_service,
        (b"valid archive", b"corrupt archive"),
    )
    hosted_blob_service.state.corrupt_downloads.add(prepared[-1][2])

    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch"):
        backend.fetch_content(*(target for target, _, _ in prepared))

    assert all(not target.exists() for target, _, _ in prepared)
    assert all(not backend.cache.contains(key) for _, key, _ in prepared)


def _download_batches(service: _ServiceFixture) -> list[dict[str, Any]]:
    return [
        body
        for path, _auth, body in service.state.control_requests
        if path.endswith("downloads:batch")
    ]


def test_fetch_content_batches_beyond_the_service_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """More objects than one mint allows are fetched in bounded groups."""
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    hosted_blob_service.state.max_batch_size = _PRESIGNED_BATCH_SIZE
    contents = tuple(f"archive number {index}".encode() for index in range(12))
    prepared = _prepare_downloads(backend, hosted_blob_service, contents)

    backend.fetch_content(*(target for target, _, _ in prepared))

    assert [target.read_bytes() for target, _, _ in prepared] == list(contents)
    assert all(backend.cache.contains(key) for _, key, _ in prepared)
    # 12 objects staged in groups of _FETCH_BATCH_SIZE, each group one mint
    # that stays inside the service's per-request limit.
    batches = _download_batches(hosted_blob_service)
    assert [len(body["objects"]) for body in batches] == [_FETCH_BATCH_SIZE, 2]


def test_fetch_content_stages_downloads_beside_the_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """In-flight objects land in the cache's staging dir, not the system temp dir.

    Observed through the destinations the backend hands the store, which is the
    actual contract, rather than through whichever temp-file call happens to
    create them.
    """
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    prepared = _prepare_downloads(backend, hosted_blob_service, (b"staged archive",))

    destinations: list[Path] = []
    real_get_many = PresignedContentStore.get_many

    def observing(self: PresignedContentStore, objects: Any) -> set[str]:
        items = list(objects)
        destinations.extend(Path(path) for _spec, path in items)
        return real_get_many(self, items)

    monkeypatch.setattr(PresignedContentStore, "get_many", observing)
    backend.fetch_content(prepared[0][0])

    assert destinations, "the store was never asked to download anything"
    assert [path.parent for path in destinations] == [backend.cache.staging_dir]
    # Staging is consumed by adopt(), so nothing is left behind.
    assert list(backend.cache.staging_dir.iterdir()) == []


def test_fetch_content_keeps_batches_that_already_succeeded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """A corrupt object fails its own group without discarding earlier groups."""
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    contents = tuple(f"archive number {index}".encode() for index in range(12))
    prepared = _prepare_downloads(backend, hosted_blob_service, contents)
    hosted_blob_service.state.corrupt_downloads.add(prepared[-1][2])

    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch"):
        backend.fetch_content(*(target for target, _, _ in prepared))

    # The first full group survives; the group containing the corrupt object
    # is discarded whole.
    for target, key, _sha in prepared[:_FETCH_BATCH_SIZE]:
        assert target.exists()
        assert backend.cache.contains(key)
    for target, key, _sha in prepared[_FETCH_BATCH_SIZE:]:
        assert not target.exists()
        assert not backend.cache.contains(key)
    # No staging files are left behind on the cache volume.
    assert list(backend.cache.cache_dir.rglob("*.tmp")) == []


def test_hosted_single_object_get_reports_an_unsupported_capability(
    hosted_blob_service: _ServiceFixture,
) -> None:
    """The key-only ContentStore.get contract is refused, not silently wrong."""
    store = _store(hosted_blob_service)
    key = compute_content_key(CATALOG_ID, _digest(b"anything"))

    with pytest.raises(ContentStoreCapabilityError, match="use get_many"):
        store.get(key, "/dev/null")


def _download_batches_seen(service: _ServiceFixture) -> int:
    return sum(
        path.endswith("downloads:batch")
        for path, _, _ in service.state.control_requests
    )


def test_a_large_object_uses_a_url_that_outlives_the_start_of_its_request(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """A URL is checked when its request starts, so object size is irrelevant.

    The service's presigned URLs live 300 s by default; a 3 GiB object at the
    assumed bandwidth takes longer than that to transfer, and must still be
    fetched with the first URL rather than minting a second.
    """
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=300)
    hosted_blob_service.state.download_expires_at = expires_at.isoformat()
    transferred: list[ContentSpec] = []
    monkeypatch.setattr(
        PresignedContentStore,
        "_download_presigned",
        lambda self, spec, local_path, request_data: transferred.append(spec),
    )
    size = 3 * 1024**3
    sha256 = "ab" * 32
    spec = ContentSpec(compute_content_key(CATALOG_ID, sha256), sha256, size)

    _store(hosted_blob_service).get_many(((spec, tmp_path / "large.zip"),))

    assert transferred == [spec]
    assert _download_batches_seen(hosted_blob_service) == 1


def test_a_reminted_url_that_is_still_expiring_is_refused(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    hosted_blob_service.state.download_expires_at = "2000-01-01T00:00:00Z"
    content = b"never fetched with an expired URL"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content

    with pytest.raises(ContentStoreError, match="expires within"):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "download.zip",
            sha256=sha256,
            size=len(content),
        )

    assert _download_batches_seen(hosted_blob_service) == 2
    assert hosted_blob_service.state.object_requests == []


def test_hosted_remote_rejection_names_the_effective_url(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """The error explains that Git URL rewrites can change what was validated."""
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.remotes[0].set_url("https://elsewhere.example/alice/demo.git")

    with pytest.raises(ValueError, match="insteadOf") as excinfo:
        _ = backend.content_store
    assert "https://elsewhere.example/alice/demo.git" in str(excinfo.value)


def test_hosted_download_is_not_hashed_a_second_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """get_many reports what it verified in transit, so the backend re-reads nothing."""
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    prepared = _prepare_downloads(
        backend, hosted_blob_service, (b"verified in transit",)
    )

    rehashed = []
    original = GitPointerBackend._verify_content

    def recording(self, local, path, sha256, size):  # type: ignore[no-untyped-def]
        rehashed.append(Path(local))
        return original(self, local, path, sha256, size)

    monkeypatch.setattr(GitPointerBackend, "_verify_content", recording)
    backend.fetch_content(prepared[0][0])

    assert prepared[0][0].read_bytes() == b"verified in transit"
    assert rehashed == [], "the hosted store already proved these bytes"


def test_hosted_get_many_reports_every_key_it_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """The verified set is the store's proof, and it covers the whole request."""
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    store = _store(hosted_blob_service)
    contents = (b"alpha payload", b"beta payload")
    requested = []
    for index, content in enumerate(contents):
        sha256 = _digest(content)
        hosted_blob_service.state.objects[sha256] = content
        spec = ContentSpec(
            key=compute_content_key(CATALOG_ID, sha256),
            sha256=sha256,
            size=len(content),
        )
        requested.append((spec, tmp_path / f"out-{index}.bin"))

    verified = store.get_many(requested)

    assert verified == {spec.key for spec, _path in requested}
    assert [path.read_bytes() for _spec, path in requested] == list(contents)


def test_unverified_store_downloads_are_still_checked_by_the_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    """A store reporting nothing verified gets its bytes proved by the backend."""
    _set_token(monkeypatch, hosted_blob_service, "read-token")
    backend = _backend(tmp_path, hosted_blob_service)
    prepared = _prepare_downloads(backend, hosted_blob_service, (b"claims nothing",))

    # Capture before patching, or the replacement calls itself.
    real_get_many = PresignedContentStore.get_many
    monkeypatch.setattr(
        PresignedContentStore,
        "get_many",
        lambda self, objects: (real_get_many(self, objects), None)[1],
    )
    rehashed = []
    original = GitPointerBackend._verify_content

    def recording(self, local, path, sha256, size):  # type: ignore[no-untyped-def]
        rehashed.append(Path(local))
        return original(self, local, path, sha256, size)

    monkeypatch.setattr(GitPointerBackend, "_verify_content", recording)
    backend.fetch_content(prepared[0][0])

    assert prepared[0][0].read_bytes() == b"claims nothing"
    assert len(rehashed) == 1, "an unproven download must be verified locally"


def test_service_http_error_survives_a_process_boundary() -> None:
    error = CatalogServiceHTTPError(
        "catalog service request failed with HTTP status 403 (forbidden)",
        status=403,
        error_code="forbidden",
    )

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is CatalogServiceHTTPError
    assert (str(restored), restored.status, restored.error_code) == (
        str(error),
        403,
        "forbidden",
    )


def test_remote_service_cannot_direct_downloads_to_loopback_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    content = b"a remote service must not reach the user's own ports"
    sha256 = _digest(content)
    store = PresignedContentStore(
        catalog_id=CATALOG_ID,
        service_url="https://catalog.example/",
        remote_url="https://catalog.example/alice/demo.git",
    )
    opened: list[str] = []

    def control_post(
        self: PresignedContentStore, suffix: str, payload: Any, *, token: Any
    ) -> dict[str, Any]:
        return {
            "downloads": [
                {
                    "sha256": sha256,
                    "size": len(content),
                    "request": {
                        "url": "http://127.0.0.1:9/objects/archive.zip",
                        "headers": {},
                        "expires_at": "2099-01-01T00:00:00Z",
                    },
                }
            ]
        }

    def record_open(self: PresignedContentStore, request: Any, **kwargs: Any) -> Any:
        opened.append(request.full_url)
        raise AssertionError("no object request may be sent")

    monkeypatch.setattr(PresignedContentStore, "_control_post", control_post)
    monkeypatch.setattr(PresignedContentStore, "_open", record_open)

    with pytest.raises(ContentStoreError, match="invalid presigned URL"):
        store.get_many(
            (
                (
                    ContentSpec(
                        compute_content_key(CATALOG_ID, sha256), sha256, len(content)
                    ),
                    tmp_path / "download.zip",
                ),
            )
        )
    assert opened == []


def test_http_failure_closes_the_error_response() -> None:
    body = io.BytesIO(b'{"error": "forbidden"}')
    error = HTTPError("https://catalog.example/", 403, "Forbidden", {}, body)

    failure = PresignedContentStore._http_failure(error, "catalog service request")

    assert (failure.status, failure.error_code) == (403, "forbidden")
    assert body.closed


def test_get_many_transfers_a_duplicated_identity_once(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    content = b"one object wanted at two destinations"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    spec = ContentSpec(compute_content_key(CATALOG_ID, sha256), sha256, len(content))
    first, second = tmp_path / "first.zip", tmp_path / "second.zip"

    verified = _store(hosted_blob_service).get_many(((spec, first), (spec, second)))

    assert verified == {spec.key}
    assert first.read_bytes() == second.read_bytes() == content
    assert [method for method, _, _ in hosted_blob_service.state.object_requests] == [
        "GET"
    ]


def test_hosted_backend_reads_pointers_checked_out_with_crlf(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    ((target, _key, _sha256),) = _prepare_downloads(
        backend, hosted_blob_service, (b"pointer checked out by core.autocrlf",)
    )
    pointer = backend._pointer_path(target)
    pointer.write_bytes(pointer.read_bytes().replace(b"\n", b"\r\n"))

    backend.fetch_content(target)

    assert target.read_bytes() == b"pointer checked out by core.autocrlf"


def test_hosted_content_store_resolves_the_remote_once_per_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    _ = backend.content_store
    commands: list[tuple[str, ...]] = []
    execute = Git.execute

    def recording(self: Git, command: Any, *args: Any, **kwargs: Any) -> Any:
        commands.append(tuple(command))
        return execute(self, command, *args, **kwargs)

    monkeypatch.setattr(Git, "execute", recording)

    _ = backend.content_store

    # One effective fetch URL and one effective push URL: nothing else is
    # needed to validate the binding.
    assert [command[1:4] for command in commands] == [
        ("remote", "get-url", "--all"),
        ("remote", "get-url", "--push"),
    ]


def test_upload_hashes_each_file_only_while_sending_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    state = hosted_blob_service.state
    stored, sent = b"the service already holds this", b"this one is uploaded"
    state.already_verified.add(_digest(stored))
    state.objects[_digest(stored)] = stored
    objects = []
    for name, content in (("stored", stored), ("sent", sent)):
        path = tmp_path / f"{name}.zip"
        path.write_bytes(content)
        sha256 = _digest(content)
        spec = ContentSpec(
            compute_content_key(CATALOG_ID, sha256), sha256, len(content)
        )
        objects.append((spec, path))
    hashed: list[Path] = []
    monkeypatch.setattr(
        "xorq.catalog.content_store.compute_sha256",
        lambda path: hashed.append(Path(path)) or _digest(Path(path).read_bytes()),
    )

    uploaded = _store(hosted_blob_service).ensure_present_many(objects)

    assert uploaded == {objects[1][0].key}
    assert state.objects[_digest(sent)] == sent
    assert hashed == []


def test_upload_never_completes_bytes_that_changed_after_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "write-token")
    content = b"the bytes the pointer describes"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(b"x" * len(content))  # same size, different bytes

    with pytest.raises(ContentIntegrityError, match="SHA256 mismatch"):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )

    assert hosted_blob_service.state.completed == []


@pytest.mark.parametrize(
    ("status", "retry_after", "expected_sleeps"),
    (
        (503, None, [1, 2]),
        (429, "5", [5, 5]),
        (429, "3600", [_CONTROL_RETRY_MAX_DELAY_SECONDS] * 2),
    ),
)
def test_control_requests_retry_statuses_that_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    status: int,
    retry_after: str | None,
    expected_sleeps: list[int],
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("xorq.catalog.content_store.time.sleep", sleeps.append)
    state = hosted_blob_service.state
    state.next_control_error = status
    state.control_error_repeats = _CONTROL_RETRY_ATTEMPTS - 1
    state.control_error_retry_after = retry_after
    content = b"fetched once the service recovers"
    sha256 = _digest(content)
    state.objects[sha256] = content
    destination = tmp_path / "download.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content
    assert sleeps == expected_sleeps


def test_control_requests_do_not_retry_other_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr("xorq.catalog.content_store.time.sleep", sleeps.append)
    hosted_blob_service.state.next_control_error = 409
    sha256 = _digest(b"missing")

    with pytest.raises(ContentStoreError, match="HTTP status 409"):
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "download.zip",
            sha256=sha256,
            size=7,
        )

    assert sleeps == []
    assert len(hosted_blob_service.state.control_requests) == 1


@pytest.mark.parametrize("token", ("", "   "))
def test_an_empty_token_is_treated_as_absent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
    token: str,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, token)
    content = b"public content with an exported but empty token"
    sha256 = _digest(content)
    hosted_blob_service.state.objects[sha256] = content
    destination = tmp_path / "public.zip"

    _store(hosted_blob_service).get(
        compute_content_key(CATALOG_ID, sha256),
        destination,
        sha256=sha256,
        size=len(content),
    )

    assert destination.read_bytes() == content
    assert [auth for _, auth, _ in hosted_blob_service.state.control_requests] == [None]


def test_an_authorization_failure_names_the_withheld_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    monkeypatch.setenv(TOKEN_ENV, "configured-but-unscoped")
    monkeypatch.delenv(TOKEN_SERVICE_ENV, raising=False)
    hosted_blob_service.state.rejected_authorizations[""] = (401, "unauthorized")
    sha256 = _digest(b"private")

    with pytest.raises(ContentStoreError) as exc_info:
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "private.zip",
            sha256=sha256,
            size=7,
        )

    message = str(exc_info.value)
    assert "HTTP status 401" in message
    assert f"{TOKEN_ENV} is set but not sent" in message
    assert TOKEN_SERVICE_ENV in message


def test_a_rejected_token_is_reported_when_the_anonymous_retry_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    token = "expired-token"
    _set_token(monkeypatch, hosted_blob_service, token)
    rejections = hosted_blob_service.state.rejected_authorizations
    rejections[f"Bearer {token}"] = (401, "unauthorized")
    rejections[""] = (403, "forbidden")
    sha256 = _digest(b"private")

    with pytest.raises(ContentStoreError) as exc_info:
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "private.zip",
            sha256=sha256,
            size=7,
        )

    message = str(exc_info.value)
    assert "HTTP status 403 (forbidden)" in message
    assert f"{TOKEN_ENV} was rejected first" in message
    assert "HTTP status 401 (unauthorized)" in message
    assert token not in message


@pytest.mark.parametrize(
    "remote_url",
    ("git@catalog.example:alice/demo.git", "ssh://git@catalog.example/alice/demo.git"),
)
def test_an_ssh_remote_is_told_to_use_https(remote_url: str) -> None:
    config = PresignedContentStoreConfig(
        catalog_id=CATALOG_ID, service_url="https://catalog.example/"
    )

    with pytest.raises(ValueError, match="must be an HTTPS URL; SSH"):
        config.validate_remote_url(remote_url)


def test_fetch_recreates_a_staging_dir_removed_since_the_cache_was_built(
    tmp_path: Path,
    hosted_blob_service: _ServiceFixture,
) -> None:
    backend = _backend(tmp_path, hosted_blob_service)
    ((target, _key, _sha256),) = _prepare_downloads(
        backend, hosted_blob_service, (b"fetched after a cache cleanup",)
    )
    staging = backend.cache.staging_dir
    assert staging is not None
    staging.rmdir()

    backend.fetch_content(target)

    assert target.read_bytes() == b"fetched after a cache cleanup"


def test_hosted_add_without_a_token_fails_before_reading_the_archive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    monkeypatch.delenv(TOKEN_ENV, raising=False)

    def refuse(*args: Any) -> str:
        raise AssertionError("read the archive before checking the token")

    monkeypatch.setattr("xorq.catalog.backend.copy_and_digest", refuse)
    backend = _backend(tmp_path, hosted_blob_service)
    backend.repo.index.add([CONTENT_STORE_YAML])
    backend.repo.index.commit("initial hosted config")
    catalog = Catalog(backend=backend)

    with pytest.raises(ContentStoreError, match=f"{TOKEN_ENV} is required"):
        catalog.add(xo.memtable({"value": [1, 2, 3]}), sync=False)
    assert not backend.repo.is_dirty(untracked_files=True)

    source = tmp_path / "source.zip"
    source.write_bytes(b"content")
    target = Path(backend.repo.working_dir) / "entries" / "entry.zip"
    with pytest.raises(ContentStoreError, match=f"{TOKEN_ENV} is required"):
        backend.stage_content(source, target)
    assert not target.exists()
    assert hosted_blob_service.state.control_requests == []


def test_a_new_pointer_version_still_parses_canonically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("xorq.catalog.content_store.POINTER_VERSION", "xorq-pointer v2")
    pointer = tmp_path / "entry.zip.pointer"
    sha256 = _digest(b"written and read under one version")

    write_pointer(pointer, sha256, 34)

    assert pointer.read_bytes().startswith(b"xorq-pointer v2\n")
    assert parse_pointer(pointer, canonical=True) == (sha256, 34)


def test_a_rejected_token_is_reported_when_the_anonymous_retry_cannot_connect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "expired-token")
    calls: list[str | None] = []

    def control_post(
        self: PresignedContentStore, suffix: str, payload: Any, *, token: Any
    ) -> dict[str, Any]:
        calls.append(token)
        if token is not None:
            raise CatalogServiceHTTPError(
                "catalog service request failed with HTTP status 401 (unauthorized)",
                status=401,
                error_code="unauthorized",
            )
        raise ContentStoreError("catalog service request failed due to a network error")

    monkeypatch.setattr(PresignedContentStore, "_control_post", control_post)
    sha256 = _digest(b"private")

    with pytest.raises(ContentStoreError) as exc_info:
        _store(hosted_blob_service).get(
            compute_content_key(CATALOG_ID, sha256),
            tmp_path / "private.zip",
            sha256=sha256,
            size=7,
        )

    message = str(exc_info.value)
    assert "network error" in message
    assert f"{TOKEN_ENV} was rejected first" in message
    assert "expired-token" not in message
    assert calls == ["expired-token", None]


def test_an_upload_names_a_blank_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hosted_blob_service: _ServiceFixture,
) -> None:
    _set_token(monkeypatch, hosted_blob_service, "\n")
    content = b"refused before any request"
    sha256 = _digest(content)
    source = tmp_path / "source.zip"
    source.write_bytes(content)

    with pytest.raises(ContentStoreError, match=f"{TOKEN_ENV} is set but blank"):
        _store(hosted_blob_service).ensure_present(
            compute_content_key(CATALOG_ID, sha256), source, sha256=sha256
        )
    assert hosted_blob_service.state.control_requests == []
