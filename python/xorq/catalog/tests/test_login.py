from __future__ import annotations

import base64
import hashlib
import json
import socket
import stat
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from xorq.catalog.exceptions import LoginError
from xorq.catalog.login import Login, Provider, Settings, login_path


AUDIENCE = "api://catalog"
CLIENT_ID = "xorq-cli"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"


def _jwt(**claims: object) -> str:
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


class FakeProvider:
    """An OIDC provider and catalog service in one local HTTP server."""

    def __init__(self) -> None:
        self.requests = []
        self.non_object = None
        handler = type("Handler", (_Handler,), {"provider": self})
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def settings(self, redirect_uris: tuple = ()) -> dict:
        return {
            "issuer": self.url,
            "client_id": CLIENT_ID,
            "audience": AUDIENCE,
            "scopes": ["openid", "offline_access"],
            "grant_types": ["authorization_code", DEVICE_GRANT],
            "redirect_uris": list(redirect_uris),
        }


class _Handler(BaseHTTPRequestHandler):
    provider: FakeProvider

    def _reply(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        provider = self.provider
        provider.requests.append((self.path, {}))
        match self.path:
            case path if path == provider.non_object:
                self._reply(200, [])
            case "/.well-known/xorq-catalog":
                self._reply(200, provider.settings())
            case "/.well-known/openid-configuration":
                self._reply(
                    200,
                    {
                        "issuer": provider.url,
                        "token_endpoint": f"{provider.url}/token",
                        "device_authorization_endpoint": f"{provider.url}/device",
                    },
                )
            case _:
                self._reply(404, {})

    def log_message(self, *args: object) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def config_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    return tmp_path / "config"


@pytest.fixture
def provider(config_home: Path) -> Iterator[FakeProvider]:
    provider = FakeProvider()
    yield provider
    provider.server.shutdown()
    provider.server.server_close()


def test_settings_and_discovery_come_from_the_service(provider: FakeProvider) -> None:
    settings = Settings.fetch(provider.url)
    discovered = Provider.discover(settings.issuer)

    assert (settings.client_id, settings.audience) == (CLIENT_ID, AUDIENCE)
    assert discovered.token_endpoint == f"{provider.url}/token"
    assert [path for path, _ in provider.requests] == [
        "/.well-known/xorq-catalog",
        "/.well-known/openid-configuration",
    ]


def _login(service_url: str, access_token: str, id_token: str) -> Login:
    return Login.from_token_response(
        {"access_token": access_token, "id_token": id_token, "expires_in": 900},
        service_url=service_url,
        settings=Settings(
            issuer="https://idp.example",
            client_id=CLIENT_ID,
            audience=AUDIENCE,
            scopes=["openid"],
        ),
        token_endpoint="https://idp.example/token",
    )


@pytest.mark.parametrize(
    "access_token, id_token, chosen",
    (
        pytest.param(_jwt(aud=AUDIENCE), _jwt(aud=CLIENT_ID), "access", id="access"),
        # an opaque access token leaves only the ID token for the service
        pytest.param("opaque.W10.sig", _jwt(aud=[AUDIENCE]), "id", id="opaque"),
    ),
)
def test_token_for_the_service_audience_is_kept(
    access_token: str, id_token: str, chosen: str
) -> None:
    login = _login("https://catalog.example", access_token, id_token)
    assert login.token == {"access": access_token, "id": id_token}[chosen]


def test_wrong_audience_is_refused() -> None:
    with pytest.raises(LoginError, match="audience 'api://catalog'"):
        _login("https://catalog.example", _jwt(aud="other"), _jwt(aud=CLIENT_ID))


def test_cache_location_and_modes(config_home: Path) -> None:
    url = "https://Catalog.example/"
    login = _login(url, _jwt(aud=AUDIENCE), _jwt(aud=CLIENT_ID)).save()

    path = login_path(url)
    assert path.parent == config_home / "xorq" / "catalog" / "logins"
    digest = hashlib.sha256(b"https://catalog.example:443/").hexdigest()
    assert path.name == f"{digest}.json"
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert login_path("https://catalog.example") == path
    assert Login.load(url) == login


def test_unreachable_service(config_home: Path) -> None:
    with pytest.raises(LoginError, match="cannot reach"):
        Settings.fetch(f"http://127.0.0.1:{_free_port()}")


def test_failed_save_keeps_previous_login(
    config_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    url = "https://catalog.example"
    previous = _login(url, _jwt(aud=AUDIENCE), _jwt(aud=CLIENT_ID)).save()

    def fail(*args: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr("xorq.catalog.login.os.replace", fail)
    with pytest.raises(OSError, match="disk full"):
        _login(url, _jwt(aud=AUDIENCE, n=2), _jwt(aud=CLIENT_ID)).save()

    assert Login.load(url) == previous
    assert [path.name for path in login_path(url).parent.iterdir()] == [
        login_path(url).name
    ]


@pytest.mark.parametrize(
    "document",
    (
        pytest.param("/.well-known/xorq-catalog", id="settings"),
        pytest.param("/.well-known/openid-configuration", id="discovery"),
    ),
)
def test_non_object_json_is_a_login_error(
    provider: FakeProvider, document: str
) -> None:
    provider.non_object = document
    with pytest.raises(LoginError, match="JSON object"):
        Provider.discover(Settings.fetch(provider.url).issuer)
