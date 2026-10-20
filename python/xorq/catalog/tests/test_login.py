from __future__ import annotations

import base64
import hashlib
import json
import socket
import stat
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl

import pytest
from click.testing import CliRunner

from xorq.catalog import login as login_module
from xorq.catalog.cli import cli
from xorq.catalog.exceptions import LoginError
from xorq.catalog.login import Login, Provider, Settings, login_path, sign_in


AUDIENCE = "api://catalog"
CLIENT_ID = "xorq-cli"
DEVICE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
# Every parameter the client may send; anything else would be provider-specific.
STANDARD_PARAMS = {
    "client_id",
    "scope",
    "response_type",
    "redirect_uri",
    "state",
    "code_challenge",
    "code_challenge_method",
    "grant_type",
    "code",
    "code_verifier",
    "device_code",
    "refresh_token",
}


def _jwt(**claims: object) -> str:
    def part(obj: dict) -> str:
        return base64.urlsafe_b64encode(json.dumps(obj).encode()).rstrip(b"=").decode()

    return f"{part({'alg': 'none'})}.{part(claims)}.sig"


class FakeProvider:
    """An OIDC provider and catalog service in one local HTTP server."""

    def __init__(self) -> None:
        self.requests = []
        self.token_url = None
        self.stall_next_body = False
        self.device_expires_in = 600
        self.non_object = None
        self.device_polls = ["authorization_pending", "slow_down"]
        self.deny = False
        self.issued = 0
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

    def tokens(self) -> dict:
        self.issued += 1
        refresh = f"refresh-{self.issued}"
        exp = int(time.time()) + 900
        return {
            "access_token": _jwt(aud=AUDIENCE, exp=exp, cid=CLIENT_ID),
            "id_token": _jwt(aud=CLIENT_ID, exp=exp),
            "refresh_token": refresh,
            "expires_in": 900,
        }

    def token(self, form: dict) -> tuple[int, dict]:
        match form:
            case {"grant_type": login_module.DEVICE_CODE_GRANT}:
                if self.deny:
                    return 400, {"error": "access_denied"}
                if self.device_polls:
                    poll = self.device_polls.pop(0)
                    if poll == "timeout":
                        threading.Event().wait(1)  # outlasts the client's timeout
                    return 400, {"error": poll}
                return 200, self.tokens()
        return 400, {"error": "unsupported_grant_type"}


class _Handler(BaseHTTPRequestHandler):
    provider: FakeProvider

    def _reply(self, status: int, body: dict, stall: bool = False) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if stall:  # headers now, the body after the client's timeout
            self.wfile.flush()
            threading.Event().wait(1)
        self.wfile.write(payload)

    def do_GET(self) -> None:
        provider = self.provider
        provider.requests.append((self.path, {}))
        match self.path:
            case path if path == provider.non_object:
                self._reply(200, [])
            case path if path.startswith("/downgrade/"):
                self.send_response(302)
                self.send_header("Location", "http://idp.example/settings")
                self.end_headers()
            case "/.well-known/xorq-catalog":
                self._reply(200, provider.settings())
            case "/.well-known/openid-configuration":
                self._reply(
                    200,
                    {
                        "issuer": provider.url,
                        "token_endpoint": provider.token_url or f"{provider.url}/token",
                        "device_authorization_endpoint": f"{provider.url}/device",
                    },
                )
            case _:
                self._reply(404, {})

    def do_POST(self) -> None:
        provider = self.provider
        length = int(self.headers["Content-Length"])
        form = dict(parse_qsl(self.rfile.read(length).decode()))
        provider.requests.append((self.path, form))
        match self.path:
            case "/device":
                self._reply(
                    200,
                    {
                        "device_code": "device-1",
                        "user_code": "ABCD-EFGH",
                        "verification_uri": f"{provider.url}/activate",
                        "expires_in": provider.device_expires_in,
                        "interval": 1,
                    },
                )
            case "/token":
                stall, provider.stall_next_body = provider.stall_next_body, False
                self._reply(*provider.token(form), stall=stall)
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


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list:
    calls = []
    monkeypatch.setattr(login_module.time, "sleep", calls.append)
    return calls


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


def test_redirect_to_plain_http_is_refused(provider: FakeProvider) -> None:
    with pytest.raises(LoginError, match="HTTPS"):
        Settings.fetch(f"{provider.url}/downgrade")


def test_well_known_settings_and_no_provider_specific_params(
    provider: FakeProvider, sleeps: list
) -> None:
    sign_in(provider.url, emit=lambda event: None)

    paths = [path for path, _ in provider.requests]
    assert paths[:2] == [
        "/.well-known/xorq-catalog",
        "/.well-known/openid-configuration",
    ]
    sent = {key for _, params in provider.requests for key in params}
    assert sent <= STANDARD_PARAMS


def test_device_sign_in_honours_interval_and_slow_down(
    provider: FakeProvider, sleeps: list
) -> None:
    events = []
    result = sign_in(provider.url, emit=events.append)

    assert events == [
        {
            "event": "device_authorization",
            "verification_uri": f"{provider.url}/activate",
            "user_code": "ABCD-EFGH",
            "expires_in": 600,
        }
    ]
    # pending at 1s, slow_down raises the interval by 5s for later polls
    assert sleeps == [1, 1, 6]
    assert result.refresh_token == "refresh-1"


def test_device_poll_timeout_backs_off(
    provider: FakeProvider, sleeps: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(login_module, "HTTP_TIMEOUT_SECONDS", 0.2)
    provider.device_polls = ["timeout"]

    result = sign_in(provider.url, emit=lambda event: None)

    # the timed-out poll counts as slow_down (RFC 8628 section 3.5)
    assert sleeps == [1, 6]
    assert result.refresh_token == "refresh-1"


def test_device_poll_with_stalled_error_body_backs_off(
    provider: FakeProvider, sleeps: list, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(login_module, "HTTP_TIMEOUT_SECONDS", 0.2)
    provider.stall_next_body = True  # the first poll's 400 body stalls

    sign_in(provider.url, emit=lambda event: None)
    assert sleeps[:2] == [1, 6]


def test_unreachable_token_endpoint_is_not_retried(
    provider: FakeProvider, sleeps: list
) -> None:
    provider.token_url = f"http://127.0.0.1:{_free_port()}/token"

    with pytest.raises(LoginError, match="cannot reach"):
        sign_in(provider.url, emit=lambda event: None)
    assert sleeps == [1]


def test_device_code_expiry_stops_polling(
    provider: FakeProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = [0.0]
    monkeypatch.setattr("xorq.catalog.login.time.monotonic", lambda: now[0])
    monkeypatch.setattr(
        "xorq.catalog.login.time.sleep",
        lambda seconds: now.__setitem__(0, now[0] + seconds),
    )
    provider.device_expires_in = 4
    provider.device_polls = ["authorization_pending", "slow_down", "slow_down"]

    with pytest.raises(LoginError, match="expired"):
        sign_in(provider.url, emit=lambda event: None)
    polls = [form for path, form in provider.requests if path == "/token"]
    assert (len(polls), now[0]) == (2, 4)


def test_settings_without_issuer_are_refused() -> None:
    with pytest.raises(LoginError, match="invalid catalog sign-in settings"):
        Settings.from_dict(
            {"issuer": None, "client_id": CLIENT_ID, "audience": AUDIENCE, "scopes": []}
        )


def test_cli_login_prints_device_code(provider: FakeProvider, sleeps: list) -> None:
    result = CliRunner().invoke(cli, ["login", provider.url])

    assert result.exit_code == 0, result.output
    assert (
        f"Open {provider.url}/activate and enter the code ABCD-EFGH." in result.output
    )
    assert result.output.splitlines()[-1].startswith(f"Signed in to {provider.url}")


@pytest.mark.parametrize(
    "field, value",
    (
        pytest.param("issuer", "http://idp.example", id="issuer"),
        pytest.param("token_endpoint", "http://idp.example/token", id="endpoint"),
    ),
)
def test_plain_http_identity_provider_is_refused(field: str, value: str) -> None:
    with pytest.raises(LoginError, match="HTTPS"):
        if field == "issuer":
            login_module.Settings.from_dict(
                {"issuer": value, "client_id": "c", "audience": "a", "scopes": []}
            )
        else:
            login_module.Provider(token_endpoint=value)
