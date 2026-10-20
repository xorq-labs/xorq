"""OIDC sign-in to a hosted catalog service (`xorq catalog login`).

The service publishes its client settings at an unauthenticated well-known
URL; everything else (discovery, the authorization-code and device grants)
is standard OAuth 2.0 / OpenID Connect, so no provider is assumed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import tempfile
import threading
import time
import webbrowser
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

import attr
from attr import field, frozen

from xorq.catalog.content_store import _is_loopback_host, _service_url_identity
from xorq.catalog.exceptions import LoginError, LoginTimeoutError


WELL_KNOWN_PATH = "/.well-known/xorq-catalog"
DEVICE_CODE_GRANT = "urn:ietf:params:oauth:grant-type:device_code"
HTTP_TIMEOUT_SECONDS = 30
CALLBACK_TIMEOUT_SECONDS = 300
# a connection that sends no request must not hold up the real callback
CALLBACK_READ_TIMEOUT_SECONDS = 5
# how often the callback wait checks whether the launcher gave up
CALLBACK_POLL_SECONDS = 1
SLOW_DOWN_SECONDS = 5
JSON_HEADERS = {"Accept": "application/json"}
CALLBACK_PAGE = b"Signed in to xorq. You can close this window.\n"


def _https_url(value: str | None, label: str) -> str | None:
    # Tokens and codes travel to these URLs: HTTPS, or loopback HTTP for
    # development, exactly as for the catalog service itself.
    if value is None:
        return None
    parts = urlsplit(value)
    if parts.scheme == "https" or (
        parts.scheme == "http" and _is_loopback_host(parts.hostname or "")
    ):
        return value
    raise LoginError(
        f"{label} must use HTTPS (loopback HTTP is allowed for development): {value}"
    )


class _NoDowngradeRedirect(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> Request | None:
        _https_url(newurl, "redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_opener = build_opener(_NoDowngradeRedirect)


def _read_json(request: Request) -> dict:
    # OAuth endpoints answer errors with a 4xx and a JSON body naming the
    # error; those bodies are returned so callers can match on "error".
    try:
        with _opener.open(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = json.load(response)
    except HTTPError as error:
        with error:
            try:
                body = json.load(error)
            except ValueError:
                body = None
            except TimeoutError:
                raise LoginTimeoutError(
                    f"cannot read {request.full_url}: timed out"
                ) from None
        if isinstance(body, dict) and "error" in body:
            return body
        raise LoginError(f"{request.full_url} returned HTTP {error.code}") from None
    except OSError as error:
        # urllib raises a bare TimeoutError on read, a URLError around one on connect
        timed_out = isinstance(error, TimeoutError) or isinstance(
            getattr(error, "reason", None), TimeoutError
        )
        raise (LoginTimeoutError if timed_out else LoginError)(
            f"cannot reach {request.full_url}: {error}"
        ) from None
    except ValueError:
        raise LoginError(f"{request.full_url} did not return JSON") from None
    if not isinstance(body, dict):
        raise LoginError(f"{request.full_url} did not return a JSON object")
    return body


def _get_json(url: str) -> dict:
    return _read_json(Request(url, headers=JSON_HEADERS))


def _post_form(url: str, form: dict) -> dict:
    return _read_json(Request(url, data=urlencode(form).encode(), headers=JSON_HEADERS))


def _failure(response: dict) -> LoginError:
    match response:
        case {"error": "access_denied"}:
            return LoginError("sign-in was denied")
        case {"error": "expired_token"}:
            return LoginError("the device code expired before sign-in finished")
        case _:
            detail = response.get("error_description") or response.get("error")
            return LoginError(f"sign-in failed: {detail}")


def _claims(token: str | None) -> dict:
    # Only read to choose a token and its expiry; the service verifies it.
    try:
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
    except (AttributeError, IndexError, ValueError):
        return {}
    return claims if isinstance(claims, dict) else {}


def _audiences(claims: dict) -> tuple:
    match claims.get("aud"):
        case str(audience):
            return (audience,)
        case list(audiences):
            return tuple(audiences)
        case _:
            return ()


def config_dir() -> Path:
    return Path(os.getenv("XDG_CONFIG_HOME") or "~/.config").expanduser() / "xorq"


def login_path(service_url: str) -> Path:
    scheme, host, port, path = _service_url_identity(service_url, label="service URL")
    digest = hashlib.sha256(f"{scheme}://{host}:{port}{path}".encode()).hexdigest()
    return config_dir() / "catalog" / "logins" / f"{digest}.json"


@frozen
class Settings:
    """Client settings published at <service>/.well-known/xorq-catalog."""

    issuer: str = field(
        converter=lambda value: _https_url(value, "issuer"),
        validator=attr.validators.instance_of(str),
    )
    client_id: str = field(validator=attr.validators.instance_of(str))
    audience: str = field(validator=attr.validators.instance_of(str))
    scopes: tuple = field(converter=tuple)
    grant_types: tuple = field(converter=tuple, default=())
    redirect_uris: tuple = field(converter=tuple, default=())

    @classmethod
    def from_dict(cls, dct: dict) -> Settings:
        names = attr.fields_dict(cls)
        try:
            return cls(**{name: value for name, value in dct.items() if name in names})
        except TypeError as error:
            raise LoginError(f"invalid catalog sign-in settings: {error}") from None

    @classmethod
    def fetch(cls, service_url: str) -> Settings:
        return cls.from_dict(_get_json(service_url.rstrip("/") + WELL_KNOWN_PATH))

    @property
    def scope(self) -> str:
        return " ".join(self.scopes)


@frozen
class Provider:
    """The endpoints from the issuer's OpenID Connect discovery document."""

    token_endpoint: str = field(
        converter=lambda value: _https_url(value, "token_endpoint")
    )
    authorization_endpoint: str | None = field(
        default=None,
        converter=lambda value: _https_url(value, "authorization_endpoint"),
    )
    device_authorization_endpoint: str | None = field(
        default=None,
        converter=lambda value: _https_url(value, "device_authorization_endpoint"),
    )

    @classmethod
    def discover(cls, issuer: str) -> Provider:
        document = _get_json(f"{issuer.rstrip('/')}/.well-known/openid-configuration")
        if document.get("issuer") != issuer or "token_endpoint" not in document:
            raise LoginError(f"{issuer} returned an invalid discovery document")
        names = attr.fields_dict(cls)
        return cls(**{name: value for name, value in document.items() if name in names})


@frozen
class Login:
    """A cached sign-in: the service's settings and the tokens it accepts."""

    service_url: str
    settings: Settings = field(
        converter=lambda value: (
            value if isinstance(value, Settings) else Settings.from_dict(value)
        )
    )
    token_endpoint: str
    token: str
    expires_at: int
    refresh_token: str | None = None

    @classmethod
    def from_token_response(
        cls,
        response: dict,
        *,
        service_url: str,
        settings: Settings,
        token_endpoint: str,
    ) -> Login:
        if "error" in response:
            raise _failure(response)
        # Send the token the service checks: the one whose aud is the
        # published audience (the access token on most providers).
        matching = (
            (response[name], claims)
            for name in ("access_token", "id_token")
            if settings.audience in _audiences(claims := _claims(response.get(name)))
        )
        token, claims = next(matching, (None, None))
        if token is None:
            raise LoginError(
                f"neither the access token nor the ID token is for audience "
                f"{settings.audience!r}; the identity provider is misconfigured"
            )
        return cls(
            service_url=service_url,
            settings=settings,
            token_endpoint=token_endpoint,
            token=token,
            expires_at=int(
                claims.get("exp") or time.time() + response.get("expires_in", 0)
            ),
            refresh_token=response.get("refresh_token"),
        )

    @classmethod
    def load(cls, service_url: str) -> Login | None:
        path = login_path(service_url)
        return cls(**json.loads(path.read_text())) if path.exists() else None

    def save(self) -> Login:
        # Written to a 0600 temporary file and renamed into place, so an
        # interrupted sign-in leaves the previous login intact.
        path = login_path(self.service_url)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.parent.chmod(0o700)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as file:
                json.dump(attr.asdict(self), file)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        return self


class _CallbackHandler(BaseHTTPRequestHandler):
    timeout = CALLBACK_READ_TIMEOUT_SECONDS

    def do_GET(self) -> None:
        parts = urlsplit(self.path)
        self.server.callbacks.append((parts.path, dict(parse_qsl(parts.query))))
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(CALLBACK_PAGE)

    def log_message(self, *args: object) -> None:
        pass


def _bind_loopback(redirect_uris: tuple) -> tuple[str, HTTPServer] | None:
    # Providers match redirect URIs exactly, so only the published loopback
    # callbacks can be used; the first free one wins.
    for uri in redirect_uris:
        parts = urlsplit(uri)
        if parts.scheme != "http" or not _is_loopback_host(parts.hostname or ""):
            continue
        server_class = (
            type("_HTTPServer6", (HTTPServer,), {"address_family": socket.AF_INET6})
            if ":" in parts.hostname
            else HTTPServer
        )
        try:
            server = server_class((parts.hostname, parts.port or 80), _CallbackHandler)
        except OSError:
            continue
        server.callbacks = []
        server.timeout = CALLBACK_POLL_SECONDS
        return uri, server
    return None


def _await_callback(
    server: HTTPServer, path: str, gave_up: Callable[[], bool]
) -> dict | None:
    deadline = time.monotonic() + CALLBACK_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if gave_up():
            return None
        server.handle_request()
        if params := next(
            (p for (got, p) in server.callbacks if got == path and p), None
        ):
            return params
    raise LoginError("timed out waiting for the browser sign-in")


def _authorization_code(
    settings: Settings, provider: Provider, open_browser: Callable[[str], bool]
) -> dict | None:
    """Sign in through the browser; None when that is not possible here."""
    if provider.authorization_endpoint is None:
        return None
    if (bound := _bind_loopback(settings.redirect_uris)) is None:
        return None
    redirect_uri, server = bound
    verifier = secrets.token_urlsafe(64)
    challenge = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
        .rstrip(b"=")
        .decode()
    )
    state = secrets.token_urlsafe(32)
    query = urlencode(
        {
            "response_type": "code",
            "client_id": settings.client_id,
            "redirect_uri": redirect_uri,
            "scope": settings.scope,
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    with server:
        endpoint = urlsplit(provider.authorization_endpoint)
        url = urlunsplit(
            (*endpoint[:3], "&".join(filter(None, (endpoint.query, query))), "")
        )
        # Some launchers return only when the browser exits, which waits on
        # this callback, so the launch runs beside the callback server.
        launched = []

        def launch() -> None:
            try:
                opened = open_browser(url)
            except (OSError, webbrowser.Error):
                opened = False  # a browser that fails to start is no browser
            launched.append(opened)

        threading.Thread(target=launch, daemon=True).start()
        params = _await_callback(
            server,
            urlsplit(redirect_uri).path or "/",
            gave_up=lambda: launched == [False],
        )
        if params is None:
            return None
    if params.get("state") != state:
        raise LoginError("the sign-in callback did not match this login attempt")
    if "error" in params:
        raise _failure(params)
    return _post_form(
        provider.token_endpoint,
        {
            "grant_type": "authorization_code",
            "code": params.get("code", ""),
            "redirect_uri": redirect_uri,
            "client_id": settings.client_id,
            "code_verifier": verifier,
        },
    )


def _device_code(
    settings: Settings, provider: Provider, emit: Callable[[dict], None]
) -> dict:
    if provider.device_authorization_endpoint is None:
        raise LoginError("the identity provider does not offer device sign-in")
    grant = _post_form(
        provider.device_authorization_endpoint,
        {"client_id": settings.client_id, "scope": settings.scope},
    )
    if "error" in grant:
        raise _failure(grant)
    emit(
        {"event": "device_authorization"}
        | {
            name: grant[name]
            for name in (
                "verification_uri",
                "verification_uri_complete",
                "user_code",
                "expires_in",
            )
            if name in grant
        }
    )
    interval = grant.get("interval", 5)
    deadline = time.monotonic() + grant["expires_in"]
    while (remaining := deadline - time.monotonic()) > 0:
        time.sleep(min(interval, remaining))
        if time.monotonic() >= deadline:
            break
        try:
            response = _post_form(
                provider.token_endpoint,
                {
                    "grant_type": DEVICE_CODE_GRANT,
                    "device_code": grant["device_code"],
                    "client_id": settings.client_id,
                },
            )
        except LoginTimeoutError:
            # RFC 8628 section 3.5: back off after a timeout, keep polling
            response = {"error": "slow_down"}
        match response.get("error"):
            case "authorization_pending":
                continue
            case "slow_down":
                interval += SLOW_DOWN_SECONDS
            case _:
                return response
    raise _failure({"error": "expired_token"})


def sign_in(
    service_url: str,
    *,
    device: bool = False,
    settings_file: str | None = None,
    emit: Callable[[dict], None] = print,
    open_browser: Callable[[str], bool] = webbrowser.open,
) -> Login:
    """Sign in to SERVICE_URL and cache the result; returns the new login."""
    login_path(service_url)  # validates the URL before any network call
    settings = (
        Settings.from_dict(json.loads(Path(settings_file).read_text()))
        if settings_file
        else Settings.fetch(service_url)
    )
    provider = Provider.discover(settings.issuer)
    # an empty grant_types publishes no restriction
    browser = not device and (
        not settings.grant_types or "authorization_code" in settings.grant_types
    )
    response = (
        browser and _authorization_code(settings, provider, open_browser)
    ) or _device_code(settings, provider, emit)
    return Login.from_token_response(
        response,
        service_url=service_url,
        settings=settings,
        token_endpoint=provider.token_endpoint,
    ).save()
