"""OIDC sign-in to a hosted catalog service: settings, discovery, token cache.

The service publishes its client settings at an unauthenticated well-known
URL; everything else is standard OAuth 2.0 / OpenID Connect, so no provider
is assumed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import attr
from attr import field, frozen

from xorq.catalog.content_store import _service_url_identity
from xorq.catalog.exceptions import LoginError


WELL_KNOWN_PATH = "/.well-known/xorq-catalog"
HTTP_TIMEOUT_SECONDS = 30
JSON_HEADERS = {"Accept": "application/json"}


def _read_json(request: Request) -> dict:
    # OAuth endpoints answer errors with a 4xx and a JSON body naming the
    # error; those bodies are returned so callers can match on "error".
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_SECONDS) as response:
            body = json.load(response)
    except HTTPError as error:
        with error:
            try:
                body = json.load(error)
            except ValueError:
                body = None
        if isinstance(body, dict) and "error" in body:
            return body
        raise LoginError(f"{request.full_url} returned HTTP {error.code}") from None
    except OSError as error:
        raise LoginError(f"cannot reach {request.full_url}: {error}") from None
    except ValueError:
        raise LoginError(f"{request.full_url} did not return JSON") from None
    if not isinstance(body, dict):
        raise LoginError(f"{request.full_url} did not return a JSON object")
    return body


def _get_json(url: str) -> dict:
    return _read_json(Request(url, headers=JSON_HEADERS))


def _failure(response: dict) -> LoginError:
    match response:
        case {"error": "access_denied"}:
            return LoginError("sign-in was denied")
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

    issuer: str = field(validator=attr.validators.instance_of(str))
    client_id: str = field(validator=attr.validators.instance_of(str))
    audience: str = field(validator=attr.validators.instance_of(str))
    scopes: tuple = field(converter=tuple)

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

    token_endpoint: str
    device_authorization_endpoint: str | None = None

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
