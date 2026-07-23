from __future__ import annotations

import pytest

import xorq.api as xo
from xorq.vendor.ibis.backends import BaseBackend


fake_env = {
    "MIXPANEL_SERVICE_ACCOUNT_USERNAME": "fake-user.abc123",
    "MIXPANEL_SERVICE_ACCOUNT_SECRET": "fake-secret-value",
    "MIXPANEL_PROJECT_ID": "1234567",
}
env_ref_kwargs = {
    "username": "${MIXPANEL_SERVICE_ACCOUNT_USERNAME}",
    "secret": "${MIXPANEL_SERVICE_ACCOUNT_SECRET}",
    "project_id": "${MIXPANEL_PROJECT_ID}",
}


@pytest.fixture
def con(monkeypatch: pytest.MonkeyPatch) -> BaseBackend:
    for name, value in fake_env.items():
        monkeypatch.setenv(name, value)
    return xo.load_backend("mixpanel").connect(**env_ref_kwargs)
