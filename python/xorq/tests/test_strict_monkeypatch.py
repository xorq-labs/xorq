import pytest


def test_rejects_nonexistent_module_attr(monkeypatch):
    with pytest.raises(AttributeError, match="patch the canonical source module"):
        monkeypatch.setattr("xorq.caching.storage.DOES_NOT_EXIST", lambda: None)


def test_allows_existing_module_attr(monkeypatch):
    monkeypatch.setattr("os.path.sep", "/")


def test_object_target_bypasses_check(monkeypatch):
    import os.path  # noqa: PLC0415

    monkeypatch.setattr(os.path, "sep", "/")
