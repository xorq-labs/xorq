"""Shared test helpers for dasher / caching test modules."""


class Probe:
    """Object with ``__dasher_tokenize__`` for testing dunder dispatch."""

    def __init__(self, payload):
        self.payload = payload

    def __dasher_tokenize__(self):
        return ("Probe.dunder", self.payload)


class MockOp:
    """Minimal op-like object for ``_parent_token`` tests."""

    schema = "test-schema"

    def op(self):
        return self

    def __hash__(self):
        return id(self)


class BombHasher:
    """Hasher that always raises RecursionError -- triggers fallback paths."""

    def tokenize(self, *args, **kwargs):
        raise RecursionError("synthetic")


class FakeRead:
    """Minimal Read-like object for normalizer tests."""

    schema = "fake-schema"

    def __init__(self, hash_path):
        self.read_kwargs = (("hash_path", hash_path),)
