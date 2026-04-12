# ADR-0006: Fix OTLP collector probe in otel_utils

- **Status:** Accepted
- **Date:** 2026-04-12
- **Context area:** `python/xorq/common/utils/otel_utils.py`

## Context

`localhost_and_listening()` determines whether an OTLP collector is accepting connections on the configured endpoint before creating the OTLP exporter. If no collector is detected, a no-op `ConsoleSpanExporter` writing to `/dev/null` is used instead.

The original implementation used `socket.bind()`, which is a server-side operation that tests whether a port is *free* — the inverse of what we need. It also leaked the socket (no `close()` call).

## Decision

Replace `socket.bind()` with `socket.create_connection()`:

- **`bind()`** asks "can I claim this port?" If something else holds it, `bind` fails — but that something may not be an OTLP collector.
- **`create_connection()`** performs a real TCP handshake, confirming a service is actively accepting connections.

A 1-second timeout is set as a safety net. In practice, `create_connection` to localhost resolves in under 1 ms when a service is listening and fails immediately with `ConnectionRefusedError` when nothing is listening. The timeout only guards against edge cases (half-open socket, firewall silently dropping packets on a non-localhost resolved address).

The socket is managed via context manager to prevent leaks.

```python
def localhost_and_listening(uri):
    parsed = urllib.parse.urlparse(uri)
    if parsed.hostname != "localhost":
        return None
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            return True
    except (OSError, TimeoutError):
        return False
```

## Consequences

- The OTLP exporter is only created when a collector is genuinely accepting connections.
- No socket leaks.
- No change to the default endpoint configuration (`localhost:4318`); the probe is the safety mechanism.
