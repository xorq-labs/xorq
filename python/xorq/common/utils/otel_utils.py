import functools
import os
import sys
import threading

from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)


_otel_template = env_templates_dir.joinpath(".env.otel.template")
if _otel_template.exists():
    OTELConfig = EnvConfigable.subclass_from_env_file(_otel_template)
else:
    OTELConfig = EnvConfigable.subclass_from_kwargs(
        "OTEL_LOG_FILE_NAME",
        "OTEL_HOST_LOG_DIR",
        "OTEL_COLLECTOR_CONTAINER_LOG_DIR",
        "OTEL_COLLECTOR_PORT_GRPC",
        "OTEL_COLLECTOR_PORT_HTTP",
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
        "OTEL_SERVICE_NAME",
        "OTEL_EXPORTER_OTLP_PROTOCOL",
        "OTEL_EXPORTER_CONSOLE_FALLBACK",
        "OTEL_EXECUTION_ID",
        "GRAFANA_CLOUD_OTLP_ENDPOINT",
        "GRAFANA_CLOUD_INSTANCE_ID",
        "GRAFANA_CLOUD_API_KEY",
        "PROMETHEUS_SCRAPE_URL",
        "PROMETHEUS_GRAFANA_ENDPOINT",
        "PROMETHEUS_GRAFANA_USERNAME",
    )
otel_config = OTELConfig.from_env()


_real_tracer = None
_tracer_lock = threading.Lock()


def is_localhost_collector_listening(uri):
    import socket  # noqa: PLC0415
    import urllib.parse  # noqa: PLC0415

    parsed = urllib.parse.urlparse(uri)
    if parsed.hostname != "localhost":
        return False
    if parsed.port is None:
        return False
    try:
        with socket.create_connection((parsed.hostname, parsed.port), timeout=1):
            return True
    except OSError:
        return False


def _should_use_otlp_exporter(endpoint):
    import urllib.parse  # noqa: PLC0415

    if not endpoint:
        return False
    parsed = urllib.parse.urlparse(endpoint)
    if parsed.hostname != "localhost":
        return True
    return is_localhost_collector_listening(endpoint)


def _get_otlp_exporter():
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    if protocol == "grpc":
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter as OTLPSpanExporterGRPC,
        )

        return OTLPSpanExporterGRPC()
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
            OTLPSpanExporter,
        )

        return OTLPSpanExporter()


_devnull = None


def _init_tracer():
    global _real_tracer, _devnull
    if _real_tracer is not None:
        return _real_tracer

    with _tracer_lock:
        if _real_tracer is not None:
            return _real_tracer

        from opentelemetry import trace  # noqa: PLC0415
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource  # noqa: PLC0415
        from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
        from opentelemetry.sdk.trace.export import (  # noqa: PLC0415
            BatchSpanProcessor,
            ConsoleSpanExporter,
        )

        resource_attributes = {
            SERVICE_NAME: otel_config.OTEL_SERVICE_NAME,
            **({"execution.id": eid} if (eid := otel_config.OTEL_EXECUTION_ID) else {}),
        }

        resource = Resource(resource_attributes)
        provider = TracerProvider(resource=resource)

        traces_endpoint = otel_config.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

        if _should_use_otlp_exporter(traces_endpoint):
            processor = BatchSpanProcessor(_get_otlp_exporter())
        else:
            if _devnull is None:
                _devnull = open(os.devnull, "w")
            processor = BatchSpanProcessor(
                ConsoleSpanExporter(
                    out=sys.stdout
                    if otel_config.get("OTEL_EXPORTER_CONSOLE_FALLBACK")
                    else _devnull
                )
            )
        provider.add_span_processor(processor)
        trace.set_tracer_provider(provider)

        _real_tracer = trace.get_tracer("xorq.tracer")
        return _real_tracer


class _LazySpan:
    __slots__ = ("_name", "_kwargs", "_cm")

    def __init__(self, name, **kwargs):
        self._name = name
        self._kwargs = kwargs
        self._cm = None

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*fargs, **fkwargs):
            with _init_tracer().start_as_current_span(self._name, **self._kwargs):
                return func(*fargs, **fkwargs)

        return wrapper

    def __enter__(self):
        self._cm = _init_tracer().start_as_current_span(self._name, **self._kwargs)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)


class _LazyTracer:
    def start_as_current_span(self, name, **kwargs):
        return _LazySpan(name, **kwargs)


def get_current_span():
    """Return the current span, initializing the tracer if needed.

    Without this, calls before any _LazySpan enters would hit the
    default no-op provider instead of our configured one.
    """
    _init_tracer()
    from opentelemetry import trace  # noqa: PLC0415

    return trace.get_current_span()


def set_span_ok(span):
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    span.set_status(StatusCode.OK)


def set_span_error(span, exc):
    from opentelemetry.trace import StatusCode  # noqa: PLC0415

    span.set_status(StatusCode.ERROR, str(exc))
    span.record_exception(exc)


def create_link(span_context):
    from opentelemetry import trace  # noqa: PLC0415

    return trace.Link(span_context)


tracer = _LazyTracer()
