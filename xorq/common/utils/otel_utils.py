import os
import sys

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)

from xorq.common.utils.env_utils import (
    EnvConfigable,
    env_templates_dir,
)


def localhost_and_listening(uri):
    import socket
    import urllib

    parsed = urllib.parse.urlparse(uri)
    localhost = "localhost"
    if parsed.hostname == localhost:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((localhost, parsed.port))
        except OSError:
            return True
        else:
            return False
    return None


OTELConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.otel.template")
)
otel_config = OTELConfig.from_env()


resource = Resource(
    attributes={
        SERVICE_NAME: otel_config.OTEL_SERVICE_NAME,
    }
)
provider = TracerProvider(resource=resource)
processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint=otel_config["OTEL_ENDPOINT_URI"])
    if otel_config["OTEL_ENDPOINT_URI"]
    and localhost_and_listening(otel_config["OTEL_ENDPOINT_URI"])
    else ConsoleSpanExporter(
        out=sys.stdout
        if otel_config["OTEL_EXPORTER_CONSOLE_FALLBACK"]
        else open(os.devnull, "w")
    )
)
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)


# Creates a tracer from the global tracer provider
tracer = trace.get_tracer("xorq.tracer")
