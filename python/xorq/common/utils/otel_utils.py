import os
import sys

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
    OTLPSpanExporter as OTLPSpanExporterGRPC,
)
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


def get_otlp_exporter():
    """Create OTLP exporter based on protocol configuration.

    SDK auto-configures from standard OTEL environment variables:
    - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT
    - OTEL_EXPORTER_OTLP_PROTOCOL (grpc or http/protobuf)
    - OTEL_EXPORTER_OTLP_HEADERS
    - OTEL_EXPORTER_OTLP_TIMEOUT
    """
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    if protocol == "grpc":
        return OTLPSpanExporterGRPC()
    else:
        return OTLPSpanExporter()


resource_attributes = {
    SERVICE_NAME: otel_config.OTEL_SERVICE_NAME,
    **({"execution.id": eid} if (eid := otel_config.OTEL_EXECUTION_ID) else {}),
}

resource = Resource(resource_attributes)

provider = TracerProvider(resource=resource)

traces_endpoint = otel_config.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT

if traces_endpoint and localhost_and_listening(traces_endpoint):
    processor = BatchSpanProcessor(get_otlp_exporter())
else:
    processor = BatchSpanProcessor(
        ConsoleSpanExporter(
            out=sys.stdout
            if otel_config.get("OTEL_EXPORTER_CONSOLE_FALLBACK")
            else open(os.devnull, "w")
        )
    )
provider.add_span_processor(processor)
trace.set_tracer_provider(provider)

tracer = trace.get_tracer("xorq.tracer")
