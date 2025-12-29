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

# Create TracerProvider with standard configuration
provider = TracerProvider(resource=resource)

# OTEL SDK automatically reads standard environment variables:
# - OTEL_EXPORTER_OTLP_ENDPOINT (set by SPCS)
# - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (set by SPCS)
# - OTEL_EXPORTER_OTLP_HEADERS
# We only need to override for local development

custom_endpoint = otel_config.get("OTEL_ENDPOINT_URI")

# Use custom endpoint for local development, otherwise let OTEL auto-configure
if custom_endpoint and localhost_and_listening(custom_endpoint):
    # Local development with OTEL collector running
    processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=custom_endpoint))
elif os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
):
    # SPCS or any environment with standard OTEL env vars - let SDK auto-configure
    processor = BatchSpanProcessor(OTLPSpanExporter())
else:
    # Fallback to console exporter
    processor = BatchSpanProcessor(
        ConsoleSpanExporter(
            out=sys.stdout
            if otel_config.get("OTEL_EXPORTER_CONSOLE_FALLBACK")
            else open(os.devnull, "w")
        )
    )

provider.add_span_processor(processor)
trace.set_tracer_provider(provider)


# Creates a tracer from the global tracer provider
tracer = trace.get_tracer("xorq.tracer")
