import logging
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

logger = logging.getLogger(__name__)


def get_otlp_exporter():
    """Create OTLP exporter based on protocol configuration.

    SDK auto-configures from standard OTEL environment variables:
    - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT
    - OTEL_EXPORTER_OTLP_PROTOCOL (grpc or http/protobuf)
    - OTEL_EXPORTER_OTLP_HEADERS
    - OTEL_EXPORTER_OTLP_TIMEOUT
    """
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "http/protobuf")

    # Support both 'grpc' and 'http/protobuf' protocol values
    if protocol == "grpc":
        # SDK will read OTEL_EXPORTER_OTLP_TRACES_ENDPOINT or OTEL_EXPORTER_OTLP_ENDPOINT
        return OTLPSpanExporterGRPC()
    else:
        # Default to HTTP exporter for 'http/protobuf' or any other value
        return OTLPSpanExporter()


# Build resource attributes
resource_attributes = {
    SERVICE_NAME: otel_config.OTEL_SERVICE_NAME,
}

# Add execution ID if available (for SPCS and other execution contexts)
# This enables filtering traces by specific runs in Snowflake Trail
execution_id = os.environ.get("EXECUTION_ID")
if execution_id:
    resource_attributes["execution.id"] = execution_id
    logger.debug(f"Added execution.id={execution_id} to telemetry resource attributes")

resource = Resource(attributes=resource_attributes)
provider = TracerProvider(resource=resource)

# Create the appropriate exporter based on configuration
# SDK auto-configures from standard OTEL environment variables:
# - OTEL_EXPORTER_OTLP_TRACES_ENDPOINT (or OTEL_EXPORTER_OTLP_ENDPOINT)
# - OTEL_EXPORTER_OTLP_PROTOCOL
# - OTEL_EXPORTER_OTLP_HEADERS
# - OTEL_EXPORTER_OTLP_TIMEOUT
traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or os.getenv(
    "OTEL_EXPORTER_OTLP_ENDPOINT"
)

if traces_endpoint and localhost_and_listening(traces_endpoint):
    # Use OTLP exporter with auto-configuration from environment
    processor = BatchSpanProcessor(get_otlp_exporter())
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
