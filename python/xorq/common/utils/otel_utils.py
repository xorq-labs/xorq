import logging
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


# Set up logging for telemetry issues
logger = logging.getLogger(__name__)

# Import Snowflake Trace ID generator for SPCS/Snowflake Trail compatibility
# This is required for proper trace ID format in SPCS
try:
    from snowflake.telemetry.trace import SnowflakeTraceIdGenerator

    HAS_SNOWFLAKE_TELEMETRY = True
except ImportError:
    HAS_SNOWFLAKE_TELEMETRY = False
    logger.debug("snowflake-telemetry-python not available, using standard trace IDs")


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


def should_use_snowflake_trace_id():
    """Determine if we should use Snowflake's trace ID generator.

    Returns True if:
    - Running in Snowpark context
    - OTEL_USE_SNOWFLAKE_TRACE_ID env var is set
    - Or SNOWFLAKE_ACCOUNT is set (indicating Snowflake environment)
    """
    return (
        os.getenv("OTEL_USE_SNOWFLAKE_TRACE_ID") is not None
        or os.getenv("SNOWFLAKE_ACCOUNT") is not None
        or os.getenv("SNOWPARK_SESSION") is not None
    )


OTELConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.otel.template")
)
otel_config = OTELConfig.from_env()

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

# Create TracerProvider with Snowflake trace ID generator if needed
provider_kwargs = {"resource": resource}
if HAS_SNOWFLAKE_TELEMETRY and should_use_snowflake_trace_id():
    # Use Snowflake-compatible trace IDs for proper Trail integration
    # Format: 16-byte big-endian with timestamp in 4 highest-order bytes
    provider_kwargs["id_generator"] = SnowflakeTraceIdGenerator()
    logger.info("Using SnowflakeTraceIdGenerator for Snowflake Trail compatibility")

provider = TracerProvider(**provider_kwargs)

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
    # Environment has OTEL env vars set - use OTLP export
    # The SDK will automatically read OTEL_EXPORTER_OTLP_ENDPOINT and OTEL_EXPORTER_OTLP_HEADERS
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
