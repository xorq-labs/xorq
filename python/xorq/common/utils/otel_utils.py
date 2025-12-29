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


OTELConfig = EnvConfigable.subclass_from_env_file(
    env_templates_dir.joinpath(".env.otel.template")
)
otel_config = OTELConfig.from_env()


resource = Resource(
    attributes={
        SERVICE_NAME: otel_config.OTEL_SERVICE_NAME,
    }
)

# Detect if we're in a Snowflake environment (SPCS or using Snowflake OTLP)
is_snowflake_env = (
    os.getenv("SNOWFLAKE_ACCOUNT") is not None
    or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") is not None
    or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") is not None
)

# Create TracerProvider with Snowflake trace ID generator if needed
provider_kwargs = {"resource": resource}
if HAS_SNOWFLAKE_TELEMETRY and is_snowflake_env:
    # Use Snowflake-compatible trace IDs for proper Trail integration
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
    # Check if OTLP export is disabled (useful for SPCS where endpoints aren't accessible)
    if os.getenv("XORQ_DISABLE_OTLP_EXPORT"):
        logger.info(
            "OTLP export disabled by XORQ_DISABLE_OTLP_EXPORT environment variable"
        )
        processor = BatchSpanProcessor(ConsoleSpanExporter(out=open(os.devnull, "w")))
    # Auto-detect SPCS - SPCS sets both SNOWFLAKE_ACCOUNT and OTLP endpoints
    # but the OTLP endpoints are not directly accessible from user containers
    elif os.getenv("SNOWFLAKE_ACCOUNT"):
        # Running in SPCS - disable direct OTLP export to avoid connection reset
        # SPCS will capture telemetry through stdout/event tables instead
        logger.info(
            "SPCS environment detected (SNOWFLAKE_ACCOUNT set) - disabling direct OTLP export"
        )
        logger.info(
            "To override this, unset SNOWFLAKE_ACCOUNT or set XORQ_FORCE_OTLP_EXPORT=1"
        )
        if not os.getenv("XORQ_FORCE_OTLP_EXPORT"):
            processor = BatchSpanProcessor(
                ConsoleSpanExporter(out=open(os.devnull, "w"))
            )
        else:
            # Force OTLP export even in SPCS (for debugging)
            logger.warning("Forcing OTLP export in SPCS environment")
            processor = BatchSpanProcessor(OTLPSpanExporter())
    else:
        # Non-SPCS environment with OTEL env vars - use OTLP
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
