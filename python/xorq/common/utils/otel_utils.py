import logging
import os
import sys

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource


# Try to import gRPC exporter - will be used if available and needed
try:
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter as OTLPSpanExporterGRPC,
    )

    HAS_GRPC_EXPORTER = True
except ImportError:
    HAS_GRPC_EXPORTER = False
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


class LoggingOTLPSpanExporter(OTLPSpanExporter):
    """Wrapper around OTLPSpanExporter that logs export attempts for debugging."""

    def export(self, spans):
        """Export spans with detailed logging."""
        logger.debug(f"Attempting to export {len(spans)} spans to OTLP endpoint")
        try:
            result = super().export(spans)
            logger.debug(f"Successfully exported {len(spans)} spans")
            return result
        except Exception as e:
            logger.error("=" * 80)
            logger.error("OTLP EXPORT FAILED - CONNECTION ERROR DETAILS")
            logger.error("=" * 80)
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error message: {str(e)}")
            logger.error(
                f"Endpoint: {self._endpoint if hasattr(self, '_endpoint') else 'unknown'}"
            )
            logger.error(f"Number of spans attempted: {len(spans)}")

            # Log first span details for debugging
            if spans:
                first_span = spans[0]
                logger.error(f"First span name: {first_span.name}")
                logger.error(f"First span trace_id: {hex(first_span.context.trace_id)}")

            logger.error("=" * 80)
            logger.error("This error typically indicates:")
            logger.error("1. The OTLP endpoint is not reachable from the container")
            logger.error("2. Network policies may be blocking the connection")
            logger.error(
                "3. The endpoint requires authentication that's not configured"
            )
            logger.error(
                "4. The endpoint is internal to SPCS and not accessible from user containers"
            )
            logger.error("=" * 80)
            raise


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
    logger.info(f"Using custom local endpoint: {custom_endpoint}")
    processor = BatchSpanProcessor(LoggingOTLPSpanExporter(endpoint=custom_endpoint))
elif os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
):
    # Fix SPCS misconfiguration: port 4317 requires gRPC but SPCS doesn't set protocol
    # Check if we have the SPCS misconfiguration and fix it
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL")

    # Detect SPCS misconfiguration: port 4317 with wrong protocol
    # Port 4317 is the standard gRPC port, but SPCS doesn't set the protocol correctly
    if (
        (traces_endpoint and ":4317" in traces_endpoint)
        or (base_endpoint and ":4317" in base_endpoint)
    ) and protocol != "grpc":
        # Fix by setting the protocol to gRPC for port 4317
        os.environ["OTEL_EXPORTER_OTLP_PROTOCOL"] = "grpc"
        logger.warning("=" * 80)
        logger.warning("SPCS ENDPOINT MISCONFIGURATION DETECTED AND FIXED")
        if traces_endpoint:
            logger.warning(f"Endpoint: {traces_endpoint}")
        else:
            logger.warning(f"Endpoint: {base_endpoint}")
        logger.warning(
            f"Port 4317 detected with wrong protocol: {protocol or 'not set'}"
        )
        logger.warning("Fixed by setting OTEL_EXPORTER_OTLP_PROTOCOL=grpc")
        logger.warning("=" * 80)

    # Log all OTLP-related environment variables for debugging
    logger.info("=" * 80)
    logger.info("OTLP CONFIGURATION DETECTED - LOGGING FOR DEBUGGING")
    logger.info("=" * 80)

    # Log all OTEL environment variables
    otel_env_vars = {
        "OTEL_EXPORTER_OTLP_ENDPOINT": os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"),
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT": os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
        ),
        "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT": os.getenv(
            "OTEL_EXPORTER_OTLP_METRICS_ENDPOINT"
        ),
        "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT": os.getenv(
            "OTEL_EXPORTER_OTLP_LOGS_ENDPOINT"
        ),
        "OTEL_EXPORTER_OTLP_HEADERS": os.getenv("OTEL_EXPORTER_OTLP_HEADERS"),
        "OTEL_EXPORTER_OTLP_TRACES_HEADERS": os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_HEADERS"
        ),
        "OTEL_EXPORTER_OTLP_TIMEOUT": os.getenv("OTEL_EXPORTER_OTLP_TIMEOUT"),
        "OTEL_EXPORTER_OTLP_PROTOCOL": os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL"),
        "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL": os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_PROTOCOL"
        ),
        "OTEL_EXPORTER_OTLP_COMPRESSION": os.getenv("OTEL_EXPORTER_OTLP_COMPRESSION"),
        "OTEL_EXPORTER_OTLP_CERTIFICATE": os.getenv("OTEL_EXPORTER_OTLP_CERTIFICATE"),
        "OTEL_SERVICE_NAME": os.getenv("OTEL_SERVICE_NAME"),
        "OTEL_RESOURCE_ATTRIBUTES": os.getenv("OTEL_RESOURCE_ATTRIBUTES"),
        "SNOWFLAKE_ACCOUNT": os.getenv("SNOWFLAKE_ACCOUNT"),
        "SNOWFLAKE_HOST": os.getenv("SNOWFLAKE_HOST"),
        "SNOWPARK_SESSION": os.getenv("SNOWPARK_SESSION"),
        "EXECUTION_ID": os.getenv("EXECUTION_ID"),
        "SPCS_IMAGE_REPOSITORY": os.getenv("SPCS_IMAGE_REPOSITORY"),
        "SPCS_CONTAINER_NAME": os.getenv("SPCS_CONTAINER_NAME"),
    }

    logger.info("OTEL Environment Variables:")
    for key, value in otel_env_vars.items():
        if value is not None:
            # Mask sensitive headers but show structure
            if "HEADERS" in key and value:
                # Show header keys but mask values
                try:
                    headers = [h.split("=")[0] for h in value.split(",")]
                    logger.info(f"  {key}: <headers present: {', '.join(headers)}>")
                except Exception:
                    logger.info(f"  {key}: <headers present but format unclear>")
            else:
                logger.info(f"  {key}: {value}")

    # Re-read endpoints after potential fix
    traces_endpoint = os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
    base_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")

    if traces_endpoint:
        effective_endpoint = traces_endpoint
        logger.info(f"\nUsing traces-specific endpoint: {traces_endpoint}")
    elif base_endpoint:
        # SDK will append /v1/traces to base endpoint
        effective_endpoint = (
            f"{base_endpoint}/v1/traces"
            if not base_endpoint.endswith("/v1/traces")
            else base_endpoint
        )
        logger.info(
            f"\nUsing base endpoint (SDK will append /v1/traces): {base_endpoint}"
        )
        logger.info(f"Effective traces endpoint will be: {effective_endpoint}")
    else:
        effective_endpoint = "UNKNOWN"
        logger.warning("No OTLP endpoint found - this should not happen")

    # Log protocol information
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf"
    logger.info(f"\nOTLP Protocol: {protocol}")

    # Log connection details for debugging
    logger.info("\nConnection attempt will be made to:")
    logger.info(f"  Endpoint: {effective_endpoint}")
    logger.info(f"  Protocol: {protocol}")
    logger.info(
        f"  Timeout: {os.getenv('OTEL_EXPORTER_OTLP_TIMEOUT') or 'default (10s)'}"
    )
    logger.info(
        f"  Compression: {os.getenv('OTEL_EXPORTER_OTLP_COMPRESSION') or 'none'}"
    )

    # Create OTLP exporter with logging - use gRPC for port 4317
    try:
        # Check if we should use gRPC exporter
        protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL") or "http/protobuf"

        if protocol == "grpc" and HAS_GRPC_EXPORTER:
            logger.info("\nCreating gRPC OTLP exporter for protocol=grpc...")
            # Use gRPC exporter for gRPC protocol
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter as OTLPSpanExporterGRPC,
            )

            # Create a logging wrapper for gRPC exporter
            class LoggingGRPCSpanExporter(OTLPSpanExporterGRPC):
                def export(self, spans):
                    logger.debug(
                        f"Attempting to export {len(spans)} spans via gRPC to OTLP endpoint"
                    )
                    try:
                        result = super().export(spans)
                        logger.debug(
                            f"Successfully exported {len(spans)} spans via gRPC"
                        )
                        return result
                    except Exception as e:
                        logger.error("=" * 80)
                        logger.error(
                            "OTLP gRPC EXPORT FAILED - CONNECTION ERROR DETAILS"
                        )
                        logger.error("=" * 80)
                        logger.error(f"Error type: {type(e).__name__}")
                        logger.error(f"Error message: {str(e)}")
                        logger.error("Protocol: gRPC")
                        logger.error(f"Number of spans attempted: {len(spans)}")
                        if spans:
                            first_span = spans[0]
                            logger.error(f"First span name: {first_span.name}")
                            logger.error(
                                f"First span trace_id: {hex(first_span.context.trace_id)}"
                            )
                        logger.error("=" * 80)
                        raise

            otlp_exporter = LoggingGRPCSpanExporter()
            logger.info("gRPC OTLP exporter created successfully")
        else:
            # Use HTTP exporter (default)
            logger.info("\nCreating HTTP OTLP exporter (LoggingOTLPSpanExporter)...")
            otlp_exporter = LoggingOTLPSpanExporter()

            # Log exporter configuration if available
            if hasattr(otlp_exporter, "_endpoint"):
                logger.info(
                    f"HTTP OTLP exporter created with endpoint: {otlp_exporter._endpoint}"
                )
            if hasattr(otlp_exporter, "_headers"):
                logger.info(
                    f"HTTP OTLP exporter headers configured: {'yes' if otlp_exporter._headers else 'no'}"
                )

        processor = BatchSpanProcessor(otlp_exporter)
        logger.info("BatchSpanProcessor created successfully")

    except Exception as e:
        logger.error(f"Failed to create OTLP exporter: {e}", exc_info=True)
        logger.error("Falling back to console exporter")
        processor = BatchSpanProcessor(ConsoleSpanExporter(out=sys.stdout))

    logger.info("=" * 80)
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

# Log final configuration
logger.info("=" * 80)
logger.info("TELEMETRY INITIALIZATION COMPLETE")
logger.info(f"TracerProvider: {provider}")
logger.info(f"Processor type: {processor.__class__.__name__}")
if hasattr(processor, "span_exporter"):
    logger.info(f"Exporter type: {processor.span_exporter.__class__.__name__}")
logger.info("=" * 80)

# Creates a tracer from the global tracer provider
tracer = trace.get_tracer("xorq.tracer")
