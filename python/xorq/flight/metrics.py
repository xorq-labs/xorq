import functools
import time
from typing import Optional

import pyarrow as pa
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    MetricExporter,
    MetricExportResult,
    PeriodicExportingMetricReader,
)

from xorq.common.utils.logging_utils import get_print_logger


__all__ = [
    "setup_console_metrics",
    "instrument_rpc",
    "instrument_reader",
    "InstrumentedWriter",
]

logger = get_print_logger()
# placeholders for instruments; initialized in setup_console_metrics()
_request_counter = None
_duration_hist = None
_streams_counter = None
_bytes_counter = None
_rows_counter = None
_throughput_hist = None


class SimpleConsoleMetricExporter(MetricExporter):
    def export(self, metrics_data, timeout_millis: int = 0, **kwargs):
        for rm in getattr(metrics_data, "resource_metrics", []):
            for scope in getattr(rm, "scope_metrics", []):
                for metric in getattr(scope, "metrics", []):
                    name = metric.name
                    data = metric.data
                    for pt in data.data_points:
                        if hasattr(pt, "value"):
                            val = pt.value
                            attrs = pt.attributes or {}
                            suffix = (
                                "{"
                                + ",".join(f"{k}={v}" for k, v in attrs.items())
                                + "}"
                                if attrs
                                else ""
                            )
                            logger.info(f"{name}{suffix} {val}")
                        else:
                            try:
                                for q, v in pt.value.quantiles:
                                    logger.info(f"{name}{{quantile={q}}} {v}")
                            except Exception:
                                pass
        return MetricExportResult.SUCCESS

    def shutdown(self, *args, **kwargs) -> None:
        return None

    def force_flush(self, *args, **kwargs) -> bool:
        return True


@functools.lru_cache(1)
def setup_console_metrics(
    interval_ms: int = 2000,
    meter_name: str = "xorq.flight_server",
    prometheus_port: Optional[int] = None,
):
    exporter = SimpleConsoleMetricExporter()

    readers = [
        PeriodicExportingMetricReader(exporter, export_interval_millis=interval_ms)
    ]
    # optional Prometheus endpoint
    if prometheus_port is not None:
        try:
            import prometheus_client
            from opentelemetry.exporter.prometheus import PrometheusMetricReader
        except ImportError:
            logger.warning(
                "Prometheus support requires 'opentelemetry-exporter-prometheus' and 'prometheus-client'"
            )
        else:
            prom_reader = PrometheusMetricReader()
            readers.append(prom_reader)
            prometheus_client.start_http_server(prometheus_port)
            logger.info(f"Prometheus metrics available at :{prometheus_port}")
    # include resource attributes for backend grouping
    import socket

    from opentelemetry.sdk.resources import Resource

    resource = Resource.create(
        {
            "service.name": "xorq-flight",
            "service.instance.id": socket.gethostname(),
        }
    )
    provider = MeterProvider(resource=resource, metric_readers=readers)
    metrics.set_meter_provider(provider)
    meter = metrics.get_meter(meter_name)
    logger.info(f"console metrics enabled, interval={interval_ms} ms")
    global \
        _request_counter, \
        _duration_hist, \
        _streams_counter, \
        _bytes_counter, \
        _rows_counter, \
        _throughput_hist
    _request_counter = meter.create_counter(
        "flight_server.requests_total", description="Total Flight RPC requests"
    )
    _duration_hist = meter.create_histogram(
        "flight_server.request_duration_seconds", unit="s"
    )
    _streams_counter = meter.create_up_down_counter("flight_server.active_streams")
    _bytes_counter = meter.create_counter("flight_server.bytes_total", unit="By")
    _rows_counter = meter.create_counter("flight_server.rows_total")
    _throughput_hist = meter.create_histogram(
        "flight_server.throughput_rows_per_sec", unit="1/s"
    )
    return meter


class _Recorder:
    """Context-manager that tracks one RPC end-to-end."""

    def __init__(self, method: str):
        self.method = method
        self.started = time.time()

        if _request_counter is not None and _streams_counter is not None:
            _request_counter.add(1, {"method": method})
            _streams_counter.add(1)
        self._bytes = 0
        self._rows = 0

    def record_batch(self, batch: pa.RecordBatch, direction: str):
        if not hasattr(batch, "get_total_buffer_size") and hasattr(batch, "data"):
            rb = batch.data
        else:
            rb = batch
        size = rb.get_total_buffer_size()
        rows = rb.num_rows
        labels = {"method": self.method, "dir": direction}

        if _bytes_counter is not None and _streams_counter is not None:
            _bytes_counter.add(size, labels)
            _rows_counter.add(rows, labels)
        self._bytes += size
        self._rows += rows

    def finish(self):
        dur = time.time() - self.started

        if (
            _duration_hist is not None
            and _throughput_hist is not None
            and _streams_counter is not None
        ):
            _duration_hist.record(dur, {"method": self.method})
            if dur > 0 and self._rows:
                thr = self._rows / dur
                _throughput_hist.record(thr, {"method": self.method})
            _streams_counter.add(-1)


class InstrumentedReader:
    def __init__(self, reader, rec: _Recorder, direction: str = "out"):
        self.reader = reader
        self.rec = rec
        self.direction = direction

    @property
    def schema(self):
        return self.reader.schema

    def __iter__(self):
        for batch in self.reader:
            if not hasattr(batch, "get_total_buffer_size") and hasattr(batch, "data"):
                rb = batch.data
            else:
                rb = batch
            self.rec.record_batch(rb, self.direction)
            yield batch


def instrument_reader(reader: pa.RecordBatchReader, rec: _Recorder, *, direction="out"):
    def gen():
        for batch in reader:
            if not hasattr(batch, "get_total_buffer_size") and hasattr(batch, "data"):
                rb = batch.data
            else:
                rb = batch
            rec.record_batch(rb, direction)
            yield rb

    return pa.RecordBatchReader.from_batches(reader.schema, gen())


class InstrumentedWriter:
    def __init__(self, raw_writer, rec: _Recorder, *, direction="out"):
        self._w = raw_writer
        self._rec = rec
        self._dir = direction

    def begin(self, *args, **kwargs):
        return self._w.begin(*args, **kwargs)

    def write_metadata(self, metadata):
        return self._w.write_metadata(metadata)

    def write_batch(self, batch):
        self._rec.record_batch(batch, self._dir)
        return self._w.write_batch(batch)

    def write_with_metadata(self, batch, metadata):
        if batch:
            self._rec.record_batch(batch, self._dir)
        return self._w.write_with_metadata(batch, metadata)


def instrument_rpc(method_name):
    def decorator(func):
        def wrapper(self, context, *args, **kwargs):
            rec = _Recorder(method_name)
            try:
                return func(self, context, rec, *args, **kwargs)
            finally:
                rec.finish()

        return wrapper

    return decorator
