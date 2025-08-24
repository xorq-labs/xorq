import datetime
import functools
import json
import operator
import time
from pathlib import Path
from queue import (
    Empty as QueueEmpty,
)
from queue import (
    Queue,
)
from threading import (
    Thread,
)

import toolz
from attr import (
    field,
    frozen,
)
from attr.validators import (
    deep_iterable,
    instance_of,
    is_callable,
    optional,
)
from toolz import (
    compose,
    identity,
)
from toolz.curried import (
    map as cmap,
)

from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.common.utils.otel_utils import (
    otel_config,
)


default_log_path = (
    Path(otel_config.OTEL_HOST_LOG_DIR, otel_config.OTEL_LOG_FILE_NAME)
    .expanduser()
    .resolve()
)

ANY = type(
    "ALL", (), {"__eq__": return_constant(True), "__contains__": return_constant(True)}
)()
NONE = type(
    "NONE",
    (),
    {"__eq__": return_constant(False), "__contains__": return_constant(False)},
)()


def dissoc_get(dct, *keys):
    values = tuple(dct.get(key) for key in keys)
    rest = toolz.dissoc(dct, *keys)
    return (values, rest)


datetime_from_unix_nano_str = compose(
    datetime.datetime.fromtimestamp,
    toolz.curried.flip(operator.truediv)(1e9),
    int,
)


def dissoc_get_all(dct, *keys):
    (values, rest) = dissoc_get(dct, *keys)
    assert not rest
    return values


def process_dict(dct, from_to_f):
    values = dissoc_get_all(dct, *from_to_f)
    processed = {
        to_: f(value) for ((_, (to_, f)), value) in zip(from_to_f.items(), values)
    }
    return processed


required_attribute = {
    "key": "service.name",
    "value": {
        "stringValue": "xorq-observability-service",
    },
}
required_scope = {"name": "xorq.tracer"}


def process_value(dct):
    ((value_type, value),) = dct.items()
    match value_type:
        case "intValue":
            value = int(value)
        case "stringValue":
            assert isinstance(value, str)
        case "arrayValue":
            if value:
                (lst,) = dissoc_get_all(value, "values")
                value = tuple(map(process_value, lst))
            else:
                value = tuple()
        case _:
            raise ValueError(f"Unhandled type {value_type}")
    return (value_type, value)


@frozen
class Link:
    trace_id = field(validator=instance_of(str))
    span_id = field(validator=instance_of(str))
    flags = field(validator=instance_of(int))

    @classmethod
    def from_dict(cls, dct):
        from_to_f = {
            "traceId": ("trace_id", identity),
            "spanId": ("span_id", identity),
            "flags": ("flags", identity),
        }
        kwargs = process_dict(dct, from_to_f)
        return cls(**kwargs)


@frozen
class Attribute:
    name = field(validator=instance_of(str))
    value_type = field(validator=instance_of(str))
    value = field()

    @classmethod
    def from_dict(cls, dct):
        (name, value) = dissoc_get_all(dct, "key", "value")
        value_type, value = process_value(value)
        return cls(name, value_type, value)


@frozen
class Event:
    time = field(validator=instance_of(datetime.datetime))
    name = field(validator=instance_of(str))
    attributes = field(
        validator=deep_iterable(instance_of(Attribute), instance_of(tuple))
    )

    @classmethod
    def from_dict(cls, dct):
        from_to_f = {
            "timeUnixNano": ("time", datetime_from_unix_nano_str),
            "name": ("name", identity),
            "attributes": ("attributes", compose(tuple, cmap(Attribute.from_dict))),
        }
        return cls(**process_dict(dct, from_to_f))


@frozen
class Span:
    trace_id = field(validator=instance_of(str))
    span_id = field(validator=instance_of(str))
    parent_span_id = field(validator=instance_of(str))
    name = field(validator=instance_of(str))
    flags = field(validator=instance_of(int))
    kind = field(validator=instance_of(int))
    links = field(validator=deep_iterable(instance_of(Link), instance_of(tuple)))
    start_datetime = field(validator=instance_of(datetime.datetime))
    end_datetime = field(validator=instance_of(datetime.datetime))
    events = field(
        validator=optional(deep_iterable(instance_of(Event), instance_of(tuple)))
    )
    attributes = field(
        validator=optional(deep_iterable(instance_of(Attribute), instance_of(tuple)))
    )
    status = field()

    @property
    def duration(self):
        return (self.end_datetime - self.start_datetime).total_seconds()

    @property
    def cache_event_dct(self):
        if self.name == "cache.set_default":
            (event,) = self.events
            (attribute,) = event.attributes
            return {
                "duration": self.duration,
                "name": event.name,
                "key": attribute.value,
            }
        else:
            return None

    @property
    def is_cache_hit(self):
        return self.name == "cache.set_default" and any(
            event.name == "cache.hit" for event in self.events
        )

    @property
    def is_cache_miss(self):
        return self.name == "cache.set_default" and any(
            event.name == "cache.miss" for event in self.events
        )

    def get_depth(self, trace):
        lineage = trace.get_lineage(self.span_id)
        return len(lineage) - 1

    @classmethod
    def from_dict(cls, dct):
        # FIXME: ensure we get all the fields
        kwargs = {
            to_: dct[from_]
            for to_, from_ in (
                ("trace_id", "traceId"),
                ("span_id", "spanId"),
                ("parent_span_id", "parentSpanId"),
                ("name", "name"),
                ("flags", "flags"),
                ("kind", "kind"),
            )
        } | {
            to_: f(dct.get(from_, default))
            for (to_, (from_, default, f)) in (
                (
                    "status",
                    ("status", None, compose(tuple, operator.methodcaller("items"))),
                ),
                (
                    "links",
                    (
                        "links",
                        (),
                        compose(
                            tuple,
                            cmap(Link.from_dict),
                        ),
                    ),
                ),
                (
                    "events",
                    (
                        "events",
                        (),
                        compose(
                            tuple,
                            cmap(Event.from_dict),
                        ),
                    ),
                ),
                (
                    "attributes",
                    (
                        "attributes",
                        (),
                        compose(
                            tuple,
                            cmap(Attribute.from_dict),
                        ),
                    ),
                ),
                (
                    "start_datetime",
                    ("startTimeUnixNano", None, datetime_from_unix_nano_str),
                ),
                (
                    "end_datetime",
                    ("endTimeUnixNano", None, datetime_from_unix_nano_str),
                ),
            )
        }
        return cls(**kwargs)

    @classmethod
    def spans_from_line(cls, line):
        assert line
        dct = json.loads(line)
        (resource_spans,) = dissoc_get_all(dct, "resourceSpans")
        resources, scope_spanss = zip(
            *(
                dissoc_get_all(resource_span, "resource", "scopeSpans")
                for resource_span in resource_spans
            )
        )
        assert all(
            dissoc_get_all(resource, "attributes") == ([required_attribute],)
            for resource in resources
        )
        scope_spans = tuple(scope_span for (scope_span,) in scope_spanss)
        (scopes, spanss) = zip(
            *(
                dissoc_get_all(scope_span, "scope", "spans")
                for scope_span in scope_spans
            )
        )
        assert all(scope == required_scope for scope in scopes)
        return tuple(Span.from_dict(dct) for spans in spanss for dct in spans)


@frozen
class Trace:
    spans = field(validator=deep_iterable(instance_of(Span), instance_of(tuple)))

    def __attrs_post_init__(self):
        end_datetimes = list(span.end_datetime for span in self.spans)
        if not sorted(end_datetimes) == end_datetimes:
            raise ValueError

    def combine_with(self, other):
        return Trace(
            tuple(
                sorted(
                    self.spans + other.spans,
                    key=operator.attrgetter("end_datetime"),
                )
            )
        )

    @property
    def closed(self):
        trace_ids = set(span.trace_id for span in self.spans)
        closed_trace_ids = set(
            span.trace_id for span in self.spans if not span.parent_span_id
        )
        return trace_ids == closed_trace_ids and bool(
            toolz.excepts(Exception, operator.attrgetter("trace_metrics"))(self)
        )

    @property
    @functools.cache
    def trace_id(self):
        dct = toolz.groupby(
            compose(bool, operator.attrgetter("links")),
            self.spans,
        )
        with_links = dct.get(True, ())
        without_links = dct.get(False, ())
        if without_links:
            trace_id, *rest = set(span.trace_id for span in without_links)
            assert not rest
            if with_links:
                # FIXME: handle links within links
                for span in with_links:
                    assert any(link.trace_id == trace_id for link in span.links)
            return trace_id
        elif with_links:
            # for now, require that there only be one linked traceId
            trace_id, *rest = set(
                link.trace_id for span in with_links for link in span.links
            )
            assert not rest
            return trace_id
        else:
            raise ValueError

    @property
    @functools.cache
    def parent_span(self):
        (parent_span, *rest) = (
            span for span in self.spans if not span.parent_span_id and not span.links
        )
        assert not rest
        return parent_span

    @staticmethod
    def calc_duration(span):
        return (span.end_datetime - span.start_datetime).total_seconds()

    @property
    def cache_event_dcts(self):
        return tuple(filter(None, (span.cache_event_dct for span in self.spans)))

    def get_lineage(self, span_id):
        dct = {span.span_id: span for span in self.spans}
        lineage = ()
        while span_id:
            span = dct[span_id]
            span_id = span.parent_span_id
            lineage += (span,)
        return lineage

    def get_depth(self, depth):
        return self.get_depths().get(depth, ())

    def get_duration_delta(self, parent_span_id, lossless_leafs=True):
        parent_span = next(
            span for span in self.spans if span.span_id == parent_span_id
        )
        parent_depth = parent_span.get_depth(self)
        child_spans = tuple(
            span
            for span in self.get_depth(parent_depth + 1)
            if span.parent_span_id == parent_span_id
            # metrics_recorder spans can come up in non-parent spans because true parent went out of context
            and span.name != "otel_instrument_reader.metrics_recorder"
            # and span.name != "to_pyarrow_batches"
        )
        if child_spans or not lossless_leafs:
            return parent_span.duration - sum(
                child_span.duration for child_span in child_spans
            )
        else:
            return 0

    @functools.cache
    def get_depths(self):
        spans = tuple(span for span in self.spans if span != self.parent_span)
        depths = {
            0: (self.parent_span,),
        }
        depth = 1
        while spans:
            parent_span_ids = set(span.span_id for span in depths[depth - 1])
            dct = toolz.groupby(
                lambda span: span.parent_span_id in parent_span_ids,
                spans,
            )
            at_depth = dct.get(True, ())
            if not at_depth:
                if rest := dct.get(False):
                    depths[-1] = rest
                break
            else:
                depths[depth] = at_depth
            spans = dct.get(False, ())
            depth += 1
        return depths

    @property
    def duration(self):
        return self.calc_duration(self.parent_span)

    @property
    def start_datetime(self):
        return self.parent_span.start_datetime

    def get_spans_named(self, name):
        return tuple(span for span in self.spans if span.name == name)

    @property
    def trace_metrics(self):
        return TraceMetrics(self)

    @classmethod
    def accrue_spans(cls, spans, partials=()):
        def combine_traces(traces):
            def combine(traces):
                (trace, *others) = traces
                for other in others:
                    trace = trace.combine_with(other)
                return trace

            dct = toolz.valmap(
                combine,
                toolz.groupby(operator.attrgetter("trace_id"), traces),
            )
            return tuple(dct.values())

        def split_by_closed(traces):
            dct = toolz.groupby(operator.attrgetter("closed"), traces)
            closed = tuple(dct.get(True, ()))
            not_closed = tuple(dct.get(False, ()))
            return (closed, not_closed)

        maybe_partials = (
            tuple(
                toolz.valmap(
                    compose(Trace, tuple),
                    toolz.groupby(
                        operator.attrgetter("trace_id"),
                        spans,
                    ),
                ).values()
            )
            + partials
        )
        (traces, partials) = split_by_closed(combine_traces(maybe_partials))
        return (traces, partials)

    @classmethod
    def process_line(cls, line, partials=()):
        spans = Span.spans_from_line(line.strip())
        return cls.accrue_spans(spans, partials)

    @classmethod
    def process_lines(cls, lines, partials=()):
        spans = (span for line in lines for span in Span.spans_from_line(line))
        return cls.accrue_spans(spans, partials)

    @classmethod
    def process_path(cls, path=default_log_path):
        lines = path.read_text().strip().split("\n")
        (traces, partials) = cls.process_lines(lines)
        return (traces, partials)


@frozen
class TraceMetric:
    name = field(validator=instance_of(str))
    f = field(validator=is_callable())

    def calc_metric(self, trace):
        return self.f(trace)

    @classmethod
    def from_name_key_operator(cls, name, key, operator=sum):
        def f(trace):
            value = operator(
                attribute.value
                for el in trace.get_spans_named(name)
                for event in el.events
                for attribute in event.attributes
                if attribute.name == key
            )
            return value

        return cls(key, f)


def get_cache_metric(trace):
    # FIXME: deal with exception: has multiple attributes
    def process_span(span):
        assert span.name == "cache.set_default"
        (event,) = span.events
        (attribute,) = event.attributes
        return (event.name, attribute.value)

    return tuple(
        process_span(span) for span in trace.spans if span.name == "cache.set_default"
    )


n_batches_metric = TraceMetric.from_name_key_operator(
    "otel_instrument_reader.metrics_recorder", "n_batches"
)
sum_buffer_size_metric = TraceMetric.from_name_key_operator(
    "otel_instrument_reader.metrics_recorder", "sum_buffer_size"
)
cache_hit_metric = TraceMetric("trace_metrics", get_cache_metric)


@frozen
class TraceMetrics:
    trace = field(validator=instance_of(Trace))
    trace_metrics = field(
        validator=deep_iterable(instance_of(TraceMetric), instance_of(tuple)),
        default=(n_batches_metric, sum_buffer_size_metric, cache_hit_metric),
    )

    def __attrs_post_init__(self):
        self.check_validity()

    def check_validity(self):
        (oir, mr) = (
            self.trace.get_spans_named(name)
            for name in (
                "otel_instrument_reader",
                "otel_instrument_reader.metrics_recorder",
            )
        )
        assert len(oir) == len(mr)
        oir_ids, mr_ids = (
            set(
                attribute.value
                for el in which
                for event in el.events
                for attribute in event.attributes
                if attribute.name == "reader_id"
            )
            for which in (oir, mr)
        )
        assert oir_ids == mr_ids

    @property
    @functools.cache
    def default_metrics(self):
        return {
            "trace_id": self.trace.trace_id,
            "duration": self.trace.duration,
            "start_datetime": self.trace.start_datetime,
        }

    @property
    @functools.cache
    def metrics(self):
        dct = {
            metric.name: metric.calc_metric(self.trace) for metric in self.trace_metrics
        } | self.default_metrics
        return dct


@frozen
class FileTailer:
    path = field(validator=instance_of(Path))
    sleep_duration = field(validator=instance_of(int), default=1)
    _fh = field(init=False)

    def __attrs_post_init__(self):
        if not self.path.exists():
            raise ValueError
        fh = self.path.open("r")
        object.__setattr__(self, "_fh", fh)

    def get_line(self):
        content = self._fh.readline()
        while not content or not content[-1] == "\n":
            time.sleep(self.sleep_duration)
            content += self._fh.readline()
        return content

    def __iter__(self):
        return self

    def __next__(self):
        return self.get_line()


@frozen
class TraceFileProcessor:
    path = field(
        validator=instance_of(Path),
        default=default_log_path,
    )
    file_tailer = field(validator=instance_of(FileTailer), init=False)
    f = field(validator=is_callable(), default=toolz.identity)
    queue = field(validator=instance_of(Queue), init=False, factory=Queue)
    thread = field(validator=instance_of(Thread), init=False)

    def __attrs_post_init__(self):
        object.__setattr__(self, "file_tailer", FileTailer(self.path))
        object.__setattr__(self, "thread", Thread(target=self._run, daemon=True))
        self.thread.start()

    def _run(self):
        partials = ()
        for line in self.file_tailer:
            (traces, partials) = Trace.process_line(line, partials=partials)
            if traces is not None:
                for trace in traces:
                    self.queue.put(self.f(trace))

    def gen_existing(self):
        while True:
            try:
                yield self.queue.get(block=False)
            except QueueEmpty:
                break


def main():
    TraceFileProcessor(f=print)


if __name__ == "__main__":
    import sys

    sys.exit(main())
