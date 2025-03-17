import functools
import urllib
from abc import (
    ABC,
    abstractmethod,
)
from typing import Callable

import dask
import pandas as pd
import pyarrow as pa
import requests
import toolz

import xorq as xo
import xorq.vendor.ibis.expr.operations as ops
from xorq.common.utils.func_utils import (
    return_constant,
)
from xorq.common.utils.rbr_utils import (
    copy_rbr_batches,
    excepts_print_exc,
    make_filtered_reader,
)


def schemas_equal(s0, s1):
    def schema_to_dct(s):
        return {name: s.field(name) for name in s.names}

    return schema_to_dct(s0) == schema_to_dct(s1)


def replace_one_unbound(unbound_expr, table):
    (unbound, *rest) = unbound_expr.op().find(ops.UnboundTable)
    if rest:
        raise ValueError
    dt = table.op()
    if not isinstance(dt, ops.DatabaseTable):
        raise ValueError
    if not unbound.schema == dt.schema:
        raise ValueError

    def _replace_unbound(node, kwargs):
        if isinstance(node, ops.UnboundTable):
            return dt
        elif kwargs:
            return node.__recreate__(kwargs)
        else:
            return node

    return unbound_expr.op().replace(_replace_unbound).to_expr()


@excepts_print_exc
def streaming_exchange(f, context, reader, writer, options=None, **kwargs):
    started = False
    for chunk in (chunk for chunk in reader if chunk.data):
        out = f(chunk.data, metadata=chunk.app_metadata)
        if not started:
            writer.begin(out.schema, options=options)
            started = True
        writer.write_batch(out)


@excepts_print_exc
def streaming_expr_exchange(
    unbound_expr, make_connection, context, reader, writer, options=None, **kwargs
):
    filtered_reader = copy_rbr_batches(make_filtered_reader(reader))
    t = make_connection().read_record_batches(filtered_reader)
    bound_expr = replace_one_unbound(unbound_expr, t)
    started = False
    for batch in bound_expr.to_pyarrow_batches():
        if not started:
            writer.begin(batch.schema, options=options)
            started = True
        writer.write_batch(batch)


class AbstractExchanger(ABC):
    @classmethod
    @property
    @abstractmethod
    def exchange_f(cls):
        # return a function with the signature (context, reader, writer, **kwargs)
        def f(context, reader, writer, **kwargs):
            raise NotImplementedError

        return f

    @classmethod
    @property
    @abstractmethod
    def schema_in_required(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def schema_in_condition(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def calc_schema_out(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def description(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def command(cls):
        pass

    @classmethod
    @property
    def query_result(cls):
        return {
            "schema-in-required": cls.schema_in_required,
            "schema-in-condition": cls.schema_in_condition,
            "calc-schema-out": cls.calc_schema_out,
            "description": cls.description,
            "command": cls.command,
        }


class EchoExchanger(AbstractExchanger):
    @classmethod
    @property
    def exchange_f(cls):
        def exchange_echo(context, reader, writer, options=None, **kwargs):
            """Run a simple echo server."""
            started = False
            for chunk in reader:
                if not started and chunk.data:
                    writer.begin(chunk.data.schema, options=options)
                    started = True
                if chunk.app_metadata and chunk.data:
                    writer.write_with_metadata(chunk.data, chunk.app_metadata)
                elif chunk.app_metadata:
                    writer.write_metadata(chunk.app_metadata)
                elif chunk.data:
                    writer.write_batch(chunk.data)
                else:
                    assert False, "Should not happen"

        return exchange_echo

    @classmethod
    @property
    def schema_in_required(cls):
        return None

    @classmethod
    @property
    def schema_in_condition(cls):
        def condition(schema_in):
            return True

        return condition

    @classmethod
    @property
    def calc_schema_out(cls):
        def f(schema_in):
            return schema_in

        return f

    @classmethod
    @property
    def description(cls):
        return "echo's data back"

    @classmethod
    @property
    def command(cls):
        return "echo"


class RowSumAppendExchanger(AbstractExchanger):
    @classmethod
    @property
    def exchange_f(cls):
        def exchange_transform(context, reader, writer):
            """Sum rows in an uploaded table."""
            for field in reader.schema:
                if not pa.types.is_integer(field.type):
                    raise pa.ArrowInvalid("Invalid field: " + repr(field))
            table = reader.read_all()
            result = table.append_column(
                "sum",
                pa.array(table.to_pandas().sum(axis=1)),
            )
            writer.begin(result.schema)
            writer.write_table(result)

        return exchange_transform

    @classmethod
    @property
    def schema_in_required(cls):
        return None

    @classmethod
    @property
    def schema_in_condition(cls):
        def condition(schema_in):
            return all(pa.types.is_integer(t) for t in schema_in.types)

        return condition

    @classmethod
    @property
    def calc_schema_out(cls):
        def f(schema_in):
            return schema_in.append(pa.field("sum", pa.int64()))

        return f

    @classmethod
    @property
    def description(cls):
        return "sums all the values sent"

    @classmethod
    @property
    def command(cls):
        return "row-sum-append"


class RowSumExchanger(AbstractExchanger):
    @classmethod
    @property
    def exchange_f(cls):
        def exchange_transform(context, reader, writer):
            """Sum rows in an uploaded table."""
            for field in reader.schema:
                if not pa.types.is_integer(field.type):
                    raise pa.ArrowInvalid("Invalid field: " + repr(field))
            table = reader.read_all()
            sums = [0] * table.num_rows
            for column in table:
                for row, value in enumerate(column):
                    sums[row] += value.as_py()
            result = pa.Table.from_arrays([pa.array(sums)], names=["sum"])
            writer.begin(result.schema)
            writer.write_table(result)

        return exchange_transform

    @classmethod
    @property
    def schema_in_required(cls):
        return None

    @classmethod
    @property
    def schema_in_condition(cls):
        def condition(schema_in):
            return all(pa.types.is_integer(t) for t in schema_in.types)

        return condition

    @classmethod
    @property
    def calc_schema_out(cls):
        def f(schema_in):
            return pa.schema((pa.field("sum", pa.int64()),))

        return f

    @classmethod
    @property
    def description(cls):
        return "sums all the values sent"

    @classmethod
    @property
    def command(cls):
        return "row-sum"


class UrlOperatorExchanger(AbstractExchanger):
    url_field_name = "url"
    url_field_typ = pa.string()
    scheme_field_name = "scheme_url"
    scheme_field_typ = pa.string()
    length_field_name = "response-length"
    length_field_typ = pa.int64()
    schemes = ("http", "https")

    @classmethod
    @property
    def exchange_f(cls):
        def exchange_transform(context, reader, writer):
            """fetch the url and return the length of the response content"""
            if not cls.schema_in_condition(reader.schema):
                raise pa.ArrowInvalid("Input does not satisfy schema_in_condition")
            table = reader.read_all()

            def f(url):
                """Return a row for each scheme"""
                parsed = urllib.parse.urlparse(url)
                scheme_urls = tuple(
                    parsed._replace(scheme=scheme).geturl() for scheme in cls.schemes
                )
                lengths = tuple(
                    len(requests.get(scheme_url).content) for scheme_url in scheme_urls
                )
                df = pd.DataFrame(
                    {
                        cls.url_field_name: [url] * len(cls.schemes),
                        cls.scheme_field_name: scheme_urls,
                        cls.length_field_name: lengths,
                    }
                )
                table = pa.Table.from_pandas(df)
                return table

            result = pa.concat_tables(
                map(
                    f,
                    table.column(cls.url_field_name).to_pylist(),
                )
            ).combine_chunks()
            writer.begin(result.schema)
            writer.write_table(result)

        return exchange_transform

    @classmethod
    @property
    def schema_in_required(cls):
        return pa.schema((pa.field(cls.url_field_name, cls.url_field_typ),))

    @classmethod
    @property
    def schema_in_condition(cls):
        def condition(schema_in):
            return all(field in schema_in for field in cls.schema_in_required)

        return condition

    @classmethod
    @property
    def calc_schema_out(cls):
        def f(schema_in):
            return pa.schema(
                tuple(field for field in schema_in)
                + (
                    pa.field(cls.scheme_field_name, cls.scheme_field_typ),
                    pa.field(cls.length_field_name, cls.length_field_typ),
                )
            )

        return f

    @classmethod
    @property
    def description(cls):
        return (
            f"fetches the content from field `{cls.url_field_name}` ({cls.url_field_typ})"
            f"\nfor each scheme in {cls.schemes}"
            f"\nCalculates its length and puts it in field `{cls.length_field_name}` ({cls.length_field_typ})"
        )

    @classmethod
    @property
    def command(cls):
        return "url-response-length"


class PandasUDFExchanger(AbstractExchanger):
    def __init__(self, f, schema_in, name, typ, append=True):
        self.f = f
        self.schema_in = schema_in
        self.name = name
        self.typ = typ
        self.append = append

    @property
    def exchange_f(self):
        def f(batch, metadata=None, **kwargs):
            df = batch.to_pandas()
            series = self.f(df).rename(self.name)
            if self.append:
                out = df.assign(**{series.name: series})
            else:
                out = series.to_frame()
            return pa.RecordBatch.from_pandas(out)

        return functools.partial(streaming_exchange, f)

    @property
    def schema_in_required(self):
        return self.schema_in

    @property
    def schema_in_condition(self):
        def condition(schema_in):
            return all(el in schema_in for el in self.schema_in_required)

        return condition

    @property
    def calc_schema_out(self):
        def f(schema_in):
            # FIXME: what to send if schema_in does not match schema_in_required?
            field = pa.field(self.name, self.typ)
            if self.append:
                schema_out = pa.schema(tuple(schema_in) + (field,))
            else:
                schema_out = pa.schema((field,))
            return schema_out

        return f

    @property
    def description(self):
        return f"a custom udf for {self.f.__name__}"

    @property
    def command(self):
        return f"custom-udf-{self.f.__name__}"

    @property
    def query_result(self):
        return {
            "schema-in-required": self.schema_in_required,
            "schema-in-condition": self.schema_in_condition,
            "calc-schema-out": self.calc_schema_out,
            "description": self.description,
            "command": self.command,
        }


class UnboundExprExchanger(AbstractExchanger):
    def __init__(self, unbound_expr, make_connection=xo.connect):
        if unbound_expr.op().find(ops.DatabaseTable):
            raise ValueError("unbound_expr must be unbound")
        self.unbound_expr = self.set_one_unbound_name(unbound_expr)
        self.make_connection = make_connection
        self._schema_in_required = self.get_one_unbound(
            self.unbound_expr
        ).schema.to_pyarrow()
        self._schema_in_condition = toolz.curried.operator.eq(self._schema_in_required)
        self._schema_out = self.unbound_expr.schema().to_pyarrow()

    @staticmethod
    def get_one_unbound(expr):
        (unbound, *rest) = expr.op().find(ops.UnboundTable)
        if rest:
            raise ValueError("expr must only have one unbound table")
        return unbound

    @staticmethod
    def set_one_unbound_name(expr, name="fixed-name"):
        def set_name(op, kwargs):
            if isinstance(op, ops.UnboundTable):
                op = op.copy(name=name)
            if kwargs:
                op = op.__recreate__(kwargs)
            return op

        return expr.op().replace(set_name).to_expr()

    @property
    def op_hash(self):
        return dask.base.tokenize(self.unbound_expr)

    @property
    def exchange_f(self):
        return functools.partial(
            streaming_expr_exchange, self.unbound_expr, self.make_connection
        )

    @property
    def schema_in_required(self):
        return self._schema_in_required

    def schema_in_condition(self, schema_in):
        return self._schema_in_condition(schema_in)

    def calc_schema_out(self, schema_in):
        return self._schema_out

    @classmethod
    @property
    def description(cls):
        return "run the given unbound expr on the rbr"

    @property
    def command(self):
        return f"execute-unbound-expr-{self.op_hash}"

    @property
    def query_result(self):
        return {
            "schema-in-required": self._schema_in_required,
            "schema-in-condition": self._schema_in_condition,
            "calc-schema-out": return_constant(self._schema_out),
            "description": self.description,
            "command": self.command,
        }


def make_udxf(
    process_df,
    maybe_schema_in,
    maybe_schema_out,
    name=None,
    description=None,
    command=None,
    do_wraps=True,
):
    def process_batch(process_df, batch, metadata=None, **kwargs):
        df = batch.to_pandas()
        out = process_df(df)
        return pa.RecordBatch.from_pandas(out)

    if do_wraps:
        exchange_f = excepts_print_exc(
            functools.partial(
                streaming_exchange, functools.partial(process_batch, process_df)
            ),
            Exception,
        )
    else:
        exchange_f = process_df

    if isinstance(maybe_schema_in, pa.Schema):
        schema_in_required = maybe_schema_in
        schema_in_condition = toolz.curried.operator.eq(maybe_schema_in)
    elif isinstance(maybe_schema_in, Callable):
        schema_in_required = None
        schema_in_condition = maybe_schema_in
    else:
        raise ValueError

    if isinstance(maybe_schema_out, pa.Schema):
        calc_schema_out = return_constant(maybe_schema_out)
    elif isinstance(maybe_schema_out, Callable):
        calc_schema_out = maybe_schema_out
    else:
        raise ValueError

    name = name or process_df.__name__
    typ = type(
        name,
        (AbstractExchanger,),
        {
            "exchange_f": exchange_f,
            "schema_in_required": schema_in_required,
            "schema_in_condition": schema_in_condition,
            "calc_schema_out": calc_schema_out,
            "description": description or name,
            "command": command or dask.base.tokenize(process_df),
        },
    )
    return typ


exchangers = {
    exchanger.command: exchanger
    for exchanger in (
        EchoExchanger,
        RowSumExchanger,
        RowSumAppendExchanger,
        UrlOperatorExchanger,
    )
}
