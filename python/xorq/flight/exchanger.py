import functools
import urllib
from abc import (
    ABC,
    abstractmethod,
)

import pandas as pd
import pyarrow as pa
import requests


def schemas_equal(s0, s1):
    def schema_to_dct(s):
        return {name: s.field(name) for name in s.names}

    return schema_to_dct(s0) == schema_to_dct(s1)


def streaming_exchange(f, context, reader, writer, options=None, **kwargs):
    started = False
    for chunk in (chunk for chunk in reader if chunk.data):
        out = f(chunk.data, metadata=chunk.app_metadata)
        if not started:
            writer.begin(out.schema, options=options)
            started = True
        writer.write_batch(out)


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


exchangers = {
    exchanger.command: exchanger
    for exchanger in (
        EchoExchanger,
        RowSumExchanger,
        RowSumAppendExchanger,
        UrlOperatorExchanger,
    )
}
