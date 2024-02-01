from abc import ABC, abstractmethod

import connectorx as cx
import sqlglot.expressions as exp
from sqlglot import parse_one, TokenType, select, and_
from sqlglot import parser, planner
from sqlglot.dialects import DuckDB, Postgres
from sqlglot.dialects.dialect import rename_func

import letsql
from letsql.db.optimizer import optimize
from letsql.db.planner import LetSQLPlan
from letsql.db.predict import LetSQLPredict


class Context:
    def __init__(self, tables):
        self.tables = tables

    def __contains__(self, table: str) -> bool:
        return table in self.tables


class Table(ABC):
    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def schema(self):
        pass

    @property
    @abstractmethod
    def target(self):
        pass


class CSVTable(Table):
    @property
    def target(self):
        return "LetSQL (local)"

    @property
    def name(self):
        return self._name

    @property
    def schema(self):
        return None

    def __init__(self, path, name):
        self.path = path
        self._name = name


class PostgreSQLTable(Table):
    @property
    def target(self):
        return "PostgreSQL (remote)"

    @property
    def name(self):
        return self._name

    @property
    def schema(self):
        if self._schema is None:
            arrow = cx.read_sql(
                self.connection,
                f"select * from {self.name} limit 1",
                return_type="arrow2",
            )
            self._schema = dict(zip(arrow.schema.names, map(str, arrow.schema.types)))
        return self._schema

    def __init__(self, connection, name):
        self.connection = connection
        self._name = name
        self._schema = None


class ArrowTable:
    def __init__(self, data):
        self.data = data

    @property
    def columns(self):
        return self.data.schema.names

    def fetchall(self):
        return self.data

    def to_arrow(self):
        return self.data.to_arrow_table()


class PythonExecutor:
    def __init__(self, tables=None):
        self.generator = LetSQL().generator(identify=True, comments=False)
        self.tables = tables or {}

    def execute(self, plan):
        finished = set()
        queue = set(plan.leaves)
        contexts = {}

        while queue:
            node = queue.pop()
            try:
                context = Context(
                    {
                        name: table
                        for dep in node.dependencies
                        for name, table in contexts[dep].tables.items()
                    }
                )

                if isinstance(node, planner.Scan):
                    contexts[node] = self.scan(node, context)
                elif isinstance(node, planner.Join):
                    contexts[node] = self.join(node)
                else:
                    raise NotImplementedError

                finished.add(node)

                for dep in node.dependents:
                    if all(d in contexts for d in dep.dependencies):
                        queue.add(dep)

                for dep in node.dependencies:
                    if all(d in finished for d in dep.dependents):
                        contexts.pop(dep)
            except Exception as e:
                raise ValueError(f"Step '{node.id}' failed: {e}") from e

        root = plan.root
        return contexts[root].tables[root.name]

    def scan(self, step, context):
        source = step.source

        if source and isinstance(source, exp.Expression):
            source = source.name or source.alias

        if source in context:
            if not step.projections and not step.condition:
                return Context({step.name: context.tables[source]})
            table_iter = ArrowTable(context.tables[source])
        else:
            context, table_iter = self.scan_table(step)

        return Context({step.name: self._project_and_filter(step, table_iter)})

    def _project_and_filter(self, step, table):
        condition = self.generate(step.condition)
        projections = self.generate_tuple(step.projections) or (exp.Star(),)

        expression = select(*projections).from_(step.source)

        if condition:
            expression = expression.where(condition)

        query = self.generate(expression)
        table_name = step.source.name

        if isinstance(table, CSVTable):
            letsql.register(table.path, table_name=table_name)
            return letsql.ba.con.sql(query)
        elif isinstance(table, PostgreSQLTable):
            source = cx.read_sql(
                table.connection, expression.sql(dialect=Postgres), return_type="arrow2"
            )
            letsql.register(source, table_name=table_name)
            return letsql.ba.con.sql(f"select * from {table_name};")
        elif isinstance(table, ArrowTable):
            letsql.register(table.to_arrow(), table_name=table_name)
            return letsql.ba.con.sql(query)
        else:
            raise ValueError

    def scan_table(self, step):
        table = self.tables[step.source.name]
        context = Context({step.source.alias_or_name: table})
        return context, table

    def join(self, step):
        source = step.name

        expression = select(*step.projections).from_(source)
        for name, join in step.joins.items():
            on = and_(
                *map(
                    lambda x, y: exp.EQ(this=x, expression=y),
                    join["source_key"],
                    join["join_key"],
                )
            )
            expression = expression.join(name, on=on)
            if join["condition"]:
                expression = expression.where(join["condition"])

        query = self.generator.generate(expression)
        return Context({source: ArrowTable(letsql.ba.con.sql(query))})

    def generate(self, expression):
        """Convert an SQL expression into literal DuckDB expressions."""
        if not expression:
            return None

        sql = self.generator.generate(expression)
        return sql

    def generate_tuple(self, expressions):
        """Convert an array of SQL expressions into tuple of DuckDB expressions."""
        if not expressions:
            return tuple()
        return tuple(self.generate(expression) for expression in expressions)


class Connection:
    def __init__(self):
        self.tables = {}

    def register_table(self, name: str, table: Table):
        self.tables[name] = table


class LetSQL(DuckDB):
    class Parser(DuckDB.Parser):
        FUNCTION_PARSERS = {
            **parser.Parser.FUNCTION_PARSERS,
            "PREDICT": lambda self: self._parse_predict(),
        }

        def _parse_predict(self) -> LetSQLPredict:
            this = self._parse_string_as_identifier()

            self._match(TokenType.COMMA)

            projections = self._parse_projections()
            return self.expression(LetSQLPredict, this=this, expressions=projections)

    class Generator(DuckDB.Generator):
        TRANSFORMS = {
            **DuckDB.Generator.TRANSFORMS,
            LetSQLPredict: rename_func("PREDICT_XGB"),
        }


class FutureExecution:
    def __init__(self, plan: LetSQLPlan, con: Connection):
        self.plan = plan
        self.con = con

    def execute(self):
        return PythonExecutor(tables=self.con.tables).execute(self.plan).to_pandas()

    def explain(self):
        print(self.plan)


con = Connection()


def sql(query):
    node = parse_one(query, dialect=LetSQL)
    node = optimize(
        node,
        schema={table.name: table.schema for table in con.tables.values()},
        dialect=LetSQL,
        constant_propagation=True,
        leave_tables_isolated=True,
        sources=con.tables,
    )
    plan = LetSQLPlan(node, tables=con.tables)
    return FutureExecution(plan, con)

    # return PythonExecutor(tables=con.tables).execute(plan)


def register_postgres(name, connection):
    con.register_table(name, PostgreSQLTable(connection, name))


def register_csv(name, path):
    con.register_table(name, CSVTable(path, name))


__all__ = ["sql", "register_postgres", "register_csv"]
