from __future__ import annotations

import importlib


__all__ = [
    "BigQueryCompiler",
    "ClickHouseCompiler",
    "DatabricksCompiler",
    "DataFusionCompiler",
    "DruidCompiler",
    "DuckDBCompiler",
    "ExasolCompiler",
    "FlinkCompiler",
    "ImpalaCompiler",
    "MSSQLCompiler",
    "MySQLCompiler",
    "OracleCompiler",
    "PostgresCompiler",
    "PySparkCompiler",
    "RisingWaveCompiler",
    "SnowflakeCompiler",
    "SQLiteCompiler",
    "TrinoCompiler",
]

_CLASS_TO_MODULE = {
    "BigQueryCompiler": "bigquery",
    "ClickHouseCompiler": "clickhouse",
    "DatabricksCompiler": "databricks",
    "DataFusionCompiler": "datafusion",
    "DruidCompiler": "druid",
    "DuckDBCompiler": "duckdb",
    "ExasolCompiler": "exasol",
    "FlinkCompiler": "flink",
    "ImpalaCompiler": "impala",
    "MSSQLCompiler": "mssql",
    "MySQLCompiler": "mysql",
    "OracleCompiler": "oracle",
    "PostgresCompiler": "postgres",
    "PySparkCompiler": "pyspark",
    "RisingWaveCompiler": "risingwave",
    "SnowflakeCompiler": "snowflake",
    "SQLiteCompiler": "sqlite",
    "TrinoCompiler": "trino",
}

_SUBMODULES = frozenset(_CLASS_TO_MODULE.values())


def __getattr__(name):
    if name in _CLASS_TO_MODULE:
        module = importlib.import_module(f".{_CLASS_TO_MODULE[name]}", __name__)
        cls = getattr(module, name)
        globals()[name] = cls
        return cls
    if name in _SUBMODULES:
        module = importlib.import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return __all__ + sorted(_SUBMODULES)
