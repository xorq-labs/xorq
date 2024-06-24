# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2024-06-24
### Details
#### Added
- Add docker start to ci-test by @mesejo
- Poetry: add poetry checks to .pre-commit-config.yaml by @dlovell
- Add source cache by default by @mesejo
- Test_cache: add test_parquet_cache_storage by @dlovell
- Add rust files by @dlovell
- Add new cases to DataFusionBackend.register by @dlovell
- Add client tests for new register types by @dlovell
- Add faster function for CachedNode removal by @mesejo
- Add optimizations for predict_xgb in datafusion by @mesejo in [#16](https://github.com/letsql/letsql/pull/16)
- Lint: add args to poetry pre-commit invocation by @dlovell in [#20](https://github.com/letsql/letsql/pull/20)
- Add TableProvider for ibis Table by @mesejo in [#21](https://github.com/letsql/letsql/pull/21)
- Add filter pushdown for ibis.Table TableProvider by @mesejo in [#24](https://github.com/letsql/letsql/pull/24)
- Add .sql implementation by @mesejo in [#28](https://github.com/letsql/letsql/pull/28)
- Add automatic testing for examples dir by @mesejo in [#45](https://github.com/letsql/letsql/pull/45)
- Add docs by @mesejo in [#51](https://github.com/letsql/letsql/pull/51)
- Add better snowflake caching by @dlovell in [#49](https://github.com/letsql/letsql/pull/49)
- Add docs-preview workflow by @mesejo in [#54](https://github.com/letsql/letsql/pull/54)
- Add missing extras to poetry install in docs workflow by @mesejo in [#58](https://github.com/letsql/letsql/pull/58)
- Add start of services to workflow by @mesejo in [#59](https://github.com/letsql/letsql/pull/59)
- Add docs deploy workflow by @mesejo in [#55](https://github.com/letsql/letsql/pull/55)
- Add array functions by @mesejo in [#60](https://github.com/letsql/letsql/pull/60)
- Add registering of arbitrary expressions by @mesejo in [#64](https://github.com/letsql/letsql/pull/64)
- Add generic functions by @mesejo in [#66](https://github.com/letsql/letsql/pull/66)
- Add hashing of duckdb parquet files by @mesejo in [#67](https://github.com/letsql/letsql/pull/67)
- Add numeric functions by @mesejo in [#80](https://github.com/letsql/letsql/pull/80)
- Add `ls` accessor for Expr by @dlovell in [#81](https://github.com/letsql/letsql/pull/81)
- Add greatest and least functions by @mesejo in [#98](https://github.com/letsql/letsql/pull/98)
- Add temporal functions by @mesejo in [#99](https://github.com/letsql/letsql/pull/99)
- Add StructColumn and StructField ops by @mesejo in [#102](https://github.com/letsql/letsql/pull/102)

#### Changed
- Improve performance and ux of predict_xgb by @mesejo in [#8](https://github.com/letsql/letsql/pull/8)
- Fetch only the required features for the model by @mesejo in [#9](https://github.com/letsql/letsql/pull/9)
- Organize the letsql package by @mesejo
- Lint by @dlovell
- Define CacheStorage with deterministic hashing for keys by @mesejo
- Define KEY_PREFIX to identify letsql cache by @dlovell
- Conftest: define expected_tables, enforce test fixture table list by @dlovell
- Lint by @dlovell
- Update poetry.lock by @dlovell
- Enable registration of pyarrow.RecordBatchReader and ir.Expr by @mesejo in [#13](https://github.com/letsql/letsql/pull/13)
- Update CONTRIBUTING.md with instructions to run Postgres by @mesejo
- Register more dask normalize_token types by @dlovell in [#17](https://github.com/letsql/letsql/pull/17)
- Enable flake to work on both linux and macos by @dlovell in [#18](https://github.com/letsql/letsql/pull/18)
- Clean up development and ci/cd workflows by @mesejo in [#19](https://github.com/letsql/letsql/pull/19)
- Temporal readme by @mesejo
- Publish test coverage by @mesejo in [#31](https://github.com/letsql/letsql/pull/31)
- Update project files README, CHANGELOG and pyproject.toml by @mesejo in [#30](https://github.com/letsql/letsql/pull/30)
- Expose TableProvider trait in Python by @mesejo in [#29](https://github.com/letsql/letsql/pull/29)
- Clear warnings, bump up datafusion version to 37.1.0 by @mesejo in [#33](https://github.com/letsql/letsql/pull/33)
- Update ibis version by @mesejo in [#34](https://github.com/letsql/letsql/pull/34)
- Xgboost  is being deprecated by @hussainsultan in [#40](https://github.com/letsql/letsql/pull/40)
- Drop connection handling by @mesejo in [#36](https://github.com/letsql/letsql/pull/36)
- Refactor _register_and_transform_cache_tables by @mesejo in [#44](https://github.com/letsql/letsql/pull/44)
- Improve postgres table caching / cache invalidation by @dlovell in [#47](https://github.com/letsql/letsql/pull/47)
- Make engines optional extras by @dlovell in [#50](https://github.com/letsql/letsql/pull/50)
- SourceStorage: special case for cross-source caching by @dlovell in [#63](https://github.com/letsql/letsql/pull/63)
- Problem with multi-engine execution by @mesejo in [#70](https://github.com/letsql/letsql/pull/70)
- Clean test_execute and move tests from test_isolated_execution by @mesejo in [#79](https://github.com/letsql/letsql/pull/79)
- Move cache related tests to test_cache.py by @mesejo in [#88](https://github.com/letsql/letsql/pull/88)
- Give ParquetCacheStorage a default path by @dlovell in [#92](https://github.com/letsql/letsql/pull/92)
- Update to datafusion version 39.0.0 by @mesejo in [#97](https://github.com/letsql/letsql/pull/97)
- Make cache default path configurable by @mesejo in [#101](https://github.com/letsql/letsql/pull/101)

#### Fixed
- Filter bug solved by @mesejo
- Set stable ibis dependency by @mesejo
- Failing ci by @mesejo
- Pyproject: specify rev when using git ref, don't use git@github.com by @dlovell
- Pyproject: make pyarrow,datafusion core dependencies by @dlovell
- Run `poetry lock --no-update` by @dlovell
- Use _load_into_cache in _put by @mesejo
- _cached: special case for name == "datafusion" by @dlovell
- ParquetCacheStorage: properly create cache dir by @dlovell
- Local cache with parquet storage by @mesejo
- Fix mac build with missing source files by @hussainsultan
- Allow for multiple execution of letsql tables by @mesejo in [#41](https://github.com/letsql/letsql/pull/41)
- Fix import order using ruff by @mesejo in [#37](https://github.com/letsql/letsql/pull/37)
- Mismatched table names causing table not found error by @mesejo in [#43](https://github.com/letsql/letsql/pull/43)
- Ensure nonnull-ability of columns works by @dlovell in [#53](https://github.com/letsql/letsql/pull/53)
- Explicitly install poetry-plugin-export per warning message by @dlovell in [#61](https://github.com/letsql/letsql/pull/61)
- Update make_native_op to replace tables by @mesejo in [#75](https://github.com/letsql/letsql/pull/75)
- Normalize_memory_databasetable: incrementally tokenize RecordBatchs by @dlovell in [#73](https://github.com/letsql/letsql/pull/73)
- Cannot create table by @mesejo in [#74](https://github.com/letsql/letsql/pull/74)
- Handle case of table names during con.register by @mesejo in [#77](https://github.com/letsql/letsql/pull/77)
- Use sqlglot to generate escaped name string by @dlovell in [#85](https://github.com/letsql/letsql/pull/85)
- Register table on caching nodes by @mesejo in [#87](https://github.com/letsql/letsql/pull/87)
- Ensure snowflake tables have their Namespace bound on creation by @dlovell in [#91](https://github.com/letsql/letsql/pull/91)
- Change name of parameter in replace_table function by @mesejo in [#94](https://github.com/letsql/letsql/pull/94)
- Return native_dts, not sources by @dlovell in [#95](https://github.com/letsql/letsql/pull/95)
- Displace offsets in TimestampBucket by @mesejo in [#104](https://github.com/letsql/letsql/pull/104)

#### Removed
- Pyproject: remove redundant and conflicting dependency specifications by @dlovell
- Remove macos test suite by @mesejo
- Remove optimizer.py by @mesejo in [#14](https://github.com/letsql/letsql/pull/14)
- Remove redundant item setting _sources on registering the cache nodes by @mesejo in [#90](https://github.com/letsql/letsql/pull/90)

## [0.1.2.post] - 2024-02-01
### Details
#### Added
- Add missing dependencies by @mesejo

## [0.1.2] - 2024-02-01
### Details
#### Added
- Add CONTRIBUTING.md
- Address problems with schema
- Nix: add flake.nix and related files by @dlovell
- Add db package for showing predict udf working by @mesejo in [#1](https://github.com/letsql/letsql/pull/1)

#### Removed
- Remove xgboost as dependency by @mesejo

## [0.1.1] - 2023-11-09
### Details
#### Added
- Add register and client functions
- Add testing of api
- Add isnan/isinf and fix offset
- Add udf support
- Add new string ops, remove typo

#### Changed
- Test array, temporal, string and udf
- Start adding wrapper
- Prepare for release

