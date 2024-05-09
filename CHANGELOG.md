# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
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

#### Removed
- Pyproject: remove redundant and conflicting dependency specifications by @dlovell
- Remove macos test suite by @mesejo
- Remove optimizer.py by @mesejo in [#14](https://github.com/letsql/letsql/pull/14)

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

