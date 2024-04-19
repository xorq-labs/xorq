# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Details
#### Added
- Add docker start to ci-test
- Poetry: add poetry checks to .pre-commit-config.yaml
- Add source cache by default
- Test_cache: add test_parquet_cache_storage
- Add rust files
- Add new cases to DataFusionBackend.register
- Add client tests for new register types
- Add faster function for CachedNode removal
- Add optimizations for predict_xgb in datafusion

#### Changed
- Improve performance and ux of predict_xgb
- Fetch only the required features for the model
- Organize the letsql package
- Lint
- Define CacheStorage with deterministic hashing for keys
- Define KEY_PREFIX to identify letsql cache
- Conftest: define expected_tables, enforce test fixture table list
- Lint
- Update poetry.lock
- Enable registration of pyarrow.RecordBatchReader and ir.Expr
- Update CONTRIBUTING.md with instructions to run Postgres
- Update requirements-dev.txt

#### Fixed
- Filter bug solved
- Set stable ibis dependency
- Failing ci
- Pyproject: specify rev when using git ref, don't use git@github.com
- Pyproject: make pyarrow,datafusion core dependencies
- Run `poetry lock --no-update`
- Use _load_into_cache in _put
- _cached: special case for name == "datafusion"
- ParquetCacheStorage: properly create cache dir
- Local cache with parquet storage

#### Removed
- Pyproject: remove redundant and conflicting dependency specifications
- Remove macos test suite
- Remove optimizer.py
- Remove pixi files

## [0.1.2.post] - 2024-02-01
### Details
#### Added
- Add missing dependencies

## [0.1.2] - 2024-02-01
### Details
#### Added
- Add CONTRIBUTING.md
- Address problems with schema
- Nix: add flake.nix and related files
- Add db package for showing predict udf working
- Add db package for showing predict udf working

#### Removed
- Remove xgboost as dependency

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

