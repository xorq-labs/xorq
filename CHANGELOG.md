# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.8] - 2024-10-24
### Details
Some major changes were introduced in this version the most important removing the need for registering expressions
for execution, updating to datafusion 42, as well as removing heavy rust dependencies such as candle. 

#### Changed
- Update dependency ruff to v0.6.4 by @renovate[bot] in [#258](https://github.com/letsql/letsql/pull/258)
- Update changelog command by @mesejo in [#259](https://github.com/letsql/letsql/pull/259)
- Update dependency ruff to v0.6.5 by @renovate[bot] in [#265](https://github.com/letsql/letsql/pull/265)
- Update actions/create-github-app-token action to v1.11.0 by @renovate[bot] in [#263](https://github.com/letsql/letsql/pull/263)
- Update dependency ruff to v0.6.8 by @renovate[bot] in [#273](https://github.com/letsql/letsql/pull/273)
- Disable test_examples temporarily by @mesejo in [#284](https://github.com/letsql/letsql/pull/284)
- Update dependency coverage to v7.6.3 by @renovate[bot] in [#283](https://github.com/letsql/letsql/pull/283)
- Update dependency ruff to v0.6.9 by @renovate[bot] in [#285](https://github.com/letsql/letsql/pull/285)
- Update dependency black to v24.10.0 by @renovate[bot] in [#287](https://github.com/letsql/letsql/pull/287)
- Update codecov/codecov-action action to v4.6.0 by @renovate[bot] in [#286](https://github.com/letsql/letsql/pull/286)
- Update to datafusion v42 by @mesejo in [#293](https://github.com/letsql/letsql/pull/293)
- Update dependency pre-commit to v4 by @renovate[bot] in [#291](https://github.com/letsql/letsql/pull/291)
- Update tests and workflows by @mesejo in [#299](https://github.com/letsql/letsql/pull/299)
- Only run ruff on repo files by @dlovell in [#301](https://github.com/letsql/letsql/pull/301)
- Set postgres env vars by @mesejo in [#303](https://github.com/letsql/letsql/pull/303)
- Update dependency ruff to v0.7.0 by @renovate[bot] in [#302](https://github.com/letsql/letsql/pull/302)

#### Fixed
- Fix pre-release workflow by @mesejo in [#257](https://github.com/letsql/letsql/pull/257)
- Update dependency fsspec to v2024.9.0 by @renovate[bot] in [#255](https://github.com/letsql/letsql/pull/255)
- Update dependency datafusion to v40 by @renovate[bot] in [#226](https://github.com/letsql/letsql/pull/226)
- Update rust crate arrow-ord to v53 by @renovate[bot] in [#251](https://github.com/letsql/letsql/pull/251)
- Enable build on macos by @dlovell in [#260](https://github.com/letsql/letsql/pull/260)
- Enable build on macos by @dlovell in [#262](https://github.com/letsql/letsql/pull/262)
- Update rust crate datafusion-common to v42 by @renovate[bot] in [#269](https://github.com/letsql/letsql/pull/269)
- Fix `nix run` issues re SSL and macos temp user dirs by @dlovell
- Fix `nix run` issues re IPYTHONDIR by @dlovell in [#264](https://github.com/letsql/letsql/pull/264)
- Docs deployment by @mesejo in [#294](https://github.com/letsql/letsql/pull/294)

#### Removed
- Remove the requirement of table registration for expr execution by @dlovell in [#209](https://github.com/letsql/letsql/pull/209)
- Remove segment_anything by @mesejo in [#295](https://github.com/letsql/letsql/pull/295)
- Remove tensor functions by @mesejo in [#297](https://github.com/letsql/letsql/pull/297)

## [0.1.7] - 2024-09-05
### Details
In this release, the segment_anything function has been refactored and cleaned up for improved performance and maintainability. 
The output of segment_anything has also been modified to return the mask and iou_score. 
Additionally, support for reading CSV files from HTTP sources has been added, along with basic S3 support, enhancing the data ingestion capabilities of the project.

#### Changed
- Update dependency ruff to v0.6.3 by @renovate[bot] in [#242](https://github.com/letsql/letsql/pull/242)
- Refactor and clean segment anything function by @mesejo in [#243](https://github.com/letsql/letsql/pull/243)
- Reading from csv in HTTP, add basic s3 support by @mesejo in [#230](https://github.com/letsql/letsql/pull/230)
- Change output of segment_anything to mask and iou_score by @mesejo in [#244](https://github.com/letsql/letsql/pull/244)
- Bump quinn-proto from 0.11.6 to 0.11.8 by @dependabot[bot] in [#249](https://github.com/letsql/letsql/pull/249)
- Update actions/create-github-app-token action to v1.10.4 by @renovate[bot] in [#253](https://github.com/letsql/letsql/pull/253)
- Bump cryptography from 43.0.0 to 43.0.1 by @dependabot[bot] in [#254](https://github.com/letsql/letsql/pull/254)

#### Fixed
- Fix typo in README by @mesejo in [#241](https://github.com/letsql/letsql/pull/241)
- Update rust crate arrow to v53 by @renovate[bot] in [#250](https://github.com/letsql/letsql/pull/250)

## [0.1.6] - 2024-08-29
### Details
This update includes new workflows for testing Snowflake and S3, a dependency update for ruff, 
and several fixes addressing PyPI release issues, in-memory table registration, and Dask version compatibility.

#### Added
- Add workflow for testing snowflake by @mesejo in [#233](https://github.com/letsql/letsql/pull/233)
- Add ci workflow for testing s3 by @mesejo in [#235](https://github.com/letsql/letsql/pull/235)

#### Changed
- Update dependency ruff to v0.6.2 by @renovate[bot] in [#229](https://github.com/letsql/letsql/pull/229)

#### Fixed
- Issues with release to pypi by @mesejo in [#228](https://github.com/letsql/letsql/pull/228)
- Registration of in-memory tables by @mesejo in [#232](https://github.com/letsql/letsql/pull/232)
- Improve snowflake workflow by @mesejo in [#234](https://github.com/letsql/letsql/pull/234)
- Checkout PR ref by @mesejo in [#236](https://github.com/letsql/letsql/pull/236)
- Fix dask version by @mesejo in [#237](https://github.com/letsql/letsql/pull/237)

## [0.1.5] - 2024-08-21
### Details
The library has seen a lot of active development, with numerous new features and improvements added in various pull requests:
- New functionality, such as a pyarrow-based UDAF, postgres and sqlite readers, image/array manipulation functions, 
and xgboost prediction functions, have been added.
- Existing functionality has been enhanced by wrapping ibis backends, updating dependencies, and improving the build/testing process.
- Numerous dependency updates have been made to keep the library up-to-date.
- Some bug fixes and stability improvements have been implemented as well.


#### Added
- Add pyarrow udaf based on PyAggregator by @mesejo in [#108](https://github.com/letsql/letsql/pull/108)
- Add unit tests based on workflow diagram by @mesejo in [#110](https://github.com/letsql/letsql/pull/110)
- Add postgres read_parquet by @mesejo in [#118](https://github.com/letsql/letsql/pull/118)
- Add wrapper for snowflake backend by @mesejo in [#119](https://github.com/letsql/letsql/pull/119)
- Add read_sqlite and read_postgres by @mesejo in [#120](https://github.com/letsql/letsql/pull/120)
- Add ibis udf and model registration method by @hussainsultan in [#182](https://github.com/letsql/letsql/pull/182)
- Add udf signature and return a partial with model_name by @hussainsultan in [#195](https://github.com/letsql/letsql/pull/195)
- Add image and array manipulation functions by @mesejo in [#181](https://github.com/letsql/letsql/pull/181)
- Add example predict_xgb.py by @dlovell in [#213](https://github.com/letsql/letsql/pull/213)
- Add connectors for using environment variables or fixed examples server by @dlovell in [#217](https://github.com/letsql/letsql/pull/217)
- Add workflow for testing library only dependencies by @mesejo in [#223](https://github.com/letsql/letsql/pull/223)
- Add duckdb and xgboost as dependencies for examples by @mesejo in [#216](https://github.com/letsql/letsql/pull/216)

#### Changed
- Wrap ibis backends by @mesejo in [#115](https://github.com/letsql/letsql/pull/115)
- Unpin pyarrow version by @mesejo in [#121](https://github.com/letsql/letsql/pull/121)
- Update README by @mesejo in [#125](https://github.com/letsql/letsql/pull/125)
- Use options.backend as ParquetCacheStorage's default backend by @mesejo in [#123](https://github.com/letsql/letsql/pull/123)
- Change to publish on release by @mesejo in [#122](https://github.com/letsql/letsql/pull/122)
- Configure Renovate by @renovate[bot] in [#124](https://github.com/letsql/letsql/pull/124)
- Update dependency black to v24 [security] by @renovate[bot] in [#126](https://github.com/letsql/letsql/pull/126)
- Update dependency pure-eval to v0.2.3 by @renovate[bot] in [#130](https://github.com/letsql/letsql/pull/130)
- Update dependency blackdoc to v0.3.9 by @renovate[bot] in [#128](https://github.com/letsql/letsql/pull/128)
- Update dependency pytest to v7.4.4 by @renovate[bot] in [#131](https://github.com/letsql/letsql/pull/131)
- Update actions/create-github-app-token action to v1.10.3 by @renovate[bot] in [#127](https://github.com/letsql/letsql/pull/127)
- Update dependency connectorx to v0.3.3 by @renovate[bot] in [#129](https://github.com/letsql/letsql/pull/129)
- Update dependency snowflake/snowflake-connector-python to v3.11.0 by @renovate[bot] in [#141](https://github.com/letsql/letsql/pull/141)
- Update dependency importlib-metadata to v8.1.0 by @renovate[bot] in [#139](https://github.com/letsql/letsql/pull/139)
- Update dependency ruff to v0.5.4 by @renovate[bot] in [#133](https://github.com/letsql/letsql/pull/133)
- Update dependency black to v24.4.2 by @renovate[bot] in [#136](https://github.com/letsql/letsql/pull/136)
- Update dependency sqlalchemy to v2.0.31 by @renovate[bot] in [#134](https://github.com/letsql/letsql/pull/134)
- Update codecov/codecov-action action to v4.5.0 by @renovate[bot] in [#135](https://github.com/letsql/letsql/pull/135)
- Update dependency codespell to v2.3.0 by @renovate[bot] in [#137](https://github.com/letsql/letsql/pull/137)
- Update dependency coverage to v7.6.0 by @renovate[bot] in [#138](https://github.com/letsql/letsql/pull/138)
- Update dependency sqlglot to v23.17.0 by @renovate[bot] in [#142](https://github.com/letsql/letsql/pull/142)
- Update dependency pre-commit to v3.7.1 by @renovate[bot] in [#140](https://github.com/letsql/letsql/pull/140)
- Update dependency structlog to v24.4.0 by @renovate[bot] in [#143](https://github.com/letsql/letsql/pull/143)
- Update actions/checkout action to v4 by @renovate[bot] in [#148](https://github.com/letsql/letsql/pull/148)
- Update actions/setup-python action to v5 by @renovate[bot] in [#149](https://github.com/letsql/letsql/pull/149)
- Update dependency datafusion/datafusion to v39 by @renovate[bot] in [#150](https://github.com/letsql/letsql/pull/150)
- Update dependency numpy to v2 by @renovate[bot] in [#152](https://github.com/letsql/letsql/pull/152)
- Update dependency duckb/duckdb to v1 by @renovate[bot] in [#151](https://github.com/letsql/letsql/pull/151)
- Update dependency pyarrow to v17 by @renovate[bot] in [#153](https://github.com/letsql/letsql/pull/153)
- Disable pip_requirements manager by @mesejo in [#163](https://github.com/letsql/letsql/pull/163)
- Update dependency pytest-cov to v5 by @renovate[bot] in [#159](https://github.com/letsql/letsql/pull/159)
- Update extractions/setup-just action to v2 by @renovate[bot] in [#161](https://github.com/letsql/letsql/pull/161)
- Update github artifact actions to v4 by @renovate[bot] in [#162](https://github.com/letsql/letsql/pull/162)
- Range for datafusion-common by @renovate[bot] in [#166](https://github.com/letsql/letsql/pull/166)
- Update dependency pytest to v8 by @renovate[bot] in [#158](https://github.com/letsql/letsql/pull/158)
- Update dependencies ranges by @mesejo in [#172](https://github.com/letsql/letsql/pull/172)
- Enable plugin development for backends by @mesejo in [#132](https://github.com/letsql/letsql/pull/132)
- Include pre-commit dependencies in renovatebot scan by @mesejo in [#176](https://github.com/letsql/letsql/pull/176)
- Update dependency ruff to v0.5.5 by @renovate[bot] in [#174](https://github.com/letsql/letsql/pull/174)
- Bump object_store from 0.10.1 to 0.10.2 by @dependabot[bot] in [#175](https://github.com/letsql/letsql/pull/175)
- Update dependency pre-commit to v3.8.0 by @renovate[bot] in [#178](https://github.com/letsql/letsql/pull/178)
- Lock file maintenance, update Cargo TOML by @renovate[bot] in [#179](https://github.com/letsql/letsql/pull/179)
- Refactor flake by @dlovell in [#180](https://github.com/letsql/letsql/pull/180)
- Use poetry2nix overlays by @dlovell
- Enable editable install by @dlovell
- Update dependency ruff to v0.5.6 by @renovate[bot] in [#183](https://github.com/letsql/letsql/pull/183)
- Update dependency coverage to v7.6.1 by @renovate[bot] in [#187](https://github.com/letsql/letsql/pull/187)
- Lock file maintenance by @renovate[bot] in [#188](https://github.com/letsql/letsql/pull/188)
- Collapse ifs by @dlovell
- Enable `nix run` to drop into an ipython shell by @dlovell
- Make key_prefix settable in config/CacheStorage by @dlovell in [#196](https://github.com/letsql/letsql/pull/196)
- Update dependency ruff to v0.5.7 by @renovate[bot] in [#197](https://github.com/letsql/letsql/pull/197)
- Bump aiohttp from 3.9.5 to 3.10.2 by @dependabot[bot] in [#212](https://github.com/letsql/letsql/pull/212)
- Lock file maintenance by @renovate[bot] in [#207](https://github.com/letsql/letsql/pull/207)
- Return wrapper with model_name partialized by @hussainsultan
- Update links to data files by @mesejo in [#214](https://github.com/letsql/letsql/pull/214)
- Update dependency ruff to v0.6.0 by @renovate[bot] in [#215](https://github.com/letsql/letsql/pull/215)
- Update gbdt-rs repo url by @mesejo in [#220](https://github.com/letsql/letsql/pull/220)
- Make gbdt-rs dependency unambiguous by @mesejo in [#222](https://github.com/letsql/letsql/pull/222)
- Use postgres.connect_examples() and TemporaryDirectory by @mesejo in [#219](https://github.com/letsql/letsql/pull/219)
- Update dependency ruff to v0.6.1 by @renovate[bot] in [#218](https://github.com/letsql/letsql/pull/218)

#### Fixed
- Register cache tables when executing to_pyarrow by @mesejo in [#114](https://github.com/letsql/letsql/pull/114)
- Update dependency fsspec to v2024.6.1 by @renovate[bot] in [#144](https://github.com/letsql/letsql/pull/144)
- Update rust crate pyo3 to 0.21 by @renovate[bot] in [#146](https://github.com/letsql/letsql/pull/146)
- Update tokio-prost monorepo to 0.13.1 by @renovate[bot] in [#147](https://github.com/letsql/letsql/pull/147)
- Update rust crate datafusion range to v40 by @renovate[bot] in [#165](https://github.com/letsql/letsql/pull/165)
- Update rust crate datafusion-* to v40 by @renovate[bot] in [#167](https://github.com/letsql/letsql/pull/167)
- Widen dependency dask range to v2024 by @renovate[bot] in [#164](https://github.com/letsql/letsql/pull/164)
- Enable build on macos by @dlovell
- Conditionally include libiconv in maturinOverride by @dlovell
- Update dependency attrs to v24 by @renovate[bot] in [#185](https://github.com/letsql/letsql/pull/185)
- Return proper type in get_log_path by @dlovell
- Use pandas backend in SourceStorage by @mesejo
- Update rust crate datafusion to v41 by @renovate[bot] in [#203](https://github.com/letsql/letsql/pull/203)

#### Removed
- Remove warnings and deprecated palmerpenguins package by @mesejo in [#113](https://github.com/letsql/letsql/pull/113)
- Remove  so that the udf keeps its metadata by @hussainsultan in [#198](https://github.com/letsql/letsql/pull/198)

## New Contributors
* @renovate[bot] made their first contribution in [#218](https://github.com/letsql/letsql/pull/218)
* @dependabot[bot] made their first contribution in [#212](https://github.com/letsql/letsql/pull/212)

## [0.1.4] - 2024-07-02
### Details
#### Changed
- Api letsql api methods by @mesejo in [#105](https://github.com/letsql/letsql/pull/105)
- Prepare for release 0.1.4 by @mesejo in [#107](https://github.com/letsql/letsql/pull/107)
- 0.1.4 by @mesejo in [#109](https://github.com/letsql/letsql/pull/109)

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
- Add SnapshotStorage by @dlovell in [#103](https://github.com/letsql/letsql/pull/103)

#### Changed
- Improve performance and ux of predict_xgb by @mesejo
- Improve performance and ux of predict_xgb by @mesejo in [#8](https://github.com/letsql/letsql/pull/8)
- Fetch only the required features for the model by @mesejo
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
- V0.1.3 by @mesejo in [#106](https://github.com/letsql/letsql/pull/106)

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
- Add db package for showing predict udf working by @mesejo
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

