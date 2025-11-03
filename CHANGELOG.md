# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.2] - 2025-11-03
### Details
#### Added
- Add Multiply and Hash ops by @mesejo in [#1183](https://github.com/xorq-labs/xorq/pull/1183)
- Add `storage` arg to `Pipeline.fit` by @dlovell in [#1257](https://github.com/xorq-labs/xorq/pull/1257)
- Add better error message for ValueError by @mesejo in [#1263](https://github.com/xorq-labs/xorq/pull/1263)
- Add reduction ops by @mesejo in [#1265](https://github.com/xorq-labs/xorq/pull/1265)
- Add xorq-psql by @dlovell in [#1270](https://github.com/xorq-labs/xorq/pull/1270)
- Add valid default xorq.expr.udf.agg.pandas_df name by @mesejo in [#1278](https://github.com/xorq-labs/xorq/pull/1278)
- Add sklearn_classifier_comparison by @dlovell in [#1273](https://github.com/xorq-labs/xorq/pull/1273)
- Add sqlite backend by @mesejo in [#1244](https://github.com/xorq-labs/xorq/pull/1244)
- Add pandas tests by @mesejo in [#1295](https://github.com/xorq-labs/xorq/pull/1295)
- Add tagging for ml ops by @mesejo in [#1297](https://github.com/xorq-labs/xorq/pull/1297)
- Add keypair auth by @dlovell in [#1334](https://github.com/xorq-labs/xorq/pull/1334)
- Add benchmark for import by @mesejo in [#1342](https://github.com/xorq-labs/xorq/pull/1342)
- Add sops by @dlovell in [#1337](https://github.com/xorq-labs/xorq/pull/1337)
- Add default database/schema from env vars by @dlovell in [#1353](https://github.com/xorq-labs/xorq/pull/1353)
- Add from_ibis function by @mesejo in [#1354](https://github.com/xorq-labs/xorq/pull/1354)

#### Changed
- Test weather_lib example by @mesejo in [#1254](https://github.com/xorq-labs/xorq/pull/1254)
- Easily uncache by @dlovell in [#1256](https://github.com/xorq-labs/xorq/pull/1256)
- Improve no-extra runner by @mesejo in [#1231](https://github.com/xorq-labs/xorq/pull/1231)
- Add translation for First and Any ops by @mesejo in [#1264](https://github.com/xorq-labs/xorq/pull/1264)
- Better metadata by @dlovell in [#1271](https://github.com/xorq-labs/xorq/pull/1271)
- Split slow test workflow by @mesejo in [#1272](https://github.com/xorq-labs/xorq/pull/1272)
- Train test split default by @dlovell in [#1196](https://github.com/xorq-labs/xorq/pull/1196)
- Update readme with bottoms-up messaging by @hussainsultan in [#1280](https://github.com/xorq-labs/xorq/pull/1280)
- Update xorq_datafusion version by @mesejo in [#1284](https://github.com/xorq-labs/xorq/pull/1284)
- Move tests to the dedicated package by @mesejo in [#1293](https://github.com/xorq-labs/xorq/pull/1293)
- Move pandas backend out of vendored ibis by @mesejo in [#1296](https://github.com/xorq-labs/xorq/pull/1296)
- Allow for specifying compiler in to_sql by @mesejo in [#1283](https://github.com/xorq-labs/xorq/pull/1283)
- Improve testing by @mesejo in [#1287](https://github.com/xorq-labs/xorq/pull/1287)
- Update quickstart.qmd by @mesejo in [#1299](https://github.com/xorq-labs/xorq/pull/1299)
- Update quarto version by @mesejo in [#1315](https://github.com/xorq-labs/xorq/pull/1315)
- Chore: use Cloudfare CDN polyfill version by @mesejo in [#1314](https://github.com/xorq-labs/xorq/pull/1314)
- Change Bitnami Catalog (docker images) by @mesejo in [#1316](https://github.com/xorq-labs/xorq/pull/1316)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.8.22 by @renovate[bot] in [#1320](https://github.com/xorq-labs/xorq/pull/1320)
- Update dependency blackdoc to v0.4.3 by @renovate[bot] in [#1317](https://github.com/xorq-labs/xorq/pull/1317)
- Update dependency coverage to v7.10.7 by @renovate[bot] in [#1318](https://github.com/xorq-labs/xorq/pull/1318)
- Update dependency pytest to v8.4.2 by @renovate[bot] in [#1319](https://github.com/xorq-labs/xorq/pull/1319)
- Update actions/create-github-app-token action to v2.1.4 by @renovate[bot] in [#1321](https://github.com/xorq-labs/xorq/pull/1321)
- Update codecov/codecov-action action to v5.5.1 by @renovate[bot] in [#1322](https://github.com/xorq-labs/xorq/pull/1322)
- Update dependency black to v25.9.0 by @renovate[bot] in [#1323](https://github.com/xorq-labs/xorq/pull/1323)
- Update dependency ipython to <9.7.0,>=8.19.0 by @renovate[bot] in [#1324](https://github.com/xorq-labs/xorq/pull/1324)
- Update dependency pre-commit to v4.3.0 by @renovate[bot] in [#1325](https://github.com/xorq-labs/xorq/pull/1325)
- Update dependency pytest-cov to v6.3.0 by @renovate[bot] in [#1326](https://github.com/xorq-labs/xorq/pull/1326)
- Make CachedNode inherit DatabaseTable by @mesejo in [#1307](https://github.com/xorq-labs/xorq/pull/1307)
- Ensure expr is indeed an ibis Expr by @mesejo in [#1328](https://github.com/xorq-labs/xorq/pull/1328)
- Update postgres Backend to latest ibis code by @mesejo in [#1329](https://github.com/xorq-labs/xorq/pull/1329)
- Make Read inherit from DatabaseTable by @mesejo in [#1309](https://github.com/xorq-labs/xorq/pull/1309)
- Add ValueError messages by @ryanping in [#1330](https://github.com/xorq-labs/xorq/pull/1330)
- Use evolve instead of defining clone by @mesejo in [#1335](https://github.com/xorq-labs/xorq/pull/1335)
- Update snowflake Backend code by @mesejo in [#1343](https://github.com/xorq-labs/xorq/pull/1343)
- Create common ancestor for custom ops by @mesejo in [#1338](https://github.com/xorq-labs/xorq/pull/1338)
- Apply security updates described by dependabot by @mesejo in [#1356](https://github.com/xorq-labs/xorq/pull/1356)
- RemoteTable registration by @mesejo in [#1355](https://github.com/xorq-labs/xorq/pull/1355)

#### Fixed
- Timeout peek port by @dlovell in [#1255](https://github.com/xorq-labs/xorq/pull/1255)
- Skip re.search on Path object in _maybe_glob_pattern by @mesejo in [#1261](https://github.com/xorq-labs/xorq/pull/1261)
- Uncached fittedstep model by @dlovell in [#1260](https://github.com/xorq-labs/xorq/pull/1260)
- Use pytest importorskip to sklearn test by @mesejo in [#1262](https://github.com/xorq-labs/xorq/pull/1262)
- _ParquetStorage._put: write to a tmp location by @dlovell in [#1268](https://github.com/xorq-labs/xorq/pull/1268)
- _transform_deferred_reads: squelch warning re Path by @dlovell in [#1275](https://github.com/xorq-labs/xorq/pull/1275)
- Faster do_explode_encode by @dlovell in [#1269](https://github.com/xorq-labs/xorq/pull/1269)
- Pipeline infer features by @dlovell in [#1274](https://github.com/xorq-labs/xorq/pull/1274)
- Not execute doc code by @mesejo in [#1281](https://github.com/xorq-labs/xorq/pull/1281)
- Enable count by @dlovell in [#1290](https://github.com/xorq-labs/xorq/pull/1290)
- Train test splits pandas by @dlovell in [#1291](https://github.com/xorq-labs/xorq/pull/1291)
- Custom_hash for nameless Series by @mesejo in [#1289](https://github.com/xorq-labs/xorq/pull/1289)
- Distinct: register from_yaml loader by @dlovell in [#1303](https://github.com/xorq-labs/xorq/pull/1303)
- Fixup template tests by @dlovell in [#1311](https://github.com/xorq-labs/xorq/pull/1311)
- Fixup pipeline tagging tests by @dlovell in [#1312](https://github.com/xorq-labs/xorq/pull/1312)
- Default env var reference by @dlovell in [#1310](https://github.com/xorq-labs/xorq/pull/1310)
- Mermaid diagrams in vignettes by @mesejo in [#1327](https://github.com/xorq-labs/xorq/pull/1327)
- Residual fixups for psycopg update by @dlovell in [#1333](https://github.com/xorq-labs/xorq/pull/1333)
- Parse multiline env vars by @dlovell in [#1331](https://github.com/xorq-labs/xorq/pull/1331)
- Fix `UnboundLocalError` on connect by @dlovell in [#1341](https://github.com/xorq-labs/xorq/pull/1341)
- Snowflake GitHub Action by @mesejo in [#1344](https://github.com/xorq-labs/xorq/pull/1344)
- Create table from expr by @dlovell in [#1347](https://github.com/xorq-labs/xorq/pull/1347)
- Default create_object_udfs to False by @dlovell in [#1352](https://github.com/xorq-labs/xorq/pull/1352)

#### Removed
- Remove vendored ibis conftest.py by @mesejo in [#1300](https://github.com/xorq-labs/xorq/pull/1300)
- Remove commented code by @mesejo in [#1313](https://github.com/xorq-labs/xorq/pull/1313)
- Remove redundant normalization by @mesejo in [#1336](https://github.com/xorq-labs/xorq/pull/1336)
- Remove replace_cache_table by @mesejo in [#1348](https://github.com/xorq-labs/xorq/pull/1348)
- Remove legacy_replace_cache_table by @mesejo in [#1349](https://github.com/xorq-labs/xorq/pull/1349)

## New Contributors
* @ryanping made their first contribution in [#1330](https://github.com/xorq-labs/xorq/pull/1330)
## [0.3.1] - 2025-08-30
### Details
#### Added
- Add xorq app by @hussainsultan in [#1163](https://github.com/xorq-labs/xorq/pull/1163)
- Add CLI section in 10_minutes by @mesejo in [#1158](https://github.com/xorq-labs/xorq/pull/1158)
- Add Tag(Node) class, Expr.tag method by @dlovell in [#1205](https://github.com/xorq-labs/xorq/pull/1205)
- Add limit option to run by @hussainsultan in [#1203](https://github.com/xorq-labs/xorq/pull/1203)
- Support group by alias in aggregate by @hussainsultan in [#1224](https://github.com/xorq-labs/xorq/pull/1224)
- Add runner per backend by @mesejo in [#1210](https://github.com/xorq-labs/xorq/pull/1210)
- LETSQLAccessor: add property `tokenized` by @dlovell in [#1238](https://github.com/xorq-labs/xorq/pull/1238)
- Add listing and fetching of UnboundExprExchanger by @mesejo in [#1207](https://github.com/xorq-labs/xorq/pull/1207)
- Add catalog by @dlovell in [#1249](https://github.com/xorq-labs/xorq/pull/1249)

#### Changed
- Make xorq cli app to use all deps by @hussainsultan in [#1164](https://github.com/xorq-labs/xorq/pull/1164)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.7 by @renovate[bot] in [#1168](https://github.com/xorq-labs/xorq/pull/1168)
- Update dependency ruff to v0.12.7 by @renovate[bot] in [#1167](https://github.com/xorq-labs/xorq/pull/1167)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.8.4 by @renovate[bot] in [#1171](https://github.com/xorq-labs/xorq/pull/1171)
- Update bitnami/minio docker tag to v2025.7.23 by @renovate[bot] in [#1169](https://github.com/xorq-labs/xorq/pull/1169)
- Properly configure pin setup-python by @mesejo in [#1177](https://github.com/xorq-labs/xorq/pull/1177)
- Update dependency coverage to v7.10.2 by @renovate[bot] in [#1170](https://github.com/xorq-labs/xorq/pull/1170)
- Update dependency pytest-codspeed to v4 by @renovate[bot] in [#1173](https://github.com/xorq-labs/xorq/pull/1173)
- Make deferred read and sql outputs optional in BuildManager by @hussainsultan in [#1199](https://github.com/xorq-labs/xorq/pull/1199)
- Improve speed of ibis_yaml package by @mesejo in [#1202](https://github.com/xorq-labs/xorq/pull/1202)
- Move into_backend tests to test_into_backend.py by @mesejo in [#1178](https://github.com/xorq-labs/xorq/pull/1178)
- Bank marketing uses pipeline lib by @dlovell in [#1204](https://github.com/xorq-labs/xorq/pull/1204)
- Use local files for reducing execution time by @mesejo in [#1206](https://github.com/xorq-labs/xorq/pull/1206)
- Delay sklearn import by @dlovell in [#1215](https://github.com/xorq-labs/xorq/pull/1215)
- Delay pandas import by @dlovell in [#1219](https://github.com/xorq-labs/xorq/pull/1219)
- Use peek_port in test_serve_command by @mesejo in [#1221](https://github.com/xorq-labs/xorq/pull/1221)
- Reduce combinations and outdated workflows by @mesejo in [#1230](https://github.com/xorq-labs/xorq/pull/1230)
- Update penguins hexdigest by @mesejo in [#1236](https://github.com/xorq-labs/xorq/pull/1236)
- Move from xorq to xorq.api by @mesejo in [#1222](https://github.com/xorq-labs/xorq/pull/1222)
- Warn when a Read op targets a local file by @mesejo in [#1195](https://github.com/xorq-labs/xorq/pull/1195)
- Update project urls by @mesejo in [#1248](https://github.com/xorq-labs/xorq/pull/1248)
- Update CLI references with new commands by @sanajitjana in [#1245](https://github.com/xorq-labs/xorq/pull/1245)
- Disable weather_flight example temporarily by @mesejo in [#1251](https://github.com/xorq-labs/xorq/pull/1251)
- Update google-site-verification by @mesejo in [#1253](https://github.com/xorq-labs/xorq/pull/1253)

#### Fixed
- Update dependency datafusion to v48 by @renovate[bot] in [#1174](https://github.com/xorq-labs/xorq/pull/1174)
- Update dependency fsspec to >=2024.6.1,<2025.7.1 by @renovate[bot] in [#1172](https://github.com/xorq-labs/xorq/pull/1172)
- Update dependency pyarrow to v21 by @renovate[bot] in [#1175](https://github.com/xorq-labs/xorq/pull/1175)
- Ensure preserve_index=False in RecordBatch.from_pandas by @mesejo in [#1160](https://github.com/xorq-labs/xorq/pull/1160)
- Xorq serve inference by @dlovell in [#1176](https://github.com/xorq-labs/xorq/pull/1176)
- Fix riscv issuee in nix run by @hussainsultan in [#1185](https://github.com/xorq-labs/xorq/pull/1185)
- Fix pyarrow 21.0.0 build on darwin by @hussainsultan in [#1192](https://github.com/xorq-labs/xorq/pull/1192)
- Read_record_batches schema mismatch by @mesejo in [#1190](https://github.com/xorq-labs/xorq/pull/1190)
- Pointing to main of xorq-weather-lib by @mesejo in [#1188](https://github.com/xorq-labs/xorq/pull/1188)
- Register with table_name default value by @mesejo in [#1182](https://github.com/xorq-labs/xorq/pull/1182)
- Update flake inputs by @dlovell in [#1191](https://github.com/xorq-labs/xorq/pull/1191)
- .sql on deferred_read_* nodes by @mesejo in [#1184](https://github.com/xorq-labs/xorq/pull/1184)
- Rename serve cmd by @hussainsultan in [#1200](https://github.com/xorq-labs/xorq/pull/1200)
- Better serve-unbound con detection by @dlovell in [#1212](https://github.com/xorq-labs/xorq/pull/1212)
- Wrong arg limit in serve_unbound_parser by @mesejo in [#1226](https://github.com/xorq-labs/xorq/pull/1226)
- Use item accessor (square brackets) for tag column by @mesejo in [#1227](https://github.com/xorq-labs/xorq/pull/1227)
- Fix readme broken links by @hussainsultan in [#1234](https://github.com/xorq-labs/xorq/pull/1234)
- SourceStorage._put: don't pull local by @dlovell in [#1235](https://github.com/xorq-labs/xorq/pull/1235)
- Update reference by @mesejo in [#1243](https://github.com/xorq-labs/xorq/pull/1243)
- Allow None for the con in deferred_read_parquet by @mesejo
- Fix CLI command docs reference by @mesejo in [#1250](https://github.com/xorq-labs/xorq/pull/1250)
- Replace computed kwargs expr by @dlovell in [#1247](https://github.com/xorq-labs/xorq/pull/1247)

#### Removed
- Remove spurious riscv64 entry by @dlovell in [#1165](https://github.com/xorq-labs/xorq/pull/1165)
- Remove bind value in FlightURL by @mesejo in [#1181](https://github.com/xorq-labs/xorq/pull/1181)
- Remove row_number hack for gen_splits by @hussainsultan in [#1193](https://github.com/xorq-labs/xorq/pull/1193)
- Remove print in elide_cache_node by @mesejo in [#1228](https://github.com/xorq-labs/xorq/pull/1228)

### New Contributors
* @sanajitjana made their first contribution in [#1245](https://github.com/xorq-labs/xorq/pull/1245)

## [0.3.0] - 2025-07-28
### Details
#### Added
- Add expression format concept page by @mesejo in [#1159](https://github.com/xorq-labs/xorq/pull/1159)

## [0.2.5] - 2025-07-26
### Details

#### Added
- Add `xorq init` by @dlovell in [#1117](https://github.com/xorq-labs/xorq/pull/1117)
- Add cryptography as a core dependency by @dlovell in [#1123](https://github.com/xorq-labs/xorq/pull/1123)
- Add google verification to docs by @mesejo in [#1126](https://github.com/xorq-labs/xorq/pull/1126)
- Add penguins template to xorq init by @mesejo in [#1134](https://github.com/xorq-labs/xorq/pull/1134)
- Add DropNull op translation by @mesejo in [#1135](https://github.com/xorq-labs/xorq/pull/1135)
- Add caching concept by @mesejo in [#1154](https://github.com/xorq-labs/xorq/pull/1154)

#### Changed
- Test installation for multiple Python versions and OS by @mesejo in [#1107](https://github.com/xorq-labs/xorq/pull/1107)
- Uv-build by @dlovell in [#1132](https://github.com/xorq-labs/xorq/pull/1132)
- Run slow tests in a different runner by @mesejo in [#1087](https://github.com/xorq-labs/xorq/pull/1087)
- Refactor reference structure by @mesejo in [#1129](https://github.com/xorq-labs/xorq/pull/1129)
- Readme graphic by @hussainsultan in [#1150](https://github.com/xorq-labs/xorq/pull/1150)
- Move functions to translate.py by @mesejo in [#1152](https://github.com/xorq-labs/xorq/pull/1152)
- Make con optional in deferred_read by @mesejo in [#1142](https://github.com/xorq-labs/xorq/pull/1142)
- Xorq serve inference by @dlovell in [#1143](https://github.com/xorq-labs/xorq/pull/1143)
- Update quickstart tutorial to reflect CLI-centric approach by @hussainsultan in [#1157](https://github.com/xorq-labs/xorq/pull/1157)
- Update 10_minutes_xorq_tour by @mesejo in [#1151](https://github.com/xorq-labs/xorq/pull/1151)

#### Fixed
- Use Path.name instead of Path.stem in run_command by @mesejo in [#1118](https://github.com/xorq-labs/xorq/pull/1118)
- Update for xorq-hash-cache rename by @dlovell in [#1122](https://github.com/xorq-labs/xorq/pull/1122)
- Fix quoting in google indexing by @mesejo in [#1127](https://github.com/xorq-labs/xorq/pull/1127)
- Use print_exc by @dlovell in [#1125](https://github.com/xorq-labs/xorq/pull/1125)
- Use port and host args in serve by @mesejo in [#1130](https://github.com/xorq-labs/xorq/pull/1130)
- Better type inference for LogisticRegression by @dlovell in [#1136](https://github.com/xorq-labs/xorq/pull/1136)
- Update for xorq-hash-cache by @dlovell in [#1131](https://github.com/xorq-labs/xorq/pull/1131)
- Port and host args handling by @mesejo in [#1144](https://github.com/xorq-labs/xorq/pull/1144)
- Use expr.ls.uncached when finding backend by @mesejo in [#1141](https://github.com/xorq-labs/xorq/pull/1141)
- Fix run commands by @dlovell in [#1153](https://github.com/xorq-labs/xorq/pull/1153)
- Deferred_read_* methods by @mesejo in [#1155](https://github.com/xorq-labs/xorq/pull/1155)

#### Removed
- Serve_command: remove spurious kwargs by @dlovell in [#1145](https://github.com/xorq-labs/xorq/pull/1145)

## [0.2.4] - 2025-07-09
### Details
#### Added
- Add base methods from Expr by @mesejo in [#1093](https://github.com/xorq-labs/xorq/pull/1093)
- Add cache_dir to serve command by @mesejo in [#1106](https://github.com/xorq-labs/xorq/pull/1106)

#### Changed
- Feature store by @hussainsultan in [#1091](https://github.com/xorq-labs/xorq/pull/1091)
- Rename weather micro-libs to xorq-prefix by @dlovell in [#1096](https://github.com/xorq-labs/xorq/pull/1096)
- Singledispatch sklearn by @dlovell in [#1100](https://github.com/xorq-labs/xorq/pull/1100)
- Use snapshot in test_build_file_stability by @mesejo in [#1092](https://github.com/xorq-labs/xorq/pull/1092)
- Improve grid listing style by @mesejo in [#1101](https://github.com/xorq-labs/xorq/pull/1101)
- Improve ml section documentation by @mesejo in [#1099](https://github.com/xorq-labs/xorq/pull/1099)

#### Fixed
- Override base, not just editables by @dlovell in [#1095](https://github.com/xorq-labs/xorq/pull/1095)
- Correct backend reference in README by @IndexSeek in [#1097](https://github.com/xorq-labs/xorq/pull/1097)
- Deferred_read_csv: schema: use columns not types by @dlovell in [#1103](https://github.com/xorq-labs/xorq/pull/1103)
- Fix border output color by @mesejo in [#1104](https://github.com/xorq-labs/xorq/pull/1104)
- Generic handling of cached node by @dlovell in [#1105](https://github.com/xorq-labs/xorq/pull/1105)
- Structfield by @dlovell in [#1109](https://github.com/xorq-labs/xorq/pull/1109)
- Fetch penguins by @mesejo in [#1114](https://github.com/xorq-labs/xorq/pull/1114)

## New Contributors
* @IndexSeek made their first contribution in [#1097](https://github.com/xorq-labs/xorq/pull/1097)

## [0.2.3] - 2025-07-02
### Details

#### Added
- Support list of paths in read_csv_rbr by @mesejo in [#927](https://github.com/xorq-labs/xorq/pull/927)
- Add snapshot filtering by @mesejo in [#937](https://github.com/xorq-labs/xorq/pull/937)
- Add python 3.13 to the test suite by @mesejo in [#947](https://github.com/xorq-labs/xorq/pull/947)
- Add upsert by @mesejo in [#938](https://github.com/xorq-labs/xorq/pull/938)
- Add xorq-cachix-{use,push} by @dlovell in [#948](https://github.com/xorq-labs/xorq/pull/948)
- Add tls_utils by @dlovell in [#966](https://github.com/xorq-labs/xorq/pull/966)
- Add set ops by @mesejo in [#970](https://github.com/xorq-labs/xorq/pull/970)
- Add missing string ops translation by @mesejo in [#969](https://github.com/xorq-labs/xorq/pull/969)
- Add .gitignore to .gitignore.template by @dlovell in [#999](https://github.com/xorq-labs/xorq/pull/999)
- Add lineage_utils by @hussainsultan in [#993](https://github.com/xorq-labs/xorq/pull/993)
- Add back opaque_ops by @mesejo in [#1012](https://github.com/xorq-labs/xorq/pull/1012)
- Add pipeline by @dlovell in [#1015](https://github.com/xorq-labs/xorq/pull/1015)
- Add ci-docs-preview workflow for netlify by @mesejo in [#1030](https://github.com/xorq-labs/xorq/pull/1030)
- Add YAML and build outputs in getting started by @hussainsultan in [#1063](https://github.com/xorq-labs/xorq/pull/1063)
- Add serve command by @hussainsultan in [#1019](https://github.com/xorq-labs/xorq/pull/1019)
- Add TSLKwargs to_disk and from_disk by @mesejo in [#1077](https://github.com/xorq-labs/xorq/pull/1077)

#### Changed
- Update postgres docker tag to v17.5 by @renovate[bot] in [#924](https://github.com/xorq-labs/xorq/pull/924)
- Update dependency ruff to v0.11.9 by @renovate[bot] in [#925](https://github.com/xorq-labs/xorq/pull/925)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.9 by @renovate[bot] in [#926](https://github.com/xorq-labs/xorq/pull/926)
- Update xorq-datafusion version by @mesejo in [#929](https://github.com/xorq-labs/xorq/pull/929)
- Update dependency ruff to v0.11.10 by @renovate[bot] in [#940](https://github.com/xorq-labs/xorq/pull/940)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.10 by @renovate[bot] in [#941](https://github.com/xorq-labs/xorq/pull/941)
- Update codecov/codecov-action action to v5.4.3 by @renovate[bot] in [#944](https://github.com/xorq-labs/xorq/pull/944)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.4 by @renovate[bot] in [#945](https://github.com/xorq-labs/xorq/pull/945)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.5 by @renovate[bot] in [#949](https://github.com/xorq-labs/xorq/pull/949)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.6 by @renovate[bot] in [#952](https://github.com/xorq-labs/xorq/pull/952)
- Disable cargo manager by @mesejo in [#953](https://github.com/xorq-labs/xorq/pull/953)
- Disable dask dependency updates by @mesejo in [#956](https://github.com/xorq-labs/xorq/pull/956)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.11 by @renovate[bot] in [#960](https://github.com/xorq-labs/xorq/pull/960)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.7 by @renovate[bot] in [#961](https://github.com/xorq-labs/xorq/pull/961)
- Update dependency ruff to v0.11.11 by @renovate[bot] in [#959](https://github.com/xorq-labs/xorq/pull/959)
- Update dependency coverage to v7.8.1 by @renovate[bot] in [#957](https://github.com/xorq-labs/xorq/pull/957)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.8 by @renovate[bot] in [#964](https://github.com/xorq-labs/xorq/pull/964)
- Update bitnami/minio docker tag to v2025.5.24 by @renovate[bot] in [#967](https://github.com/xorq-labs/xorq/pull/967)
- Update dependency coverage to v7.8.2 by @renovate[bot] in [#963](https://github.com/xorq-labs/xorq/pull/963)
- Update duckdb darwin override by @hussainsultan in [#968](https://github.com/xorq-labs/xorq/pull/968)
- Update dependency quartodoc to <0.10.1,>=0.7.2 by @renovate[bot] in [#972](https://github.com/xorq-labs/xorq/pull/972)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.12 by @renovate[bot] in [#975](https://github.com/xorq-labs/xorq/pull/975)
- Update dependency ruff to v0.11.12 by @renovate[bot] in [#974](https://github.com/xorq-labs/xorq/pull/974)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.9 by @renovate[bot] in [#977](https://github.com/xorq-labs/xorq/pull/977)
- Update dependency ipython to <9.4.0,>=8.19.0 by @renovate[bot] in [#980](https://github.com/xorq-labs/xorq/pull/980)
- Update dependency pytest to v8.4.0 by @renovate[bot] in [#982](https://github.com/xorq-labs/xorq/pull/982)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.10 by @renovate[bot] in [#984](https://github.com/xorq-labs/xorq/pull/984)
- Update dependency python to v3.13.4 by @renovate[bot] in [#986](https://github.com/xorq-labs/xorq/pull/986)
- Make normalize_filenames accepts glob pattern by @mesejo in [#934](https://github.com/xorq-labs/xorq/pull/934)
- Versioned pins by @dlovell in [#991](https://github.com/xorq-labs/xorq/pull/991)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.11 by @renovate[bot] in [#989](https://github.com/xorq-labs/xorq/pull/989)
- No .gitignore in version control by @dlovell in [#994](https://github.com/xorq-labs/xorq/pull/994)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.12 by @renovate[bot] in [#998](https://github.com/xorq-labs/xorq/pull/998)
- Update trinodb/trino docker tag to v476 by @renovate[bot] in [#997](https://github.com/xorq-labs/xorq/pull/997)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.13 by @renovate[bot] in [#996](https://github.com/xorq-labs/xorq/pull/996)
- Update dependency ruff to v0.11.13 by @renovate[bot] in [#995](https://github.com/xorq-labs/xorq/pull/995)
- Update dependency quartodoc to <0.11.1,>=0.7.2 by @renovate[bot] in [#1003](https://github.com/xorq-labs/xorq/pull/1003)
- Update dependency blackdoc to v0.3.10 by @renovate[bot] in [#1004](https://github.com/xorq-labs/xorq/pull/1004)
- Update dependency quartodoc to <0.11.2,>=0.7.2 by @renovate[bot] in [#1006](https://github.com/xorq-labs/xorq/pull/1006)
- Update dependency coverage to v7.9.0 by @renovate[bot] in [#1007](https://github.com/xorq-labs/xorq/pull/1007)
- Update dependency pytest-cov to v6.2.0 by @renovate[bot] in [#1008](https://github.com/xorq-labs/xorq/pull/1008)
- Make FlightUDXF use mtls by default by @mesejo in [#1000](https://github.com/xorq-labs/xorq/pull/1000)
- Make udwf buildable by @mesejo in [#1005](https://github.com/xorq-labs/xorq/pull/1005)
- Update dependency pytest-cov to v6.2.1 by @renovate[bot] in [#1009](https://github.com/xorq-labs/xorq/pull/1009)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.13 by @renovate[bot] in [#1011](https://github.com/xorq-labs/xorq/pull/1011)
- Update dependency coverage to v7.9.1 by @renovate[bot] in [#1017](https://github.com/xorq-labs/xorq/pull/1017)
- Update dependency blackdoc to v0.4.0 by @renovate[bot] in [#1018](https://github.com/xorq-labs/xorq/pull/1018)
- Flight backend create by @hussainsultan in [#1021](https://github.com/xorq-labs/xorq/pull/1021)
- Update actions/create-github-app-token action to v1.12.0 by @renovate[bot] in [#1031](https://github.com/xorq-labs/xorq/pull/1031)
- Update astral-sh/setup-uv action to v6 by @renovate[bot] in [#1033](https://github.com/xorq-labs/xorq/pull/1033)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.0 by @renovate[bot] in [#1024](https://github.com/xorq-labs/xorq/pull/1024)
- Update dependency trino to v0.335.0 by @renovate[bot] in [#1023](https://github.com/xorq-labs/xorq/pull/1023)
- Update actions/create-github-app-token action to v2 by @renovate[bot] in [#1032](https://github.com/xorq-labs/xorq/pull/1032)
- Update extractions/setup-just action to v3 by @renovate[bot] in [#1034](https://github.com/xorq-labs/xorq/pull/1034)
- Update dependency ruff to v0.12.0 by @renovate[bot] in [#1022](https://github.com/xorq-labs/xorq/pull/1022)
- Update dependency pytest to v8.4.1 by @renovate[bot] in [#1029](https://github.com/xorq-labs/xorq/pull/1029)
- Set environment variables in ci-docs-preview workflow by @mesejo in [#1042](https://github.com/xorq-labs/xorq/pull/1042)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.14 by @renovate[bot] in [#1044](https://github.com/xorq-labs/xorq/pull/1044)
- Clean commands by @dlovell in [#1039](https://github.com/xorq-labs/xorq/pull/1039)
- Revert mintlify and migrate to netlify by @mesejo in [#1013](https://github.com/xorq-labs/xorq/pull/1013)
- Readme rewrite by @hussainsultan in [#1045](https://github.com/xorq-labs/xorq/pull/1045)
- Update astral-sh/setup-uv action to v6 by @renovate[bot] in [#1046](https://github.com/xorq-labs/xorq/pull/1046)
- Update extractions/setup-just action to v3 by @renovate[bot] in [#1047](https://github.com/xorq-labs/xorq/pull/1047)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.15 by @renovate[bot] in [#1050](https://github.com/xorq-labs/xorq/pull/1050)
- Update with penguins🐧 by @hussainsultan in [#1055](https://github.com/xorq-labs/xorq/pull/1055)
- Improve wording in README by @hussainsultan in [#1057](https://github.com/xorq-labs/xorq/pull/1057)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.1 by @renovate[bot] in [#1060](https://github.com/xorq-labs/xorq/pull/1060)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.17 by @renovate[bot] in [#1062](https://github.com/xorq-labs/xorq/pull/1062)
- Update xorq-datafusion version to 0.2.3 by @mesejo in [#1073](https://github.com/xorq-labs/xorq/pull/1073)
- Update dependency blackdoc to v0.4.1 by @renovate[bot] in [#1058](https://github.com/xorq-labs/xorq/pull/1058)
- Update dependency blackdoc to v0.4.1 by @renovate[bot] in [#1076](https://github.com/xorq-labs/xorq/pull/1076)
- Update dependency ruff to v0.12.1 by @renovate[bot] in [#1059](https://github.com/xorq-labs/xorq/pull/1059)
- Update dependency ipython to <9.5.0,>=8.19.0 by @renovate[bot] in [#1079](https://github.com/xorq-labs/xorq/pull/1079)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.18 by @renovate[bot] in [#1084](https://github.com/xorq-labs/xorq/pull/1084)
- Configure monthly PR creation by @mesejo in [#1085](https://github.com/xorq-labs/xorq/pull/1085)
- Make pyiceberg an optional by @dlovell in [#1035](https://github.com/xorq-labs/xorq/pull/1035)

#### Fixed
- Fix walking and checking by @dlovell in [#914](https://github.com/xorq-labs/xorq/pull/914)
- Classproperty by @dlovell in [#946](https://github.com/xorq-labs/xorq/pull/946)
- Update dependency fsspec to >=2024.6.1,<2025.5.1 by @renovate[bot] in [#954](https://github.com/xorq-labs/xorq/pull/954)
- Update dependency fsspec to >=2024.6.1,<2025.5.2 by @renovate[bot] in [#965](https://github.com/xorq-labs/xorq/pull/965)
- Update dependency datafusion to v47 by @renovate[bot] in [#971](https://github.com/xorq-labs/xorq/pull/971)
- Ensure expr results in a non-empty result by @dlovell in [#988](https://github.com/xorq-labs/xorq/pull/988)
- Xorqify all types by @dlovell in [#990](https://github.com/xorq-labs/xorq/pull/990)
- Clear out registry by @dlovell in [#1002](https://github.com/xorq-labs/xorq/pull/1002)
- Enable building of DatabaseTable by @mesejo in [#973](https://github.com/xorq-labs/xorq/pull/973)
- Use cloudpickle by @dlovell in [#1014](https://github.com/xorq-labs/xorq/pull/1014)
- Pass ibis schema instead of pyarrow by @hussainsultan in [#1037](https://github.com/xorq-labs/xorq/pull/1037)
- Store parquet files inside expr build dir by @mesejo in [#1010](https://github.com/xorq-labs/xorq/pull/1010)
- Use `large_binary` to intermediate state by @dlovell in [#1043](https://github.com/xorq-labs/xorq/pull/1043)
- Build for non SQL backends by @mesejo in [#1051](https://github.com/xorq-labs/xorq/pull/1051)
- Fix links in README by @mesejo in [#1052](https://github.com/xorq-labs/xorq/pull/1052)
- Uniform caching dir by @dlovell in [#1067](https://github.com/xorq-labs/xorq/pull/1067)
- Test_default_caching_dir on windows by @mesejo in [#1070](https://github.com/xorq-labs/xorq/pull/1070)
- Update 10_minutes ParquetStorage usage by @mesejo in [#1072](https://github.com/xorq-labs/xorq/pull/1072)
- Fix theme minor issues by @mesejo in [#1078](https://github.com/xorq-labs/xorq/pull/1078)
- Build stability by @dlovell in [#1064](https://github.com/xorq-labs/xorq/pull/1064)
- Replace InMemoryTable and DatabaseTable behind opaque ops by @mesejo in [#1080](https://github.com/xorq-labs/xorq/pull/1080)

#### Removed
- Remove read_sqlite by @mesejo in [#987](https://github.com/xorq-labs/xorq/pull/987)
- Remove dead code by @mesejo in [#1001](https://github.com/xorq-labs/xorq/pull/1001)
- Remove rust toolchain from nix configs by @hussainsultan in [#1020](https://github.com/xorq-labs/xorq/pull/1020)
- Remove to_pyarrow transformation by @mesejo in [#1040](https://github.com/xorq-labs/xorq/pull/1040)
- Remove empty example section by @mesejo in [#1086](https://github.com/xorq-labs/xorq/pull/1086)

## [0.2.2] - 2025-05-08
### Details

#### Added
- Support StringToDate in build by @mesejo in [#892](https://github.com/xorq-labs/xorq/pull/892)
- Add ArrayAny and ArrayAll to xorq backend by @mesejo in [#893](https://github.com/xorq-labs/xorq/pull/893)
- Add IsInf and IsNan ops by @mesejo in [#898](https://github.com/xorq-labs/xorq/pull/898)
- Add pyiceberg backend by @mesejo in [#910](https://github.com/xorq-labs/xorq/pull/910)

#### Changed
- Update dependency ruff to v0.11.7 by @renovate[bot] in [#883](https://github.com/xorq-labs/xorq/pull/883)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.7 by @renovate[bot] in [#884](https://github.com/xorq-labs/xorq/pull/884)
- Update astral-sh/setup-uv action to v6 by @renovate[bot] in [#880](https://github.com/xorq-labs/xorq/pull/880)
- Test library as an installed wheel by @mesejo in [#885](https://github.com/xorq-labs/xorq/pull/885)
- Update dependency ipython to <9.3.0,>=8.19.0 by @renovate[bot] in [#889](https://github.com/xorq-labs/xorq/pull/889)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.17 by @renovate[bot] in [#891](https://github.com/xorq-labs/xorq/pull/891)
- Ensure requests is installed for examples by @mesejo in [#890](https://github.com/xorq-labs/xorq/pull/890)
- Run tests mark with s3 in workflow by @mesejo in [#900](https://github.com/xorq-labs/xorq/pull/900)
- Ensure dependencies are resolved against PyPI index by @mesejo in [#901](https://github.com/xorq-labs/xorq/pull/901)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.0 by @renovate[bot] in [#902](https://github.com/xorq-labs/xorq/pull/902)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.2 by @renovate[bot] in [#903](https://github.com/xorq-labs/xorq/pull/903)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.8 by @renovate[bot] in [#905](https://github.com/xorq-labs/xorq/pull/905)
- Trace utils by @dlovell in [#874](https://github.com/xorq-labs/xorq/pull/874)
- Darwin override for grpcio by @hussainsultan in [#881](https://github.com/xorq-labs/xorq/pull/881)
- Update dependency ruff to v0.11.8 by @renovate[bot] in [#904](https://github.com/xorq-labs/xorq/pull/904)
- Update dependency trino to v0.334.0 by @renovate[bot] in [#908](https://github.com/xorq-labs/xorq/pull/908)
- Migrate to xorq-datafusion by @mesejo in [#792](https://github.com/xorq-labs/xorq/pull/792)
- Update astral-sh/setup-uv action to v6 by @renovate[bot] in [#911](https://github.com/xorq-labs/xorq/pull/911)
- Hook up kwargs in client.upload_data by @hussainsultan in [#917](https://github.com/xorq-labs/xorq/pull/917)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.7.3 by @renovate[bot] in [#921](https://github.com/xorq-labs/xorq/pull/921)

#### Fixed
- DevShells.impure: don't require commands to build by @dlovell in [#861](https://github.com/xorq-labs/xorq/pull/861)
- Register translation for PosixPath by @mesejo in [#888](https://github.com/xorq-labs/xorq/pull/888)
- Update dependency pyarrow to v20 by @renovate[bot] in [#895](https://github.com/xorq-labs/xorq/pull/895)
- Change _translate_type to translate yaml by @mesejo in [#897](https://github.com/xorq-labs/xorq/pull/897)
- Ensure inclusion of env_templates by @dlovell in [#906](https://github.com/xorq-labs/xorq/pull/906)
- Flight server issues by @mesejo in [#922](https://github.com/xorq-labs/xorq/pull/922)

#### Removed
- Remove double prefix by @mesejo in [#896](https://github.com/xorq-labs/xorq/pull/896)
- Remove old rust build related code by @dlovell in [#912](https://github.com/xorq-labs/xorq/pull/912)

## [0.2.1] - 2025-04-25
### Details
This release includes numerous additions, like MCP server support, GC Storage integration, and OTEL instrumentation,
alongside extensive updates to dependencies and developer tools through Renovate bot automation.
Multiple fixes addressed type handling, schema management, and file reading functionality across various object storage, including S3 and GCS. Several refactoring efforts improved code organization by factoring out helpers, enabling features like memtable, AggUDF,
and ExprScalarUDF while removing redundant components like the compiler and external dependencies.

#### Added
- Add mcp server example by @hussainsultan in [#731](https://github.com/xorq-labs/xorq/pull/731)
- Add files for pure-pypi uv env by @dlovell in [#742](https://github.com/xorq-labs/xorq/pull/742)
- Add workflow for checking PR titles by @mesejo in [#716](https://github.com/xorq-labs/xorq/pull/716)
- Add default value to expr_name by @dlovell in [#748](https://github.com/xorq-labs/xorq/pull/748)
- Add top level --pdb and drop into post mortem on failure by @dlovell in [#753](https://github.com/xorq-labs/xorq/pull/753)
- Additional checks by @dlovell in [#810](https://github.com/xorq-labs/xorq/pull/810)
- Add test for query schema action by @hussainsultan in [#822](https://github.com/xorq-labs/xorq/pull/822)
- Add normalization GCS path by @mesejo in [#829](https://github.com/xorq-labs/xorq/pull/829)
- Add GCStorage by @dlovell in [#838](https://github.com/xorq-labs/xorq/pull/838)
- Add virtualenv-default by @dlovell in [#841](https://github.com/xorq-labs/xorq/pull/841)
- Add read_record_batches to Snowflake Backend by @mesejo in [#834](https://github.com/xorq-labs/xorq/pull/834)
- Add env_utils, EnvConfigable by @dlovell in [#843](https://github.com/xorq-labs/xorq/pull/843)
- Add env vars to reduce uv actions by @dlovell in [#844](https://github.com/xorq-labs/xorq/pull/844)
- Add otel instrumentation by @dlovell in [#845](https://github.com/xorq-labs/xorq/pull/845)
- Add get_object_metadata to SessionContext by @mesejo in [#865](https://github.com/xorq-labs/xorq/pull/865)
- Add normalization for s3 and gcs objects by @mesejo in [#871](https://github.com/xorq-labs/xorq/pull/871)

#### Changed
- Update README with xorq info by @mesejo in [#741](https://github.com/xorq-labs/xorq/pull/741)
- Make github-actions dependencies pin by @mesejo in [#744](https://github.com/xorq-labs/xorq/pull/744)
- Update dependency node to v22 by @renovate[bot] in [#743](https://github.com/xorq-labs/xorq/pull/743)
- Pin dependencies by @renovate[bot] in [#745](https://github.com/xorq-labs/xorq/pull/745)
- Enable writing to stdout by @dlovell in [#757](https://github.com/xorq-labs/xorq/pull/757)
- Update to datafusion 46 by @mesejo in [#724](https://github.com/xorq-labs/xorq/pull/724)
- Define fixture fixture_dir by @dlovell in [#761](https://github.com/xorq-labs/xorq/pull/761)
- Enable renovatebot for pre-commit dependencies by @mesejo in [#770](https://github.com/xorq-labs/xorq/pull/770)
- Make git diff message explicit and descriptive by @mesejo in [#774](https://github.com/xorq-labs/xorq/pull/774)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.11 by @renovate[bot] in [#773](https://github.com/xorq-labs/xorq/pull/773)
- Upgrade requirements-dev.txt to uv 0.6.11 format by @mesejo in [#775](https://github.com/xorq-labs/xorq/pull/775)
- Update pre-commit hook codespell-project/codespell to v2.4.1 by @renovate[bot] in [#771](https://github.com/xorq-labs/xorq/pull/771)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.2 by @renovate[bot] in [#772](https://github.com/xorq-labs/xorq/pull/772)
- Enable xorq buildable by @dlovell in [#764](https://github.com/xorq-labs/xorq/pull/764)
- Update dependency coverage to v7.8.0 by @renovate[bot] in [#769](https://github.com/xorq-labs/xorq/pull/769)
- Centralised postgres connection fixture by @mesejo in [#776](https://github.com/xorq-labs/xorq/pull/776)
- Register Alias in ibis_yaml translate by @dlovell in [#783](https://github.com/xorq-labs/xorq/pull/783)
- Register Round by @dlovell in [#790](https://github.com/xorq-labs/xorq/pull/790)
- Update dependency pytest-cov to v6.1.0 by @renovate[bot] in [#791](https://github.com/xorq-labs/xorq/pull/791)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.12 by @renovate[bot] in [#796](https://github.com/xorq-labs/xorq/pull/796)
- Enable memtable by @dlovell in [#784](https://github.com/xorq-labs/xorq/pull/784)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.4 by @renovate[bot] in [#800](https://github.com/xorq-labs/xorq/pull/800)
- Update bitnami/minio docker tag to v2025.4.3 by @renovate[bot] in [#801](https://github.com/xorq-labs/xorq/pull/801)
- Update dependency ruff to v0.11.4 by @renovate[bot] in [#799](https://github.com/xorq-labs/xorq/pull/799)
- Enable memtable test by @mesejo in [#804](https://github.com/xorq-labs/xorq/pull/804)
- Enable `AggUDF` by @dlovell in [#807](https://github.com/xorq-labs/xorq/pull/807)
- Update dependency pytest-cov to v6.1.1 by @renovate[bot] in [#808](https://github.com/xorq-labs/xorq/pull/808)
- Factor out helpers by @dlovell in [#811](https://github.com/xorq-labs/xorq/pull/811)
- Make PR description compact by @mesejo in [#814](https://github.com/xorq-labs/xorq/pull/814)
- Update codecov badge in README by @mesejo in [#825](https://github.com/xorq-labs/xorq/pull/825)
- Rationalize helpers by @dlovell in [#818](https://github.com/xorq-labs/xorq/pull/818)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.13 by @renovate[bot] in [#821](https://github.com/xorq-labs/xorq/pull/821)
- Update dependency ipython to <9.2.0,>=8.19.0 by @renovate[bot] in [#815](https://github.com/xorq-labs/xorq/pull/815)
- Enable exprscalarudf by @dlovell in [#813](https://github.com/xorq-labs/xorq/pull/813)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.14 by @renovate[bot] in [#831](https://github.com/xorq-labs/xorq/pull/831)
- Update bitnami/minio docker tag to v2025.4.8 by @renovate[bot] in [#827](https://github.com/xorq-labs/xorq/pull/827)
- Update dependency ruff to v0.11.5 by @renovate[bot] in [#832](https://github.com/xorq-labs/xorq/pull/832)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.5 by @renovate[bot] in [#833](https://github.com/xorq-labs/xorq/pull/833)
- Use gcs built in uri ctor by @dlovell in [#839](https://github.com/xorq-labs/xorq/pull/839)
- Update google-github-actions/auth action to v2 by @renovate[bot] in [#835](https://github.com/xorq-labs/xorq/pull/835)
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.6 by @renovate[bot] in [#847](https://github.com/xorq-labs/xorq/pull/847)
- Update dependency ruff to v0.11.6 by @renovate[bot] in [#846](https://github.com/xorq-labs/xorq/pull/846)
- Update dependency ruff to v0.11.6 by @renovate[bot] in [#848](https://github.com/xorq-labs/xorq/pull/848)
- Update codecov/codecov-action action to v5.4.2 by @renovate[bot] in [#842](https://github.com/xorq-labs/xorq/pull/842)
- Gcstorage improvement by @dlovell in [#863](https://github.com/xorq-labs/xorq/pull/863)
- Update pre-commit hook astral-sh/uv-pre-commit to v0.6.16 by @renovate[bot] in [#864](https://github.com/xorq-labs/xorq/pull/864)
- Update bitnami/minio docker tag to v2025.4.22 by @renovate[bot] in [#869](https://github.com/xorq-labs/xorq/pull/869)
- Avoid double rust compilation and use cargo cache by @mesejo in [#873](https://github.com/xorq-labs/xorq/pull/873)
- Update trinodb/trino docker tag to v475 by @renovate[bot] in [#876](https://github.com/xorq-labs/xorq/pull/876)
- Use python 3.10 by @mesejo in [#877](https://github.com/xorq-labs/xorq/pull/877)

#### Fixed
- Specify the correct type when raising on incorrect type by @dlovell in [#747](https://github.com/xorq-labs/xorq/pull/747)
- Require explicit variable for tests by @dlovell in [#749](https://github.com/xorq-labs/xorq/pull/749)
- Read_csv_rbr: ensure Schema by @dlovell in [#751](https://github.com/xorq-labs/xorq/pull/751)
- Update dependency fsspec to >=2024.6.1,<2025.3.2 by @renovate[bot] in [#768](https://github.com/xorq-labs/xorq/pull/768)
- Test cli on select examples by @dlovell in [#763](https://github.com/xorq-labs/xorq/pull/763)
- Dont raise blocking raise by @dlovell in [#765](https://github.com/xorq-labs/xorq/pull/765)
- Read with schema by @dlovell in [#766](https://github.com/xorq-labs/xorq/pull/766)
- Fix Mod deserialization by @dlovell in [#778](https://github.com/xorq-labs/xorq/pull/778)
- Update dependency fsspec to >=2024.6.1,<2025.3.3 by @renovate[bot] in [#780](https://github.com/xorq-labs/xorq/pull/780)
- Read: capture read_kwargs by @dlovell in [#777](https://github.com/xorq-labs/xorq/pull/777)
- Typo in modification time caching strategy by @hussainsultan in [#803](https://github.com/xorq-labs/xorq/pull/803)
- Enable `CachedNode` by @dlovell in [#806](https://github.com/xorq-labs/xorq/pull/806)
- Sundry fixes by @dlovell in [#819](https://github.com/xorq-labs/xorq/pull/819)
- Update rust crate tokio to v1.44.2 [security] by @renovate[bot] in [#823](https://github.com/xorq-labs/xorq/pull/823)
- Prevent optional dependency leakage by @dlovell in [#840](https://github.com/xorq-labs/xorq/pull/840)
- Subprocess not monkeypatch by @dlovell in [#849](https://github.com/xorq-labs/xorq/pull/849)
- Enable __main__ `curry`/`functools.lru_cache` by @dlovell in [#856](https://github.com/xorq-labs/xorq/pull/856)
- Clean up api by @dlovell in [#862](https://github.com/xorq-labs/xorq/pull/862)
- Read_csv from s3 URL by @mesejo in [#852](https://github.com/xorq-labs/xorq/pull/852)
- Update otel_utils to last EnvConfigable version by @mesejo in [#870](https://github.com/xorq-labs/xorq/pull/870)
- Skip setting env variables in profiles.py example by @mesejo in [#875](https://github.com/xorq-labs/xorq/pull/875)
- Use env var contextmanager whenever get_con is invoked by @dlovell in [#879](https://github.com/xorq-labs/xorq/pull/879)
- Include env_templates files in wheel by @mesejo in [#882](https://github.com/xorq-labs/xorq/pull/882)

#### Removed
- Remove redundant compiler by @mesejo in [#756](https://github.com/xorq-labs/xorq/pull/756)
- Remove tenacity from examples by @mesejo in [#759](https://github.com/xorq-labs/xorq/pull/759)
- Remove commintlint by @mesejo in [#779](https://github.com/xorq-labs/xorq/pull/779)

## [0.2.0] - 2025-03-27
### Details
This release enhances xorq with ExprScalarUDF walk_nodes support, multi-duck vignette, import_from_gist functionality, 
and type annotations for deferred reading functions. Key changes centralize parquet fixtures and generalize OpenAI invocation. 
Fixes address Postgres HTTP Parquet reading, backend detection, command hashes, and dictionary registration. 
PyO3 warnings and unused exchanger components were removed.

#### Added
- Add ExprScalarUDF to walk_nodes by @dlovell in [#693](https://github.com/xorq-labs/xorq/pull/693)
- Add hn part4 by @hussainsultan in [#713](https://github.com/xorq-labs/xorq/pull/713)
- Add multi-duck vignette by @hussainsultan in [#719](https://github.com/xorq-labs/xorq/pull/719)
- Add import_from_gist by @dlovell in [#733](https://github.com/xorq-labs/xorq/pull/733)
- Add uv section by @mesejo in [#730](https://github.com/xorq-labs/xorq/pull/730)
- Add type annotations to deferred_read_csv and deferred_read_parquet by @soheil-star01 in [#738](https://github.com/xorq-labs/xorq/pull/738)

#### Changed
- ParquetStorage to build and run by @mesejo in [#684](https://github.com/xorq-labs/xorq/pull/684)
- Centralize definition of parquet_dir fixture by @dlovell in [#722](https://github.com/xorq-labs/xorq/pull/722)
- Generalize openai invocation by @dlovell in [#732](https://github.com/xorq-labs/xorq/pull/732)
- Connect_examples pass kwargs by @dlovell in [#736](https://github.com/xorq-labs/xorq/pull/736)

#### Fixed
- Postgres deferred_read_parquet fails for http source by @mesejo in [#708](https://github.com/xorq-labs/xorq/pull/708)
- Spurious backend in _find_backend by @mesejo in [#717](https://github.com/xorq-labs/xorq/pull/717)
- Update expected command hashes by @dlovell in [#734](https://github.com/xorq-labs/xorq/pull/734)
- Register dict by @dlovell in [#735](https://github.com/xorq-labs/xorq/pull/735)

#### Removed
- Remove PyO3 migration warnings by @mesejo in [#707](https://github.com/xorq-labs/xorq/pull/707)
- Remove unused exchanger by @mesejo in [#715](https://github.com/xorq-labs/xorq/pull/715)

## New Contributors
* @soheil-star01 made their first contribution in [#738](https://github.com/xorq-labs/xorq/pull/738)

## [0.1.17] - 2025-03-24
### Details
This release enhances xorq documentation with new quickstart guides, HN dataset tutorials, normalization helpers, 
In the functionality side we added ListActionsAction. We upgraded to DataFusion 45, improved testing infrastructure, and refined core concepts documentation. 
Key fixes address expression optimization, Darwin builds for Python 3.10/3.11, and various dependency updates to ruff, async-trait, and arrow.

#### Added
- Add quickstart by @hussainsultan in [#654](https://github.com/xorq-labs/xorq/pull/654)
- Add helpers to identify normalization issues by @dlovell in [#669](https://github.com/xorq-labs/xorq/pull/669)
- Add exchangers to FlightServer constructor by @dlovell in [#671](https://github.com/xorq-labs/xorq/pull/671)
- Add hn quickstart example by @hussainsultan in [#675](https://github.com/xorq-labs/xorq/pull/675)
- Add part 1 of hn data tutorial by @hussainsultan in [#661](https://github.com/xorq-labs/xorq/pull/661)
- Add hn data tutorial part 2 by @hussainsultan in [#663](https://github.com/xorq-labs/xorq/pull/663)
- ToolsPackages: add gh by @dlovell in [#673](https://github.com/xorq-labs/xorq/pull/673)
- Add ListActionsAction by @dlovell in [#706](https://github.com/xorq-labs/xorq/pull/706)

#### Changed
- Update dependency ruff to v0.10.0 by @renovate[bot] in [#653](https://github.com/xorq-labs/xorq/pull/653)
- Update extractions/setup-just action to v3 by @renovate[bot] in [#658](https://github.com/xorq-labs/xorq/pull/658)
- Update dependency ruff to v0.11.0 by @renovate[bot] in [#657](https://github.com/xorq-labs/xorq/pull/657)
- Update CONTRIBUTING.md by @dlovell in [#664](https://github.com/xorq-labs/xorq/pull/664)
- Mark snapshot tests by @dlovell in [#665](https://github.com/xorq-labs/xorq/pull/665)
- Update dependency coverage to v7.7.0 by @renovate[bot] in [#666](https://github.com/xorq-labs/xorq/pull/666)
- Update readme to use similar language as docs by @hussainsultan in [#674](https://github.com/xorq-labs/xorq/pull/674)
- Update dependency pre-commit to v4.2.0 by @renovate[bot] in [#681](https://github.com/xorq-labs/xorq/pull/681)
- Update trinodb/trino docker tag to v473 by @renovate[bot] in [#685](https://github.com/xorq-labs/xorq/pull/685)
- Update to datafusion 45 by @mesejo in [#672](https://github.com/xorq-labs/xorq/pull/672)
- Change url in test_read_csv_from_url by @mesejo in [#688](https://github.com/xorq-labs/xorq/pull/688)
- Move tests to xorq/tests by @mesejo in [#689](https://github.com/xorq-labs/xorq/pull/689)
- Improve core concepts by @hussainsultan in [#686](https://github.com/xorq-labs/xorq/pull/686)
- Update dependency ruff to v0.11.1 by @renovate[bot] in [#691](https://github.com/xorq-labs/xorq/pull/691)
- Update dependency ruff to v0.11.2 by @renovate[bot] in [#698](https://github.com/xorq-labs/xorq/pull/698)
- Convert expr to table by @mesejo in [#695](https://github.com/xorq-labs/xorq/pull/695)
- Update trinodb/trino docker tag to v474 by @renovate[bot] in [#702](https://github.com/xorq-labs/xorq/pull/702)
- Update dependency coverage to v7.7.1 by @renovate[bot] in [#697](https://github.com/xorq-labs/xorq/pull/697)
- Hotfix darwin build for 3.10,3.11 by @dlovell in [#704](https://github.com/xorq-labs/xorq/pull/704)

#### Fixed
- Update rust crate async-trait to v0.1.88 by @renovate[bot] in [#656](https://github.com/xorq-labs/xorq/pull/656)
- Invoke inspect when explicitly requested in normalize_seq_with_caller by @dlovell in [#668](https://github.com/xorq-labs/xorq/pull/668)
- Warning for not using raw string in regex by @ghoersti in [#676](https://github.com/xorq-labs/xorq/pull/676)
- Update flake inputs to build with rust 1.85.0 by @dlovell in [#690](https://github.com/xorq-labs/xorq/pull/690)
- Avoid duplicated execution of expr by removing read_all by @mesejo in [#677](https://github.com/xorq-labs/xorq/pull/677)
- Update rust crate arrow to v54.3.0 by @renovate[bot] in [#700](https://github.com/xorq-labs/xorq/pull/700)
- Make action_body a dict by @mesejo in [#692](https://github.com/xorq-labs/xorq/pull/692)

## [0.1.16] - 2025-03-13
### Details
#### Added
- Add tutorials intro section by @mesejo in [#634](https://github.com/xorq-labs/xorq/pull/634)

#### Changed
- Update bitnami/minio docker tag to v2025.3.12 by @renovate[bot] in [#649](https://github.com/xorq-labs/xorq/pull/649)
- Update tutorial with new sklearn functions by @mesejo in [#645](https://github.com/xorq-labs/xorq/pull/645)
- Expose deferred_reads as top level by @mesejo in [#646](https://github.com/xorq-labs/xorq/pull/646)
- Make udf top level by @mesejo in [#647](https://github.com/xorq-labs/xorq/pull/647)

#### Fixed
- Infer suffix in read_(parquet/csv) by @mesejo in [#639](https://github.com/xorq-labs/xorq/pull/639)
- Special case for postgres in case of complex types by @dlovell in [#644](https://github.com/xorq-labs/xorq/pull/644)
- Update rust crate tokio to v1.44.1 by @renovate[bot] in [#650](https://github.com/xorq-labs/xorq/pull/650)

## [0.1.15] - 2025-03-11
### Details
This release adds a new CLI to xorq. Additionally, it improves the architecture 
through consolidated serialization, better caching, and an improved Flight server implementation. Dependencies were updated across the board, 
the codebase was streamlined by removing unnecessary components, and several critical bugs were fixed to improve stability and reliability.

#### Added
- FlightServer: add con and client as properties by @dlovell in [#568](https://github.com/xorq-labs/xorq/pull/568)
- Add xorq build cli command by @mesejo in [#567](https://github.com/xorq-labs/xorq/pull/567)
- Add importing of notebooks by @mesejo in [#577](https://github.com/xorq-labs/xorq/pull/577)
- Add deferred_fit_predict by @dlovell in [#595](https://github.com/xorq-labs/xorq/pull/595)
- Add overview image by @hussainsultan in [#605](https://github.com/xorq-labs/xorq/pull/605)
- Add to_json/to_csv by @mesejo in [#604](https://github.com/xorq-labs/xorq/pull/604)
- Add run cli command by @mesejo in [#581](https://github.com/xorq-labs/xorq/pull/581)
- Add examples tab by @mesejo in [#608](https://github.com/xorq-labs/xorq/pull/608)
- Add run section of build and run tutorial by @mesejo in [#637](https://github.com/xorq-labs/xorq/pull/637)

#### Changed
- Update dependency ruff to v0.9.7 by @renovate[bot] in [#558](https://github.com/xorq-labs/xorq/pull/558)
- Consolidate YAML/SQL serialization by @hussainsultan in [#525](https://github.com/xorq-labs/xorq/pull/525)
- Move custom types to source file by @mesejo in [#566](https://github.com/xorq-labs/xorq/pull/566)
- Extract postgres extra by @mesejo in [#571](https://github.com/xorq-labs/xorq/pull/571)
- Update codecov/codecov-action action to v5.4.0 by @renovate[bot] in [#574](https://github.com/xorq-labs/xorq/pull/574)
- Update cli docstring by @mesejo in [#575](https://github.com/xorq-labs/xorq/pull/575)
- Migrate docs to mintlify by @mesejo in [#541](https://github.com/xorq-labs/xorq/pull/541)
- Update readme by @hussainsultan in [#580](https://github.com/xorq-labs/xorq/pull/580)
- Change api by @dlovell in [#582](https://github.com/xorq-labs/xorq/pull/582)
- Update dependency ruff to v0.9.8 by @renovate[bot] in [#583](https://github.com/xorq-labs/xorq/pull/583)
- Replace pydantic AnyURL with attrs by @mesejo in [#573](https://github.com/xorq-labs/xorq/pull/573)
- Update dependency ruff to v0.9.9 by @renovate[bot] in [#585](https://github.com/xorq-labs/xorq/pull/585)
- Update dependency ipython to v9 by @renovate[bot] in [#586](https://github.com/xorq-labs/xorq/pull/586)
- Update bitnami/minio docker tag to v2025.2.28 by @renovate[bot] in [#587](https://github.com/xorq-labs/xorq/pull/587)
- Default pdb to ipython TerminalPdb by @dlovell in [#590](https://github.com/xorq-labs/xorq/pull/590)
- Update dependency pytest to v8.3.5 by @renovate[bot] in [#591](https://github.com/xorq-labs/xorq/pull/591)
- Create cache module by @mesejo in [#579](https://github.com/xorq-labs/xorq/pull/579)
- Update rust toolchain version by @mesejo in [#597](https://github.com/xorq-labs/xorq/pull/597)
- Profile docs& guide by @ghoersti in [#602](https://github.com/xorq-labs/xorq/pull/602)
- Update overview page with benefits and painpoints by @hussainsultan in [#607](https://github.com/xorq-labs/xorq/pull/607)
- Update for ParquetCacheStorage -> ParquetStorage change by @dlovell in [#606](https://github.com/xorq-labs/xorq/pull/606)
- Use connect_env by @mesejo in [#611](https://github.com/xorq-labs/xorq/pull/611)
- Update overview page by @hussainsultan in [#615](https://github.com/xorq-labs/xorq/pull/615)
- Repr of opaque nodes by @mesejo in [#596](https://github.com/xorq-labs/xorq/pull/596)
- Refactor FlightServer code by @mesejo in [#616](https://github.com/xorq-labs/xorq/pull/616)
- Update dependency ruff to v0.9.10 by @renovate[bot] in [#617](https://github.com/xorq-labs/xorq/pull/617)
- Update trinodb/trino docker tag to v472 by @renovate[bot] in [#622](https://github.com/xorq-labs/xorq/pull/622)
- Bump ring from 0.17.8 to 0.17.13 by @dependabot[bot] in [#626](https://github.com/xorq-labs/xorq/pull/626)
- Expose datatypes as part of xorq by @mesejo in [#578](https://github.com/xorq-labs/xorq/pull/578)
- Enable serde for connections by @dlovell in [#630](https://github.com/xorq-labs/xorq/pull/630)

#### Fixed
- Update dependency datafusion to v45 by @renovate[bot] in [#563](https://github.com/xorq-labs/xorq/pull/563)
- Update rust crate async-trait to v0.1.87 by @renovate[bot] in [#594](https://github.com/xorq-labs/xorq/pull/594)
- Dedupe yaml serialization by @hussainsultan in [#589](https://github.com/xorq-labs/xorq/pull/589)
- Default to dynamic FlightUrl port by @dlovell in [#603](https://github.com/xorq-labs/xorq/pull/603)
- Default_backend should xorq by @mesejo in [#609](https://github.com/xorq-labs/xorq/pull/609)
- Flight caching by @dlovell in [#610](https://github.com/xorq-labs/xorq/pull/610)
- Mark remote tables so clean up won't interfere with others by @dlovell in [#613](https://github.com/xorq-labs/xorq/pull/613)
- Update dependency fsspec to >=2024.6.1,<2025.3.1 by @renovate[bot] in [#618](https://github.com/xorq-labs/xorq/pull/618)
- Update rust crate tokio to v1.44.0 by @renovate[bot] in [#621](https://github.com/xorq-labs/xorq/pull/621)
- Update rust crate arrow-ord to v53.4.1 by @renovate[bot] in [#620](https://github.com/xorq-labs/xorq/pull/620)

#### Removed
- Remove predict_xgb related code by @mesejo in [#562](https://github.com/xorq-labs/xorq/pull/562)
- Remove sqlalchemy dependency by @mesejo in [#572](https://github.com/xorq-labs/xorq/pull/572)
- Remove netlify workflows by @mesejo in [#576](https://github.com/xorq-labs/xorq/pull/576)

## [0.1.14] - 2025-02-20
### Details
This release marks the project's rebranding to xorq, accompanied by significant architectural improvements including the 
addition of FlightServer and ExprScalarUDF. The update integrates ibis functionality, enhances logging systems, 
and includes various dependency updates. Notable bug fixes address Python identifier validation and hash stability, 
while unnecessary dependencies have been removed to streamline the codebase.

#### Added
- Add FlightServer by @mesejo in [#474](https://github.com/letsql/letsql/pull/474)
- Add ordering arg to register_record_batch_reader by @mesejo in [#479](https://github.com/letsql/letsql/pull/479)
- Add ExprScalarUDF by @dlovell in [#532](https://github.com/letsql/letsql/pull/532)

#### Changed
- Refactor out hotfixing snowflake by @mesejo in [#509](https://github.com/letsql/letsql/pull/509)
- Update trinodb/trino docker tag to v470 by @renovate[bot] in [#512](https://github.com/letsql/letsql/pull/512)
- Update dependency ruff to v0.9.5 by @renovate[bot] in [#514](https://github.com/letsql/letsql/pull/514)
- Update bitnami/minio docker tag to v2025.2.7 by @renovate[bot] in [#526](https://github.com/letsql/letsql/pull/526)
- Update dependency coverage to v7.6.11 by @renovate[bot] in [#527](https://github.com/letsql/letsql/pull/527)
- Use get_print_logger instead of print by @dlovell in [#533](https://github.com/letsql/letsql/pull/533)
- Update dependency coverage to v7.6.12 by @renovate[bot] in [#530](https://github.com/letsql/letsql/pull/530)
- Update postgres docker tag to v17.3 by @renovate[bot] in [#535](https://github.com/letsql/letsql/pull/535)
- Register ibis datatypes, raise if not registered by @mesejo in [#538](https://github.com/letsql/letsql/pull/538)
- Transform datetime to bytes by @mesejo in [#540](https://github.com/letsql/letsql/pull/540)
- Vendor ibis by @mesejo in [#529](https://github.com/letsql/letsql/pull/529)
- Update actions/create-github-app-token action to v1.11.5 by @renovate[bot] in [#539](https://github.com/letsql/letsql/pull/539)
- Update dependency ruff to v0.9.6 by @renovate[bot] in [#528](https://github.com/letsql/letsql/pull/528)
- Wrap pins with examples by @mesejo in [#511](https://github.com/letsql/letsql/pull/511)
- Enable kwargless use of default name by @dlovell in [#545](https://github.com/letsql/letsql/pull/545)
- Clean vendoring by @mesejo in [#546](https://github.com/letsql/letsql/pull/546)
- Update bitnami/minio docker tag to v2025.2.18 by @renovate[bot] in [#554](https://github.com/letsql/letsql/pull/554)
- Update trinodb/trino docker tag to v471 by @renovate[bot] in [#555](https://github.com/letsql/letsql/pull/555)
- Rename to xorq by @mesejo in [#550](https://github.com/letsql/letsql/pull/550)

#### Fixed
- Convert name to valid Python identifier by @mesejo in [#508](https://github.com/letsql/letsql/pull/508)
- Update rust crate prost to v0.13.5 by @renovate[bot] in [#531](https://github.com/letsql/letsql/pull/531)
- Ensure stable hash for normalize_read by @dlovell in [#549](https://github.com/letsql/letsql/pull/549)

#### Removed
- Remove unwanted dependencies by @mesejo in [#551](https://github.com/letsql/letsql/pull/551)
- Remove replace_fix wrapper by @mesejo in [#553](https://github.com/letsql/letsql/pull/553)

## [0.1.13] - 2025-02-05
### Details
Enable caching for BigQuery

#### Added
- Add quickgrove example by @hussainsultan in [#499](https://github.com/letsql/letsql/pull/499)
- Add quickgrove to examples optional by @mesejo in [#501](https://github.com/letsql/letsql/pull/501)

#### Changed
- Enable caching bigquery exprs by @dlovell in [#486](https://github.com/letsql/letsql/pull/486)
- Snowflake-connector-python security update by @mesejo in [#489](https://github.com/letsql/letsql/pull/489)
- Update actions/create-github-app-token action to v1.11.2 by @renovate[bot] in [#492](https://github.com/letsql/letsql/pull/492)
- Update dependency ruff to v0.9.4 by @renovate[bot] in [#493](https://github.com/letsql/letsql/pull/493)
- Enable running entirely from pypi by @dlovell
- Invoke nix fmt by @dlovell
- Update dependency trino to v0.333.0 by @renovate[bot] in [#502](https://github.com/letsql/letsql/pull/502)
- Update actions/create-github-app-token action to v1.11.3 by @renovate[bot] in [#505](https://github.com/letsql/letsql/pull/505)
- Update bitnami/minio docker tag to v2025.2.3 by @renovate[bot] in [#507](https://github.com/letsql/letsql/pull/507)

#### Fixed
- Update rust crate async-trait to v0.1.86 by @renovate[bot] in [#494](https://github.com/letsql/letsql/pull/494)
- Update dependency fsspec to v2025 by @renovate[bot] in [#495](https://github.com/letsql/letsql/pull/495)
- Enable no rust build drv by @dlovell
- Ls.register: invoke correct method by @dlovell in [#500](https://github.com/letsql/letsql/pull/500)

#### Removed
- Remove poetry.lock by @mesejo in [#488](https://github.com/letsql/letsql/pull/488)
- Remove optimizer module by @mesejo in [#506](https://github.com/letsql/letsql/pull/506)
- Remove hotfix for Postgres by @mesejo in [#503](https://github.com/letsql/letsql/pull/503)
- Remove hotfix pandas by @mesejo in [#504](https://github.com/letsql/letsql/pull/504)

## [0.1.12] - 2025-01-29
### Details
This release introduces several key features including segmentation support, SQL caching optimization, a new hash(string) function, and QuickGrove UDF integration. 
Notable infrastructure improvements include upgrading to DataFusion v44, adding LargeUtf8 as a DataType, and implementing a standalone UV + Rust shell. 
The codebase underwent maintenance with multiple dependency updates and quality improvements, while fixes addressed DuckDB integration and documentation issues. 
Several components were removed for cleanup, including pre-register functionality and MarkedRemoteTable.

#### Added
- Add segmentation by @mesejo in [#389](https://github.com/letsql/letsql/pull/389)
- Add caching for to_sql by @mesejo in [#404](https://github.com/letsql/letsql/pull/404)
- Add hash(string) function by @mesejo in [#451](https://github.com/letsql/letsql/pull/451)
- Add get_plans by @dlovell in [#447](https://github.com/letsql/letsql/pull/447)
- Add quickgrove udf by @hussainsultan in [#475](https://github.com/letsql/letsql/pull/475)
- Add LargeUtf8 as a DataType by @mesejo in [#462](https://github.com/letsql/letsql/pull/462)
- Add requirement for letsql-pytest by @dlovell
- Add stand alone uv + rust shell by @dlovell in [#482](https://github.com/letsql/letsql/pull/482)

#### Changed
- Update trinodb/trino docker tag to v468 by @renovate[bot] in [#421](https://github.com/letsql/letsql/pull/421)
- Update astral-sh/setup-uv action to v5 by @renovate[bot] in [#420](https://github.com/letsql/letsql/pull/420)
- Update actions/create-github-app-token action to v1.11.1 by @renovate[bot] in [#414](https://github.com/letsql/letsql/pull/414)
- Update bitnami/minio docker tag to v2024.12.18 by @renovate[bot] in [#415](https://github.com/letsql/letsql/pull/415)
- Update codecov/codecov-action action to v5.1.2 by @renovate[bot] in [#416](https://github.com/letsql/letsql/pull/416)
- Update dependency ruff to v0.8.6 by @renovate[bot] in [#417](https://github.com/letsql/letsql/pull/417)
- Update to datafusion v44 by @mesejo in [#435](https://github.com/letsql/letsql/pull/435)
- Update dependency coverage to v7.6.10 by @renovate[bot] in [#424](https://github.com/letsql/letsql/pull/424)
- Uv lock --upgrade-package jinja2 by @mesejo in [#436](https://github.com/letsql/letsql/pull/436)
- Update contributing workflow by @mesejo in [#437](https://github.com/letsql/letsql/pull/437)
- Update dependency ruff to v0.9.0 by @renovate[bot] in [#439](https://github.com/letsql/letsql/pull/439)
- Expose full SessionConfig by @mesejo in [#440](https://github.com/letsql/letsql/pull/440)
- Update dependency ruff to v0.9.1 by @renovate[bot] in [#441](https://github.com/letsql/letsql/pull/441)
- Update dependency trino to v0.332.0 by @renovate[bot] in [#449](https://github.com/letsql/letsql/pull/449)
- Update dependency ruff to v0.9.2 by @renovate[bot] in [#453](https://github.com/letsql/letsql/pull/453)
- Update dependency pre-commit to v4.1.0 by @renovate[bot] in [#463](https://github.com/letsql/letsql/pull/463)
- Update dependency codespell to v2.4.0 by @renovate[bot] in [#468](https://github.com/letsql/letsql/pull/468)
- Update codecov/codecov-action action to v5.2.0 by @renovate[bot] in [#469](https://github.com/letsql/letsql/pull/469)
- Update dependency ruff to v0.9.3 by @renovate[bot] in [#471](https://github.com/letsql/letsql/pull/471)
- Update codecov/codecov-action action to v5.3.1 by @renovate[bot] in [#472](https://github.com/letsql/letsql/pull/472)
- Enforce import order by @mesejo in [#476](https://github.com/letsql/letsql/pull/476)
- Update trinodb/trino docker tag to v469 by @renovate[bot] in [#478](https://github.com/letsql/letsql/pull/478)
- Update dependency codespell to v2.4.1 by @renovate[bot] in [#481](https://github.com/letsql/letsql/pull/481)
- Update dependency black to v25 by @renovate[bot] in [#483](https://github.com/letsql/letsql/pull/483)
- Update bitnami/minio docker tag to v2025 by @renovate[bot] in [#484](https://github.com/letsql/letsql/pull/484)
- Extend ruff select rules instead of overriding by @mesejo in [#485](https://github.com/letsql/letsql/pull/485)

#### Fixed
- Update rust crate object_store to v0.11.2 by @renovate[bot] in [#422](https://github.com/letsql/letsql/pull/422)
- Update rust crate async-trait to v0.1.84 by @renovate[bot] in [#425](https://github.com/letsql/letsql/pull/425)
- Update dependency dask to v2024.12.1 by @renovate[bot] in [#418](https://github.com/letsql/letsql/pull/418)
- Update dependency fsspec to >=2024.6.1,<2024.12.1 by @renovate[bot] in [#419](https://github.com/letsql/letsql/pull/419)
- Update rust crate async-trait to v0.1.85 by @renovate[bot] in [#432](https://github.com/letsql/letsql/pull/432)
- Update rust crate tokio to v1.43.0 by @renovate[bot] in [#438](https://github.com/letsql/letsql/pull/438)
- Invoke maturin on changes to rust by @dlovell in [#445](https://github.com/letsql/letsql/pull/445)
- Identify correct kwarg for duckdb by @dlovell in [#452](https://github.com/letsql/letsql/pull/452)
- Update dependency pyarrow to v19 by @renovate[bot] in [#454](https://github.com/letsql/letsql/pull/454)
- Update dependency structlog to v25 by @renovate[bot] in [#455](https://github.com/letsql/letsql/pull/455)
- Update dependency dask to v2025 by @renovate[bot] in [#458](https://github.com/letsql/letsql/pull/458)
- Failing docs deployment by @mesejo in [#465](https://github.com/letsql/letsql/pull/465)
- Fix dependency groups by @mesejo in [#448](https://github.com/letsql/letsql/pull/448)
- Get_storage_uncached args by @mesejo in [#466](https://github.com/letsql/letsql/pull/466)
- Update dependency attrs to v25 by @renovate[bot] in [#473](https://github.com/letsql/letsql/pull/473)
- Ensure dependencies are in sync by @mesejo in [#480](https://github.com/letsql/letsql/pull/480)
- Update for changes to uv/uv2nix by @dlovell

#### Removed
- Remove pre-register by @mesejo in [#450](https://github.com/letsql/letsql/pull/450)
- Remove warnings by @mesejo in [#457](https://github.com/letsql/letsql/pull/457)
- Remove uv.lock from git diff by @mesejo in [#467](https://github.com/letsql/letsql/pull/467)
- Remove MarkedRemoteTable by @mesejo in [#459](https://github.com/letsql/letsql/pull/459)

## [0.1.11] - 2024-12-17
### Details
Improve CI/CD, migrate to uv, update dependencies. 

#### Added
- Add reading parquet from http by @mesejo in [#370](https://github.com/letsql/letsql/pull/370)
- Add codspeed by @mesejo in [#374](https://github.com/letsql/letsql/pull/374)

#### Changed
- Update dependency ruff to v0.7.4 by @renovate[bot] in [#353](https://github.com/letsql/letsql/pull/353)
- Update codecov/codecov-action action to v5.0.2 by @renovate[bot] in [#355](https://github.com/letsql/letsql/pull/355)
- Update dependency coverage to v7.6.7 by @renovate[bot] in [#357](https://github.com/letsql/letsql/pull/357)
- Pin cargo dependencies version by @mesejo in [#358](https://github.com/letsql/letsql/pull/358)
- Update codecov/codecov-action action to v5.0.4 by @renovate[bot] in [#363](https://github.com/letsql/letsql/pull/363)
- Update codecov/codecov-action action to v5.0.7 by @renovate[bot] in [#365](https://github.com/letsql/letsql/pull/365)
- Update trinodb/trino docker tag to v465 by @renovate[bot] in [#369](https://github.com/letsql/letsql/pull/369)
- Update dependency ruff to v0.8.0 by @renovate[bot] in [#371](https://github.com/letsql/letsql/pull/371)
- Udf.agg.pandas_df: enableuse as a decorator by @dlovell in [#375](https://github.com/letsql/letsql/pull/375)
- Use bare datafusion by @mesejo in [#346](https://github.com/letsql/letsql/pull/346)
- Update dependency coverage to v7.6.8 by @renovate[bot] in [#378](https://github.com/letsql/letsql/pull/378)
- Update trinodb/trino docker tag to v466 by @renovate[bot] in [#382](https://github.com/letsql/letsql/pull/382)
- Update postgres docker tag to v17.2 by @renovate[bot] in [#377](https://github.com/letsql/letsql/pull/377)
- Update dependency ruff to v0.8.1 by @renovate[bot] in [#383](https://github.com/letsql/letsql/pull/383)
- Update dependency pytest to v8.3.4 by @renovate[bot] in [#384](https://github.com/letsql/letsql/pull/384)
- Move to uv by @mesejo in [#380](https://github.com/letsql/letsql/pull/380)
- Update trinodb/trino docker tag to v467 by @renovate[bot] in [#395](https://github.com/letsql/letsql/pull/395)
- Update dependency coverage to v7.6.9 by @renovate[bot] in [#392](https://github.com/letsql/letsql/pull/392)
- Update codecov/codecov-action action to v5.1.1 by @renovate[bot] in [#391](https://github.com/letsql/letsql/pull/391)
- Update astral-sh/setup-uv action to v4 by @renovate[bot] in [#394](https://github.com/letsql/letsql/pull/394)
- Update dependency ruff to v0.8.2 by @renovate[bot] in [#390](https://github.com/letsql/letsql/pull/390)
- Enable codspeed profiling by @mesejo in [#399](https://github.com/letsql/letsql/pull/399)
- Use python 3.12 in codspeed by @mesejo in [#400](https://github.com/letsql/letsql/pull/400)
- Update dependency trino to v0.331.0 by @renovate[bot] in [#401](https://github.com/letsql/letsql/pull/401)
- Enforce letsql as ls alias import by @mesejo in [#403](https://github.com/letsql/letsql/pull/403)
- Use cache for compiling rust by @mesejo in [#406](https://github.com/letsql/letsql/pull/406)
- Use maturin action with sccache by @mesejo in [#410](https://github.com/letsql/letsql/pull/410)
- Update dependency ruff to v0.8.3 by @renovate[bot] in [#407](https://github.com/letsql/letsql/pull/407)
- Update actions/cache action to v4 by @renovate[bot] in [#409](https://github.com/letsql/letsql/pull/409)
- Update bitnami/minio docker tag to v2024.12.13 by @renovate[bot] in [#411](https://github.com/letsql/letsql/pull/411)

#### Fixed
- Update rust crate datafusion to v43 by @renovate[bot] in [#340](https://github.com/letsql/letsql/pull/340)
- Update dependency ibis-framework to v9.5.0 by @renovate[bot] in [#324](https://github.com/letsql/letsql/pull/324)
- Pin dependencies by @renovate[bot] in [#359](https://github.com/letsql/letsql/pull/359)
- Update rust crate object_store to v0.11.1 by @renovate[bot] in [#360](https://github.com/letsql/letsql/pull/360)
- Update aws-sdk-rust monorepo to v0.101.0 by @renovate[bot] in [#361](https://github.com/letsql/letsql/pull/361)
- Update rust crate arrow-ord to v53.3.0 by @renovate[bot] in [#368](https://github.com/letsql/letsql/pull/368)
- Update rust crate arrow to v53.3.0 by @renovate[bot] in [#367](https://github.com/letsql/letsql/pull/367)
- Fix performance regression by only parsing metadata once by @dlovell in [#373](https://github.com/letsql/letsql/pull/373)
- Update rust crate url to v2.5.4 by @renovate[bot] in [#376](https://github.com/letsql/letsql/pull/376)
- Update rust crate tokio to v1.42.0 by @renovate[bot] in [#386](https://github.com/letsql/letsql/pull/386)
- Update dependency dask to v2024.12.0 by @renovate[bot] in [#387](https://github.com/letsql/letsql/pull/387)
- Dynamically generate `none_tokenized` by @dlovell in [#396](https://github.com/letsql/letsql/pull/396)
- Update rust crate prost to v0.13.4 by @renovate[bot] in [#393](https://github.com/letsql/letsql/pull/393)
- To_pyarrow_batches by @mesejo in [#398](https://github.com/letsql/letsql/pull/398)
- Update dependency datafusion to v43 by @renovate[bot] in [#408](https://github.com/letsql/letsql/pull/408)

#### Removed
- Remove xfail markers, clean warnings by @mesejo in [#385](https://github.com/letsql/letsql/pull/385)

## [0.1.10] - 2024-11-15
### Details
This release introduces UDWF and RemoteTables functionality.

#### Added
- Add test for to_pyarrow and to_pyarrow_batches by @mesejo in [#325](https://github.com/letsql/letsql/pull/325)
- Add udwf by @dlovell in [#354](https://github.com/letsql/letsql/pull/354)

#### Changed
- Reparametrize caching into storage and invalidation strategy by @dlovell in [#278](https://github.com/letsql/letsql/pull/278)
- Update trinodb/trino docker tag to v464 by @renovate[bot] in [#326](https://github.com/letsql/letsql/pull/326)
- Use vars instead of secrets by @mesejo in [#330](https://github.com/letsql/letsql/pull/330)
- Update bitnami/minio docker tag to v2024.10.29 by @renovate[bot] in [#327](https://github.com/letsql/letsql/pull/327)
- Update dependency ruff to v0.7.2 by @renovate[bot] in [#331](https://github.com/letsql/letsql/pull/331)
- Use env variables in workflows by @mesejo in [#332](https://github.com/letsql/letsql/pull/332)
- Update dependency quartodoc to ^0.7.2 || ^0.9.0 by @renovate[bot] in [#333](https://github.com/letsql/letsql/pull/333)
- Update dependency ruff to v0.7.3 by @renovate[bot] in [#338](https://github.com/letsql/letsql/pull/338)
- Update bitnami/minio docker tag to v2024.11.7 by @renovate[bot] in [#337](https://github.com/letsql/letsql/pull/337)
- Udate python version in pyproject by @mesejo in [#351](https://github.com/letsql/letsql/pull/351)
- Update dependency coverage to v7.6.5 by @renovate[bot] in [#349](https://github.com/letsql/letsql/pull/349)
- Update codecov/codecov-action action to v5 by @renovate[bot] in [#350](https://github.com/letsql/letsql/pull/350)
- Update postgres docker tag to v17.1 by @renovate[bot] in [#352](https://github.com/letsql/letsql/pull/352)

#### Fixed
- Update dependency pyarrow to v18 by @renovate[bot] in [#319](https://github.com/letsql/letsql/pull/319)
- Update dependency ibis-framework to v9.4.0 by @renovate[bot] in [#145](https://github.com/letsql/letsql/pull/145)
- Update dependency connectorx to v0.4.0 by @renovate[bot] in [#334](https://github.com/letsql/letsql/pull/334)
- RemoteTable bug by @mesejo in [#335](https://github.com/letsql/letsql/pull/335)

## [0.1.9] - 2024-10-30
### Details
Fix dependencies issues (adbc-driver-postgresql not installed) in release of 0.1.8

#### Changed
- Make git indifferent to changes in use nix/flake by @dlovell in [#305](https://github.com/letsql/letsql/pull/305)
- Update dependency ruff to v0.7.1 by @renovate[bot] in [#311](https://github.com/letsql/letsql/pull/311)
- Ci check for proper package installation by @mesejo in [#318](https://github.com/letsql/letsql/pull/318)
- Check proper installation of examples extras by @mesejo in [#320](https://github.com/letsql/letsql/pull/320)
- Use letsql execute/to_pyarrow/to_pyarrow_batches by @mesejo in [#316](https://github.com/letsql/letsql/pull/316)
- Update dependency pytest-cov to v6 by @renovate[bot] in [#322](https://github.com/letsql/letsql/pull/322)
- Update trinodb/trino docker tag to v463 by @renovate[bot] in [#321](https://github.com/letsql/letsql/pull/321)

#### Fixed
- Update dependency snowflake-connector-python to v3.12.3 [security] by @renovate[bot] in [#312](https://github.com/letsql/letsql/pull/312)
- Synchronize dependencies by @dlovell in [#317](https://github.com/letsql/letsql/pull/317)

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

