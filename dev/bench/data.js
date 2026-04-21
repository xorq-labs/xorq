window.BENCHMARK_DATA = {
  "lastUpdate": 1776804500477,
  "repoUrl": "https://github.com/xorq-labs/xorq",
  "entries": {
    "Benchmark": [
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a38df21f737e4bb38cca5a11dceb2999a2c379b9",
          "message": "feat(ci): add benchmark workflow with PR regression alerts (#1694)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T12:36:47+01:00",
          "tree_id": "b0cb73df5ddecd50b8af5cb13dd19f453756188d",
          "url": "https://github.com/xorq-labs/xorq/commit/a38df21f737e4bb38cca5a11dceb2999a2c379b9"
        },
        "date": 1773056259648,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.127452446209134,
            "unit": "iter/sec",
            "range": "stddev: 0.0012016626524617422",
            "extra": "mean: 89.86782957141973 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.082833006501964,
            "unit": "iter/sec",
            "range": "stddev: 0.0008071524060529112",
            "extra": "mean: 196.74067566666054 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9269727294500978,
            "unit": "iter/sec",
            "range": "stddev: 0.03120849026569381",
            "extra": "mean: 1.0787803871999813 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.420443811467016,
            "unit": "iter/sec",
            "range": "stddev: 0.0018739342920500806",
            "extra": "mean: 184.48673849998917 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.382647508698477,
            "unit": "iter/sec",
            "range": "stddev: 0.0015126291393375926",
            "extra": "mean: 185.78218216667133 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.270619163534736,
            "unit": "iter/sec",
            "range": "stddev: 0.0014356292230350736",
            "extra": "mean: 189.73102950002385 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ddda2e3fd71f3d0ac028b261159b6547be06eba1",
          "message": "fix(catalog): cli no subcommand prints help (#1696)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T14:54:55+01:00",
          "tree_id": "8c5cd0c8532a20104bd485a34e662fd2044abe31",
          "url": "https://github.com/xorq-labs/xorq/commit/ddda2e3fd71f3d0ac028b261159b6547be06eba1"
        },
        "date": 1773064550446,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.144294051515026,
            "unit": "iter/sec",
            "range": "stddev: 0.0012360365530897088",
            "extra": "mean: 89.73201850000123 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.1337433930263145,
            "unit": "iter/sec",
            "range": "stddev: 0.001972957250912619",
            "extra": "mean: 194.7896346666648 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9408693529438458,
            "unit": "iter/sec",
            "range": "stddev: 0.013870632601038142",
            "extra": "mean: 1.062846820200002 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.369607023887066,
            "unit": "iter/sec",
            "range": "stddev: 0.0022120890126589333",
            "extra": "mean: 186.23336783333144 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.31967780552145,
            "unit": "iter/sec",
            "range": "stddev: 0.0009261624447094183",
            "extra": "mean: 187.9813094999984 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.294622956530396,
            "unit": "iter/sec",
            "range": "stddev: 0.00157114041310619",
            "extra": "mean: 188.87086166666478 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "10c2b8f6c978d6f6881008f49e06d9fc7930e139",
          "message": "chore(ci): add actionlint to pre-commit config (#1699)",
          "timestamp": "2026-03-09T17:08:30+01:00",
          "tree_id": "2a1d67153a49a4f11459db1cc3c6c97d7ada28ac",
          "url": "https://github.com/xorq-labs/xorq/commit/10c2b8f6c978d6f6881008f49e06d9fc7930e139"
        },
        "date": 1773072564708,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.061856981364825,
            "unit": "iter/sec",
            "range": "stddev: 0.0010974524463720002",
            "extra": "mean: 90.4007348571432 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.066392335707592,
            "unit": "iter/sec",
            "range": "stddev: 0.003191127822294912",
            "extra": "mean: 197.3791079999998 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.923402339153203,
            "unit": "iter/sec",
            "range": "stddev: 0.02415430754979229",
            "extra": "mean: 1.082951555999999 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.333938334696266,
            "unit": "iter/sec",
            "range": "stddev: 0.0019285891403847052",
            "extra": "mean: 187.47873283333405 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.317759365105049,
            "unit": "iter/sec",
            "range": "stddev: 0.0024920246490253463",
            "extra": "mean: 188.04912583332842 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.258645741435484,
            "unit": "iter/sec",
            "range": "stddev: 0.0007148533347579905",
            "extra": "mean: 190.16302850000008 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bfe364c0ace0adef10986a414fe4fcc65f538e72",
          "message": "fix: race condition on tui refresh (#1697)\n\nCo-authored-by: Hussain Sultan <hussainz@gmail.com>",
          "timestamp": "2026-03-09T14:57:03-04:00",
          "tree_id": "611a10ea207781f9f26c45c9254d59c0449051e4",
          "url": "https://github.com/xorq-labs/xorq/commit/bfe364c0ace0adef10986a414fe4fcc65f538e72"
        },
        "date": 1773082676209,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.540820715251558,
            "unit": "iter/sec",
            "range": "stddev: 0.0006042786589668883",
            "extra": "mean: 86.64895024999986 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.268452111427782,
            "unit": "iter/sec",
            "range": "stddev: 0.0008795033941748997",
            "extra": "mean: 189.80907083332946 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9797629899557906,
            "unit": "iter/sec",
            "range": "stddev: 0.008541882686658083",
            "extra": "mean: 1.0206550055999997 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.535791544440262,
            "unit": "iter/sec",
            "range": "stddev: 0.0008271454002368187",
            "extra": "mean: 180.64264016666698 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.5451836079515004,
            "unit": "iter/sec",
            "range": "stddev: 0.001237380211512993",
            "extra": "mean: 180.33667966666656 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.384143949661417,
            "unit": "iter/sec",
            "range": "stddev: 0.0060969429243008435",
            "extra": "mean: 185.73054683333368 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "dfb737dee9e83a84c50a3c6bd207b46a130e2d68",
          "message": "feat(ml): add pipeline introspection via structured tag metadata (#1691)\n\nAdd FittedStepTagKey and FittedPipelineTagKey enums to consolidate all\ntag string constants. Refactor FittedStep.tag_kwargs into\nget_tag_kwargs() and FittedPipeline into get_tag_kwargs(which) so\ncallers can request step subsets by key. All FittedPipeline output exprs\nnow include ALL_STEPS tag metadata, enabling safe reconstruction of the\noriginal unfitted sklearn Pipeline even when pipelines are composed.\nExpose get_sklearn_pipeline_tags(), pipeline_tag_to_pipeline(), and\nget_outermost_pipeline() as composable building blocks.\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-09T15:05:49-04:00",
          "tree_id": "d9ad2054a4afd4ae540c2f916faeec51c153be94",
          "url": "https://github.com/xorq-labs/xorq/commit/dfb737dee9e83a84c50a3c6bd207b46a130e2d68"
        },
        "date": 1773083200684,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.235092066780835,
            "unit": "iter/sec",
            "range": "stddev: 0.0011460885735442739",
            "extra": "mean: 97.7030781428547 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.869941956778675,
            "unit": "iter/sec",
            "range": "stddev: 0.004597931680766498",
            "extra": "mean: 205.34125639999843 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9273981273462477,
            "unit": "iter/sec",
            "range": "stddev: 0.02197593862415748",
            "extra": "mean: 1.0782855502000017 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.070248028700052,
            "unit": "iter/sec",
            "range": "stddev: 0.0034689313055677354",
            "extra": "mean: 197.22901016666583 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.051699888087178,
            "unit": "iter/sec",
            "range": "stddev: 0.001679943264604309",
            "extra": "mean: 197.953168666686 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.0506145357543275,
            "unit": "iter/sec",
            "range": "stddev: 0.001855853799250273",
            "extra": "mean: 197.99570783333328 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f48805d5ac438890b5d4f055331cc7185d408433",
          "message": "release: 0.3.14 (#1703)\n\n## Summary\n- Bump version to 0.3.14\n- Update CHANGELOG.md with git-cliff\n\n### Changes in this release\n#### Added\n- Add HashingTag node for hash-contributing metadata tags (#1681)\n- Add benchmark workflow with PR regression alerts (#1694)\n- Add actionlint to pre-commit config (#1699)\n- Add pipeline introspection via structured tag metadata (#1691)\n\n#### Changed\n- Defer OTLP exporter imports until first use (#1690)\n- Convert test classes to standalone functions (#1667)\n- Update template commit hashes (#1695)\n\n#### Fixed\n- Defer import of xorq.ibis_yaml.translate (#1689)\n- Ensure python3.10 compat (#1693)\n- CLI no subcommand prints help (#1696)\n- Race condition on tui refresh (#1697)\n\n#### Removed\n- Remove redundant handlers (#1692)",
          "timestamp": "2026-03-09T16:12:49-04:00",
          "tree_id": "49e90a742f8b551ba87ef415d91a02fcd91deebf",
          "url": "https://github.com/xorq-labs/xorq/commit/f48805d5ac438890b5d4f055331cc7185d408433"
        },
        "date": 1773087222779,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.234451738217034,
            "unit": "iter/sec",
            "range": "stddev: 0.0010760070988997373",
            "extra": "mean: 89.01190937500125 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.135553503926689,
            "unit": "iter/sec",
            "range": "stddev: 0.0022685889882564773",
            "extra": "mean: 194.72097783333217 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9413862022195837,
            "unit": "iter/sec",
            "range": "stddev: 0.02185544098981812",
            "extra": "mean: 1.062263285399996 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.274312992690945,
            "unit": "iter/sec",
            "range": "stddev: 0.003418691122468083",
            "extra": "mean: 189.59815266666644 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.395322985498326,
            "unit": "iter/sec",
            "range": "stddev: 0.0012925505053056475",
            "extra": "mean: 185.34571566666594 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.366024758816864,
            "unit": "iter/sec",
            "range": "stddev: 0.00586020696298152",
            "extra": "mean: 186.35769399999683 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fc2a75b1a86d0e854f96002e40ac7807eb68ff21",
          "message": "chore: format with ruff (#1704)\n\nThe ruff version in the .pre-commit-config.yaml is not compatible with\nPython 3.10",
          "timestamp": "2026-03-09T23:18:22+01:00",
          "tree_id": "ae24b8708a0c911314399c8eb0945044b9c439a4",
          "url": "https://github.com/xorq-labs/xorq/commit/fc2a75b1a86d0e854f96002e40ac7807eb68ff21"
        },
        "date": 1773094759277,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.945150074737255,
            "unit": "iter/sec",
            "range": "stddev: 0.0005606212427944543",
            "extra": "mean: 83.71598462499819 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.223151145628314,
            "unit": "iter/sec",
            "range": "stddev: 0.04579811054366762",
            "extra": "mean: 191.45530583333445 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9351325241703202,
            "unit": "iter/sec",
            "range": "stddev: 0.0170636236465353",
            "extra": "mean: 1.0693671475999964 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.856428038548026,
            "unit": "iter/sec",
            "range": "stddev: 0.003564144536354357",
            "extra": "mean: 170.7525463333326 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.962881758709107,
            "unit": "iter/sec",
            "range": "stddev: 0.0015810520475401178",
            "extra": "mean: 167.7041471666693 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.85602166128361,
            "unit": "iter/sec",
            "range": "stddev: 0.001574840107293647",
            "extra": "mean: 170.7643956666658 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d128a35e263ae71d72ee18f9f48b007bc7f29051",
          "message": "fix: resolve DeprecationWarning and FutureWarning in tests (#1706)\n\n- Fix Python 3.13+ DeprecationWarning in bigquery udf/core.py: create\nast.Attribute pattern via __new__ to avoid required `value` argument\n- Fix pandas FutureWarning: use lowercase 'h' instead of deprecated 'H'\nfreq alias in date_range\n- Fix datafusion backend: use catalog().schema() with fallback to\ncatalog().database() for newer datafusion API\n- Remove redundant pytest.mark.backend marker in backends/conftest.py\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-10T13:14:36+01:00",
          "tree_id": "7679f1dd4fb75b9b7beff8c5bebda05ef2cdb29a",
          "url": "https://github.com/xorq-labs/xorq/commit/d128a35e263ae71d72ee18f9f48b007bc7f29051"
        },
        "date": 1773144930765,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.019470049757214,
            "unit": "iter/sec",
            "range": "stddev: 0.00020172129886346832",
            "extra": "mean: 90.74846571428655 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.02730993536094,
            "unit": "iter/sec",
            "range": "stddev: 0.0015713802093259377",
            "extra": "mean: 198.91353683333315 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9106737043637411,
            "unit": "iter/sec",
            "range": "stddev: 0.014042057469572941",
            "extra": "mean: 1.0980881463999979 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.320094592416967,
            "unit": "iter/sec",
            "range": "stddev: 0.0011710859319187107",
            "extra": "mean: 187.966582666661 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.267074505199654,
            "unit": "iter/sec",
            "range": "stddev: 0.0029904122135194612",
            "extra": "mean: 189.8587154999992 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.2058577130040575,
            "unit": "iter/sec",
            "range": "stddev: 0.001533732658594809",
            "extra": "mean: 192.0913046666707 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7b0c576660ef7b28c8a81ce50ef834cac6b0b05b",
          "message": "chore(deps): bump virtualenv from 20.33.0 to 20.36.1 (#1707)\n\nBumps [virtualenv](https://github.com/pypa/virtualenv) from 20.33.0 to\n20.36.1.\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/pypa/virtualenv/releases\">virtualenv's\nreleases</a>.</em></p>\n<blockquote>\n<h2>20.36.0</h2>\n<!-- raw HTML omitted -->\n<h2>What's Changed</h2>\n<ul>\n<li>release 20.35.3 by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2981\">pypa/virtualenv#2981</a></li>\n<li>fix: Prevent NameError when accessing _DISTUTILS_PATCH during file\nov… by <a href=\"https://github.com/gracetyy\"><code>@​gracetyy</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2982\">pypa/virtualenv#2982</a></li>\n<li>Upgrade pip and fix 3.15 picking old wheel by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2989\">pypa/virtualenv#2989</a></li>\n<li>release 20.35.4 by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2990\">pypa/virtualenv#2990</a></li>\n<li>fix: wrong path on migrated venv by <a\nhref=\"https://github.com/sk1234567891\"><code>@​sk1234567891</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2996\">pypa/virtualenv#2996</a></li>\n<li>test_too_many_open_files: assert on <code>errno.EMFILE</code>\ninstead of <code>strerror</code> by <a\nhref=\"https://github.com/pltrz\"><code>@​pltrz</code></a> in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3001\">pypa/virtualenv#3001</a></li>\n<li>fix: update filelock dependency version to 3.20.1 to fix CVE\nCVE-2025-68146 by <a\nhref=\"https://github.com/pythonhubdev\"><code>@​pythonhubdev</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3002\">pypa/virtualenv#3002</a></li>\n<li>fix: resolve EncodingWarning in tox upgrade environment by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3007\">pypa/virtualenv#3007</a></li>\n<li>Fix Interpreter discovery bug wrt. Microsoft Store shortcut using\nLatin-1 by <a\nhref=\"https://github.com/rahuldevikar\"><code>@​rahuldevikar</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3006\">pypa/virtualenv#3006</a></li>\n<li>Add support for PEP 440 version specifiers in the\n<code>--python</code> flag. by <a\nhref=\"https://github.com/rahuldevikar\"><code>@​rahuldevikar</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3008\">pypa/virtualenv#3008</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a href=\"https://github.com/gracetyy\"><code>@​gracetyy</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2982\">pypa/virtualenv#2982</a></li>\n<li><a\nhref=\"https://github.com/sk1234567891\"><code>@​sk1234567891</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2996\">pypa/virtualenv#2996</a></li>\n<li><a href=\"https://github.com/pltrz\"><code>@​pltrz</code></a> made\ntheir first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3001\">pypa/virtualenv#3001</a></li>\n<li><a\nhref=\"https://github.com/pythonhubdev\"><code>@​pythonhubdev</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3002\">pypa/virtualenv#3002</a></li>\n<li><a\nhref=\"https://github.com/rahuldevikar\"><code>@​rahuldevikar</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/3006\">pypa/virtualenv#3006</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/pypa/virtualenv/compare/20.35.3...20.36.0\">https://github.com/pypa/virtualenv/compare/20.35.3...20.36.0</a></p>\n<h2>20.35.4</h2>\n<!-- raw HTML omitted -->\n<h2>What's Changed</h2>\n<ul>\n<li>release 20.35.3 by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2981\">pypa/virtualenv#2981</a></li>\n<li>fix: Prevent NameError when accessing _DISTUTILS_PATCH during file\nov… by <a href=\"https://github.com/gracetyy\"><code>@​gracetyy</code></a>\nin <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2982\">pypa/virtualenv#2982</a></li>\n<li>Upgrade pip and fix 3.15 picking old wheel by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2989\">pypa/virtualenv#2989</a></li>\n</ul>\n<h2>New Contributors</h2>\n<ul>\n<li><a href=\"https://github.com/gracetyy\"><code>@​gracetyy</code></a>\nmade their first contribution in <a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2982\">pypa/virtualenv#2982</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/pypa/virtualenv/compare/20.35.3...20.35.4\">https://github.com/pypa/virtualenv/compare/20.35.3...20.35.4</a></p>\n<h2>20.35.3</h2>\n<!-- raw HTML omitted -->\n<h2>What's Changed</h2>\n<ul>\n<li>release 20.35.1 by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2976\">pypa/virtualenv#2976</a></li>\n<li>Revert out effort to extract discovery by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2978\">pypa/virtualenv#2978</a></li>\n<li>release 20.35.2 by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2980\">pypa/virtualenv#2980</a></li>\n<li>test_too_many_open_files fails by <a\nhref=\"https://github.com/gaborbernat\"><code>@​gaborbernat</code></a> in\n<a\nhref=\"https://redirect.github.com/pypa/virtualenv/pull/2979\">pypa/virtualenv#2979</a></li>\n</ul>\n<p><strong>Full Changelog</strong>: <a\nhref=\"https://github.com/pypa/virtualenv/compare/20.35.1...20.35.3\">https://github.com/pypa/virtualenv/compare/20.35.1...20.35.3</a></p>\n<h2>20.35.2</h2>\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/pypa/virtualenv/blob/main/docs/changelog.rst\">virtualenv's\nchangelog</a>.</em></p>\n<blockquote>\n<h1>Bugfixes - 20.36.1</h1>\n<ul>\n<li>Fix TOCTOU vulnerabilities in app_data and lock directory creation\nthat could be exploited via symlink attacks -\nreported by :user:<code>tsigouris007</code>, fixed by\n:user:<code>gaborbernat</code>. (:issue:<code>3013</code>)</li>\n</ul>\n<hr />\n<p>v20.36.0 (2026-01-07)</p>\n<hr />\n<h1>Features - 20.36.0</h1>\n<ul>\n<li>Add support for PEP 440 version specifiers in the\n<code>--python</code> flag. Users can now specify Python versions using\noperators like <code>&gt;=</code>, <code>&lt;=</code>, <code>~=</code>,\netc. For example: <code>virtualenv --python=&quot;&gt;=3.12&quot;\nmyenv</code> <code>. (:issue:</code>2994`)</li>\n</ul>\n<hr />\n<p>v20.35.4 (2025-10-28)</p>\n<hr />\n<h1>Bugfixes - 20.35.4</h1>\n<ul>\n<li>\n<p>Fix race condition in <code>_virtualenv.py</code> when file is\noverwritten during import, preventing <code>NameError</code> when\n<code>_DISTUTILS_PATCH</code> is accessed - by\n:user:<code>gracetyy</code>. (:issue:<code>2969</code>)</p>\n</li>\n<li>\n<p>Upgrade embedded wheels:</p>\n<ul>\n<li>pip to <code>25.3</code> from <code>25.2</code>\n(:issue:<code>2989</code>)</li>\n</ul>\n</li>\n</ul>\n<hr />\n<p>v20.35.3 (2025-10-10)</p>\n<hr />\n<h1>Bugfixes - 20.35.3</h1>\n<ul>\n<li>Accept RuntimeError in <code>test_too_many_open_files</code>, by\n:user:<code>esafak</code> (:issue:<code>2935</code>)</li>\n</ul>\n<hr />\n<p>v20.35.2 (2025-10-10)</p>\n<hr />\n<h1>Bugfixes - 20.35.2</h1>\n<ul>\n<li>Revert out changes related to the extraction of the discovery module\n- by :user:<code>gaborbernat</code>. (:issue:<code>2978</code>)</li>\n</ul>\n<hr />\n<p>v20.35.1 (2025-10-09)</p>\n<hr />\n<!-- raw HTML omitted -->\n</blockquote>\n<p>... (truncated)</p>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/d0ad11d1146e81ea74d2461be9653f1da9cf3fd1\"><code>d0ad11d</code></a>\nrelease 20.36.1</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/dec4cec5d16edaf83a00a658f32d1e032661cebc\"><code>dec4cec</code></a>\nMerge pull request <a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3013\">#3013</a>\nfrom gaborbernat/fix-sec</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/5fe5d38beb1273b489591a7b444f1018af2edf0a\"><code>5fe5d38</code></a>\nrelease 20.36.0 (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3011\">#3011</a>)</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/9719376addaa710b61d9ed013774fa26f6224b4e\"><code>9719376</code></a>\nrelease 20.36.0</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/0276db6fcf8849c519d75465f659b12aefb2acd8\"><code>0276db6</code></a>\nAdd support for PEP 440 version specifiers in the <code>--python</code>\nflag. (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3008\">#3008</a>)</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/4f900c29044e17812981b5b98ddce45604858b7f\"><code>4f900c2</code></a>\nFix Interpreter discovery bug wrt. Microsoft Store shortcut using\nLatin-1 (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3\">#3</a>...</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/13afcc62a3444d0386c8031d0a62277a8274ab07\"><code>13afcc6</code></a>\nfix: resolve EncodingWarning in tox upgrade environment (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3007\">#3007</a>)</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/31b5d31581df3e3a7bbc55e52568b26dd01b0d57\"><code>31b5d31</code></a>\n[pre-commit.ci] pre-commit autoupdate (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/2997\">#2997</a>)</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/7c284221b4751388801355fc6ebaa2abe60427bd\"><code>7c28422</code></a>\nfix: update filelock dependency version to 3.20.1 to fix CVE\nCVE-2025-68146 (...</li>\n<li><a\nhref=\"https://github.com/pypa/virtualenv/commit/365628c544cd5498fbf0a3b6c6a8c1f41d25a749\"><code>365628c</code></a>\ntest_too_many_open_files: assert on <code>errno.EMFILE</code> instead of\n<code>strerror</code> (<a\nhref=\"https://redirect.github.com/pypa/virtualenv/issues/3001\">#3001</a>)</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/pypa/virtualenv/compare/20.33.0...20.36.1\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\n[![Dependabot compatibility\nscore](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=virtualenv&package-manager=uv&previous-version=20.33.0&new-version=20.36.1)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore this major version` will close this PR and stop\nDependabot creating any more for this major version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this minor version` will close this PR and stop\nDependabot creating any more for this minor version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this dependency` will close this PR and stop\nDependabot creating any more for this dependency (unless you reopen the\nPR or upgrade to it yourself)\nYou can disable automated security fix PRs for this repo from the\n[Security Alerts\npage](https://github.com/xorq-labs/xorq/network/alerts).\n\n</details>\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-03-10T15:23:07+01:00",
          "tree_id": "f4a071ebbad5bf6d9150b9d3965ad1984bc5500c",
          "url": "https://github.com/xorq-labs/xorq/commit/7b0c576660ef7b28c8a81ce50ef834cac6b0b05b"
        },
        "date": 1773152637998,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.341095437059902,
            "unit": "iter/sec",
            "range": "stddev: 0.0010277970902031232",
            "extra": "mean: 88.17490387500371 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.125441568963989,
            "unit": "iter/sec",
            "range": "stddev: 0.002214868230439224",
            "extra": "mean: 195.10514100000384 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9344814034196498,
            "unit": "iter/sec",
            "range": "stddev: 0.01568783349528367",
            "extra": "mean: 1.0701122529999965 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.391380448020654,
            "unit": "iter/sec",
            "range": "stddev: 0.001102785932819251",
            "extra": "mean: 185.4812528333317 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.3116793494138,
            "unit": "iter/sec",
            "range": "stddev: 0.0027539521965397385",
            "extra": "mean: 188.2643763333268 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.294092568630761,
            "unit": "iter/sec",
            "range": "stddev: 0.0018792674055029893",
            "extra": "mean: 188.88978366667195 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f9d19bff4c3cf28afb6871dfe93728dc408e56c8",
          "message": "feat(ml): remappers (#1708)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-11T15:39:51-04:00",
          "tree_id": "887b346882511aac6d696ba4c8623be2f7bf1be6",
          "url": "https://github.com/xorq-labs/xorq/commit/f9d19bff4c3cf28afb6871dfe93728dc408e56c8"
        },
        "date": 1773258043388,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.251078101157441,
            "unit": "iter/sec",
            "range": "stddev: 0.0012118036604294397",
            "extra": "mean: 88.88037137499971 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.161407136878949,
            "unit": "iter/sec",
            "range": "stddev: 0.0014131987044652856",
            "extra": "mean: 193.7456149999998 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9591731133414164,
            "unit": "iter/sec",
            "range": "stddev: 0.009318138903009525",
            "extra": "mean: 1.042564669600003 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.270773566110566,
            "unit": "iter/sec",
            "range": "stddev: 0.00862105037646493",
            "extra": "mean: 189.72547149998795 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.435130506279378,
            "unit": "iter/sec",
            "range": "stddev: 0.0012348731182705987",
            "extra": "mean: 183.98822233333098 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.273642474225823,
            "unit": "iter/sec",
            "range": "stddev: 0.002244664114592134",
            "extra": "mean: 189.62225916666853 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "160b6352b6299f41cfc59699162633425fb68f9f",
          "message": "fix(lint): remove functools.cache from methods (#1684)\n\n- Replace `@property @functools.cache` combos with\n`@functools.cached_property` to fix B019 violations and eliminate memory\nleaks from instance-keyed caches\n- Convert standalone `@functools.cache` methods (`copy_sdist`) to\n`@functools.cached_property` and update call sites accordingly\n- Replace class-level `cached_property` aliases (`popened =\n_uv_build_popened`) with explicit `@property` delegates (Python 3.13\ndisallows reusing the same `cached_property` object under two names)\n- Add `# noqa: B019` for `make_deferred_other`, which takes extra\narguments and cannot be converted to a `cached_property`\n\nFix a latent bug in `SdistBuilder.maybe_packager`: the field had\n`converter=toolz.curried.excepts(Exception, Path)`, which silently\nconverted a `Sdister` object to `None` because `Path(sdister_instance)`\nraises `TypeError`. This caused the `Sdister` to be garbage-collected\nimmediately after `SdistBuilder.from_script_path` returned, cleaning up\nits\n`TemporaryDirectory` and deleting the sdist file that\n`SdistBuilder.sdist_path` pointed to. Subsequent access to `sdist_path`\nin `_uv_tool_run_xorq_build` then failed with `FileNotFoundError`. The\nfix removes the broken converter, so `maybe_packager` holds the\n`Sdister` directly, keeping it alive for the lifetime of the\n`SdistBuilder`.\n\nProof that `@frozen` (attrs) works with `@cached_property`:\n\n```python\nfrom attrs import frozen\nfrom functools import cached_property\n\n@Frozen\nclass Circle:\n    radius: float\n\n    @cached_property\n    def area(self):\n        print(\"computing...\")\n        return 3.14159 * self.radius ** 2\n\nc = Circle(radius=5.0)\nprint(c.area)  # computing... → 78.53975\nprint(c.area)  # cached → 78.53975\n```\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-12T09:33:08-04:00",
          "tree_id": "9f16309636f152c11177bc35eb7f0d60ff524f67",
          "url": "https://github.com/xorq-labs/xorq/commit/160b6352b6299f41cfc59699162633425fb68f9f"
        },
        "date": 1773322441282,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.430416477411928,
            "unit": "iter/sec",
            "range": "stddev: 0.000606884442068613",
            "extra": "mean: 87.48587612499836 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.2352900371491495,
            "unit": "iter/sec",
            "range": "stddev: 0.000914879760482777",
            "extra": "mean: 191.01138483333102 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9549165875041928,
            "unit": "iter/sec",
            "range": "stddev: 0.02093594820573095",
            "extra": "mean: 1.0472118854 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.423609138712642,
            "unit": "iter/sec",
            "range": "stddev: 0.0017229220402804424",
            "extra": "mean: 184.37906833333528 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.438948513175606,
            "unit": "iter/sec",
            "range": "stddev: 0.0018699741460870027",
            "extra": "mean: 183.85906716666747 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.295482084087549,
            "unit": "iter/sec",
            "range": "stddev: 0.0017261875847178452",
            "extra": "mean: 188.84021966666845 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "950ae96b6a1900d08972f64726b947b1b3afcb51",
          "message": "fix: do not pass None values to span.add_event (#1709)",
          "timestamp": "2026-03-12T09:34:58-04:00",
          "tree_id": "ffecd23ac3bf04e9737d10e1b7ebaa88e560c4c8",
          "url": "https://github.com/xorq-labs/xorq/commit/950ae96b6a1900d08972f64726b947b1b3afcb51"
        },
        "date": 1773322548423,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.126480288321464,
            "unit": "iter/sec",
            "range": "stddev: 0.0008848623805535523",
            "extra": "mean: 89.87568162499838 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.100601939121887,
            "unit": "iter/sec",
            "range": "stddev: 0.0013383140439944712",
            "extra": "mean: 196.05529149999867 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9368434684118192,
            "unit": "iter/sec",
            "range": "stddev: 0.010990558334228588",
            "extra": "mean: 1.067414177200004 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.320844233559524,
            "unit": "iter/sec",
            "range": "stddev: 0.0016880812294975743",
            "extra": "mean: 187.9401004999958 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.3514850855423095,
            "unit": "iter/sec",
            "range": "stddev: 0.0014161786086894792",
            "extra": "mean: 186.8640169999954 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.3210742310234815,
            "unit": "iter/sec",
            "range": "stddev: 0.0014211364687813164",
            "extra": "mean: 187.93197700000044 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3118066b781742b5a21abcae505bbb660efd368e",
          "message": "feat(catalog): add Expr kind (XOR-244) (#1682)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-03-12T09:46:19-04:00",
          "tree_id": "d98ffbe50e6fc66f0991cebfdd8286884cfbe661",
          "url": "https://github.com/xorq-labs/xorq/commit/3118066b781742b5a21abcae505bbb660efd368e"
        },
        "date": 1773323230856,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.979156687430475,
            "unit": "iter/sec",
            "range": "stddev: 0.0010373761469446565",
            "extra": "mean: 91.08167671428292 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.9341677224107405,
            "unit": "iter/sec",
            "range": "stddev: 0.001982880081146798",
            "extra": "mean: 202.66842480000236 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9117610093296938,
            "unit": "iter/sec",
            "range": "stddev: 0.013780940542030903",
            "extra": "mean: 1.0967786401999988 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.258661835737393,
            "unit": "iter/sec",
            "range": "stddev: 0.0016226287760966402",
            "extra": "mean: 190.16244649999928 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.208218942399719,
            "unit": "iter/sec",
            "range": "stddev: 0.0034181849910916635",
            "extra": "mean: 192.004216999995 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.153316658884114,
            "unit": "iter/sec",
            "range": "stddev: 0.0022785319010165594",
            "extra": "mean: 194.0497869999973 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9fac011ab3a8ba5a6aa5a71d4f1cc9b241fc02bd",
          "message": "chore(deps): update dependency black to v26 [security] (#1711)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [black](https://redirect.github.com/psf/black)\n([changelog](https://redirect.github.com/psf/black/blob/main/CHANGES.md))\n| `==25.12.0` → `==26.3.1` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/black/26.3.1?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/black/25.12.0/26.3.1?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n####\n[CVE-2026-32274](https://redirect.github.com/psf/black/security/advisories/GHSA-3936-cmfr-pm3m)\n\n### Impact\n\nBlack writes a cache file, the name of which is computed from various\nformatting options. The value of the `--python-cell-magics` option was\nplaced in the filename without sanitization, which allowed an attacker\nwho controls the value of this argument to write cache files to\narbitrary file system locations.\n\n### Patches\n\nFixed in Black 26.3.1.\n\n### Workarounds\n\nDo not allow untrusted user input into the value of the\n`--python-cell-magics` option.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My41OS4wIiwidXBkYXRlZEluVmVyIjoiNDMuNTkuMCIsInRhcmdldEJyYW5jaCI6Im1haW4iLCJsYWJlbHMiOltdfQ==-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-03-13T13:27:40+01:00",
          "tree_id": "ad67dc1f548b9c66b354388528719053df9286db",
          "url": "https://github.com/xorq-labs/xorq/commit/9fac011ab3a8ba5a6aa5a71d4f1cc9b241fc02bd"
        },
        "date": 1773404912487,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.797663742197642,
            "unit": "iter/sec",
            "range": "stddev: 0.0013677503832469182",
            "extra": "mean: 92.61262657142818 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.043441638320141,
            "unit": "iter/sec",
            "range": "stddev: 0.0017388544986011705",
            "extra": "mean: 198.2773018333326 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.8619262235500489,
            "unit": "iter/sec",
            "range": "stddev: 0.02077035062948573",
            "extra": "mean: 1.1601921053999973 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.113853330486362,
            "unit": "iter/sec",
            "range": "stddev: 0.0027209765952489647",
            "extra": "mean: 195.54725866666445 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.113068836236691,
            "unit": "iter/sec",
            "range": "stddev: 0.0016809892553823133",
            "extra": "mean: 195.5772613333361 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.059070674636393,
            "unit": "iter/sec",
            "range": "stddev: 0.002366339189442688",
            "extra": "mean: 197.66476183333262 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "248b6181c4bdeeee399381e859f87931eaaf6067",
          "message": "feat(catalog): allow opting out of assert_consistency on Catalog init (#1714)\n\nAdd check_consistency param (default True) to Catalog, from_repo_path,\nfrom_name, and from_default. Skipping the consistency check avoids an\nO(n_blobs) git tree traversal on construction, which is expensive for\nlarge catalogs used in read-only workflows.\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-13T17:45:29+01:00",
          "tree_id": "0e9dc4e3fb519d2a5085d456276bf0a7f0d9ccb1",
          "url": "https://github.com/xorq-labs/xorq/commit/248b6181c4bdeeee399381e859f87931eaaf6067"
        },
        "date": 1773420374637,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 12.259835793475935,
            "unit": "iter/sec",
            "range": "stddev: 0.0005768222852652237",
            "extra": "mean: 81.56716099999883 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.6248212185730715,
            "unit": "iter/sec",
            "range": "stddev: 0.00073224751619707",
            "extra": "mean: 177.78342833333363 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 1.0017630231804617,
            "unit": "iter/sec",
            "range": "stddev: 0.013532341335624583",
            "extra": "mean: 998.2400795999993 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.816668672108592,
            "unit": "iter/sec",
            "range": "stddev: 0.001479045895687515",
            "extra": "mean: 171.9197115000005 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.738935601412301,
            "unit": "iter/sec",
            "range": "stddev: 0.0008805985251049652",
            "extra": "mean: 174.24833966666378 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.705776320541913,
            "unit": "iter/sec",
            "range": "stddev: 0.0008516912606694631",
            "extra": "mean: 175.26098883333438 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0de0e27d86d07f0c76bd231153ba77dcebae8ddf",
          "message": "chore: remove __main__ (#1719)",
          "timestamp": "2026-03-16T14:44:17+01:00",
          "tree_id": "0a385d7473e11d58612da295b97bba912dbfbf0b",
          "url": "https://github.com/xorq-labs/xorq/commit/0de0e27d86d07f0c76bd231153ba77dcebae8ddf"
        },
        "date": 1773668710465,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.329964316648407,
            "unit": "iter/sec",
            "range": "stddev: 0.001612649163301179",
            "extra": "mean: 88.261531285724 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.102937136820458,
            "unit": "iter/sec",
            "range": "stddev: 0.002829243794690278",
            "extra": "mean: 195.96557299999992 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9560796537846458,
            "unit": "iter/sec",
            "range": "stddev: 0.008503900299508925",
            "extra": "mean: 1.0459379572000045 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.320547277877584,
            "unit": "iter/sec",
            "range": "stddev: 0.0022375632062423398",
            "extra": "mean: 187.95059000000265 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.364067178203967,
            "unit": "iter/sec",
            "range": "stddev: 0.0015968144562243628",
            "extra": "mean: 186.425704000006 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.3139395747378275,
            "unit": "iter/sec",
            "range": "stddev: 0.0011868471690017024",
            "extra": "mean: 188.1843001666681 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2868d9713a32e26df3c0c8336dcac5a746e05664",
          "message": "perf(packager): switch sdist format from tgz to zip (#1716)\n\nZip archives support random access and in-place append, eliminating the\ngunzip→tar_append→gzip pipeline that TGZAppender required. This\nsimplifies the code and improves sdist build/read performance.\n\n- Add zip_utils.py with ZipProxy, ZipAppender,\ncalc_zip_content_hexdigest, and tgz_to_zip converter (since uv build\nonly outputs .tar.gz)\n- Remove tar_utils.py (no remaining importers in common/utils)\n- Update packager.py to use zip utilities throughout Sdister,\nSdistBuilder, SdistRunner, and helper functions\n- Add ZipExtFile support to file_digest in dask_normalize_utils\n- Update tests to work with zip-based sdists\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-16T16:45:45+01:00",
          "tree_id": "00b66482272e292148c4c4e44f8e02ce50eb8f59",
          "url": "https://github.com/xorq-labs/xorq/commit/2868d9713a32e26df3c0c8336dcac5a746e05664"
        },
        "date": 1773676003683,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.45766691335469,
            "unit": "iter/sec",
            "range": "stddev: 0.0006450040797777024",
            "extra": "mean: 87.2778033749988 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.064579683707648,
            "unit": "iter/sec",
            "range": "stddev: 0.004078350723613398",
            "extra": "mean: 197.44975149999533 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9484000153162796,
            "unit": "iter/sec",
            "range": "stddev: 0.012040761504586012",
            "extra": "mean: 1.0544074060000015 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.349916605366231,
            "unit": "iter/sec",
            "range": "stddev: 0.0005932564088109913",
            "extra": "mean: 186.91880149999918 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.142351323550072,
            "unit": "iter/sec",
            "range": "stddev: 0.007607740166010183",
            "extra": "mean: 194.46357066666545 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.311377129506878,
            "unit": "iter/sec",
            "range": "stddev: 0.0014013758784788054",
            "extra": "mean: 188.2750886666642 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "610ee4232bca02ea166dfa02aa7f53a627228566",
          "message": "perf(catalog): switch catalog and download_utils from tgz to zip (#1717)\n\n## Summary\n- Extends the tgz-to-zip conversion from #1716 to cover the catalog\nsubsystem and GitHub template downloads\n- Replaces `catalog/tar_utils.py` with `catalog/zip_utils.py` (zip has\nO(1) append, random-access reads, simpler stdlib API)\n- Switches GitHub archive downloads from `.tar.gz` to `.zip`\n- Renames `REQUIRED_TGZ_NAMES` → `REQUIRED_ARCHIVE_NAMES`,\n`VALID_SUFFIXES`/`PREFERRED_SUFFIX` to `.zip` only\n\n> **Depends on #1716** (`perf/sdister/use-zip`) — the first commit in\nthis branch is from that PR.\n\n## Test plan\n- [x] `python -m pytest python/xorq/catalog/tests/ -x -q -m \"not slow\"`\n— 28 passed\n- [x] `python -m pytest python/xorq/catalog/tests/test_cli.py -x -q -m\n\"not slow\"` — 63 passed\n- [x] `python -m pytest python/xorq/tests/test_cli_run_alias.py -x -q -m\n\"not slow\"` — 10 passed\n- [x] `python -m pytest python/xorq/common/utils/tests/test_io_utils.py\n-x -q` — 19 passed\n- [ ] `python -m pytest python/xorq/ibis_yaml/tests/test_packager.py -x\n-q --snapshot-update` (slow, needs network)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-16T13:46:16-04:00",
          "tree_id": "a94bee72c97ac41b9fc078fbdac29c444ff86df6",
          "url": "https://github.com/xorq-labs/xorq/commit/610ee4232bca02ea166dfa02aa7f53a627228566"
        },
        "date": 1773683231227,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.020845685778632,
            "unit": "iter/sec",
            "range": "stddev: 0.001066703918671644",
            "extra": "mean: 90.73713837499841 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.956120213233308,
            "unit": "iter/sec",
            "range": "stddev: 0.0041002249162676845",
            "extra": "mean: 201.77073133333323 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9368522657888888,
            "unit": "iter/sec",
            "range": "stddev: 0.0069439639367685985",
            "extra": "mean: 1.0674041537999983 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.324226128590607,
            "unit": "iter/sec",
            "range": "stddev: 0.0029522513326335565",
            "extra": "mean: 187.82072283333187 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.232942795818036,
            "unit": "iter/sec",
            "range": "stddev: 0.004200584981809395",
            "extra": "mean: 191.0970631666681 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.174428764732359,
            "unit": "iter/sec",
            "range": "stddev: 0.004104804794951837",
            "extra": "mean: 193.25804749999756 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1a883fe36885c1618adfe29619a15bdff42a33a5",
          "message": "release: 0.3.15 (#1722)",
          "timestamp": "2026-03-17T11:35:26+01:00",
          "tree_id": "9a6448fc4a8cbe16832e9815a77a4fb46565d01e",
          "url": "https://github.com/xorq-labs/xorq/commit/1a883fe36885c1618adfe29619a15bdff42a33a5"
        },
        "date": 1773743777661,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.593641531067108,
            "unit": "iter/sec",
            "range": "stddev: 0.0003397378963781211",
            "extra": "mean: 86.25417625000154 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.260757471093818,
            "unit": "iter/sec",
            "range": "stddev: 0.0008635906777527565",
            "extra": "mean: 190.08669483333543 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9655275603785399,
            "unit": "iter/sec",
            "range": "stddev: 0.010061743999395002",
            "extra": "mean: 1.0357032165999953 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.386498637958818,
            "unit": "iter/sec",
            "range": "stddev: 0.002195359935097486",
            "extra": "mean: 185.64935539999396 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.4233093913568355,
            "unit": "iter/sec",
            "range": "stddev: 0.002487405922127249",
            "extra": "mean: 184.38925899999484 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.429393684649351,
            "unit": "iter/sec",
            "range": "stddev: 0.0010473346912212428",
            "extra": "mean: 184.18262849999678 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a47e0911fffc3fc6dd928a5d5216d7f28e4c8889",
          "message": "feat: add LazyBackend (#1655)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-17T14:43:26+01:00",
          "tree_id": "a92b607bb80a634332926d0b46279eaefa042bf7",
          "url": "https://github.com/xorq-labs/xorq/commit/a47e0911fffc3fc6dd928a5d5216d7f28e4c8889"
        },
        "date": 1773755057436,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.50688554914756,
            "unit": "iter/sec",
            "range": "stddev: 0.0003889262454105341",
            "extra": "mean: 86.90448825000097 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.124836708822657,
            "unit": "iter/sec",
            "range": "stddev: 0.0009042496352261191",
            "extra": "mean: 195.12816833333466 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9510314572203384,
            "unit": "iter/sec",
            "range": "stddev: 0.016442435722975816",
            "extra": "mean: 1.051489929600001 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.490982787685503,
            "unit": "iter/sec",
            "range": "stddev: 0.001178858744330371",
            "extra": "mean: 182.11676099999372 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.383063417756215,
            "unit": "iter/sec",
            "range": "stddev: 0.0038427652189170305",
            "extra": "mean: 185.76782816666557 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.3645006913374935,
            "unit": "iter/sec",
            "range": "stddev: 0.008386935951889808",
            "extra": "mean: 186.4106386666672 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6713581c9115591c7fdc19e7f3a2ef6de7ee7033",
          "message": "fix(ibis_yaml): use typed cache to prevent int/bool collision in translate_from_yaml (#1725)\n\n## Summary\n- `functools.cache` on `translate_from_yaml` treats `1` and `True` as\nthe same cache key (since `1 == True` in Python), causing `Limit(n=1)`\nto roundtrip as `Limit(n=True)` when a boolean value is cached first\n- DataFusion then rejects the query with: `Expected LIMIT to be an\ninteger or null, but got Boolean`\n- Fix: switch to `lru_cache(maxsize=None, typed=True)` to distinguish\n`int` from `bool`, matching the existing `translate_to_yaml`\nimplementation\n\n## Test plan\n- [x] Added `test_limit_not_coerced_to_bool` that filters on a boolean\ncolumn then applies `.limit(1)`, verifying the roundtripped limit is\n`int(1)` not `True`\n- [x] All existing `test_relations.py` tests pass\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-19T05:02:42-04:00",
          "tree_id": "28a1692e463f0839413ff39486ae07a5fc4fac10",
          "url": "https://github.com/xorq-labs/xorq/commit/6713581c9115591c7fdc19e7f3a2ef6de7ee7033"
        },
        "date": 1773911011618,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.41443616586924,
            "unit": "iter/sec",
            "range": "stddev: 0.0010603786016070108",
            "extra": "mean: 87.60835712500104 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.180635917843729,
            "unit": "iter/sec",
            "range": "stddev: 0.0016910444505210604",
            "extra": "mean: 193.02649633333382 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9352795106508198,
            "unit": "iter/sec",
            "range": "stddev: 0.029879044336202542",
            "extra": "mean: 1.0691990882000013 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.439883988018622,
            "unit": "iter/sec",
            "range": "stddev: 0.0011202446246968624",
            "extra": "mean: 183.82744966666684 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.394058497965787,
            "unit": "iter/sec",
            "range": "stddev: 0.0023071135249005115",
            "extra": "mean: 185.38916483332932 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.357197395325665,
            "unit": "iter/sec",
            "range": "stddev: 0.0014475291009965854",
            "extra": "mean: 186.66476633333198 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45543592+ghoersti@users.noreply.github.com",
            "name": "ghoersti",
            "username": "ghoersti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7f9dc3b2a71286e9dd813acec0d73bcfa2d2cbf2",
          "message": "fix(tui): handle scalar expressions in _entry_info (#1726)\n\n## Summary\n- `_entry_info` crashed with `AttributeError` when called with scalar\nexpressions (e.g. `StringScalar`, `FloatingScalar`) because `.columns`\nonly exists on ibis `Table` expressions\n- Wrap `len(expr.columns)` in `try/except AttributeError`, defaulting\n`column_count` to `0` for non-table expressions\n- Add unit test covering the scalar case\n\n## Test plan\n- [ ] `pytest\npython/xorq/catalog/tests/test_tui.py::test_entry_info_scalar_expression_returns_zero_column_count`\n- [ ] `xorq catalog --path .experiments/<uuid>/submissions tui` with a\ncatalog containing scalar expressions\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: ghoersti <ghoersti@users.noreply.github.com>\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-19T19:02:57-04:00",
          "tree_id": "d081c0b62f8724d49f5ca8de389349d933e7c6f3",
          "url": "https://github.com/xorq-labs/xorq/commit/7f9dc3b2a71286e9dd813acec0d73bcfa2d2cbf2"
        },
        "date": 1773961427683,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.933772915985399,
            "unit": "iter/sec",
            "range": "stddev: 0.0015840175475932336",
            "extra": "mean: 91.4597374286034 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.02205721847781,
            "unit": "iter/sec",
            "range": "stddev: 0.0033898310883763256",
            "extra": "mean: 199.12158633331956 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.8683672493463633,
            "unit": "iter/sec",
            "range": "stddev: 0.025382354142616553",
            "extra": "mean: 1.151586498399979 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.214143228635208,
            "unit": "iter/sec",
            "range": "stddev: 0.0024173126032440437",
            "extra": "mean: 191.7860626666652 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.192326769116107,
            "unit": "iter/sec",
            "range": "stddev: 0.0012697308494931596",
            "extra": "mean: 192.59188499999405 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.161871653676483,
            "unit": "iter/sec",
            "range": "stddev: 0.003411850202097734",
            "extra": "mean: 193.7281798333288 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9d4679187ccb3e794897fc2686eff52985e6786e",
          "message": "chore: expand ci-benchmark to cover all benchmark tests, drop codspeed (#1724)\n\n- Run `-m benchmark python/` instead of a single file so all\nbenchmark-marked tests are covered (ibis_yaml, test_into_backend,\nlineage_utils, gen_downstream_performance, test_api)\n- Add setup-just + download-data, full docker compose startup, and\npostgres env vars to match what codspeed was doing\n- Switch to --all-extras --all-groups to ensure all deps are available\n- Delete ci-codspeed.yaml (no longer needed)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-20T16:03:51-04:00",
          "tree_id": "9d5a52167f6ee1d89cf75c20799cc0afb57194b5",
          "url": "https://github.com/xorq-labs/xorq/commit/9d4679187ccb3e794897fc2686eff52985e6786e"
        },
        "date": 1774037237543,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.288273000019468,
            "unit": "iter/sec",
            "range": "stddev: 0.012670895415574052",
            "extra": "mean: 137.20671549999963 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.088775911467427,
            "unit": "iter/sec",
            "range": "stddev: 0.06200302704856895",
            "extra": "mean: 244.57197499999666 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7077462630749197,
            "unit": "iter/sec",
            "range": "stddev: 0.2036590965395006",
            "extra": "mean: 1.4129357542000094 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.811332196155377,
            "unit": "iter/sec",
            "range": "stddev: 0.02430469775349245",
            "extra": "mean: 207.84264300001496 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.075149948752028,
            "unit": "iter/sec",
            "range": "stddev: 0.013052697628246585",
            "extra": "mean: 197.0385131666698 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.947043257419319,
            "unit": "iter/sec",
            "range": "stddev: 0.008164072038310181",
            "extra": "mean: 202.14094519999435 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7c433b9124e006d7e9bc317403a1b946728c35b4",
          "message": "feat(expr): add ExprKind.Source to distinguish source from transformed expressions (#1727)\n\n## Summary\n- Adds `ExprKind.Source` enum variant to distinguish bare source tables\n(e.g. `DatabaseTable`, `InMemoryTable`, `Read`, `CachedNode`) from\ntransformed expressions\n- Adds `.ls.kind` and `.ls.unwrapped` accessors on `LETSQLAccessor` for\nconvenient source detection and Tag/HashingTag unwrapping\n- Updates `ExprMetadata.kind` to return `Source` when the expression\nroot is a source node\n\n## Test plan\n- [x] Unit tests for `ExprMetadata.kind` across source, expr, and\nunbound cases\n- [x] Tests for `.ls.kind` and `.ls.unwrapped` accessors\n- [x] Updated catalog and compiler tests to verify `ExprKind.Source`\nclassification\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-20T16:33:45-04:00",
          "tree_id": "731069740dc1e9bbf8ae7798f06d2f88cca3759e",
          "url": "https://github.com/xorq-labs/xorq/commit/7c433b9124e006d7e9bc317403a1b946728c35b4"
        },
        "date": 1774039039169,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.0971651626408185,
            "unit": "iter/sec",
            "range": "stddev: 0.015260012839226768",
            "extra": "mean: 140.9013284999986 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.391054024489112,
            "unit": "iter/sec",
            "range": "stddev: 0.012131681004755437",
            "extra": "mean: 227.7357542000061 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6690270012417427,
            "unit": "iter/sec",
            "range": "stddev: 0.1547173019259884",
            "extra": "mean: 1.4947079836000001 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.740907586165936,
            "unit": "iter/sec",
            "range": "stddev: 0.0317993450408753",
            "extra": "mean: 267.3148098333276 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.894147767567985,
            "unit": "iter/sec",
            "range": "stddev: 0.05255438446432482",
            "extra": "mean: 256.79559679999784 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.53863550966736,
            "unit": "iter/sec",
            "range": "stddev: 0.012780991822457467",
            "extra": "mean: 220.33053719999884 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c64271b522a78b6f2d328a9b5bb6472ecc5d97ac",
          "message": "perf(build): normalize sequential IDs for deterministic build hashes (#1728)\n\nProfile.idx and UDF class names carry session-global counters that leak\ninto build hashes and YAML, making builds non-reproducible when\nconnections or UDFs are created in different order.\n\nAdd replace_sources() as a general-purpose graph rewrite for swapping\nbackends in an expression tree (handles node.source, nested\ncache.storage.source, and opaque sub-expressions). Build\nnormalize_profiles() on top of it: sorts backends by content hash,\nassigns canonical idx=0,1,2,..., shallow-copies backends with shared\n.con so registered tables remain accessible. For UDFs, emit\n__func_name__ instead of __class__.__name__ to strip the counter suffix.\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-20T17:03:48-04:00",
          "tree_id": "04a47622d5e1eeedfdae37459190a2802604c84e",
          "url": "https://github.com/xorq-labs/xorq/commit/c64271b522a78b6f2d328a9b5bb6472ecc5d97ac"
        },
        "date": 1774040844033,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.530530557025831,
            "unit": "iter/sec",
            "range": "stddev: 0.018472935989538773",
            "extra": "mean: 132.7927683750012 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.4606790203155535,
            "unit": "iter/sec",
            "range": "stddev: 0.025570312880925487",
            "extra": "mean: 224.18111580000186 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6698653493599862,
            "unit": "iter/sec",
            "range": "stddev: 0.17179448048890866",
            "extra": "mean: 1.4928373306000027 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.9652485647507936,
            "unit": "iter/sec",
            "range": "stddev: 0.035385432869114004",
            "extra": "mean: 252.19099979999555 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.628535161062855,
            "unit": "iter/sec",
            "range": "stddev: 0.024663106045480252",
            "extra": "mean: 216.05107559998942 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.648907711518556,
            "unit": "iter/sec",
            "range": "stddev: 0.020607883002899927",
            "extra": "mean: 215.10429160000513 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "95b88b98e8bf4a1662aa0d395c28f31d20cc81ed",
          "message": "feat(catalog): add catalog schema (XOR-241) (#1686)\n\ndepends on #1682\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-03-21T08:17:27-04:00",
          "tree_id": "c07b0c409c2abf8e797496bb9e3dc369d0976d72",
          "url": "https://github.com/xorq-labs/xorq/commit/95b88b98e8bf4a1662aa0d395c28f31d20cc81ed"
        },
        "date": 1774095635667,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.546012253041351,
            "unit": "iter/sec",
            "range": "stddev: 0.019323357907162154",
            "extra": "mean: 132.52032549999626 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.168395445843866,
            "unit": "iter/sec",
            "range": "stddev: 0.0076264951905394085",
            "extra": "mean: 193.48364699998797 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7193009945376204,
            "unit": "iter/sec",
            "range": "stddev: 0.09503669014521596",
            "extra": "mean: 1.390238589400002 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.626299147592077,
            "unit": "iter/sec",
            "range": "stddev: 0.03477957333266271",
            "extra": "mean: 216.15549883333549 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.416662978069593,
            "unit": "iter/sec",
            "range": "stddev: 0.00599669886039116",
            "extra": "mean: 184.61551033333498 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.37757445771014,
            "unit": "iter/sec",
            "range": "stddev: 0.004860125901771431",
            "extra": "mean: 185.95744380000951 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a8cf1b748a0fcad1fcc8d1082a0c2cff65dab0be",
          "message": "fix(defer_utils): normalize path kwarg in Read nodes for cross-backen… (#1730)\n\n…d portability\n\nmake_read_kwargs now normalizes backend-specific path parameter names\n(paths, source, source_list) to \"path\" so Read nodes created on one\nbackend can be replayed on another. Previously, a deferred_read_parquet\ncreated with pandas (which uses \"source\") would fail when\nreplace_sources swapped it to xorq or duckdb.\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-21T18:34:00-04:00",
          "tree_id": "4691f1828a4504ec6c4933f4d7a500f146a3abac",
          "url": "https://github.com/xorq-labs/xorq/commit/a8cf1b748a0fcad1fcc8d1082a0c2cff65dab0be"
        },
        "date": 1774132653486,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.271762752726459,
            "unit": "iter/sec",
            "range": "stddev: 0.014476104349900158",
            "extra": "mean: 137.51823787499973 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.6702742698423485,
            "unit": "iter/sec",
            "range": "stddev: 0.004718521826116202",
            "extra": "mean: 214.12018700001454 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6722260329640458,
            "unit": "iter/sec",
            "range": "stddev: 0.2042989776845188",
            "extra": "mean: 1.4875948728000026 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.7666355994309866,
            "unit": "iter/sec",
            "range": "stddev: 0.014397552024903018",
            "extra": "mean: 265.4889154000102 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.076011788440166,
            "unit": "iter/sec",
            "range": "stddev: 0.03928674979170537",
            "extra": "mean: 245.33785766666938 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.7098597086713285,
            "unit": "iter/sec",
            "range": "stddev: 0.021293608297005374",
            "extra": "mean: 212.32054919998973 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9aeff8fd78e07a4370dafa7c0decf66e350dfded",
          "message": "fix(hash): stabilize ScalarUDF normalization across processes (#1738)\n\nnormalize_scalar_udf was passing computed_kwargs_expr through\nnormalize_expr -> normalize_op -> unbound_expr_to_default_sql, which\nembeds session-dependent AggUDF class names (e.g. _inner_fit_0) whose\nnumeric suffix is a process-global counter. Under pytest-xdist or\nmulti-module import, the counter value is non-deterministic, producing\ndifferent tokens for functionally identical expressions.\n\nAdd _normalize_computed_kwargs_expr that decomposes the sub-expression\ninto content-stable components (InMemoryTable data, AggUDF/ScalarUDF via\ntheir registered normalizers, Read/CachedNode), bypassing SQL generation\nentirely.\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-24T14:39:07+01:00",
          "tree_id": "4b9aa817698124ce9ce7ec14e44bd47511fbc5e1",
          "url": "https://github.com/xorq-labs/xorq/commit/9aeff8fd78e07a4370dafa7c0decf66e350dfded"
        },
        "date": 1774359758929,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.72789181376008,
            "unit": "iter/sec",
            "range": "stddev: 0.017996127194494296",
            "extra": "mean: 129.40139744443968 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.780616394208404,
            "unit": "iter/sec",
            "range": "stddev: 0.0109189730174729",
            "extra": "mean: 209.1780468333487 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7224314450471878,
            "unit": "iter/sec",
            "range": "stddev: 0.176256267546407",
            "extra": "mean: 1.3842143871999952 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.9044615532510756,
            "unit": "iter/sec",
            "range": "stddev: 0.0264228777815492",
            "extra": "mean: 256.117261333344 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.408593202831662,
            "unit": "iter/sec",
            "range": "stddev: 0.03764269981772311",
            "extra": "mean: 226.82972866666282 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.056987659174202,
            "unit": "iter/sec",
            "range": "stddev: 0.008880181813002038",
            "extra": "mean: 197.7461815999959 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "2f8525680f8fe29bf357149829fa376c173dccb4",
          "message": "fix(ibis yaml): stabilize inmemory read yaml (#1739)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-24T14:45:37+01:00",
          "tree_id": "a8528c93c523ffbe2369c89614b28cdc1b94c5bf",
          "url": "https://github.com/xorq-labs/xorq/commit/2f8525680f8fe29bf357149829fa376c173dccb4"
        },
        "date": 1774360149115,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.659368255076623,
            "unit": "iter/sec",
            "range": "stddev: 0.01885134078539925",
            "extra": "mean: 130.5590704999986 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.796912487616206,
            "unit": "iter/sec",
            "range": "stddev: 0.008624240059223413",
            "extra": "mean: 208.46742620000214 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7241310964654554,
            "unit": "iter/sec",
            "range": "stddev: 0.17027949221878852",
            "extra": "mean: 1.380965414800005 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.452619671845606,
            "unit": "iter/sec",
            "range": "stddev: 0.030117449009850954",
            "extra": "mean: 224.58688899999876 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.092781585088609,
            "unit": "iter/sec",
            "range": "stddev: 0.006841190526112492",
            "extra": "mean: 196.35634933332824 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.083475867359756,
            "unit": "iter/sec",
            "range": "stddev: 0.007209221055844399",
            "extra": "mean: 196.715795666672 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28b009973aa2206a14a7d1d74a2070fd1b7c2e5c",
          "message": "ref(hash): use __func_name__ for UDF SQL names and extract canonicalize_expr(#1735)\n\nUDF SQL names used a session-global counter suffix (e.g. add_one_0) via\ntype(op).__name__, making build hashes non-deterministic across\nsessions. Switch __sql_name__ and all backend UDF registration to use\n__func_name__ (the user-given name), which is stable. This aligns SQL\ngeneration with how DataFusion already registers UDFs.\n\nExtract canonicalize_expr as a shared primitive combining\n_sanitize_generated_names and normalize_profiles, so both the YAML\nserialization path (ExprDumper) and the hashing path (get_expr_hash)\noperate on the same canonical expression form.\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-24T16:07:57+01:00",
          "tree_id": "981ef1690da90d538c07fcc0cbf6c1c02305116a",
          "url": "https://github.com/xorq-labs/xorq/commit/28b009973aa2206a14a7d1d74a2070fd1b7c2e5c"
        },
        "date": 1774365096717,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.2937473220548865,
            "unit": "iter/sec",
            "range": "stddev: 0.008323497933243076",
            "extra": "mean: 137.10373500000372 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.469494083582187,
            "unit": "iter/sec",
            "range": "stddev: 0.03963429477248663",
            "extra": "mean: 223.73896939998303 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6861542114716288,
            "unit": "iter/sec",
            "range": "stddev: 0.19092916778291233",
            "extra": "mean: 1.457398327200019 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.10135931417374,
            "unit": "iter/sec",
            "range": "stddev: 0.039181950658823295",
            "extra": "mean: 243.82160239999848 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.782849868153104,
            "unit": "iter/sec",
            "range": "stddev: 0.016557549503346404",
            "extra": "mean: 209.08036580001408 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.764747019721515,
            "unit": "iter/sec",
            "range": "stddev: 0.006614746558310453",
            "extra": "mean: 209.87473119998867 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1fbb2442c46c0528fe758d41467932f71c20ee35",
          "message": "ref(ExprMetadata): extract explicit fields, drop expr dependency (#1740)\n\nRefactor ExprMetadata from cached_property-on-expr to explicit attrs\nfields (kind, schema_out, schema_in) with from_expr and from_dict\nclassmethods. This makes ExprMetadata a standalone value object that can\nbe constructed from either a live expression or serialized dict, without\nholding a reference to the original expr.\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-24T16:42:03+01:00",
          "tree_id": "98d699c6f72e12add307de1b191c968ae3354587",
          "url": "https://github.com/xorq-labs/xorq/commit/1fbb2442c46c0528fe758d41467932f71c20ee35"
        },
        "date": 1774367138961,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.040508558703897,
            "unit": "iter/sec",
            "range": "stddev: 0.011050449878968202",
            "extra": "mean: 142.03519414286347 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.593732603764169,
            "unit": "iter/sec",
            "range": "stddev: 0.01154437775524795",
            "extra": "mean: 217.687899200007 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6658334369331319,
            "unit": "iter/sec",
            "range": "stddev: 0.20522939557190045",
            "extra": "mean: 1.5018771130000004 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.4921736407631645,
            "unit": "iter/sec",
            "range": "stddev: 0.034986682459745386",
            "extra": "mean: 286.3546040000074 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.10669827317463,
            "unit": "iter/sec",
            "range": "stddev: 0.03696854485953038",
            "extra": "mean: 243.50461939999377 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.813226708489003,
            "unit": "iter/sec",
            "range": "stddev: 0.007659360792823017",
            "extra": "mean: 207.76083500000482 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bbe8b14f4553bbcadd7fce964ab8cfb9e75f4645",
          "message": "release: 0.3.16 (#1741)",
          "timestamp": "2026-03-24T16:44:33+01:00",
          "tree_id": "18097cb482bd8b16fadd68709c24e8535cd64658",
          "url": "https://github.com/xorq-labs/xorq/commit/bbe8b14f4553bbcadd7fce964ab8cfb9e75f4645"
        },
        "date": 1774367286811,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.550981216104259,
            "unit": "iter/sec",
            "range": "stddev: 0.017286646173318317",
            "extra": "mean: 132.43311980001522 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.048242884546638,
            "unit": "iter/sec",
            "range": "stddev: 0.04598807111314305",
            "extra": "mean: 247.02075159998458 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7117670583218857,
            "unit": "iter/sec",
            "range": "stddev: 0.23780485390311107",
            "extra": "mean: 1.4049540342 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.712636636831799,
            "unit": "iter/sec",
            "range": "stddev: 0.029157633845737486",
            "extra": "mean: 212.1954389999985 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.052704061316964,
            "unit": "iter/sec",
            "range": "stddev: 0.006906162891050931",
            "extra": "mean: 197.91382749999306 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.042195903826251,
            "unit": "iter/sec",
            "range": "stddev: 0.011774443362783379",
            "extra": "mean: 198.3262886000034 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "712f41a8535df7f66974920e0fa668e3de34bbb4",
          "message": "fix(caching): avoid wrapping Expr without backend (#1731)\n\nWhen an expression containing a ParquetSnapshotCache is serialized and\ndeserialized, cache.storage.source is reconstructed from the YAML\nprofile as a fresh backend instance with a different Profile.idx. The\nidentity comparison in maybe_prevent_cross_source_caching treated this\nnew instance as a different backend and wrapped the parent expression\nwith into_backend, which changed the SQL and produced a different cache\nkey at execute time than at build time.\n\nFix by comparing backends via Profile.almost_equals(), which ignores the\nsession-scoped idx counter and compares only connection parameters. Add\n_backends_equivalent() as a named helper to make the intent explicit.\n\nAdd two regression tests in test_compiler.py:\n- test_memtable_cache_key_stable_across_roundtrip: asserts calc_key on\nthe sanitized expr equals calc_key on the loaded expr\n- test_memtable_creates_same_key: end-to-end check that\nsanitized.execute() and loaded.execute() write to the same cache file\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-24T12:31:35-04:00",
          "tree_id": "ed6025cb30b365971c5c32bea25a296064c58a6f",
          "url": "https://github.com/xorq-labs/xorq/commit/712f41a8535df7f66974920e0fa668e3de34bbb4"
        },
        "date": 1774370099931,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.919294831775625,
            "unit": "iter/sec",
            "range": "stddev: 0.014632435591286817",
            "extra": "mean: 100.81361799999979 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.749387740916942,
            "unit": "iter/sec",
            "range": "stddev: 0.05500343880728346",
            "extra": "mean: 266.71021219999034 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6959651067437568,
            "unit": "iter/sec",
            "range": "stddev: 0.27387412128822597",
            "extra": "mean: 1.436853644399997 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.210184748965802,
            "unit": "iter/sec",
            "range": "stddev: 0.008772473471635869",
            "extra": "mean: 191.93177366666228 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.112321109813066,
            "unit": "iter/sec",
            "range": "stddev: 0.007531770085546423",
            "extra": "mean: 195.6058664000011 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.979067846201776,
            "unit": "iter/sec",
            "range": "stddev: 0.04709618074481935",
            "extra": "mean: 251.315141799995 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5b9358f4e64f3d69ef46a8b7bd53ea9551dd251f",
          "message": "feat(tui): load expr lazily (#1720)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-03-24T14:33:47-04:00",
          "tree_id": "788bfd3be8ae85d705a434f969e9ad20c31b49fb",
          "url": "https://github.com/xorq-labs/xorq/commit/5b9358f4e64f3d69ef46a8b7bd53ea9551dd251f"
        },
        "date": 1774377446075,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.49514788700546,
            "unit": "iter/sec",
            "range": "stddev: 0.021389126628770308",
            "extra": "mean: 117.71425445455077 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.697111135625072,
            "unit": "iter/sec",
            "range": "stddev: 0.057709434447242025",
            "extra": "mean: 270.4814552000016 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6578538951948548,
            "unit": "iter/sec",
            "range": "stddev: 0.168517720203355",
            "extra": "mean: 1.520094366400008 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.894842936150626,
            "unit": "iter/sec",
            "range": "stddev: 0.03436455381024649",
            "extra": "mean: 256.7497628000183 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.486201710889254,
            "unit": "iter/sec",
            "range": "stddev: 0.04844635418202476",
            "extra": "mean: 222.9057150000017 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.709041787577795,
            "unit": "iter/sec",
            "range": "stddev: 0.010039984904401751",
            "extra": "mean: 212.3574274999953 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c1d572a3b4d19b94512780d2421ba09ac07268c",
          "message": "ref(cli): simplify xorq run, use match in replace_cache_table (#1745)\n\n## Summary\n- Remove `--alias` / `--name` from `xorq run` — alias-based execution\nnow lives in `xorq catalog run` (#1744). `BUILD_PATH` becomes a required\npositional argument.\n- Delete `_resolve_alias` helper and `test_cli_run_alias.py` (143 lines)\n- Convert `replace_cache_table` from `if/elif` to `match` statement\n\n## Test plan\n- [x] 15 CLI tests pass\n- [x] 23 relation tests pass\n- [ ] CI green\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-24T16:13:20-04:00",
          "tree_id": "b549a24383a96bb60424e60c175b7a0905c691df",
          "url": "https://github.com/xorq-labs/xorq/commit/5c1d572a3b4d19b94512780d2421ba09ac07268c"
        },
        "date": 1774383420127,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.2795850587745745,
            "unit": "iter/sec",
            "range": "stddev: 0.020024162077535046",
            "extra": "mean: 137.37046712499534 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.801884917307833,
            "unit": "iter/sec",
            "range": "stddev: 0.011280717517379807",
            "extra": "mean: 208.25155479999466 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7395453447948385,
            "unit": "iter/sec",
            "range": "stddev: 0.17123351572521534",
            "extra": "mean: 1.352182130600005 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.3272749476211265,
            "unit": "iter/sec",
            "range": "stddev: 0.04302942666056022",
            "extra": "mean: 231.09231839999893 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.184252946420046,
            "unit": "iter/sec",
            "range": "stddev: 0.0035414231074130547",
            "extra": "mean: 192.8918226666667 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.019838149374617,
            "unit": "iter/sec",
            "range": "stddev: 0.009310413437196978",
            "extra": "mean: 199.20960999999218 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "97ea343ccab457c18284f98e9bb48f656cb10247",
          "message": "fix(tui): make test_j_k_moves_cursor resilient to slow CI (#1746)\n\n## Summary\n- Replace fixed `3x pilot.pause()` with a poll loop (up to 20\niterations) in `test_j_k_moves_cursor`, so the test waits for the async\n`_do_refresh` to actually populate both rows before asserting\n\n## Test plan\n- [x] Fixes flaky CI failure in `test_tui.py::test_j_k_moves_cursor`\n- [ ] CI green\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-24T16:13:54-04:00",
          "tree_id": "8504ec2e1152c0bd0fe9363e0b6eb0389586c796",
          "url": "https://github.com/xorq-labs/xorq/commit/97ea343ccab457c18284f98e9bb48f656cb10247"
        },
        "date": 1774383435518,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.07136570477175,
            "unit": "iter/sec",
            "range": "stddev: 0.023024563811801572",
            "extra": "mean: 123.8947703000008 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.794114196198805,
            "unit": "iter/sec",
            "range": "stddev: 0.010301473366892761",
            "extra": "mean: 208.5891072000095 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7389354170805974,
            "unit": "iter/sec",
            "range": "stddev: 0.18481830579451103",
            "extra": "mean: 1.353298240800018 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.066798209830199,
            "unit": "iter/sec",
            "range": "stddev: 0.030292499236279715",
            "extra": "mean: 245.8936855000123 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.476671936075309,
            "unit": "iter/sec",
            "range": "stddev: 0.040758362639146097",
            "extra": "mean: 223.38022849999106 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.079770919196541,
            "unit": "iter/sec",
            "range": "stddev: 0.008319157362658551",
            "extra": "mean: 196.8592709999939 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "a065d884cf974f15ff16aa08746ccb305e43c745",
          "message": "ref(graph): extract replace_unbound utility and simplify exchanger (#1742)\n\n## Summary\n- Extract a reusable `replace_unbound()` helper in `graph_utils` that\nreplaces a single `UnboundTable` node in an expression graph\n- Refactor `replace_one_unbound` and\n`UnboundExprExchanger.set_one_unbound_name` in `flight/exchanger.py` to\nuse it, eliminating duplicated inline `replace_nodes` callbacks\n\n## Test plan\n- [x] All 66 flight tests pass (`python -m pytest\npython/xorq/flight/tests/ -x -q`)\n- [ ] CI green\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-24T16:33:15-04:00",
          "tree_id": "73159c3b707b2e1d5bf2fcdb0ff464a4201bb8bb",
          "url": "https://github.com/xorq-labs/xorq/commit/a065d884cf974f15ff16aa08746ccb305e43c745"
        },
        "date": 1774384600404,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.831531109542153,
            "unit": "iter/sec",
            "range": "stddev: 0.02414844761390839",
            "extra": "mean: 113.23064909090745 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.0114145313950615,
            "unit": "iter/sec",
            "range": "stddev: 0.06402858234126016",
            "extra": "mean: 249.28862180000806 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7210380601982016,
            "unit": "iter/sec",
            "range": "stddev: 0.17764174715232448",
            "extra": "mean: 1.3868893408000076 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.071102815797565,
            "unit": "iter/sec",
            "range": "stddev: 0.014262704227181518",
            "extra": "mean: 197.19576516665902 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.18084998551492,
            "unit": "iter/sec",
            "range": "stddev: 0.010888093641681725",
            "extra": "mean: 193.0185206666645 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.775122076974908,
            "unit": "iter/sec",
            "range": "stddev: 0.03390813169280806",
            "extra": "mean: 209.41872979999516 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ad57e3683815701719b7ef3ffa2b8330c1dce6a8",
          "message": "feat(catalog): add bind(), ExprComposer, and Catalog.source/bind (#1748)\n\n## Summary\n- Add `ExprKind.Composed` variant and `sources` field to `ExprMetadata`\nfor tracking composed expression provenance\n- Add `bind()` function and `ExprComposer` class for chaining catalog\nentries through unbound transforms with schema validation and provenance\ntagging (`HashingTag`)\n- Add `Catalog.source()` and `Catalog.bind()` convenience methods;\nrefactor `check_consistency` out of `__attrs_post_init__` into callers\n- Add `safe_eval` utility for restricted inline code evaluation\n(AST-whitelisted)\n\n## Test plan\n- [x] `test_bind.py` covers schema validation, single/multi-step bind,\nprovenance tagging, error cases, and ExprComposer with inline code\n- [ ] Run full catalog test suite: `python -m pytest\npython/xorq/catalog/tests/ -x -q`\n- [ ] Verify no regressions in `python/xorq/ibis_yaml/tests/`\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-03-25T16:01:31-04:00",
          "tree_id": "5e2050bd8d3ce39b4069c0bf7a6cd1f72e2e8621",
          "url": "https://github.com/xorq-labs/xorq/commit/ad57e3683815701719b7ef3ffa2b8330c1dce6a8"
        },
        "date": 1774469077814,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.225268311973808,
            "unit": "iter/sec",
            "range": "stddev: 0.007953874489986012",
            "extra": "mean: 121.57658110000682 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.532299452933284,
            "unit": "iter/sec",
            "range": "stddev: 0.005457160137213551",
            "extra": "mean: 180.75666519999913 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.798997674547794,
            "unit": "iter/sec",
            "range": "stddev: 0.19637688947407225",
            "extra": "mean: 1.2515680981999935 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.676016041045612,
            "unit": "iter/sec",
            "range": "stddev: 0.00984374916225649",
            "extra": "mean: 176.17991083333587 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.735057325089699,
            "unit": "iter/sec",
            "range": "stddev: 0.056523475184172466",
            "extra": "mean: 211.19068500000822 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.076851078060554,
            "unit": "iter/sec",
            "range": "stddev: 0.03548886609925667",
            "extra": "mean: 245.2873506666625 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9de2a7b767af75988c1b0885553c2ce86d5ba712",
          "message": "fix(tui): eliminate race in test_j_k_moves_cursor (#1749)\n\n## Summary\n- Replace the racy polling loop in `test_j_k_moves_cursor` with a\ndeterministic `_populate_table()` helper that calls `_render_refresh()`\ndirectly\n- Add module-level docstring warning against waiting for the async\n`_do_refresh` worker in tests\n- The helper matches the pattern already used by every other multi-row\npilot test\n\nFixes the flaky failure seen in [CI run\n#23593493819](https://github.com/xorq-labs/xorq/actions/runs/23593493819/job/68708365740?pr=1718).\n\n## Test plan\n- [x] `pytest\npython/xorq/catalog/tests/test_tui.py::test_j_k_moves_cursor` passes\ndeterministically\n- [ ] Full TUI test suite passes in CI\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-26T14:26:08+01:00",
          "tree_id": "c11658bab42a9a757b2408e2ae4853896075083a",
          "url": "https://github.com/xorq-labs/xorq/commit/9de2a7b767af75988c1b0885553c2ce86d5ba712"
        },
        "date": 1774531779080,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.92571392289171,
            "unit": "iter/sec",
            "range": "stddev: 0.02177740561372316",
            "extra": "mean: 126.17159914285025 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.737196115430191,
            "unit": "iter/sec",
            "range": "stddev: 0.010500745153971624",
            "extra": "mean: 211.09533480000096 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7148485052208444,
            "unit": "iter/sec",
            "range": "stddev: 0.20195074406536406",
            "extra": "mean: 1.3988977981999995 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.9093185707456133,
            "unit": "iter/sec",
            "range": "stddev: 0.012366946065534226",
            "extra": "mean: 255.79905600000075 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.422723763339441,
            "unit": "iter/sec",
            "range": "stddev: 0.026374781722653153",
            "extra": "mean: 226.10500983333756 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.118037592729267,
            "unit": "iter/sec",
            "range": "stddev: 0.0039178327527824575",
            "extra": "mean: 195.38738859999967 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e2dc4d774f3468996aa0300cc8fda41ab4235e70",
          "message": "feat(catalog): add ExprComposer.from_expr to recover composer from ta… (#1750)\n\n…gged expr\n\nWalks HashingTag nodes (SOURCE, TRANSFORM, CODE) embedded during\ncomposition and reconstructs the original ExprComposer fields. This\nenables round-tripping: build an expr via ExprComposer, then recover the\nrecipe from the expression's provenance tags.\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-26T14:44:17-04:00",
          "tree_id": "39a1a653fcb5ca345c446426b89660a777a84348",
          "url": "https://github.com/xorq-labs/xorq/commit/e2dc4d774f3468996aa0300cc8fda41ab4235e70"
        },
        "date": 1774550877890,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.1209485491658855,
            "unit": "iter/sec",
            "range": "stddev: 0.00799508934433635",
            "extra": "mean: 140.43072957143264 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.660113759472459,
            "unit": "iter/sec",
            "range": "stddev: 0.008254910361403757",
            "extra": "mean: 214.58703619999255 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6964890674242387,
            "unit": "iter/sec",
            "range": "stddev: 0.21510674112879283",
            "extra": "mean: 1.4357727159999911 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.782383183306556,
            "unit": "iter/sec",
            "range": "stddev: 0.04090062614467218",
            "extra": "mean: 264.38357816666286 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.537353581868632,
            "unit": "iter/sec",
            "range": "stddev: 0.027482182719598958",
            "extra": "mean: 220.39278666666462 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.014464389680994,
            "unit": "iter/sec",
            "range": "stddev: 0.013423779863604362",
            "extra": "mean: 199.4230933333275 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7b49e0bdec82d869a3c012bdb4e48800df81e10f",
          "message": "ref(catalog): hoist work outside sync context in _add_zip (#1756)\n\nMove BuildZip construction, md5sum computation, and ensure_dirs() before\nthe maybe_synchronizing context to minimize time spent holding the\npull/push lock. Cache BuildZip.md5sum with cached_property so the\neagerly-computed hash is reused inside _add().\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-29T09:11:42-04:00",
          "tree_id": "e6df5f2dfb1b378754ec5418d1221e695f63a692",
          "url": "https://github.com/xorq-labs/xorq/commit/7b49e0bdec82d869a3c012bdb4e48800df81e10f"
        },
        "date": 1774790104968,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.357136497617827,
            "unit": "iter/sec",
            "range": "stddev: 0.02447350972131739",
            "extra": "mean: 119.65821071428552 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.810607730251565,
            "unit": "iter/sec",
            "range": "stddev: 0.01192018947367316",
            "extra": "mean: 207.87394359999212 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7332815898752421,
            "unit": "iter/sec",
            "range": "stddev: 0.20814873148102098",
            "extra": "mean: 1.363732587599992 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.1382371070610455,
            "unit": "iter/sec",
            "range": "stddev: 0.039242892172879414",
            "extra": "mean: 241.64879250000126 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.666987376334846,
            "unit": "iter/sec",
            "range": "stddev: 0.03573274381639956",
            "extra": "mean: 214.27098883334375 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.209716366505331,
            "unit": "iter/sec",
            "range": "stddev: 0.005187956001053331",
            "extra": "mean: 191.94902940000134 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6c6779deb2b0c6080447d52f4c92ee037f91263b",
          "message": "ref: use pyyaml-12 (#1718)",
          "timestamp": "2026-03-29T10:32:08-04:00",
          "tree_id": "164a08f34bcdff44fc1e30030e7f54ccfc8c38ef",
          "url": "https://github.com/xorq-labs/xorq/commit/6c6779deb2b0c6080447d52f4c92ee037f91263b"
        },
        "date": 1774794935155,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.082729501239509,
            "unit": "iter/sec",
            "range": "stddev: 0.02030216158216564",
            "extra": "mean: 110.0990621666682 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.020456775733639,
            "unit": "iter/sec",
            "range": "stddev: 0.009409742379004738",
            "extra": "mean: 199.18506316666176 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6635974224124213,
            "unit": "iter/sec",
            "range": "stddev: 0.2499054311380193",
            "extra": "mean: 1.5069377400000008 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.092857939075986,
            "unit": "iter/sec",
            "range": "stddev: 0.047878409254613154",
            "extra": "mean: 244.32805019999364 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.9574051947110798,
            "unit": "iter/sec",
            "range": "stddev: 0.02702936712280511",
            "extra": "mean: 252.69082916666247 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.6138939301405095,
            "unit": "iter/sec",
            "range": "stddev: 0.04003680489435473",
            "extra": "mean: 216.73666866666488 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5999fbcedf10b24a08a440c700a232675940f3f1",
          "message": "feat(catalog): compose and run commands (#1753)\n\n## Summary\n- Add `xorq catalog compose` command to assemble, build, and persist\ncomposed expressions to catalog (with `--dry-run`, `--alias`, `--code`)\n- Add `xorq catalog run` command that composes and executes in one shot\n— accepts multiple entries, inline code, all output formats, `--limit`,\nand Arrow IPC stdin via shared `read_pyarrow_stream`/`maybe_open`\nmachinery\n\n## Test plan\n- [x] `python -m pytest python/xorq/catalog/tests/test_bind.py -v` —\nExprComposer, bind, source, provenance tests\n- [x] `python -m pytest python/xorq/catalog/tests/test_cli.py -v` — run\n(single, multi-entry, piped arrow, code, limit, formats), compose\n(alias, dry-run, code), roundtrip tests\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: dlovell <dlovell@gmail.com>\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-29T11:30:39-04:00",
          "tree_id": "028e97d4430776946630f5e378ad58e801438221",
          "url": "https://github.com/xorq-labs/xorq/commit/5999fbcedf10b24a08a440c700a232675940f3f1"
        },
        "date": 1774798426564,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.696205185932151,
            "unit": "iter/sec",
            "range": "stddev: 0.015537860068127957",
            "extra": "mean: 129.9341657142788 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.470342515713793,
            "unit": "iter/sec",
            "range": "stddev: 0.016149514437676607",
            "extra": "mean: 182.80390983333442 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.8020129620677198,
            "unit": "iter/sec",
            "range": "stddev: 0.1437602517613917",
            "extra": "mean: 1.2468626410000128 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 6.032132057558482,
            "unit": "iter/sec",
            "range": "stddev: 0.009956590111558545",
            "extra": "mean: 165.7788639999954 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.936789292906865,
            "unit": "iter/sec",
            "range": "stddev: 0.012902924758668487",
            "extra": "mean: 168.4412147142861 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.11421853256682,
            "unit": "iter/sec",
            "range": "stddev: 0.045525939192591625",
            "extra": "mean: 195.53329479999775 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5becdfbebd89563c4e3e41c1c445df1a54797bfe",
          "message": "feat(direnv): add worktree helper script and envrcs documentation (#1760)\n\nAdd setup-worktree script that copies gitignored direnv files\n(.envrc.secrets, .envrc.user, .env.*) from the main worktree into new\nworktrees, and a README documenting the composable .envrcs/ layout.\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T08:47:16-04:00",
          "tree_id": "33dbe54166d7b84fc65f26c7631ad0b5c27d3950",
          "url": "https://github.com/xorq-labs/xorq/commit/5becdfbebd89563c4e3e41c1c445df1a54797bfe"
        },
        "date": 1774875047557,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.006271071942288,
            "unit": "iter/sec",
            "range": "stddev: 0.022816738765313965",
            "extra": "mean: 124.90209125000362 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.220928410706953,
            "unit": "iter/sec",
            "range": "stddev: 0.010934987198085847",
            "extra": "mean: 191.5368151666712 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7419717725998026,
            "unit": "iter/sec",
            "range": "stddev: 0.20760702420542704",
            "extra": "mean: 1.3477601667999977 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.552421067130173,
            "unit": "iter/sec",
            "range": "stddev: 0.030888832446126974",
            "extra": "mean: 219.66333633334045 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.5197511682926015,
            "unit": "iter/sec",
            "range": "stddev: 0.012482897374756557",
            "extra": "mean: 181.1675869999997 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.54862458363387,
            "unit": "iter/sec",
            "range": "stddev: 0.0063131147731386675",
            "extra": "mean: 180.22484400000374 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "95d0bd653e0766b84e3c4f7600fd5ca1be2e6398",
          "message": "fix(cli): restore --pdb behavior for catalog commands (#1762)\n\n## Summary\n- `click_context_catalog` and `click_context` were catching all\nexceptions and\nwrapping them as `ClickException`, which `PdbGroup` re-raises without\nentering\nthe debugger. Now they check whether `--pdb` is active and re-raise the\n  original exception so `post_mortem` fires.\n- Removed the unnecessary `import pdb as pdb_module` alias (the `--pdb`\noption\n  is already mapped to `use_pdb`, so no shadowing risk).\n\n## Test plan\n- [x] `test_pdb_flag_invokes_post_mortem` — mocks `pdb.post_mortem`,\ninvokes a\n  failing catalog command with `--pdb`, asserts `post_mortem` is called\n- [x] `test_no_pdb_flag_wraps_exception` — same failing command without\n`--pdb`,\n  asserts clean `Error:` output\n- [x] Full catalog test suite passes (93 tests)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-30T11:36:19-04:00",
          "tree_id": "ef9977cd0b8032cdbcd501ef767e44bda219844d",
          "url": "https://github.com/xorq-labs/xorq/commit/95d0bd653e0766b84e3c4f7600fd5ca1be2e6398"
        },
        "date": 1774885192185,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.894442806604657,
            "unit": "iter/sec",
            "range": "stddev: 0.00911272156862094",
            "extra": "mean: 101.06683312500309 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.17033897681224,
            "unit": "iter/sec",
            "range": "stddev: 0.05479829008935984",
            "extra": "mean: 239.78866120000362 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6689740912996067,
            "unit": "iter/sec",
            "range": "stddev: 0.19827587284829928",
            "extra": "mean: 1.4948262018000036 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.541409124263588,
            "unit": "iter/sec",
            "range": "stddev: 0.05206991761466325",
            "extra": "mean: 220.1959727999963 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.165820952940358,
            "unit": "iter/sec",
            "range": "stddev: 0.024206777571284407",
            "extra": "mean: 240.04872300000577 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.799154799731911,
            "unit": "iter/sec",
            "range": "stddev: 0.029422104161474904",
            "extra": "mean: 208.3700238333345 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "philip@gizmodata.com",
            "name": "Philip Moore",
            "username": "prmoore77"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "946e12530d637ff97dbfe4ed9090f22094489de2",
          "message": "chore: bump adbc-driver-gizmosql from >=1.1.3 to >=1.1.5 (#1763)\n\n## Summary\n- Bump `adbc-driver-gizmosql` optional dependency from `>=1.1.3` to\n`>=1.1.5` in the `[gizmosql]` extras group\n\n## Changes in adbc-driver-gizmosql 1.1.4-1.1.5\n- **1.1.4**: Strip SQL comments before DDL/DML keyword detection — fixes\ndbt integration where query comment prefixes prevented DDL/DML from\nbeing routed through `execute_update()`\n- **1.1.5**: Thread-safe `adbc_get_info()` with cached result — prevents\nconcurrent map writes crash in the Go ADBC driver\n(apache/arrow-adbc#1178)\n\nGenerated with [Claude Code](https://claude.com/claude-code)",
          "timestamp": "2026-03-31T13:07:34-04:00",
          "tree_id": "ddb6b59fc46ac5ad9daeeba913fbf4775e3d15b8",
          "url": "https://github.com/xorq-labs/xorq/commit/946e12530d637ff97dbfe4ed9090f22094489de2"
        },
        "date": 1774977068151,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.078780796450951,
            "unit": "iter/sec",
            "range": "stddev: 0.027411044322760245",
            "extra": "mean: 123.78105375000459 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.205327130397308,
            "unit": "iter/sec",
            "range": "stddev: 0.004474431315201847",
            "extra": "mean: 192.11088466666126 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6858395305784112,
            "unit": "iter/sec",
            "range": "stddev: 0.2017905023885121",
            "extra": "mean: 1.4580670192000127 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.948745017992734,
            "unit": "iter/sec",
            "range": "stddev: 0.04456905595602717",
            "extra": "mean: 253.24501720000399 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.227981780380563,
            "unit": "iter/sec",
            "range": "stddev: 0.02449929316454986",
            "extra": "mean: 236.51946766667228 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.902777921978407,
            "unit": "iter/sec",
            "range": "stddev: 0.025604757911969863",
            "extra": "mean: 203.96599966666903 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3b477e71d0a0ad87ba56e1da05044e4f0361374e",
          "message": "feat(catalog): fuse catalog source wrappers for database backends (#1761)\n\n## Summary\n\n- Adds `fuse_catalog_source()` to strip catalog-created `RemoteTable` +\n`HashingTag` wrappers when the source is a database table (not a\ndeferred `Read`)\n- Integrates fuse in `catalog run` before execution so composed queries\npush down to the backend as a single query\n- Skips fusion when the source contains `Read` ops, preserving the\n`RemoteTable` boundary for cross-engine data transfer\n\n## Test plan\n\n- [x] `test_fuse_strips_catalog_wrappers` — all CatalogTag HashingTags\nremoved\n- [x] `test_fuse_strips_catalog_remote_tables` — no RemoteTables left\nafter fuse\n- [x] `test_fuse_preserves_correctness` — fused expression produces\nidentical results\n- [x] `test_fuse_chained_transforms` — multi-transform chain fully\nstripped\n- [x] `test_fuse_bare_source` — source-only expression fused\n- [x] `test_fuse_noop_without_catalog_tags` — plain expressions returned\nunchanged\n- [x] `test_fuse_idempotent` — double-fuse returns same object\n- [x] `test_fuse_skips_read_source` — deferred reads preserve wrappers\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-03-31T14:06:16-04:00",
          "tree_id": "462e8a8d6269348ccc3e2493a985b771c362d1a2",
          "url": "https://github.com/xorq-labs/xorq/commit/3b477e71d0a0ad87ba56e1da05044e4f0361374e"
        },
        "date": 1774980592581,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.568028614474297,
            "unit": "iter/sec",
            "range": "stddev: 0.029122130295445543",
            "extra": "mean: 116.71296222221548 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.251789029154167,
            "unit": "iter/sec",
            "range": "stddev: 0.008760163278509274",
            "extra": "mean: 190.41130449999363 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7541752367394848,
            "unit": "iter/sec",
            "range": "stddev: 0.18225827890703636",
            "extra": "mean: 1.3259517831999972 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.305097755885,
            "unit": "iter/sec",
            "range": "stddev: 0.016670675596442405",
            "extra": "mean: 232.28276259999348 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.202106299200314,
            "unit": "iter/sec",
            "range": "stddev: 0.02344291061028665",
            "extra": "mean: 192.22982816666465 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.55811222542249,
            "unit": "iter/sec",
            "range": "stddev: 0.010319412787649945",
            "extra": "mean: 179.9172019999986 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "aa4115c6e360c3f9747930ac811e55413b5ed82a",
          "message": "feat: add xorq param  (#1747)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: Hussain Sultan <hussainz@gmail.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-03-31T15:39:55-04:00",
          "tree_id": "aaf122753f2d59586ded12a8d77651ed366b5a40",
          "url": "https://github.com/xorq-labs/xorq/commit/aa4115c6e360c3f9747930ac811e55413b5ed82a"
        },
        "date": 1774986198105,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.685662665754068,
            "unit": "iter/sec",
            "range": "stddev: 0.0023086526447782876",
            "extra": "mean: 93.58333977777988 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.38746617620059,
            "unit": "iter/sec",
            "range": "stddev: 0.05371580017344665",
            "extra": "mean: 227.9219850000004 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.734190770790132,
            "unit": "iter/sec",
            "range": "stddev: 0.18152409653547513",
            "extra": "mean: 1.3620438171999978 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.876978819003222,
            "unit": "iter/sec",
            "range": "stddev: 0.03390858734681332",
            "extra": "mean: 257.9327993999982 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.730988066534315,
            "unit": "iter/sec",
            "range": "stddev: 0.02662109975514199",
            "extra": "mean: 211.37233616667098 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.428675186840294,
            "unit": "iter/sec",
            "range": "stddev: 0.01880455463066153",
            "extra": "mean: 184.20700549999935 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "eeddd9938c8b40597b5ed7334b921b149e92f946",
          "message": "release: 0.3.17 (#1764)\n\n## Summary\n- Bump version from 0.3.16 to 0.3.17\n- Update CHANGELOG.md with git-cliff generated release notes\n\n## Highlights\n### Added\n- `bind()`, `ExprComposer`, and `Catalog.source/bind`\n- `ExprComposer.from_expr` to recover composer from table\n- Worktree helper script and envrcs documentation\n- `xorq param`\n\n### Changed\n- Lazy expr loading in TUI\n- Compose and run commands for catalog\n- Fuse catalog source wrappers for database backends\n- Bump adbc-driver-gizmosql to >=1.1.5\n\n### Fixed\n- Avoid wrapping Expr without backend\n- TUI test race conditions\n- Restore `--pdb` behavior for catalog commands\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-03-31T22:30:04-04:00",
          "tree_id": "4e53aa8d09c18f63fc7c1b8f7ebe8370942073be",
          "url": "https://github.com/xorq-labs/xorq/commit/eeddd9938c8b40597b5ed7334b921b149e92f946"
        },
        "date": 1775010815935,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.05013596207677,
            "unit": "iter/sec",
            "range": "stddev: 0.00533752350515361",
            "extra": "mean: 99.50114145454397 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.770123318327184,
            "unit": "iter/sec",
            "range": "stddev: 0.03718324607246915",
            "extra": "mean: 265.24331316666405 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6752102275348538,
            "unit": "iter/sec",
            "range": "stddev: 0.17091511479571908",
            "extra": "mean: 1.481020220399995 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.171409694364862,
            "unit": "iter/sec",
            "range": "stddev: 0.014172841737310185",
            "extra": "mean: 193.37087159999555 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.2009934215172136,
            "unit": "iter/sec",
            "range": "stddev: 0.058378410089016265",
            "extra": "mean: 238.03893500000868 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.7033576043613508,
            "unit": "iter/sec",
            "range": "stddev: 0.025302217345478138",
            "extra": "mean: 270.0252330000012 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "bc0094fe6ccdacad5426d8dba73a82761c79a07d",
          "message": "fix: use namespace-aware table lookup in cross-database expr building (#1773)\n\n## Summary\n- `_find_missing_tables` was checking `table_name in\nbackend.list_tables()` which only searches the default schema, causing\nfalse `ValueError` for tables in non-default schemas/catalogs (e.g.\n`CREDIT_CARD_ACCOUNTS` in a specific catalog/schema)\n- Now propagates the `DatabaseTable` node's `namespace` and uses\n`backend.table(name, database=...)` to test reachability directly\n- Adds `_namespace_to_database` helper to convert `Namespace(catalog,\ndatabase)` to the `database` kwarg format\n\n## Test plan\n- [x] Unit tests for `_namespace_to_database` (catalog+db, db-only,\nempty)\n- [x] `_find_missing_tables` correctly finds table in non-default schema\n- [x] `_find_missing_tables` still detects truly missing tables\n- [x] End-to-end `replace_sources` with cross-schema table (no transfer\nneeded)\n- [x] End-to-end `replace_sources` with catalog + schema namespace\n- [x] All 75 existing tests still pass\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-01T09:15:36-04:00",
          "tree_id": "73d0365d1acae022f0d4d76e2838a62d7f54d339",
          "url": "https://github.com/xorq-labs/xorq/commit/bc0094fe6ccdacad5426d8dba73a82761c79a07d"
        },
        "date": 1775049545603,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.318512456931578,
            "unit": "iter/sec",
            "range": "stddev: 0.011310115296727395",
            "extra": "mean: 136.63978928571348 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.0374861932368065,
            "unit": "iter/sec",
            "range": "stddev: 0.011253698380441364",
            "extra": "mean: 198.51171033333515 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7673498518986442,
            "unit": "iter/sec",
            "range": "stddev: 0.15607743811274324",
            "extra": "mean: 1.3031865419999917 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.745258335406764,
            "unit": "iter/sec",
            "range": "stddev: 0.03103191935801584",
            "extra": "mean: 210.73668266667292 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.403105283041942,
            "unit": "iter/sec",
            "range": "stddev: 0.007475128267185568",
            "extra": "mean: 185.07875520000994 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.107867245827535,
            "unit": "iter/sec",
            "range": "stddev: 0.018744013284144205",
            "extra": "mean: 195.7764271999963 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1faba5fe05f677ccc80606eed0d849d6b9876aff",
          "message": "feat(catalog): CatalogBackend abstraction with optional git-annex (#1752)\n\n## Summary\n\n- Introduce `CatalogBackend` ABC with `GitBackend` (plain git) and\n`GitAnnexBackend` implementations, decoupling `Catalog` from git-annex\nas a hard dependency\n- Add `Annex` wrapper, `RemoteConfig` hierarchy\n(`DirectoryRemoteConfig`, `S3RemoteConfig`), and auto-detection logic so\n`clone_from` / `from_repo_path` pick the right backend automatically\n(`annex=None` default)\n- Promote entry metadata to a git-tracked sidecar YAML so `entry.kind`,\n`.columns`, `.backends`, `.composed_from` work without fetching annex\ncontent; `entry.expr` / `entry.lazy_expr` auto-fetch on access\n- Add `Catalog.fetch_entries()` for bulk content fetch, `embedcreds`\nsupport for credential-free clones, `autoenable` field for git-annex\nnative auto-enable on clone, and `remote.log` as single source of truth\nfor remote config\n- Rename `ExprMetadata.sources` → `composed_from` (backwards-compatible\n`from_dict`)\n- Extract `BuildZip` and zip helpers into `zip_utils.py`; expose public\nAPI via `xo.catalog`\n- ADR-0003 documents the design, sidecar guidelines, and MinIO testing\ngaps\n\n## Test plan\n\n- [ ] `test_annex.py` — 31 tests for `RemoteConfig` round-trips,\n`from_env`, `embedcreds`, `from_dict` dispatch\n- [ ] `test_git_backend.py` — plain-git backend: init, add, remove,\nalias, clone\n- [ ] `test_catalog.py` — both backends via parametrized fixtures:\nauto-detection, `is_content_local`, sidecar metadata after drop,\n`fetch_entries` bulk, directory remote round-trip, S3/MinIO\n(`@pytest.mark.s3`)\n- [ ] `test_bind.py` — compose/bind with `composed_from` rename\n- [ ] Verify `xo.catalog` import works and CLI `xorq catalog schema` /\n`xorq catalog run` handle `ValueError`\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-01T15:00:39-04:00",
          "tree_id": "a5c21c88a7afc885f3cbb6b2388137268f0f1dd4",
          "url": "https://github.com/xorq-labs/xorq/commit/1faba5fe05f677ccc80606eed0d849d6b9876aff"
        },
        "date": 1775070262989,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.206258030491854,
            "unit": "iter/sec",
            "range": "stddev: 0.016774647162583242",
            "extra": "mean: 108.62176540000519 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.456421233358871,
            "unit": "iter/sec",
            "range": "stddev: 0.0529202027381431",
            "extra": "mean: 289.316588600002 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.640642559197733,
            "unit": "iter/sec",
            "range": "stddev: 0.2254908017040121",
            "extra": "mean: 1.5609328253999935 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.174176232331286,
            "unit": "iter/sec",
            "range": "stddev: 0.05375391809128115",
            "extra": "mean: 239.56822720000446 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.932746324945053,
            "unit": "iter/sec",
            "range": "stddev: 0.006944476740559963",
            "extra": "mean: 202.72682480000412 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.811781424320536,
            "unit": "iter/sec",
            "range": "stddev: 0.0065660836182044055",
            "extra": "mean: 207.82323880000604 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c9a67856a0bfbdcabaf0b3042fad4a46db85ba6f",
          "message": "fix(flake): git-annex on darwin (#1776)",
          "timestamp": "2026-04-03T07:36:05-04:00",
          "tree_id": "4911f38c6d9e0c36c042deaf068fc47e03b0a18a",
          "url": "https://github.com/xorq-labs/xorq/commit/c9a67856a0bfbdcabaf0b3042fad4a46db85ba6f"
        },
        "date": 1775216375909,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.209254954788369,
            "unit": "iter/sec",
            "range": "stddev: 0.018393339406260874",
            "extra": "mean: 138.7105888571471 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.379256759094006,
            "unit": "iter/sec",
            "range": "stddev: 0.029614226322932376",
            "extra": "mean: 228.34925080001085 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6238794056700844,
            "unit": "iter/sec",
            "range": "stddev: 0.24836414883236624",
            "extra": "mean: 1.6028738741999973 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.442176517567027,
            "unit": "iter/sec",
            "range": "stddev: 0.019154539442379236",
            "extra": "mean: 290.5138637999926 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.010287850755695,
            "unit": "iter/sec",
            "range": "stddev: 0.04756825232764665",
            "extra": "mean: 249.35865883332062 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.577499618668971,
            "unit": "iter/sec",
            "range": "stddev: 0.022943579088369796",
            "extra": "mean: 218.45987620000642 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8163ae83f183279e42bd1c57b20ab1fa7014df45",
          "message": "feat: parquet embedded provenance (#1777)\n\ndepends on #1776\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-03T13:27:41-04:00",
          "tree_id": "4f132815ce709987dae4b3dfb81f6de7ad929304",
          "url": "https://github.com/xorq-labs/xorq/commit/8163ae83f183279e42bd1c57b20ab1fa7014df45"
        },
        "date": 1775237474327,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.182663392836165,
            "unit": "iter/sec",
            "range": "stddev: 0.01466635141092103",
            "extra": "mean: 139.22412137500118 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.829134953105196,
            "unit": "iter/sec",
            "range": "stddev: 0.007433598864618276",
            "extra": "mean: 207.07642460001807 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.631431385974437,
            "unit": "iter/sec",
            "range": "stddev: 0.17557853519619715",
            "extra": "mean: 1.583703348000006 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.208536911551343,
            "unit": "iter/sec",
            "range": "stddev: 0.03308791807139567",
            "extra": "mean: 311.66853539998556 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.9690709227517025,
            "unit": "iter/sec",
            "range": "stddev: 0.033387902243576746",
            "extra": "mean: 251.9481308000195 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.224072219983098,
            "unit": "iter/sec",
            "range": "stddev: 0.051656689617794495",
            "extra": "mean: 236.73837660001027 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7d452c5e72cc85770a00e4c8d5bf51926ef69668",
          "message": "feat(tui): lazygit-style horizontal layout with SQL, info, and inline panels (#1755)\n\n## Summary\n- **Drop Expression Detail**\n- **Cache SQL queries in expr_metadata.json at build time** —\npre-compute SQL plans during `build_expr` and store them in the metadata\nZIP, eliminating runtime expression deserialization for SQL display\n- **Cache lineage chain in expr_metadata.json at build time** — extract\nlineage during `build_expr` and persist it alongside SQL queries in\nmetadata\n- **Move `extract_lineage_chain` to `lineage_utils`** — relocate lineage\nextraction logic to a shared utility module\n- **Lazygit-style horizontal layout** — redesign TUI with left column\n(expressions, revisions, git log) and right column (SQL, info, schema)\npanels\n- **Read lineage, SQL, and cache info from metadata** — TUI now reads\npre-computed lineage, sql_queries, and parquet_cache_paths directly from\n`ExprMetadata` instead of loading full expressions; removes\n`maybe_expr`, `maybe_sqls`, `maybe_lineage`, `_build_lineage_chain`,\n`maybe_cache_path`, `maybe_cache_info`, `_check_cached`\n\n## Test plan\n\n- [x] `pytest python/xorq/catalog/tests/test_tui.py` — 45 passing (3\npre-existing failures from missing local parquet fixture)\n- [ ] Manual: `xorq catalog --name flights tui` — verify schema panel\nshows side-by-side \"In | Out\" for expressions with schema_in\n- [ ] Manual: verify SQL panel reads from cached metadata without\nloading expressions\n- [ ] Manual: verify lineage displays from cached metadata\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-03T19:13:06-04:00",
          "tree_id": "caec20d266e15cf59796fb11b96c0dff8beb4211",
          "url": "https://github.com/xorq-labs/xorq/commit/7d452c5e72cc85770a00e4c8d5bf51926ef69668"
        },
        "date": 1775258195189,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.67094584788186,
            "unit": "iter/sec",
            "range": "stddev: 0.004581635655935121",
            "extra": "mean: 93.71240509092229 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.052869879412731,
            "unit": "iter/sec",
            "range": "stddev: 0.026853595220199703",
            "extra": "mean: 246.7387381666697 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7214459235706865,
            "unit": "iter/sec",
            "range": "stddev: 0.18851782141047616",
            "extra": "mean: 1.3861052746000042 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.729866591615553,
            "unit": "iter/sec",
            "range": "stddev: 0.051278395552969945",
            "extra": "mean: 211.4224535999938 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.7874328549606866,
            "unit": "iter/sec",
            "range": "stddev: 0.022466861133791274",
            "extra": "mean: 264.03108339999335 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.366764793329902,
            "unit": "iter/sec",
            "range": "stddev: 0.03896003962636287",
            "extra": "mean: 229.00248750000665 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "07348355edac8562007fecf5e4fef2c008c8a1d7",
          "message": "feat(catalog): replay, annex target support, and enableremote fixes (#1774)\n\n## Summary\n\n- **enableremote fix**: `from_repo_path` now auto-detects annex via\n`_has_annex_branch()` instead of checking `.git/annex` (fixes submodule\ndetection), calls `Annex.init_repo_path` before reading `remote.log`,\nand ensures the special remote is enabled after clone. Adds missing\n`enableremote` implementations for `DirectoryRemoteConfig` and\n`RsyncRemoteConfig`.\n\n- **`from_dict` kwargs fix**: `RemoteConfig.from_dict` now filters\n`kwargs` to valid attrs fields (previously only filtered the dict),\npreventing `TypeError` when remote.log contains unexpected keys. Catalog\ninit uses `_try_resolve_annex_remote` for graceful degradation when\ncredentials are unavailable instead of failing hard.\n\n- **Constants extraction**: `MAIN_BRANCH`, `ANNEX_BRANCH`,\n`DEFAULT_REMOTE` pulled into `catalog/constants.py`. All `Repo.init`\ncalls use `initial_branch=MAIN_BRANCH`.\n\n- **Replay module** (`catalog/replay.py`): Parses a catalog's git log\ninto typed `CatalogOp` objects (`AddEntry`, `AddAlias`, `RemoveEntry`,\n`RemoveAlias`, etc.) and replays them into a target catalog. Each op\nverifies the source commit's diff at parse time. Unrecognized commits\nfall through to `UnknownOp` (replayed via `git format-patch`/`am`).\n\n- **CLI commands**: `xorq catalog log` (inspect history, `--json`) and\n`xorq catalog replay` (replay into a new catalog with `--dry-run`,\n`--preserve-commits`/`--no-preserve-commits`, `--force`).\n\n- **Annex target support**: `xorq catalog init` and `xorq catalog\nreplay` accept `--env-file`, `--env-prefix`, `--gcs`, and `--remote-url`\nto create annex-backed catalogs and push to a git remote. Enables\ngit-to-annex catalog migration.\n\n## Test plan\n\n- [ ] `python -m pytest python/xorq/catalog/tests/test_cli.py -x -q` —\nlog, replay, init tests\n- [ ] `python -m pytest python/xorq/catalog/tests/test_catalog.py -x -q`\n— enableremote tests\n- [ ] `python -m pytest python/xorq/catalog/tests/test_catalog_ctor.py\n-x -q` — constructor tests\n- [ ] Manual: `xorq catalog --path <git-catalog> replay /tmp/new\n--env-file .env.catalog.s3 --gcs --remote-url <url>`\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-04T07:10:33-04:00",
          "tree_id": "ddf7877d047ecefc1e4b416f4a4f3b6665e02b1c",
          "url": "https://github.com/xorq-labs/xorq/commit/07348355edac8562007fecf5e4fef2c008c8a1d7"
        },
        "date": 1775301240338,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.42789970418584,
            "unit": "iter/sec",
            "range": "stddev: 0.017052742734817748",
            "extra": "mean: 106.06816272727376 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.6409849943014447,
            "unit": "iter/sec",
            "range": "stddev: 0.04533060180897935",
            "extra": "mean: 274.650953400004 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6917471192258712,
            "unit": "iter/sec",
            "range": "stddev: 0.20836044579387433",
            "extra": "mean: 1.4456149829999903 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.1477665663606675,
            "unit": "iter/sec",
            "range": "stddev: 0.011969334020878658",
            "extra": "mean: 194.2590028333342 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.133559716186939,
            "unit": "iter/sec",
            "range": "stddev: 0.002783514082420139",
            "extra": "mean: 194.79660416666414 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.210971966055474,
            "unit": "iter/sec",
            "range": "stddev: 0.05044982474602417",
            "extra": "mean: 237.47486519999939 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "0f68f22acce13711b4f60e19aebddb0982c9eca8",
          "message": "feat(catalog): add embed-readonly command to verify and embed read-on… (#1779)\n\n…ly S3 creds\n\nAdds S3RemoteConfig.assert_readonly(), Catalog.embed_readonly(), and the\n`xorq catalog embed-readonly` CLI command. The command verifies\ncredentials cannot write before embedding them into the git-annex\nbranch.\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-06T08:05:20-04:00",
          "tree_id": "230a8c99f1e840fb108dcbdce24c49e2301ed151",
          "url": "https://github.com/xorq-labs/xorq/commit/0f68f22acce13711b4f60e19aebddb0982c9eca8"
        },
        "date": 1775477327878,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.864868033565207,
            "unit": "iter/sec",
            "range": "stddev: 0.018088838019226557",
            "extra": "mean: 112.80483772727155 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.7916104679217972,
            "unit": "iter/sec",
            "range": "stddev: 0.05790016676510867",
            "extra": "mean: 263.7401728000043 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7058958799657897,
            "unit": "iter/sec",
            "range": "stddev: 0.20304668394603947",
            "extra": "mean: 1.4166395192000039 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.022340504777172,
            "unit": "iter/sec",
            "range": "stddev: 0.009682449137356483",
            "extra": "mean: 199.1103548333323 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.097701016875436,
            "unit": "iter/sec",
            "range": "stddev: 0.006251293243990253",
            "extra": "mean: 196.16685966666125 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.522247988564179,
            "unit": "iter/sec",
            "range": "stddev: 0.04600452722423799",
            "extra": "mean: 221.128961200003 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "32166bea32a8d015b86a8e7e3dc879eb1b4466e5",
          "message": "ref(packager): overhaul sdist build pipeline (#1781)\n\n## Summary\n\n- **Structured args**: Replace opaque `sys.argv` passthrough with\nexplicit typed fields on\n`SdistPackager`/`PackagedBuilder`/`PackagedRunner`, add validated\n`python_version` resolved from `requires-python`, consolidate on\n`tomlkit`\n- **Guaranteed lockfile + requirements**: `SdistPackager` always embeds\n`uv.lock` and derives `requirements.txt` via `uv export`; if a\npre-existing `requirements.txt` doesn't match the lockfile, the build\nerrors out (or overwrites it when `overwrite_requirements=True`)\n- **Hardened zip_utils**: Replace asserts with proper exceptions, add\nFIPS-safe hashing, simplify `ZipAppender` → `append_toplevel`\n- **SdistArchive**: Validated-path type that guarantees the sdist\ncontains `pyproject.toml`, `uv.lock`, and `requirements.txt`; replaces\nduplicated validation\n- **Rename classes**: `Sdister` → `SdistPackager`, `SdistBuilder` →\n`PackagedBuilder`, `SdistRunner` → `PackagedRunner` for clarity\n- **ADR-0004**: Records the decision to use uv as the sole packaging and\nexecution runtime — covers build, lock/export, isolated execution,\nPython version selection, hash-pinned requirements, and\n`--with`/`--with-requirements` resolution semantics\n- **Tests**: Unit tests for pure helpers, validation paths, archive\nrejection, and `ZipProxy` error handling\n\n## Test plan\n\n- [x] `python -m pytest python/xorq/ibis_yaml/tests/test_packager.py -x\n-q -m \"not slow\"` — unit tests pass\n- [x] `python -m pytest python/xorq/ibis_yaml/tests/test_packager.py -x\n-q` — full suite including slow integration tests\n- [x] `python -m pytest python/xorq/tests/test_cli.py -x -q -m \"not\nslow\"` — CLI tests unaffected by renames\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-06T08:15:24-04:00",
          "tree_id": "637554572da400bc542735995cb4f4c6febb36dc",
          "url": "https://github.com/xorq-labs/xorq/commit/32166bea32a8d015b86a8e7e3dc879eb1b4466e5"
        },
        "date": 1775477933267,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.145789900302,
            "unit": "iter/sec",
            "range": "stddev: 0.0064151303577945845",
            "extra": "mean: 98.56305027272779 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.799502291492403,
            "unit": "iter/sec",
            "range": "stddev: 0.036247550517032934",
            "extra": "mean: 263.19236659999774 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6785180837462401,
            "unit": "iter/sec",
            "range": "stddev: 0.19595168967876087",
            "extra": "mean: 1.4738000710000108 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.9697915815770415,
            "unit": "iter/sec",
            "range": "stddev: 0.006637198282629187",
            "extra": "mean: 201.21568150000257 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.856003142105943,
            "unit": "iter/sec",
            "range": "stddev: 0.02177839411080985",
            "extra": "mean: 205.9306740000011 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.8313077584683364,
            "unit": "iter/sec",
            "range": "stddev: 0.04216663414426505",
            "extra": "mean: 261.0074843999939 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5d3c0ce6173e4c628bf230e47c1f697b3139c702",
          "message": "perf(ibis_yaml): defer sklearn import in translate.py (#1792)\n\n## Summary\n- Defers the `sklearn.base.BaseEstimator` singledispatch registration in\n`translate.py` from module-load time to first use via\n`_ensure_sklearn_to_yaml_registered()` (guarded by `functools.cache`)\n- sklearn import pulls in scipy/joblib/numpy transitively, costing\n~0.48s — this removes that penalty from CLI cold-start and any codepath\nthat doesn't serialize sklearn estimators\n- The registration is triggered in `ExprDumper.dump_expr()`, the single\nentrypoint for YAML serialization\n\n## Test plan\n- [x] Existing `test_benchmark.py` and ibis_yaml tests pass\n- [x] Verify sklearn estimator round-trip still works (e.g.\n`test_sklearn_estimator_yaml`)\n- [x] Measure CLI cold-start improvement (~0.48s reduction)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-06T15:10:30-04:00",
          "tree_id": "fbb0cba8ece8e2a1758fc3387820569054d62cac",
          "url": "https://github.com/xorq-labs/xorq/commit/5d3c0ce6173e4c628bf230e47c1f697b3139c702"
        },
        "date": 1775502850037,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.470237801977433,
            "unit": "iter/sec",
            "range": "stddev: 0.004984151757353388",
            "extra": "mean: 95.50881450000475 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.520798731216718,
            "unit": "iter/sec",
            "range": "stddev: 0.02698380142032177",
            "extra": "mean: 284.02646000000686 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7004968470447087,
            "unit": "iter/sec",
            "range": "stddev: 0.16307296541216312",
            "extra": "mean: 1.4275581741999985 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.0287554756578645,
            "unit": "iter/sec",
            "range": "stddev: 0.013051349736234834",
            "extra": "mean: 198.85635816666536 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.9581431030590855,
            "unit": "iter/sec",
            "range": "stddev: 0.021718658458617892",
            "extra": "mean: 201.68841019998354 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.9806727035134117,
            "unit": "iter/sec",
            "range": "stddev: 0.03971636348677435",
            "extra": "mean: 251.2138209999989 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "58d6245cd6a317148bcc030d0e5c48e87b7599de",
          "message": "feat(cli): add RunLogger to run_cached_command and catalog run (#1778)\n\n## Summary\n- Add `RunLogger` instrumentation to `run_cached_command` in `cli.py`\nand `catalog run` in `catalog/cli.py` so all execution paths produce\nstructured run logs to `~/.local/share/xorq/runs/`\n- Refactor: push span status handling into\n`RunLogger.from_expr_hash(span=...)` to eliminate redundant try/except\nblocks in callers\n- Refactor: add `log_span_event` to log to both RunLogger and OTel span\nin one call\n- Refactor: extract `_resolve_single_entry` in `catalog/cli.py` to\nflatten deeply nested logic\n- Add unit tests for `log_span_event`, `from_expr_hash` span\nintegration, and `_NullRunLogger`\n\n## Changes\n\n### `logging_utils.py`\n- `RunLogger.from_expr_hash` accepts optional `span` — sets\n`StatusCode.OK`/`ERROR` and calls `finalize(span_context=...)`\nautomatically\n- `RunLogger.log_span_event` logs to both run.jsonl and the OTel span\n- `_NullRunLogger` updated to match `RunLogger` interface\n\n### `cli.py` / `catalog/cli.py`\n- Removed outer try/except for span status (now handled by context\nmanager)\n- Removed explicit `rl.finalize()` calls (now handled by context\nmanager)\n- Replaced paired `span.add_event` / `rl.log_event` with single\n`rl.log_span_event`\n- Extracted `_resolve_single_entry` to reduce nesting in catalog `run`\n\n### `test_logging.py`\n- 8 new unit tests covering span ok/error paths, None span, None fields,\nspan_context finalization, NullRunLogger\n\n## Test plan\n- [x] `python -m pytest python/xorq/common/utils/tests/test_logging.py`\n— 11 passed\n- [x] Pre-commit hooks pass (ruff check, ruff format, codespell)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-07T07:27:07+02:00",
          "tree_id": "78c50b34fbdadedd72e34866bfd071bd21beb463",
          "url": "https://github.com/xorq-labs/xorq/commit/58d6245cd6a317148bcc030d0e5c48e87b7599de"
        },
        "date": 1775539846033,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.986273377008065,
            "unit": "iter/sec",
            "range": "stddev: 0.010656316411354184",
            "extra": "mean: 100.1374549090909 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.613434405699524,
            "unit": "iter/sec",
            "range": "stddev: 0.050928678043376814",
            "extra": "mean: 276.7450264000047 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6862945376279076,
            "unit": "iter/sec",
            "range": "stddev: 0.22831427547100375",
            "extra": "mean: 1.4571003340000004 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.089537968146676,
            "unit": "iter/sec",
            "range": "stddev: 0.005878384649283062",
            "extra": "mean: 196.48148933333212 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.044939276453855,
            "unit": "iter/sec",
            "range": "stddev: 0.009530771064576548",
            "extra": "mean: 198.2184413333338 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.327941974755586,
            "unit": "iter/sec",
            "range": "stddev: 0.05669208124846778",
            "extra": "mean: 231.05670220000434 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "628f6fa8bd29c3be45215fa4f898d693efcc548f",
          "message": "chore(deps): update actions/create-github-app-token action to v2.2.2 (#1766)\n\nThis PR contains the following updates:\n\n| Package | Type | Update | Change |\n|---|---|---|---|\n|\n[actions/create-github-app-token](https://redirect.github.com/actions/create-github-app-token)\n| action | patch | `v2.2.1` → `v2.2.2` |\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMC4wIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T07:38:05+02:00",
          "tree_id": "2550ffbe01a290c96bbfecf3045756598ad2a945",
          "url": "https://github.com/xorq-labs/xorq/commit/628f6fa8bd29c3be45215fa4f898d693efcc548f"
        },
        "date": 1775540492275,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.877528948720654,
            "unit": "iter/sec",
            "range": "stddev: 0.01807384793379261",
            "extra": "mean: 126.94336085713839 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.765650808530698,
            "unit": "iter/sec",
            "range": "stddev: 0.011335738786099326",
            "extra": "mean: 209.834929200008 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6951308256004185,
            "unit": "iter/sec",
            "range": "stddev: 0.19762236841587053",
            "extra": "mean: 1.4385781253999936 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.1194530997588235,
            "unit": "iter/sec",
            "range": "stddev: 0.043835810021137504",
            "extra": "mean: 242.7506699999924 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.8885534950106795,
            "unit": "iter/sec",
            "range": "stddev: 0.016191087656153198",
            "extra": "mean: 204.559488000001 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.091939166748747,
            "unit": "iter/sec",
            "range": "stddev: 0.010112869834852375",
            "extra": "mean: 196.38883483333322 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4ba2bcb647c3ab495bdf40b1ad1ed1bc2d44ee0d",
          "message": "chore(deps): update codecov/codecov-action action to v5.5.4 (#1767)\n\nThis PR contains the following updates:\n\n| Package | Type | Update | Change |\n|---|---|---|---|\n|\n[codecov/codecov-action](https://redirect.github.com/codecov/codecov-action)\n| action | patch | `v5.5.2` → `v5.5.4` |\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMC4wIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T07:39:02+02:00",
          "tree_id": "b8234733074b0e41dc35dd5676c6a984ff17538c",
          "url": "https://github.com/xorq-labs/xorq/commit/4ba2bcb647c3ab495bdf40b1ad1ed1bc2d44ee0d"
        },
        "date": 1775540540755,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.75335938219891,
            "unit": "iter/sec",
            "range": "stddev: 0.00784574980635983",
            "extra": "mean: 128.97635085714197 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.769758651042155,
            "unit": "iter/sec",
            "range": "stddev: 0.01143348925765395",
            "extra": "mean: 209.6542138000018 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7279588251300445,
            "unit": "iter/sec",
            "range": "stddev: 0.17852436251612833",
            "extra": "mean: 1.3737040687999866 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.131134047934475,
            "unit": "iter/sec",
            "range": "stddev: 0.010969233388096605",
            "extra": "mean: 194.8886914000127 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.172859889837985,
            "unit": "iter/sec",
            "range": "stddev: 0.0104746862610809",
            "extra": "mean: 193.31666066666267 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.611533502472723,
            "unit": "iter/sec",
            "range": "stddev: 0.04431042532988916",
            "extra": "mean: 216.8476059999989 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3a9d4c901763710d5e542370fac07cc60c9d0f63",
          "message": "chore(deps): update dependency codespell to v2.4.2 (#1768)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [codespell](https://redirect.github.com/codespell-project/codespell) |\n`==2.4.1` → `==2.4.2` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/codespell/2.4.2?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/codespell/2.4.1/2.4.2?slim=true)\n|\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMC4wIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T07:39:39+02:00",
          "tree_id": "b92e8b48be5de985e4cb8413c3945d9d14a68295",
          "url": "https://github.com/xorq-labs/xorq/commit/3a9d4c901763710d5e542370fac07cc60c9d0f63"
        },
        "date": 1775540595763,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.290630155887245,
            "unit": "iter/sec",
            "range": "stddev: 0.016383651044801342",
            "extra": "mean: 107.63532540000256 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.45032735393084,
            "unit": "iter/sec",
            "range": "stddev: 0.01924867306082411",
            "extra": "mean: 224.70257139999603 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6356046473793792,
            "unit": "iter/sec",
            "range": "stddev: 0.2046528512919987",
            "extra": "mean: 1.5733050476000074 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.095629203409343,
            "unit": "iter/sec",
            "range": "stddev: 0.06005501942264838",
            "extra": "mean: 244.16272819999563 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.573655612343492,
            "unit": "iter/sec",
            "range": "stddev: 0.04329304886079389",
            "extra": "mean: 279.8255088000019 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.666278288763998,
            "unit": "iter/sec",
            "range": "stddev: 0.024161159709483057",
            "extra": "mean: 272.75616340000397 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6afebd75c458c6950d8af243c95a84ff37d548bc",
          "message": "chore(deps): update dependency requests to v2.33.0 [security] (#1751)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [requests](https://redirect.github.com/psf/requests)\n([changelog](https://redirect.github.com/psf/requests/blob/master/HISTORY.md))\n| `2.32.4` → `2.33.0` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/requests/2.33.0?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/requests/2.32.4/2.33.0?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n####\n[CVE-2026-25645](https://redirect.github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2)\n\n### Impact\nThe `requests.utils.extract_zipped_paths()` utility function uses a\npredictable filename when extracting files from zip archives into the\nsystem temporary directory. If the target file already exists, it is\nreused without validation. A local attacker with write access to the\ntemp directory could pre-create a malicious file that would be loaded in\nplace of the legitimate one.\n\n### Affected usages\n**Standard usage of the Requests library is not affected by this\nvulnerability.** Only applications that call `extract_zipped_paths()`\ndirectly are impacted.\n\n### Remediation\nUpgrade to at least Requests 2.33.0, where the library now extracts\nfiles to a non-deterministic location.\n\nIf developers are unable to upgrade, they can set `TMPDIR` in their\nenvironment to a directory with restricted write access.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My45MS41IiwidXBkYXRlZEluVmVyIjoiNDMuOTEuNSIsInRhcmdldEJyYW5jaCI6Im1haW4iLCJsYWJlbHMiOltdfQ==-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T07:40:08+02:00",
          "tree_id": "809440ce4f8809e531becc0a2aa995caad378b1f",
          "url": "https://github.com/xorq-labs/xorq/commit/6afebd75c458c6950d8af243c95a84ff37d548bc"
        },
        "date": 1775540603428,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.81134738798159,
            "unit": "iter/sec",
            "range": "stddev: 0.015538570030050164",
            "extra": "mean: 128.0188871818175 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.856274670949683,
            "unit": "iter/sec",
            "range": "stddev: 0.007516314281098404",
            "extra": "mean: 205.91915980000408 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7396333090447139,
            "unit": "iter/sec",
            "range": "stddev: 0.16534294399318755",
            "extra": "mean: 1.352021316200006 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.755685382243685,
            "unit": "iter/sec",
            "range": "stddev: 0.029050042352826436",
            "extra": "mean: 210.27463333333665 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.237100848309985,
            "unit": "iter/sec",
            "range": "stddev: 0.007610909543481173",
            "extra": "mean: 190.94533960000035 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.838762780499595,
            "unit": "iter/sec",
            "range": "stddev: 0.03610071140148381",
            "extra": "mean: 206.6643985999974 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5d28a6071d9d68004519058f5163c8d08c97add4",
          "message": "chore(deps): update pre-commit hook codespell-project/codespell to v2.4.2 (#1771)\n\nThis PR contains the following updates:\n\n| Package | Type | Update | Change |\n|---|---|---|---|\n|\n[codespell-project/codespell](https://redirect.github.com/codespell-project/codespell)\n| repository | patch | `v2.4.1` → `v2.4.2` |\n\nNote: The `pre-commit` manager in Renovate is not supported by the\n`pre-commit` maintainers or community. Please do not report any problems\nthere, instead [create a Discussion in the Renovate\nrepository](https://redirect.github.com/renovatebot/renovate/discussions/new)\nif you have any questions.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMC4wIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T07:49:13+02:00",
          "tree_id": "a80b160a2eb6119f958f12175c31b3ed58d25854",
          "url": "https://github.com/xorq-labs/xorq/commit/5d28a6071d9d68004519058f5163c8d08c97add4"
        },
        "date": 1775541160943,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.599869095763912,
            "unit": "iter/sec",
            "range": "stddev: 0.015699235971250645",
            "extra": "mean: 116.280839727267 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.697046626907585,
            "unit": "iter/sec",
            "range": "stddev: 0.021340492960230963",
            "extra": "mean: 212.89973879999025 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7306799065951429,
            "unit": "iter/sec",
            "range": "stddev: 0.16256993168980688",
            "extra": "mean: 1.3685883394000087 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.156986672806688,
            "unit": "iter/sec",
            "range": "stddev: 0.008292110360927996",
            "extra": "mean: 193.9116897999952 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.142524884324338,
            "unit": "iter/sec",
            "range": "stddev: 0.011548946585695162",
            "extra": "mean: 194.45700750000108 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.820855282205341,
            "unit": "iter/sec",
            "range": "stddev: 0.019410247005686852",
            "extra": "mean: 207.4320720000003 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b576703404d3a75d58d4b721eb1c6e456b58a192",
          "message": "chore(deps): update pre-commit hook rhysd/actionlint to v1.7.12 (#1772)\n\nThis PR contains the following updates:\n\n| Package | Type | Update | Change |\n|---|---|---|---|\n| [rhysd/actionlint](https://redirect.github.com/rhysd/actionlint) |\nrepository | patch | `v1.7.11` → `v1.7.12` |\n\nNote: The `pre-commit` manager in Renovate is not supported by the\n`pre-commit` maintainers or community. Please do not report any problems\nthere, instead [create a Discussion in the Renovate\nrepository](https://redirect.github.com/renovatebot/renovate/discussions/new)\nif you have any questions.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMC4wIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T08:26:10+02:00",
          "tree_id": "daef22acdc458224504aebb273645a2b5a848162",
          "url": "https://github.com/xorq-labs/xorq/commit/b576703404d3a75d58d4b721eb1c6e456b58a192"
        },
        "date": 1775543379169,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.969189022962794,
            "unit": "iter/sec",
            "range": "stddev: 0.020630069760218927",
            "extra": "mean: 125.48328281818303 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.278220184866928,
            "unit": "iter/sec",
            "range": "stddev: 0.050379920327602676",
            "extra": "mean: 233.7420601999952 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6422499665007922,
            "unit": "iter/sec",
            "range": "stddev: 0.19082336122476307",
            "extra": "mean: 1.557026161399989 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.4943009286676356,
            "unit": "iter/sec",
            "range": "stddev: 0.02951170765338569",
            "extra": "mean: 286.18027480000023 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.034607461993266,
            "unit": "iter/sec",
            "range": "stddev: 0.04318565326432592",
            "extra": "mean: 247.85558680000008 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.761494402712952,
            "unit": "iter/sec",
            "range": "stddev: 0.018692707246758162",
            "extra": "mean: 210.01809839999623 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "17ae7b1e40183ca9ebb05731f29ce9b08542f8d4",
          "message": "chore(deps): update dependency ruff to v0.15.9 (#1770)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [ruff](https://docs.astral.sh/ruff)\n([source](https://redirect.github.com/astral-sh/ruff),\n[changelog](https://redirect.github.com/astral-sh/ruff/blob/main/CHANGELOG.md))\n| `==0.15.4` → `==0.15.9` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/ruff/0.15.9?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/ruff/0.15.4/0.15.9?slim=true)\n|\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMDAuMCIsInVwZGF0ZWRJblZlciI6IjQzLjEwMi4xMSIsInRhcmdldEJyYW5jaCI6Im1haW4iLCJsYWJlbHMiOltdfQ==-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-07T08:25:42+02:00",
          "tree_id": "818a2b954d02af2e4559f96de9fc37b75fe16c74",
          "url": "https://github.com/xorq-labs/xorq/commit/17ae7b1e40183ca9ebb05731f29ce9b08542f8d4"
        },
        "date": 1775543382187,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.9648454899434,
            "unit": "iter/sec",
            "range": "stddev: 0.012225650016572102",
            "extra": "mean: 100.35278530000369 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.5728451228547664,
            "unit": "iter/sec",
            "range": "stddev: 0.029854285985119543",
            "extra": "mean: 279.8889863999989 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6860779345254294,
            "unit": "iter/sec",
            "range": "stddev: 0.21755226893739874",
            "extra": "mean: 1.4575603581999985 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.1705985100762755,
            "unit": "iter/sec",
            "range": "stddev: 0.003946847493794763",
            "extra": "mean: 193.40120840000168 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.711937931571576,
            "unit": "iter/sec",
            "range": "stddev: 0.03943597771001374",
            "extra": "mean: 212.22690420000276 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.8050122622911835,
            "unit": "iter/sec",
            "range": "stddev: 0.05752735332536402",
            "extra": "mean: 262.8112424000051 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f5fd8e30324eb2c88dcb2f454cd214627ca39094",
          "message": "release: 0.3.18 (#1794)",
          "timestamp": "2026-04-07T12:34:48+02:00",
          "tree_id": "2b4046661c7c5509cc7f8f377fc6a4d469aa57af",
          "url": "https://github.com/xorq-labs/xorq/commit/f5fd8e30324eb2c88dcb2f454cd214627ca39094"
        },
        "date": 1775558303666,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.835190720137774,
            "unit": "iter/sec",
            "range": "stddev: 0.01762906390430912",
            "extra": "mean: 113.1837479999986 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.029978234329196,
            "unit": "iter/sec",
            "range": "stddev: 0.06519178804701552",
            "extra": "mean: 248.1402980000098 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7196008564484423,
            "unit": "iter/sec",
            "range": "stddev: 0.2123477333808424",
            "extra": "mean: 1.3896592688000056 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.205690442253586,
            "unit": "iter/sec",
            "range": "stddev: 0.010850662093168045",
            "extra": "mean: 192.0974770000138 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.165442570589794,
            "unit": "iter/sec",
            "range": "stddev: 0.008245721933723573",
            "extra": "mean: 193.59425379998356 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.048926217033972,
            "unit": "iter/sec",
            "range": "stddev: 0.04956358684645827",
            "extra": "mean: 246.97906219998913 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "be0f1d827f1d151303d489558707f3f6727106a4",
          "message": "feat(lineage): replace flat chain with full DAG data model (#1788)\n\n## Summary\n\n- Replace `extract_lineage_chain()` (flat first-child walk) with\n`extract_lineage_dag()` (full BFS via `graph_utils.bfs()`) producing\n`{\"nodes\": (...), \"edges\": (...), \"root\": \"...\"}`\n- Change `ExprMetadata.lineage` from `tuple[str, ...]` to\n`Optional[dict]` with backward-compat `_parse_lineage()` shim for old\nlist format\n- Edges stored as `(source, target)` tuples internally, serialized to\nlists for JSON\n- BFS traverses all children including opaque sub-expressions\n(`ExprScalarUDF.computed_kwargs_expr`, `RemoteTable.remote_expr`)\n- Each node carries `id`, `type`, `label`, and optionally `schema` and\n`tag_metadata`\n\n## Files changed\n\n| File | Change |\n|------|--------|\n| `lineage_utils.py` | Replace `extract_lineage_chain` with\n`extract_lineage_dag` using `bfs()` |\n| `core.py` | `ExprMetadata.lineage` type change + `_parse_lineage()` +\n`to_dict()`/`from_dict()` |\n| `compiler.py` | Call `extract_lineage_dag` with try/except fallback |\n| `tui.py` | Render DAG node labels in `lineage_text` |\n| `test_lineage_utils.py` | 8 new tests: structure, fields, schema,\nedges, root, multi-join, round-trip, backward-compat |\n\n## Test plan\n\n- [x] `python -m pytest\npython/xorq/common/utils/tests/test_lineage_utils.py -x -q` — 14 pass\n- [x] `python -m pytest python/xorq/ibis_yaml/tests/ -x -q` — compiler\ntests pass (pre-existing failures unrelated)\n- [x] `python -m pytest python/xorq/catalog/tests/test_tui.py -x -q` —\n81/82 TUI tests pass (pre-existing failure unrelated)\n- [x] `pre-commit run --files` on all changed files — clean\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-07T08:10:57-04:00",
          "tree_id": "a9dd67d87f5755f85e85aa3b1451ea7abf5121af",
          "url": "https://github.com/xorq-labs/xorq/commit/be0f1d827f1d151303d489558707f3f6727106a4"
        },
        "date": 1775564047969,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.845427178903892,
            "unit": "iter/sec",
            "range": "stddev: 0.018421901756984837",
            "extra": "mean: 101.56999608333213 msec\nrounds: 12"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.404835179824316,
            "unit": "iter/sec",
            "range": "stddev: 0.05654387435130491",
            "extra": "mean: 227.02325040000346 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7385715699359257,
            "unit": "iter/sec",
            "range": "stddev: 0.13602701861279543",
            "extra": "mean: 1.353964924600001 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.517542495577312,
            "unit": "iter/sec",
            "range": "stddev: 0.006849779951099782",
            "extra": "mean: 181.240108399993 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.17065100426388,
            "unit": "iter/sec",
            "range": "stddev: 0.02897972080154252",
            "extra": "mean: 239.7707214000036 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.59967851068346,
            "unit": "iter/sec",
            "range": "stddev: 0.02747042815882527",
            "extra": "mean: 217.40649866666692 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8975c28ce6db26ef68bc6ece99cc4f5ccc8bbc8e",
          "message": "fix(packager): include env_templates in wheel, fallback in otel_utils (#1785)\n\n## Summary\n- **Packaging**: add `[tool.hatch.build.targets.wheel.force-include]`\nfor `env_templates/` — `.gitignore` contains `.env.*` which hatchling\ntreats as an exclusion pattern in non-VCS contexts (sdist rebuilds,\nstaging dirs, `uv tool run`). Inside a git repo hatchling uses `git\nls-files` so tracked files override `.gitignore`, but outside one (e.g.\nthe staging dirs created by\n`WheelPackager.from_script_and_requirements`) the pattern excludes the\ndotfiles. `force-include` overrides this unconditionally.\n- **Resilience**: `otel_utils.py` now falls back to\n`subclass_from_kwargs` with the same env var names if the template file\nis missing. This is the only template loaded unconditionally at import\ntime (via `cli.py → _lazy_span → otel_utils`), so it's the only one that\nneeds a fallback.\n\n## Test plan\n- [x] `uv build --sdist` succeeds and the resulting sdist contains\n`xorq/env_templates/.env.otel.template`\n- [x] `uv build --wheel` succeeds and the wheel contains\n`xorq/env_templates/.env.otel.template`\n- [x] `python -c \"from xorq.common.utils.otel_utils import tracer\"`\nworks in a fresh venv installed from the wheel\n- [x] Temporarily rename `env_templates/` and verify the import still\nsucceeds via the fallback path\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-07T14:16:12+02:00",
          "tree_id": "8f532a105067363696b9a062fe13007975a03372",
          "url": "https://github.com/xorq-labs/xorq/commit/8975c28ce6db26ef68bc6ece99cc4f5ccc8bbc8e"
        },
        "date": 1775564379367,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.122937485153102,
            "unit": "iter/sec",
            "range": "stddev: 0.01755294881874706",
            "extra": "mean: 109.61381699999865 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.4821096338374966,
            "unit": "iter/sec",
            "range": "stddev: 0.039034936854609764",
            "extra": "mean: 287.18222719999176 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7090190320854285,
            "unit": "iter/sec",
            "range": "stddev: 0.22436934808263032",
            "extra": "mean: 1.4103993753999986 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.1274559981190375,
            "unit": "iter/sec",
            "range": "stddev: 0.01221065513414093",
            "extra": "mean: 195.0284898333289 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.095909968642264,
            "unit": "iter/sec",
            "range": "stddev: 0.00985058436583444",
            "extra": "mean: 196.23580600000992 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.924453451630118,
            "unit": "iter/sec",
            "range": "stddev: 0.04425374301209538",
            "extra": "mean: 254.81255220000776 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f6308e54983216253e9731a4b7b914be6b7f3575",
          "message": "chore(deps): loose attrs (#1496)",
          "timestamp": "2026-04-07T08:19:19-04:00",
          "tree_id": "e59a1b27ad6e526aaa3eb705464663ac585a1578",
          "url": "https://github.com/xorq-labs/xorq/commit/f6308e54983216253e9731a4b7b914be6b7f3575"
        },
        "date": 1775564565482,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.70268892519887,
            "unit": "iter/sec",
            "range": "stddev: 0.010296832511618014",
            "extra": "mean: 103.06421320000254 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.5684216078253614,
            "unit": "iter/sec",
            "range": "stddev: 0.04685700065739531",
            "extra": "mean: 280.23594460000254 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.669006514363828,
            "unit": "iter/sec",
            "range": "stddev: 0.1844241704946654",
            "extra": "mean: 1.4947537558000021 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.849261642596529,
            "unit": "iter/sec",
            "range": "stddev: 0.0045096916266755115",
            "extra": "mean: 206.2169612000048 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.768387061681134,
            "unit": "iter/sec",
            "range": "stddev: 0.007526531950454102",
            "extra": "mean: 209.71451920000845 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.337563095840698,
            "unit": "iter/sec",
            "range": "stddev: 0.042699326986946395",
            "extra": "mean: 230.54419680001956 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c1219d55b993438dfe372eb2777297c82d454745",
          "message": "fix(cache): SnapshotStrategy fails with HashingTagNode (#1786)\n\ncloses #1784\n\n---------\n\nCo-authored-by: dlovell <dlovell@gmail.com>\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-07T14:26:01+02:00",
          "tree_id": "3d8fb35edfafa6307b7d079d37476f2d53c4271d",
          "url": "https://github.com/xorq-labs/xorq/commit/c1219d55b993438dfe372eb2777297c82d454745"
        },
        "date": 1775564974884,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.458028542237269,
            "unit": "iter/sec",
            "range": "stddev: 0.015324756322518594",
            "extra": "mean: 105.73027936363712 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.6755638617645827,
            "unit": "iter/sec",
            "range": "stddev: 0.05447529820745664",
            "extra": "mean: 272.06709980000596 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6922791648252633,
            "unit": "iter/sec",
            "range": "stddev: 0.24531602730764515",
            "extra": "mean: 1.4445039672000064 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.895496852660932,
            "unit": "iter/sec",
            "range": "stddev: 0.009066541397065635",
            "extra": "mean: 204.2693581666697 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.955645105549308,
            "unit": "iter/sec",
            "range": "stddev: 0.004151993456268343",
            "extra": "mean: 201.79007549999994 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.802347816366187,
            "unit": "iter/sec",
            "range": "stddev: 0.007521764569362756",
            "extra": "mean: 208.23148140000285 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f2751317df8af9850fe28a850a074ec76c831a9a",
          "message": "fix(deps): clean up dependency declarations (#1796)\n\n## Summary\n- Move `pytest-mock` from core dependencies to the `test` dependency\ngroup — it's only used in test files, not runtime code\n- Remove duplicate `quickgrove` entry in the `dev` dependency group\n- Remove redundant `fsspec` from `examples` extra — already pulled in\ntransitively via `pins[gcs]` → `gcsfs` → `fsspec`\n\n## Test plan\n- [ ] `uv sync --group test` installs pytest-mock\n- [ ] `uv pip install xorq[examples]` still pulls in fsspec transitively\n- [ ] CI passes\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-07T23:04:53+02:00",
          "tree_id": "5b4461df3e17fd239980576f69fa547fb33182c2",
          "url": "https://github.com/xorq-labs/xorq/commit/f2751317df8af9850fe28a850a074ec76c831a9a"
        },
        "date": 1775596111037,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.868215355430124,
            "unit": "iter/sec",
            "range": "stddev: 0.01988981497471974",
            "extra": "mean: 127.09362350000575 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.530349384522294,
            "unit": "iter/sec",
            "range": "stddev: 0.006968418338233802",
            "extra": "mean: 220.7335273999945 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7230784729900656,
            "unit": "iter/sec",
            "range": "stddev: 0.16924237561837802",
            "extra": "mean: 1.3829757590000042 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.721937721975994,
            "unit": "iter/sec",
            "range": "stddev: 0.027572365821935566",
            "extra": "mean: 268.6772521999899 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.361774541283775,
            "unit": "iter/sec",
            "range": "stddev: 0.016235383375013922",
            "extra": "mean: 229.26448639999535 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.749257935808745,
            "unit": "iter/sec",
            "range": "stddev: 0.007005611258143117",
            "extra": "mean: 210.55921020000596 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d8a4b2bc864893d3d50e3da8f4f0d16f032c7ae9",
          "message": "feat(tui): replace catalog DataTable with Tree widget and add view switching (#1791)\n\n## Summary\n- Replace flat `#catalog-table` (DataTable) with `#catalog-tree` (Tree)\nwidget grouped by kind (source/transform/analytic)\n- Add `1`/`2`/`3` keys to switch detail pane between SQL, Lineage, and\nData views\n- Add `v` key to toggle revisions panel; `h`/`l` collapse/expand tree\nbranches\n- Add `netext>=0.4.1` and `networkx>=3.4.2` dependencies for upcoming\nlineage graph\n- Remove profiles panel (superseded by consolidated view switching)\n\n## Test plan\n- [x] All 97 existing TUI tests pass (`python -m pytest\npython/xorq/catalog/tests/test_tui.py -x -q`)\n- [x] New tests: `test_catalog_tree_exists`, `test_j_k_moves_cursor`\n(tree), `test_render_refresh_populates_tree`,\n`test_view_switching_1_2_3`, `test_v_toggles_revisions`,\n`test_tree_entry_hashes_helper`, `test_tree_label_with_alias`,\n`test_tree_label_without_alias`\n- [x] Pre-commit passes on all changed files\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-08T11:27:47-04:00",
          "tree_id": "0c0e0e9115fc1db8fb799b01804dbb1c2e27fa86",
          "url": "https://github.com/xorq-labs/xorq/commit/d8a4b2bc864893d3d50e3da8f4f0d16f032c7ae9"
        },
        "date": 1775662362002,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.197985033910728,
            "unit": "iter/sec",
            "range": "stddev: 0.017046433290242347",
            "extra": "mean: 138.92776871428023 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.01349163145806,
            "unit": "iter/sec",
            "range": "stddev: 0.013781051136172574",
            "extra": "mean: 249.15960760000644 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.5710430794326917,
            "unit": "iter/sec",
            "range": "stddev: 0.2565931781590035",
            "extra": "mean: 1.7511813662000066 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.9855710244661156,
            "unit": "iter/sec",
            "range": "stddev: 0.0576514125992782",
            "extra": "mean: 250.90507580001142 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.8230278185961293,
            "unit": "iter/sec",
            "range": "stddev: 0.0468434467768532",
            "extra": "mean: 261.5727761999949 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.3717430000273367,
            "unit": "iter/sec",
            "range": "stddev: 0.05172182260372362",
            "extra": "mean: 296.58250940000244 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b6fdbe23eb2df2a02907e678157289889675999d",
          "message": "chore: remove CLAUDE.md (#1802)",
          "timestamp": "2026-04-08T18:10:22-04:00",
          "tree_id": "dc958ce175e5fc05f43a22e88dd42bd77b47777a",
          "url": "https://github.com/xorq-labs/xorq/commit/b6fdbe23eb2df2a02907e678157289889675999d"
        },
        "date": 1775686439668,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.6033924512296585,
            "unit": "iter/sec",
            "range": "stddev: 0.013871214575929547",
            "extra": "mean: 131.52024000001146 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.234161691600665,
            "unit": "iter/sec",
            "range": "stddev: 0.05995578870112951",
            "extra": "mean: 236.17425900000626 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7105743373194647,
            "unit": "iter/sec",
            "range": "stddev: 0.20059544094061268",
            "extra": "mean: 1.407312292999984 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.372589915343852,
            "unit": "iter/sec",
            "range": "stddev: 0.059194594164441804",
            "extra": "mean: 228.69741260000183 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.0151467663244995,
            "unit": "iter/sec",
            "range": "stddev: 0.009366785374006406",
            "extra": "mean: 199.39595919998965 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.968597406082407,
            "unit": "iter/sec",
            "range": "stddev: 0.009835860453106457",
            "extra": "mean: 201.26404259999617 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c62223c4e058e50827c171d4d2af3656e2b12e04",
          "message": "chore: add databricks workflow (#1804)",
          "timestamp": "2026-04-09T12:10:21+02:00",
          "tree_id": "9d014100ef6848a76e092d5ed73effed6fa76f8a",
          "url": "https://github.com/xorq-labs/xorq/commit/c62223c4e058e50827c171d4d2af3656e2b12e04"
        },
        "date": 1775729632574,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.211908953904084,
            "unit": "iter/sec",
            "range": "stddev: 0.02074122341702122",
            "extra": "mean: 108.55513281817574 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.758508953253515,
            "unit": "iter/sec",
            "range": "stddev: 0.057096705563135695",
            "extra": "mean: 266.06295540000247 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6872343201341505,
            "unit": "iter/sec",
            "range": "stddev: 0.22471458911880735",
            "extra": "mean: 1.455107771400003 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.310809512412448,
            "unit": "iter/sec",
            "range": "stddev: 0.033111898516034945",
            "extra": "mean: 231.97499150000075 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.898212628790626,
            "unit": "iter/sec",
            "range": "stddev: 0.011225213097639132",
            "extra": "mean: 204.15610260000108 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.937916420407013,
            "unit": "iter/sec",
            "range": "stddev: 0.0066833473393922375",
            "extra": "mean: 202.51456583333058 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "ff24aeb94f1ae2430b0640d90bbcf357c7d0a8c6",
          "message": "fix(dev): worktree setup improvements (#1795)\n\n## Summary\n- Fix `dev/cleanup-worktree` to tolerate mixed state when manifest\ncontents are removed but other files remain\n- Symlink `ci/ibis-testing-data` from the main worktree so test data is\navailable without duplication\n- Symlink `.claude/settings.json` from the main worktree so Claude Code\npermissions (e.g. `git:*`) carry over automatically\n\n## Test plan\n- [ ] Run `dev/setup-worktree` in a worktree and verify\n`ci/ibis-testing-data` and `.claude/settings.json` are symlinked\n- [ ] Run `dev/cleanup-worktree` and verify symlinks are removed\n- [ ] Confirm Claude Code does not re-prompt for permissions in the new\nworktree\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-09T14:19:00-04:00",
          "tree_id": "316c55deb404df28017af99533951079316fe80d",
          "url": "https://github.com/xorq-labs/xorq/commit/ff24aeb94f1ae2430b0640d90bbcf357c7d0a8c6"
        },
        "date": 1775758950636,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.1028778555678045,
            "unit": "iter/sec",
            "range": "stddev: 0.022496197408674683",
            "extra": "mean: 140.78800457143157 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.516408712178678,
            "unit": "iter/sec",
            "range": "stddev: 0.014878232193875044",
            "extra": "mean: 221.41485940000507 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6931101473412685,
            "unit": "iter/sec",
            "range": "stddev: 0.1779841055585986",
            "extra": "mean: 1.4427721247999954 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.7503861463206767,
            "unit": "iter/sec",
            "range": "stddev: 0.019698020069279284",
            "extra": "mean: 266.63921020000885 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.086158759361147,
            "unit": "iter/sec",
            "range": "stddev: 0.04270614592770122",
            "extra": "mean: 244.72862140000302 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.591913851670013,
            "unit": "iter/sec",
            "range": "stddev: 0.005938941761747018",
            "extra": "mean: 217.77412039999717 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9780571f7671ced0f1d2dc26d2b68cccff5168ba",
          "message": "fix(ibis_yaml): skip _MISSING sentinel when binding default params (#1793)\n\n## Summary\n\n- Fix `InputTypeError` during `xorq build` when expressions have\nrequired named parameters (no default value)\n- `_extract_sql_queries` filtered defaults with `is not None`, but the\nsentinel for required parameters is `_MISSING`, not `None` — so\n`_MISSING` values leaked into `bind_params` which tried to\n`dt.infer(_MISSING)` and crashed\n\n## Repro\n\n```\n$ xorq build expr.py\nBuilding expr from expr.py\nTraceback (most recent call last):\n  ...\n  File \".../xorq/ibis_yaml/compiler.py\", line 364, in _extract_sql_queries\n    clean = bind_params(clean, defaults)\n  File \".../xorq/expr/api.py\", line 769, in bind_params\n    if name in named and not dt.infer(value).castable(named[name].dtype):\n  File \".../xorq/vendor/ibis/expr/datatypes/value.py\", line 35, in infer\n    raise InputTypeError(\nxorq.common.exceptions.InputTypeError: Unable to infer datatype of value _MISSING\n  with type <class 'xorq.expr.operations._MissingSentinel'>\n```\n\n## Fix\n\nAdd `_MISSING` sentinel check alongside the existing `None` check when\ncollecting defaults to bind.\n\n## Test plan\n\n- [ ] `xorq build expr.py` completes without error for expressions with\nrequired named parameters\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-09T22:15:59+02:00",
          "tree_id": "45a84aba4ea4595b74240a94aa992be0e0b4a47f",
          "url": "https://github.com/xorq-labs/xorq/commit/9780571f7671ced0f1d2dc26d2b68cccff5168ba"
        },
        "date": 1775765975213,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.228000816843041,
            "unit": "iter/sec",
            "range": "stddev: 0.02184381794704515",
            "extra": "mean: 121.53620572727225 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.946820032220511,
            "unit": "iter/sec",
            "range": "stddev: 0.05445098096958616",
            "extra": "mean: 253.36853260000115 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6387571258844313,
            "unit": "iter/sec",
            "range": "stddev: 0.20521390105318885",
            "extra": "mean: 1.5655402648000005 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.50133670986744,
            "unit": "iter/sec",
            "range": "stddev: 0.022713916474643454",
            "extra": "mean: 285.6052082000019 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.799370473187879,
            "unit": "iter/sec",
            "range": "stddev: 0.04182780750385856",
            "extra": "mean: 263.2014979999951 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.025389364098686,
            "unit": "iter/sec",
            "range": "stddev: 0.03335041871453176",
            "extra": "mean: 248.4231733999991 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "7497aff64dbeaa55aa602ecaea94e4a58b09a38f",
          "message": "feat(metadata): add cache_keys to ExprMetadata (XOR-281) (#1783)\n\nStore the cache key for the root CachedNode in ExprMetadata instead of\nonly baking in the full filesystem path. parquet_cache_paths embeds a\nmachine-specific path at build time; cache_keys are portable — the path\ncan be resolved at read time from key + local cache directory.\n\n- Add `cache_keys: tuple[str, ...]` field to ExprMetadata\n- In from_expr(), compute key only when root node is CachedNode\n(expr.ls.is_cached), not for deeply nested cached nodes\n- Serialize/deserialize via to_dict()/from_dict(); omitted when empty\n- Add CatalogEntry.cache_keys property delegating to metadata.cache_keys\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-10T11:12:13-04:00",
          "tree_id": "e4558867c618b3f0c868db7e861e7343afec0f62",
          "url": "https://github.com/xorq-labs/xorq/commit/7497aff64dbeaa55aa602ecaea94e4a58b09a38f"
        },
        "date": 1775834137252,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.602535843221304,
            "unit": "iter/sec",
            "range": "stddev: 0.0066787195472507764",
            "extra": "mean: 94.31705912499666 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.6306066790638,
            "unit": "iter/sec",
            "range": "stddev: 0.026911069607724065",
            "extra": "mean: 215.95442439999601 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7591451554939865,
            "unit": "iter/sec",
            "range": "stddev: 0.15296023034434475",
            "extra": "mean: 1.3172711342000014 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.757707830028834,
            "unit": "iter/sec",
            "range": "stddev: 0.031036910172756307",
            "extra": "mean: 266.1196786000062 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.544882162306471,
            "unit": "iter/sec",
            "range": "stddev: 0.031700287231004036",
            "extra": "mean: 220.0277068333302 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.084227628212899,
            "unit": "iter/sec",
            "range": "stddev: 0.017597687670114187",
            "extra": "mean: 196.6867089999861 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f1691e01b13fb085193cc80fffe53eef1ab65cbb",
          "message": "ci: add faulthandler_timeout, verbose output, and step timeout (#1811)\n\n## Summary\n\n- `faulthandler_timeout = 300` in pyproject.toml — dumps thread\ntracebacks if any test blocks >5 min\n- `-v` on pytest invocation — streams test names in real time (essential\nfor xdist debugging)\n- `--durations=20` — surfaces the 20 slowest tests at the end\n- `timeout-minutes: 20` on pytest step — kills hung steps so logs are\nalways available\n\nThese were essential for diagnosing the Rich infinite recursion hang in\n#1805.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-12T02:19:57-04:00",
          "tree_id": "91589cb01206463a7daef1093605bdaf26f80061",
          "url": "https://github.com/xorq-labs/xorq/commit/f1691e01b13fb085193cc80fffe53eef1ab65cbb"
        },
        "date": 1775975000437,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.505001389213038,
            "unit": "iter/sec",
            "range": "stddev: 0.01990550143000453",
            "extra": "mean: 133.24447900000433 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.493430275568621,
            "unit": "iter/sec",
            "range": "stddev: 0.007609223325494484",
            "extra": "mean: 222.54712740000286 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7068754602715269,
            "unit": "iter/sec",
            "range": "stddev: 0.19479090099972507",
            "extra": "mean: 1.4146763556000053 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.8332153026287363,
            "unit": "iter/sec",
            "range": "stddev: 0.017807544484367326",
            "extra": "mean: 260.87759780000397 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.343713432638415,
            "unit": "iter/sec",
            "range": "stddev: 0.025473177424202995",
            "extra": "mean: 230.21776540000474 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.655964922601014,
            "unit": "iter/sec",
            "range": "stddev: 0.00822222782128677",
            "extra": "mean: 214.7782503999963 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "1db68ce287b2bb2d3462e38a334187499175e993",
          "message": "fix(catalog): prevent hang when annex content is missing with no remote (#1812)\n\n## Summary\n\n`GitAnnexBackend.fetch_content` now checks for configured remotes before\ncalling `git-annex get`. With no git remotes and no annex special\nremote, `git-annex get` blocks indefinitely waiting for content that has\nno source. The guard raises `AnnexError` instead.\n\n## Test plan\n\n- [x] `test_annex_fetch_content_no_remote_raises` — drops annex content,\nverifies `fetch_content` raises instead of blocking\n- [x] All 92 catalog tests pass (verified in main worktree)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-12T02:22:14-04:00",
          "tree_id": "3c1c3f8ecd17054191f03978c7d7e367c343dcdc",
          "url": "https://github.com/xorq-labs/xorq/commit/1db68ce287b2bb2d3462e38a334187499175e993"
        },
        "date": 1775975140939,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.644756078797295,
            "unit": "iter/sec",
            "range": "stddev: 0.005254714054761001",
            "extra": "mean: 93.94296990908462 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.944669618239483,
            "unit": "iter/sec",
            "range": "stddev: 0.027437939473692538",
            "extra": "mean: 253.50665499999536 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7107549176929844,
            "unit": "iter/sec",
            "range": "stddev: 0.2288136973859934",
            "extra": "mean: 1.4069547393999984 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.42509108634041,
            "unit": "iter/sec",
            "range": "stddev: 0.05448360362743175",
            "extra": "mean: 225.9840488000009 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.6492423380657013,
            "unit": "iter/sec",
            "range": "stddev: 0.033255827209953494",
            "extra": "mean: 274.0294853999899 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.061456504195516,
            "unit": "iter/sec",
            "range": "stddev: 0.033667606351722654",
            "extra": "mean: 246.2170895999975 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45543592+ghoersti@users.noreply.github.com",
            "name": "ghoersti",
            "username": "ghoersti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9696ad33932a59dc128099a19c22fe897c0791fa",
          "message": "XOR-253 add expr builder roundtrip fitted bsl  (#1801)\n\nTagged expressions can now round-trip through the catalog as\nExprKind.ExprBuilder entries. A TagHandler registry dispatches\nextract_metadata (sidecar) and from_tagged (recovery) per tag name.\nBuilt-in handlers cover BSL semantic models and ML fitted pipelines.\nThird parties register handlers inline or via entry points.\n\nSee the examples for end-to-end usage:\n- examples/semantic_builder_example.py — BSL: catalog a query, recover\nthe SemanticModel, issue new queries\n- examples/fitted_pipeline_builder_example.py — ML: catalog predictions,\nrecover the FittedPipeline, predict/transform on new data\n- examples/custom_builder_example.py — Third-party: register a custom\nTagHandler, full round-trip\nFor full context see\n[XOR-253](https://linear.app/xorq-labs/issue/XOR-253/add-exprbuilder-kind)\n\n---------\n\nCo-authored-by: ghoersti <ghoersti@users.noreply.github.com>\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-12T02:44:51-04:00",
          "tree_id": "1ebce99435a1f59667be16d20fef70ac3a74457d",
          "url": "https://github.com/xorq-labs/xorq/commit/9696ad33932a59dc128099a19c22fe897c0791fa"
        },
        "date": 1775976509519,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.846452793188525,
            "unit": "iter/sec",
            "range": "stddev: 0.012543788260286022",
            "extra": "mean: 127.44612455555662 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.854371062306762,
            "unit": "iter/sec",
            "range": "stddev: 0.010091533177591091",
            "extra": "mean: 205.9999095999899 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7247664159558805,
            "unit": "iter/sec",
            "range": "stddev: 0.1540423084723437",
            "extra": "mean: 1.3797548809999967 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.366957916290975,
            "unit": "iter/sec",
            "range": "stddev: 0.03685706341057939",
            "extra": "mean: 228.99236016667146 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.108276896511343,
            "unit": "iter/sec",
            "range": "stddev: 0.00917768291933111",
            "extra": "mean: 195.76072719999615 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.115563059078613,
            "unit": "iter/sec",
            "range": "stddev: 0.0049109228291824274",
            "extra": "mean: 195.48190266666646 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b54cc7bc2744c601eb0746fa55b884dd2967254f",
          "message": "feat(tui): add DataViewScreen for full-screen data exploration (#1797)\n\n## Summary\n- Adds `DataViewScreen` — a full-screen data viewer pushed via `e` from\nthe catalog tree\n- Executes selected entry (up to 50K rows) via `xorq catalog run`\nsubprocess with Arrow IPC output\n- Column sorting with `[`/`]`, vim-style navigation (`hjkl`, `g/G`),\n`q`/`escape` to return\n- NaN values display as \"—\" instead of \"nan\"\n- Stats panel removed — not performant enough for interactive use, will\nrevisit later\n- Tests bypass the subprocess with an in-process monkeypatch to avoid\n`WorkerCancelled` in CI (memtable data isn't accessible cross-process)\n- `settle()` test helper hardened to catch `WorkerCancelled` alongside\n`CancelledError`/`TimeoutError`\n\n## Test plan\n- [x] `test_data_view_screen_construction` — DataViewScreen instantiates\ncorrectly\n- [x] `test_e_pushes_data_view_screen` — `e` on tree leaf pushes\nDataViewScreen\n- [x] `test_e_on_branch_does_nothing` — `e` on branch node is no-op\n- [x] `test_data_view_escape_returns` — escape pops back to\nCatalogScreen\n- [x] `test_data_view_loads_data` — background worker loads and renders\nrows\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-12T08:34:52-04:00",
          "tree_id": "7b49c40098c1075d88602310af381ab274d90edf",
          "url": "https://github.com/xorq-labs/xorq/commit/b54cc7bc2744c601eb0746fa55b884dd2967254f"
        },
        "date": 1775997501860,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.523505361188452,
            "unit": "iter/sec",
            "range": "stddev: 0.005840315315029683",
            "extra": "mean: 105.0033534999983 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.6973048630782297,
            "unit": "iter/sec",
            "range": "stddev: 0.025938885997959942",
            "extra": "mean: 270.4672828000014 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.678278897619069,
            "unit": "iter/sec",
            "range": "stddev: 0.14199701005356208",
            "extra": "mean: 1.4743197872000053 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.8410031948182235,
            "unit": "iter/sec",
            "range": "stddev: 0.008546549065628729",
            "extra": "mean: 206.56875439999567 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.136945180474818,
            "unit": "iter/sec",
            "range": "stddev: 0.058653615907943754",
            "extra": "mean: 241.7242569999985 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.5641047739936984,
            "unit": "iter/sec",
            "range": "stddev: 0.04697182711472428",
            "extra": "mean: 280.5753656000036 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45543592+ghoersti@users.noreply.github.com",
            "name": "ghoersti",
            "username": "ghoersti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "54e16eb441b429fba36efd01f1f123158eec5cd6",
          "message": "fix(otel): collector probe (#1813)\n\n## Summary\n\n- **`socket.bind()` → `socket.create_connection()`** — `bind()` checks\nif a port is free (server-side); `create_connection()` checks if a\nservice is accepting connections (client-side). Adds 1s timeout and\ncontext manager to prevent socket leak.\n- **Fix silent discard of remote OTLP endpoints** —\n`localhost_and_listening()` returned `None` for non-localhost URIs,\nwhich is falsy, so `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` pointing at a\nremote collector was silently ignored (traces sent to `/dev/null`).\nSplit the logic into `is_localhost_collector_listening` (pure TCP probe)\nand `_should_use_otlp_exporter` (routing policy that trusts remote\nendpoints and only probes localhost).\n- **Guard `parsed.port is None`** — prevents `TypeError` on portless\nURIs.\n- **Add tests** — 10 tests covering the probe and routing functions.\n\n## Test plan\n\n- [x] `ruff check` passes\n- [x] `pytest python/xorq/common/utils/tests/test_otel_utils.py` — 10/10\npass\n- [ ] Set `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to a remote endpoint and\nverify OTLP exporter is created\n- [ ] Verify no collector on localhost falls back to console exporter\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: ghoersti <ghoersti@users.noreply.github.com>\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-12T17:15:00-04:00",
          "tree_id": "4715c38f0f7f408233e5d58ea4524732012b5b19",
          "url": "https://github.com/xorq-labs/xorq/commit/54e16eb441b429fba36efd01f1f123158eec5cd6"
        },
        "date": 1776028715483,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.447359741032741,
            "unit": "iter/sec",
            "range": "stddev: 0.02389376280079937",
            "extra": "mean: 134.2757748749932 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.807691600406936,
            "unit": "iter/sec",
            "range": "stddev: 0.011382852488924313",
            "extra": "mean: 208.0000305999988 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7020249263699855,
            "unit": "iter/sec",
            "range": "stddev: 0.19256142873449514",
            "extra": "mean: 1.424450845599995 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.646913076407305,
            "unit": "iter/sec",
            "range": "stddev: 0.012083396643623168",
            "extra": "mean: 274.2045064000081 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.090509342164834,
            "unit": "iter/sec",
            "range": "stddev: 0.04410231783052364",
            "extra": "mean: 244.4683330000089 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.995915743956872,
            "unit": "iter/sec",
            "range": "stddev: 0.009135155392085686",
            "extra": "mean: 200.16350380000176 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f386dbf2944715f91dcb23e1cfe6190696472649",
          "message": "chore(cli): split uv into its own subcommand group (XOR-287) (#1818)\n\nReplace top-level `xorq uv-build` / `xorq uv-run` commands with a `uv`\nsubgroup so the interface becomes `xorq uv build` / `xorq uv run`.\nUpdate tests, README, and docs accordingly.\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-13T08:39:55-04:00",
          "tree_id": "737dc6d99939d7e07c756797689be7a869cf66ef",
          "url": "https://github.com/xorq-labs/xorq/commit/f386dbf2944715f91dcb23e1cfe6190696472649"
        },
        "date": 1776084212071,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.798738551365833,
            "unit": "iter/sec",
            "range": "stddev: 0.016677614190112502",
            "extra": "mean: 128.22586542856536 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.879962806641376,
            "unit": "iter/sec",
            "range": "stddev: 0.011729999086975395",
            "extra": "mean: 204.91959459999407 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7238960705127603,
            "unit": "iter/sec",
            "range": "stddev: 0.17063632990270822",
            "extra": "mean: 1.381413770200004 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.598159985892538,
            "unit": "iter/sec",
            "range": "stddev: 0.032300788259396356",
            "extra": "mean: 217.47829633333046 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.07916611794753,
            "unit": "iter/sec",
            "range": "stddev: 0.008837741238262172",
            "extra": "mean: 196.88271199999576 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.125632213288802,
            "unit": "iter/sec",
            "range": "stddev: 0.011871914063006207",
            "extra": "mean: 195.09788419999836 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "c0aaf41af184b04736de2cc1f08e3a4945683939",
          "message": "chore(deps): update dependency cryptography to v46.0.7 [security] (#1814)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [cryptography](https://redirect.github.com/pyca/cryptography)\n([changelog](https://cryptography.io/en/latest/changelog/)) | `46.0.0` →\n`46.0.7` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/cryptography/46.0.7?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/cryptography/46.0.0/46.0.7?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n####\n[CVE-2026-26007](https://redirect.github.com/pyca/cryptography/security/advisories/GHSA-r6ph-v2qm-q3c2)\n\n## Vulnerability Summary\n\nThe `public_key_from_numbers` (or\n`EllipticCurvePublicNumbers.public_key()`),\n`EllipticCurvePublicNumbers.public_key()`, `load_der_public_key()` and\n`load_pem_public_key()` functions do not verify that the point belongs\nto the expected prime-order subgroup of the curve.\n\nThis missing validation allows an attacker to provide a public key point\n`P` from a small-order subgroup. This can lead to security issues in\nvarious situations, such as the most commonly used signature\nverification (ECDSA) and shared key negotiation (ECDH). When the victim\ncomputes the shared secret as `S = [victim_private_key]P` via ECDH, this\nleaks information about `victim_private_key mod (small_subgroup_order)`.\nFor curves with cofactor > 1, this reveals the least significant bits of\nthe private key. When these weak public keys are used in ECDSA , it's\neasy to forge signatures on the small subgroup.\n\nOnly SECT curves are impacted by this.\n\n## Credit\n\nThis vulnerability was discovered by:\n- XlabAI Team of Tencent Xuanwu Lab\n- Atuin Automated Vulnerability Discovery Engine\n\n####\n[CVE-2026-34073](https://redirect.github.com/pyca/cryptography/security/advisories/GHSA-m959-cc7f-wv43)\n\n## Summary\n\nIn versions of cryptography prior to 46.0.5, DNS name constraints were\nonly validated against SANs within child certificates, and not the \"peer\nname\" presented during each validation. Consequently, cryptography would\nallow a peer named `bar.example.com` to validate against a wildcard leaf\ncertificate for `*.example.com`, even if the leaf's parent certificate\n(or upwards) contained an excluded subtree constraint for\n`bar.example.com`.\n\nThis behavior resulted from a gap between RFC 5280 (which defines Name\nConstraint semantics) and RFC 9525 (which defines service identity\nsemantics): put together, neither states definitively whether Name\nConstraints should be applied to peer names. To close this gap,\ncryptography now conservatively rejects any validation where the peer\nname would be rejected by a name constraint if it were a SAN instead.\n\nIn practice, exploitation of this bypass requires an uncommon X.509\ntopology, one that the Web PKI avoids because it exhibits these kinds of\nproblems. Consequently, we consider this a medium-to-low impact\nseverity.\n\nSee CVE-2025-61727 for a similar bypass in Go's `crypto/x509`.\n\n## Remediation\n\nUsers should upgrade to 46.0.6 or newer. \n\n## Attribution\n\nReporter: @&#8203;1seal\n\n####\n[CVE-2026-39892](https://redirect.github.com/pyca/cryptography/security/advisories/GHSA-p423-j2cm-9vmq)\n\nIf a non-contiguous buffer was passed to APIs which accepted Python\nbuffers (e.g. `Hash.update()`), this could lead to buffer overflows. For\nexample:\n\n```python\nh = Hash(SHA256())\nb.update(buf[::-1])\n```\n\nwould read past the end of the buffer on Python >3.11\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMTAuMiIsInVwZGF0ZWRJblZlciI6IjQzLjExMC4yIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-13T15:24:05+02:00",
          "tree_id": "147d995f5ece58491e8a7b5e86c65c6fbdb43534",
          "url": "https://github.com/xorq-labs/xorq/commit/c0aaf41af184b04736de2cc1f08e3a4945683939"
        },
        "date": 1776086851641,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.888469368048264,
            "unit": "iter/sec",
            "range": "stddev: 0.00550058694433732",
            "extra": "mean: 101.12788570000646 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.7651819175808114,
            "unit": "iter/sec",
            "range": "stddev: 0.03493989213756036",
            "extra": "mean: 265.59141679999243 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7362795602740072,
            "unit": "iter/sec",
            "range": "stddev: 0.17274303389278942",
            "extra": "mean: 1.358179764800002 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.425084734133219,
            "unit": "iter/sec",
            "range": "stddev: 0.03441378365427879",
            "extra": "mean: 225.98437320000357 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.6930903311016117,
            "unit": "iter/sec",
            "range": "stddev: 0.02200061141349212",
            "extra": "mean: 270.7759383999985 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.380625575270205,
            "unit": "iter/sec",
            "range": "stddev: 0.027441259632164122",
            "extra": "mean: 228.27789840000605 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "4b3d8033337c2171c545f2e7034c97031ac0063d",
          "message": "fix(dev): handle missing parent dirs in worktree setup/cleanup (#1806)\n\n## Summary\n\n- **setup-worktree** now creates parent directories before symlinking,\nso targets like `ci/ibis-testing-data` no longer fail when `ci/` doesn't\nexist.\n- **cleanup-worktree** removes empty parent directories left behind.\n- Users can now list additional paths in `.envrcs/.worktree-symlinks`\n(one per line) to have them automatically symlinked into new worktrees.\nCleanup is automatic via the existing manifest.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-13T15:27:52+02:00",
          "tree_id": "b7a6410bf4400a0b91277b190c60680419902348",
          "url": "https://github.com/xorq-labs/xorq/commit/4b3d8033337c2171c545f2e7034c97031ac0063d"
        },
        "date": 1776087084936,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.633496104183598,
            "unit": "iter/sec",
            "range": "stddev: 0.02666275344946063",
            "extra": "mean: 131.00157337500207 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.86459137964106,
            "unit": "iter/sec",
            "range": "stddev: 0.012281846415444964",
            "extra": "mean: 205.56711180000207 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7149563967011384,
            "unit": "iter/sec",
            "range": "stddev: 0.21630114459571728",
            "extra": "mean: 1.3986866956000028 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.014099714713485,
            "unit": "iter/sec",
            "range": "stddev: 0.034740005705832966",
            "extra": "mean: 249.12186319999705 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.354379573360482,
            "unit": "iter/sec",
            "range": "stddev: 0.044140069768182015",
            "extra": "mean: 229.65384233332978 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.008925971196691,
            "unit": "iter/sec",
            "range": "stddev: 0.004900844654113241",
            "extra": "mean: 199.64359740000077 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "375ce76ccfb5994d14550919d775210340d05fed",
          "message": "chore(deps): update dependency pytest to v9.0.3 [security] (#1822)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [pytest](https://redirect.github.com/pytest-dev/pytest)\n([changelog](https://docs.pytest.org/en/stable/changelog.html)) |\n`==9.0.2` → `==9.0.3` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/pytest/9.0.3?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/pytest/9.0.2/9.0.3?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n#### [CVE-2025-71176](https://nvd.nist.gov/vuln/detail/CVE-2025-71176)\n\npytest through 9.0.2 on UNIX relies on directories with the\n`/tmp/pytest-of-{user}` name pattern, which allows local users to cause\na denial of service or possibly gain privileges.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMTAuMiIsInVwZGF0ZWRJblZlciI6IjQzLjExMC4yIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-14T10:05:33+02:00",
          "tree_id": "9009f756ec3d83c81151a70458f909f2944575c3",
          "url": "https://github.com/xorq-labs/xorq/commit/375ce76ccfb5994d14550919d775210340d05fed"
        },
        "date": 1776154143589,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.535965771085443,
            "unit": "iter/sec",
            "range": "stddev: 0.016552276359007228",
            "extra": "mean: 132.6969933750064 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.325568827418542,
            "unit": "iter/sec",
            "range": "stddev: 0.05048153868614354",
            "extra": "mean: 231.18346740000675 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7115347672855271,
            "unit": "iter/sec",
            "range": "stddev: 0.24651225399668",
            "extra": "mean: 1.405412702199999 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.060437010062417,
            "unit": "iter/sec",
            "range": "stddev: 0.007515971393799058",
            "extra": "mean: 197.61139166667854 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.135222466764003,
            "unit": "iter/sec",
            "range": "stddev: 0.01796064266427133",
            "extra": "mean: 194.73353033333277 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.027919205742837,
            "unit": "iter/sec",
            "range": "stddev: 0.0118907412229564",
            "extra": "mean: 198.88943299999937 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "f0478324b28bf7154a3d1ad720b6789933d98c33",
          "message": "fix(packaging): exclude env_templates from VCS scan (#1825)\n\nexclude env_templates from VCS scan to prevent duplicate wheel entries\n\nIn a git repo, hatchling includes env_templates/ via `git ls-files`,\nwhile force-include (added in 8975c28) also adds the same files —\nproducing a ZIP with duplicate entries that PyPI rejects with HTTP 400.\n\nAdding `exclude = [\"python/xorq/env_templates/**\"]` to the wheel target\nprevents the VCS scan from including these files, so force-include\nremains the single code path that writes them in both git and non-git\ncontexts.\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-14T08:51:41-04:00",
          "tree_id": "64cf41623e9f60b3ea0a4285ef7ab96ae1b61853",
          "url": "https://github.com/xorq-labs/xorq/commit/f0478324b28bf7154a3d1ad720b6789933d98c33"
        },
        "date": 1776171319280,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.4968183815156175,
            "unit": "iter/sec",
            "range": "stddev: 0.011693399518773257",
            "extra": "mean: 133.38991944444464 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.969080154338481,
            "unit": "iter/sec",
            "range": "stddev: 0.0604593105563913",
            "extra": "mean: 251.9475448000037 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6208789870469495,
            "unit": "iter/sec",
            "range": "stddev: 0.26974891738169565",
            "extra": "mean: 1.6106198161999998 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.3234237102540867,
            "unit": "iter/sec",
            "range": "stddev: 0.029213347765971648",
            "extra": "mean: 300.89452539999684 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.7084351340188606,
            "unit": "iter/sec",
            "range": "stddev: 0.02534457343484676",
            "extra": "mean: 269.6555187999991 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.1378422540808595,
            "unit": "iter/sec",
            "range": "stddev: 0.03745287650462483",
            "extra": "mean: 241.6718517999982 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6abcaf2e028576a2fd3471ebf0420a601cec7a0c",
          "message": "release: 0.3.19 (#1824)",
          "timestamp": "2026-04-14T15:09:33+02:00",
          "tree_id": "4f2063a1dbdc7b7059c8b010735677c0e2364449",
          "url": "https://github.com/xorq-labs/xorq/commit/6abcaf2e028576a2fd3471ebf0420a601cec7a0c"
        },
        "date": 1776172395706,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.231117586607969,
            "unit": "iter/sec",
            "range": "stddev: 0.01692089581987501",
            "extra": "mean: 121.49018520000254 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.247928656887813,
            "unit": "iter/sec",
            "range": "stddev: 0.05693367642270225",
            "extra": "mean: 235.4088499999989 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6731792902539865,
            "unit": "iter/sec",
            "range": "stddev: 0.23270158523534265",
            "extra": "mean: 1.4854883602000086 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.250562407851844,
            "unit": "iter/sec",
            "range": "stddev: 0.04547586251456713",
            "extra": "mean: 235.26298499999712 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.721085055514203,
            "unit": "iter/sec",
            "range": "stddev: 0.017277857910969984",
            "extra": "mean: 211.81571359999225 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.92720171963681,
            "unit": "iter/sec",
            "range": "stddev: 0.008921929091143593",
            "extra": "mean: 202.95495433333124 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "8247e842f1a1147c231da6b62aa82bf09fd39359",
          "message": "chore(deps): update dependency requests to v2.33.0 [security] (#1815)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [requests](https://redirect.github.com/psf/requests)\n([changelog](https://redirect.github.com/psf/requests/blob/master/HISTORY.md))\n| `2.32.4` → `2.33.0` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/requests/2.33.0?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/requests/2.32.4/2.33.0?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n####\n[CVE-2026-25645](https://redirect.github.com/psf/requests/security/advisories/GHSA-gc5v-m9x4-r6x2)\n\n### Impact\nThe `requests.utils.extract_zipped_paths()` utility function uses a\npredictable filename when extracting files from zip archives into the\nsystem temporary directory. If the target file already exists, it is\nreused without validation. A local attacker with write access to the\ntemp directory could pre-create a malicious file that would be loaded in\nplace of the legitimate one.\n\n### Affected usages\n**Standard usage of the Requests library is not affected by this\nvulnerability.** Only applications that call `extract_zipped_paths()`\ndirectly are impacted.\n\n### Remediation\nUpgrade to at least Requests 2.33.0, where the library now extracts\nfiles to a non-deterministic location.\n\nIf developers are unable to upgrade, they can set `TMPDIR` in their\nenvironment to a directory with restricted write access.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMTAuMiIsInVwZGF0ZWRJblZlciI6IjQzLjExMC4yIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-14T15:33:24+02:00",
          "tree_id": "1a1e2d2eea3bffaa819eea16d5af51657a67d3cc",
          "url": "https://github.com/xorq-labs/xorq/commit/8247e842f1a1147c231da6b62aa82bf09fd39359"
        },
        "date": 1776173826938,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.464493686424441,
            "unit": "iter/sec",
            "range": "stddev: 0.030206091502262254",
            "extra": "mean: 133.9675592222262 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.7296886811452605,
            "unit": "iter/sec",
            "range": "stddev: 0.015061720585268036",
            "extra": "mean: 211.43040639999526 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7328620426423118,
            "unit": "iter/sec",
            "range": "stddev: 0.14087865897117505",
            "extra": "mean: 1.3645132941999976 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.7477130115395285,
            "unit": "iter/sec",
            "range": "stddev: 0.0071434777044375635",
            "extra": "mean: 266.8293962000064 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.4813137261576905,
            "unit": "iter/sec",
            "range": "stddev: 0.03521853698314748",
            "extra": "mean: 223.1488490000023 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.783323909334918,
            "unit": "iter/sec",
            "range": "stddev: 0.01976266706912433",
            "extra": "mean: 209.05964533333096 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5eaec96df86e78abdebec645807ec6bbc14d8b89",
          "message": "chore(deps): update dependency uv to v0.11.6 [security] (#1807)\n\nThis PR contains the following updates:\n\n| Package | Change |\n[Age](https://docs.renovatebot.com/merge-confidence/) |\n[Confidence](https://docs.renovatebot.com/merge-confidence/) |\n|---|---|---|---|\n| [uv](https://pypi.org/project/uv/)\n([source](https://redirect.github.com/astral-sh/uv),\n[changelog](https://redirect.github.com/astral-sh/uv/blob/main/CHANGELOG.md))\n| `0.10.4` → `0.11.6` |\n![age](https://developer.mend.io/api/mc/badges/age/pypi/uv/0.11.6?slim=true)\n|\n![confidence](https://developer.mend.io/api/mc/badges/confidence/pypi/uv/0.10.4/0.11.6?slim=true)\n|\n\n### GitHub Vulnerability Alerts\n\n####\n[GHSA-pjjw-68hj-v9mw](https://redirect.github.com/astral-sh/uv/security/advisories/GHSA-pjjw-68hj-v9mw)\n\n## Impact\n\nWheel RECORD entries can contain relative paths that traverse outside of\nthe wheel’s installation prefix. In versions 0.11.5 and earlier of uv,\nthese wheels were not rejected on installation and the RECORD was\nrespected without validation on uninstall.\n\nuv uses the RECORD to determine files to remove on uninstall.\nConsequently, a malicious or malformed wheel could induce deletion of\narbitrary files outside of the wheel’s installation prefix on uninstall.\n\nuv does not use the RECORD file to determine wheel file paths. Invalid\nRECORD entries cannot be used to create or modify files in arbitrary\nlocations.\n\nStandards-compliant Python packaging tooling does not produce RECORD\nfiles that exhibit this behavior; an attacker must manually manipulate\nthe RECORD. A user must install *and* uninstall the malformed wheel to\nbe affected. An attack must guess the depth of the installation prefix\npath in order to target system files.\n\nAbsolute paths in RECORD files are not allowed by the specification and,\nwhen present, uv always treats them as rooted in the wheel’s\ninstallation prefix. Absolute paths cannot be used to delete arbitrary\nfiles.\n\nOnly files can be deleted, attempts to delete a directory via an invalid\nRECORD entry will fail.\n\n## Patches\n\nVersions\n[0.11.6](https://redirect.github.com/astral-sh/uv/releases/tag/0.11.6)\nand newer of uv address the validation gap above, by [removing invalid\nentries from RECORD files on wheel\ninstallation](https://redirect.github.com/astral-sh/uv/pull/18943) and\n[ignoring RECORD paths that would escape the installation prefix on\nuninstall](https://redirect.github.com/astral-sh/uv/pull/18942).\n\n## Workarounds\n\nUsers are advised to upgrade to 0.11.6 or newer to address this\nadvisory.\n\nUsers should experience no breaking changes as a result of the patch\nabove.\n\n---\n\n- [ ] <!-- rebase-check -->If you want to rebase/retry this PR, check\nthis box\n\n<!--renovate-debug:eyJjcmVhdGVkSW5WZXIiOiI0My4xMTAuMiIsInVwZGF0ZWRJblZlciI6IjQzLjExMC4yIiwidGFyZ2V0QnJhbmNoIjoibWFpbiIsImxhYmVscyI6W119-->\n\nCo-authored-by: renovate[bot] <29139614+renovate[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-14T15:37:22+02:00",
          "tree_id": "a435e73b2486244c0eecece70a4c0de95b58cbec",
          "url": "https://github.com/xorq-labs/xorq/commit/5eaec96df86e78abdebec645807ec6bbc14d8b89"
        },
        "date": 1776174089064,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.559350028652288,
            "unit": "iter/sec",
            "range": "stddev: 0.0041993473520120575",
            "extra": "mean: 86.51005441666608 msec\nrounds: 12"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.111024741587291,
            "unit": "iter/sec",
            "range": "stddev: 0.02540186629844965",
            "extra": "mean: 243.24835360000634 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.8028068255020954,
            "unit": "iter/sec",
            "range": "stddev: 0.2256587773469103",
            "extra": "mean: 1.2456296686000088 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.185579661806432,
            "unit": "iter/sec",
            "range": "stddev: 0.031186803343424004",
            "extra": "mean: 238.91553399999452 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.351568529573392,
            "unit": "iter/sec",
            "range": "stddev: 0.02415245360215867",
            "extra": "mean: 186.86110333332806 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.812313548609815,
            "unit": "iter/sec",
            "range": "stddev: 0.004642967614912205",
            "extra": "mean: 172.0485296666728 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "05d8c32dc568641536cb5e3cce09c29135aa62b8",
          "message": "feat(catalog): CLI-settable default catalog (#1828)\n\n## Summary\n\n- Adds `xorq catalog default [--set NAME | --unset]` subcommand to\npersist a default catalog name across invocations\n- Resolution order: CLI flags (`--name`/`--path`) >\n`XORQ_DEFAULT_CATALOG` env var > config file\n(`~/.config/xorq/catalog-default`) > hardcoded `\"default\"`\n- Wired through the existing `env_config` / `.env.xorq.template` pattern\n\nCloses XOR-9\n\n## Test plan\n\n- [x] 12 new tests in `test_catalog_default.py` covering resolution\norder, `from_default()` integration, and CLI set/show/unset behavior\n- [x] All 40 existing `test_catalog_ctor.py` tests still pass\n- [ ] Manual: `xorq catalog default`, `xorq catalog default --set foo`,\n`xorq catalog default --unset`\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-16T20:14:50-04:00",
          "tree_id": "273268de0e9d2a678f58c8866956feb38b84a3dd",
          "url": "https://github.com/xorq-labs/xorq/commit/05d8c32dc568641536cb5e3cce09c29135aa62b8"
        },
        "date": 1776385102939,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.358700666394471,
            "unit": "iter/sec",
            "range": "stddev: 0.013916249883643083",
            "extra": "mean: 119.6358190000062 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.857118126917672,
            "unit": "iter/sec",
            "range": "stddev: 0.019911660592037052",
            "extra": "mean: 205.8834012000034 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7146536937605584,
            "unit": "iter/sec",
            "range": "stddev: 0.1984958363971833",
            "extra": "mean: 1.3992791315999908 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.618134103228627,
            "unit": "iter/sec",
            "range": "stddev: 0.02540802995955074",
            "extra": "mean: 216.53767033332372 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.181160225810038,
            "unit": "iter/sec",
            "range": "stddev: 0.01450137999135241",
            "extra": "mean: 193.00696300000197 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.084089182753809,
            "unit": "iter/sec",
            "range": "stddev: 0.010171391708042796",
            "extra": "mean: 196.69206499999822 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "e2b40b43db685dcae05f9f0546d23ac76c18a3ac",
          "message": "feat: add DataBricks backend (#1803)\n\nCo-authored-by: ghoersti <ghoersti@users.noreply.github.com>",
          "timestamp": "2026-04-17T12:04:05+02:00",
          "tree_id": "c40b1b7b73de696fdc986c02653ab7160f765ef0",
          "url": "https://github.com/xorq-labs/xorq/commit/e2b40b43db685dcae05f9f0546d23ac76c18a3ac"
        },
        "date": 1776420465982,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.337481378362013,
            "unit": "iter/sec",
            "range": "stddev: 0.02400836106266137",
            "extra": "mean: 136.28654689999848 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.842841976058617,
            "unit": "iter/sec",
            "range": "stddev: 0.010345649087662345",
            "extra": "mean: 206.49032219999413 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7171138940821853,
            "unit": "iter/sec",
            "range": "stddev: 0.182200696539394",
            "extra": "mean: 1.3944786292000004 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.8522023817235715,
            "unit": "iter/sec",
            "range": "stddev: 0.021427025885895637",
            "extra": "mean: 259.5917609999958 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.7798900418193098,
            "unit": "iter/sec",
            "range": "stddev: 0.061994309293537525",
            "extra": "mean: 264.55796040000337 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.966851922221646,
            "unit": "iter/sec",
            "range": "stddev: 0.03275021437722716",
            "extra": "mean: 252.08906699999716 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "45543592+ghoersti@users.noreply.github.com",
            "name": "ghoersti",
            "username": "ghoersti"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "434add6466398d0893f81bc145e639a1066f196f",
          "message": "refactor: drop BSL builtin tag handler, rely on entry-point discovery… (#1827)\n\n… (XOR-296)\n\nBSL now ships its own TagHandler via the xorq.from_tag_node entry point\npending (boring-semantic-layer#235), xorq no longer needs BSL-specific\ncode. Remove _bsl_extract_metadata, _bsl_from_tag_node, and the BSL\nentry from _builtin_handlers — only the ML pipeline handler remains\nbuilt-in.\n\nCo-authored-by: ghoersti <ghoersti@users.noreply.github.com>\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-17T07:13:38-04:00",
          "tree_id": "54fd9c78df1e3c9ef04e7dc5bff1d8a3bb1bf82b",
          "url": "https://github.com/xorq-labs/xorq/commit/434add6466398d0893f81bc145e639a1066f196f"
        },
        "date": 1776424607261,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.499135346350636,
            "unit": "iter/sec",
            "range": "stddev: 0.015582747348038719",
            "extra": "mean: 105.27273941666475 msec\nrounds: 12"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.59011418661805,
            "unit": "iter/sec",
            "range": "stddev: 0.047279571410031286",
            "extra": "mean: 217.85950400000615 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7786556441004265,
            "unit": "iter/sec",
            "range": "stddev: 0.14134132223844276",
            "extra": "mean: 1.2842647549999981 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.705498789709612,
            "unit": "iter/sec",
            "range": "stddev: 0.007374703534733497",
            "extra": "mean: 175.26951400000144 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.992173822282218,
            "unit": "iter/sec",
            "range": "stddev: 0.02857490290371495",
            "extra": "mean: 250.49009500000352 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.0831828046174365,
            "unit": "iter/sec",
            "range": "stddev: 0.027168242195684922",
            "extra": "mean: 196.72713699999633 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b634749e86a0000524127ea9370a6797f839ea4a",
          "message": "fix(caching): resolve ParquetCache \"relation already exists\" (#1823)\n\n- Pass mode=\"replace\" in ParquetStorage.get() for ADBC-backed sources\n(postgres, sqlite, snowflake, databricks) so cache hits do not fail when\nthe materialized table already exists from a prior execution.\n- Extract shared adbc_ingest logic into ADBCBase mixin; PgADBC and\nSQLiteADBC now inherit from it instead of duplicating the method body.\n\n- Add regression tests in test_cache.py covering both postgres and\nsqlite.\n\ncloses #1820\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-18T02:15:26-04:00",
          "tree_id": "a546d061db8b2bff9e950c39fd504a45845233d9",
          "url": "https://github.com/xorq-labs/xorq/commit/b634749e86a0000524127ea9370a6797f839ea4a"
        },
        "date": 1776493141875,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.3677596390592,
            "unit": "iter/sec",
            "range": "stddev: 0.005943622933470654",
            "extra": "mean: 106.74910955555106 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.313490071703264,
            "unit": "iter/sec",
            "range": "stddev: 0.019087337558267675",
            "extra": "mean: 301.79658859999563 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6406687420467718,
            "unit": "iter/sec",
            "range": "stddev: 0.19324518109774125",
            "extra": "mean: 1.5608690331999924 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.488898742879551,
            "unit": "iter/sec",
            "range": "stddev: 0.01159154313819235",
            "extra": "mean: 222.7717881999979 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.601504132860534,
            "unit": "iter/sec",
            "range": "stddev: 0.010777588606958101",
            "extra": "mean: 217.32024380000894 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.362386405093867,
            "unit": "iter/sec",
            "range": "stddev: 0.021611134731879256",
            "extra": "mean: 229.2323300000021 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "fc4ff21cb19c1bd4335681703cc4eeaf47772786",
          "message": "feat(xorq-datafusion): rename backend to xorq-datafusion (#1837)\n\n## Summary\n- Rename the DataFusion-based backend's `name` from `\"xorq\"` to\n`\"xorq-datafusion\"` to distinguish it from other backends\n- Update the entry point key in `pyproject.toml` and all string\ncomparisons that check backend identity (`relations.py`,\n`dask_normalize_expr.py`, `compiler.py`, `test_into_backend.py`)\n- No module path changes — `xorq.backends.xorq` remains the import path\n\n## Test plan\n- [ ] Verify `xo.connect().name` returns `\"xorq-datafusion\"`\n- [ ] Run `pytest python/xorq/tests/test_into_backend.py -v`\n- [ ] Run `pytest python/xorq/ibis_yaml/ -v`\n- [ ] Verify `load_backend(\"xorq-datafusion\")` resolves correctly\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-18T06:50:47-04:00",
          "tree_id": "613fd817edebb8eae22d8054d750d35191abd39e",
          "url": "https://github.com/xorq-labs/xorq/commit/fc4ff21cb19c1bd4335681703cc4eeaf47772786"
        },
        "date": 1776509664285,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.158411255980008,
            "unit": "iter/sec",
            "range": "stddev: 0.011163652541844574",
            "extra": "mean: 122.57288442857218 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.456805741809859,
            "unit": "iter/sec",
            "range": "stddev: 0.03205567475208037",
            "extra": "mean: 224.37594499999705 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7087830736532544,
            "unit": "iter/sec",
            "range": "stddev: 0.19369436538432264",
            "extra": "mean: 1.4108689064000033 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.412775079524112,
            "unit": "iter/sec",
            "range": "stddev: 0.03804526233807466",
            "extra": "mean: 226.61476780000385 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.1318019599510185,
            "unit": "iter/sec",
            "range": "stddev: 0.00944324813891386",
            "extra": "mean: 194.86332633333822 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.917324849300129,
            "unit": "iter/sec",
            "range": "stddev: 0.0061293527611889975",
            "extra": "mean: 203.36260683333288 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "5c431bf51447a72ab7b8c0bc76267d6dc517354b",
          "message": "docs: add ADR template (#1817)\n\n## Summary\n\n- Adds `docs/adr/template.md` with the common structure extracted from\nADR-0002 through ADR-0005\n- Includes HTML comment guidance on when to write an ADR vs. a commit\nmessage, how to title decisions (not problems), and when Rationale\nwarrants its own section\n\n## Test plan\n\n- [x] Reviewed against existing ADRs for consistency\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-18T07:41:33-04:00",
          "tree_id": "fa284b5e7475f70d10d486438e79e06ee8b7a27e",
          "url": "https://github.com/xorq-labs/xorq/commit/5c431bf51447a72ab7b8c0bc76267d6dc517354b"
        },
        "date": 1776512715377,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.936511148905495,
            "unit": "iter/sec",
            "range": "stddev: 0.015595326731913745",
            "extra": "mean: 125.999948999997 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.846381065598588,
            "unit": "iter/sec",
            "range": "stddev: 0.008115307752802043",
            "extra": "mean: 206.3395318000005 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7082084658468635,
            "unit": "iter/sec",
            "range": "stddev: 0.1632641113842091",
            "extra": "mean: 1.4120136205999985 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.238481727967089,
            "unit": "iter/sec",
            "range": "stddev: 0.035285142377843644",
            "extra": "mean: 235.9335404000035 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.92667102205739,
            "unit": "iter/sec",
            "range": "stddev: 0.005156452890322864",
            "extra": "mean: 202.97681649999788 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.933187859433354,
            "unit": "iter/sec",
            "range": "stddev: 0.009245031168962251",
            "extra": "mean: 202.708680166675 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b2fd1d626685a08f86efd8fbea6a057b1f0d2529",
          "message": "fix(compiler): use except_ in sge.Star to emit EXCLUDE clauses (#1839)\n\n## Summary\n\n- **Fix one-character typo** (`\"except\"` → `\"except_\"`) in `sge.Star()`\nacross all four vendored compilers that override `visit_DropColumns`:\nDuckDB, Snowflake, BigQuery, and ClickHouse\n- `except` is a Python reserved word; sqlglot names the arg slot\n`except_` — passing `\"except\"` was silently ignored, causing `SELECT *\nEXCLUDE (...)` to render as bare `SELECT *`\n- This made `DropColumns` a no-op on wide tables, leading to column\ncount mismatches downstream (e.g., `ArrowInvalid: tried to rename a\ntable of 72 columns but only 70 names were provided`)\n- Bug was inherited from upstream ibis when the vendor copy was created\n(0c806fdc, Feb 2025)\n\n### Why it went undetected\n\nThe `drop_columns_to_select` rewrite converts `DropColumns` → explicit\n`Select` when ≥50% of columns are dropped. Most test tables are narrow\n(3–5 columns), so the rewrite always kicked in and `visit_DropColumns`\nwas never exercised.\n\n## Test plan\n\n- [x] Added 7 regression tests in `test_drop_columns.py` using a\n10-column table (dropping 2 = 20% < 50% threshold) to ensure\n`visit_DropColumns` runs\n- [x] `test_drop_columns_generates_exclude` — SQL contains `EXCLUDE`,\nnot bare `SELECT *`\n- [x] `test_drop_columns_lists_all_dropped` — every dropped column\nappears in `EXCLUDE`\n- [x] `test_drop_columns_preserves_remaining` — dropped columns absent\nfrom result schema\n- [x] `test_sge_star_except_underscore` — regression guard proving\n`\"except\"` is silently ignored by sqlglot\n- [x] Pre-commit passes on all changed files\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-18T08:06:53-04:00",
          "tree_id": "49dba1573df53e2f73f055f537026815c4f68ceb",
          "url": "https://github.com/xorq-labs/xorq/commit/b2fd1d626685a08f86efd8fbea6a057b1f0d2529"
        },
        "date": 1776514234926,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.627386869529853,
            "unit": "iter/sec",
            "range": "stddev: 0.012197831102036084",
            "extra": "mean: 131.10650044444898 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.524439564299585,
            "unit": "iter/sec",
            "range": "stddev: 0.030001579482699933",
            "extra": "mean: 221.0218493999946 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6826201441007616,
            "unit": "iter/sec",
            "range": "stddev: 0.16300250644650088",
            "extra": "mean: 1.4649435834000086 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.9288264152523156,
            "unit": "iter/sec",
            "range": "stddev: 0.03453112668771585",
            "extra": "mean: 254.52893416666217 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.230573305109196,
            "unit": "iter/sec",
            "range": "stddev: 0.04895639931980356",
            "extra": "mean: 236.3745827999992 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.994388285345388,
            "unit": "iter/sec",
            "range": "stddev: 0.006631442599740383",
            "extra": "mean: 200.2247207999858 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "759d3d01a49f0378b64ef06ba3e544583c641f32",
          "message": "feat(metadata): compute synthetic cache keys for uncached expressions (XOR-284) (#1787)\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-19T11:02:54-04:00",
          "tree_id": "5c0f6a16631edfc142359593d458faaf77e35c81",
          "url": "https://github.com/xorq-labs/xorq/commit/759d3d01a49f0378b64ef06ba3e544583c641f32"
        },
        "date": 1776611193178,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.299970645195918,
            "unit": "iter/sec",
            "range": "stddev: 0.015103839428671433",
            "extra": "mean: 136.98685222222036 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.6498491412709155,
            "unit": "iter/sec",
            "range": "stddev: 0.007739882824843555",
            "extra": "mean: 215.0607406000006 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.688406306237774,
            "unit": "iter/sec",
            "range": "stddev: 0.1493233179233083",
            "extra": "mean: 1.4526305046000005 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.8957971531787154,
            "unit": "iter/sec",
            "range": "stddev: 0.03690857070536055",
            "extra": "mean: 256.6868758000055 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.675791887251157,
            "unit": "iter/sec",
            "range": "stddev: 0.018644500476699927",
            "extra": "mean: 213.86751680000202 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.938658468957033,
            "unit": "iter/sec",
            "range": "stddev: 0.009635409247570328",
            "extra": "mean: 202.48413740000615 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b1c7038ecfdc2eb8c5346483ee2247f4f40dd990",
          "message": "fix(catalog): portable deferred reads via read_path/hash_path (#1832)\n\n## Summary\n\nMake catalog-loaded deferred reads execute end-to-end. Fixes a chain of\nregressions that surface when a catalog entry is loaded from a zip and\neither executed locally or served over flight.\n\n### Catalog / deferred reads\n\n- **Portable deferred reads** (`fab04ad5` + `a7462020`): split `path` in\n`read_kwargs` into `hash_path` (absolute, used only by `normalize_read`\n  for cache-key computation) and `read_path` (relative to build root,\n  used by `deferred_reads_to_memtables` for file I/O). Also drop the\n  dead `try_names` fallback in `normalize_read` — `make_read_kwargs`\n  already normalizes backend-specific param names into `hash_path`.\n- **database_table re-registration** (`34c9fbc0`):\n  `deferred_reads_to_memtables` now dispatches `database_table` Reads\n  through `make_dt` instead of converting to bare memtables, preserving\n  the backend connection.\n- **Read hash disambiguation** (`0c18e10f`): include `node.name` in the\n  Read content hash so a memtable and a database_table sharing identical\n  data don't collide in `Registry.register_node`'s `setdefault`.\n- **Extract-dir lifetime** (`39cc0532`): replace contextmanager-scoped\n  cleanup with `weakref.finalize` keyed on the loaded expression, plus\n  an atexit sweep. Fixes premature cleanup of files referenced by\n  deferred reads after the context exits.\n\n### ibis_yaml\n\n- **Per-RemoteTable Read rename** (`cf1a3429`): thread the enclosing\n  RemoteTable's name through a stack on `TranslationContext` and mix\n  it into `_read_to_yaml`'s tokenize input. Prevents\n  content-equivalent Reads wrapped by distinct RemoteTables from\n  collapsing to the same `table_name`, which was causing \"table not\n  found\" failures on zip roundtrips of multi-RemoteTable expressions.\n\n### dask_normalize\n\n- **Tempdir canonicalization** (`12fe6c68`): strip\n  `.../xorq-catalog-<random>/` prefixes from DatabaseTable plan paths so\n  tokens match across `load_expr_from_zip` calls that pick fresh\n  `mkdtemp` extract dirs. Preserves the same-path-same-token cache\n  semantic for `test_parquet_cache_storage`.\n\n### flight (UnboundExprExchanger chain)\n\n- **Zip-bundle serialization** (`05a09141`):\n  `UnboundExprExchanger.__reduce__` now serializes via the build-zip\n  bundle. Default cloudpickle walked into `Backend.__reduce__` which\n  returns only `(Profile.get_con, (profile,))`, dropping all table\n  state — so a client fetching the exchanger got a backend missing the\n  Read-backed tables the server registered at load time.\n- **In-process registration** (`c77a3705`): `FlightServer.serve()`\n  registers exchangers directly on `self.server.exchangers` instead of\n  pickling them through its own gRPC loopback. The loopback path\n  exceeded gRPC's initial-metadata size limit for exchangers carrying\n  UDF closures, manifesting as `ArrowInvalid: received initial metadata\n  size exceeds limit` during startup.\n- **Skip redundant AddExchange** (`bda7404f`): mark server-originated\n  exchangers with `_xorq_server_has_command=True`; `FlightUDXF.to_rbr`\n  skips `AddExchangeAction` when set, and uses the user-provided\n  `command` string so the server's key matches across cloudpickle\n  (unbound_expr token isn't stable across roundtrip).\n\n## Test plan\n\n- [x] `test_catalog_entry_roundtrip_execute` — catalog add → load →\n      execute, data roundtrip verified against git and annex backends\n- [x] `test_read_kwargs_contains_hash_path_and_read_path` — YAML has\n      both keys; `read_path` relative\n- [x] `test_memtable_cache_key_stable_across_roundtrip` — cache keys\n      unchanged by `hash_path` rename\n- [x] Parametrized multi-RemoteTable join-order tests for\n      `build_expr`/`load_expr` and catalog roundtrip paths\n- [x] `test_flight_exchange` — command hash stable across two\n      cloudpickle roundtrips of UnboundExprExchanger\n- [x] `test_parquet_cache_storage` — cache hits preserved under\n      content-change (schema-change still invalidates)\n- [x] Updated snapshot files for `path` → `hash_path` rename and\n      tuple-of-paths DT token form\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.6 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T11:55:52-04:00",
          "tree_id": "89ce684fd115391745398c3553bb5a660add238f",
          "url": "https://github.com/xorq-labs/xorq/commit/b1c7038ecfdc2eb8c5346483ee2247f4f40dd990"
        },
        "date": 1776614367677,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.548724606460834,
            "unit": "iter/sec",
            "range": "stddev: 0.013211530042691882",
            "extra": "mean: 132.4727092499991 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.439494120426339,
            "unit": "iter/sec",
            "range": "stddev: 0.04137298233467999",
            "extra": "mean: 225.25088959999948 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7036496883741719,
            "unit": "iter/sec",
            "range": "stddev: 0.18711616578873788",
            "extra": "mean: 1.4211617179999962 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.33460629474327,
            "unit": "iter/sec",
            "range": "stddev: 0.029524353795766387",
            "extra": "mean: 230.7014598333268 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.541773269623331,
            "unit": "iter/sec",
            "range": "stddev: 0.04756209141337121",
            "extra": "mean: 220.17831816666936 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.126972367418533,
            "unit": "iter/sec",
            "range": "stddev: 0.0072733714994903656",
            "extra": "mean: 195.04688699999897 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9ff7230856b3dd51cb3b0443bc2a60bb10940d32",
          "message": "docs(adr): add ADR-0006 on read_kwargs hash_path/read_path split (#1838)\n\n## Summary\n- Adds ADR-0006 documenting the split of the `read_kwargs` `\"path\"` key\ninto `hash_path` (identity for dask tokenization) and `read_path`\n(location for I/O).\n- Motivated by catalog roundtrip failures where zipped build tmpdirs\nextract to a different path on load, leaving absolute paths pointing at\nnonexistent dirs.\n\n## Test plan\n- [ ] Review ADR for accuracy against current `relations.py`,\n`compiler.py`, `defer_utils.py`, and `dask_normalize_expr.py` behavior\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-19T11:59:57-04:00",
          "tree_id": "6c337ab32c45011ddb537dd2aa3cf0b56ef333d4",
          "url": "https://github.com/xorq-labs/xorq/commit/9ff7230856b3dd51cb3b0443bc2a60bb10940d32"
        },
        "date": 1776614599669,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.78615302681743,
            "unit": "iter/sec",
            "range": "stddev: 0.014918314520236245",
            "extra": "mean: 128.4331294999923 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.338246318622366,
            "unit": "iter/sec",
            "range": "stddev: 0.033846884797221845",
            "extra": "mean: 230.50788879999686 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7333709264843933,
            "unit": "iter/sec",
            "range": "stddev: 0.14384633864149296",
            "extra": "mean: 1.363566462599988 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.4623657814082645,
            "unit": "iter/sec",
            "range": "stddev: 0.033558047647463216",
            "extra": "mean: 224.09637599999996 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.93936853990389,
            "unit": "iter/sec",
            "range": "stddev: 0.011764705022655678",
            "extra": "mean: 202.45502879998867 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.82259017907958,
            "unit": "iter/sec",
            "range": "stddev: 0.008670168246872704",
            "extra": "mean: 207.35744960001057 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "125b9c42aeae46e34c60bf379dc0a3be931a419d",
          "message": "fix(caching): use xorq-datafusion in SnapshotStrategy.normalize_backend (#1842)",
          "timestamp": "2026-04-19T17:58:52+02:00",
          "tree_id": "a1f05f789f533d3bd597e959d648b2ef43a767d0",
          "url": "https://github.com/xorq-labs/xorq/commit/125b9c42aeae46e34c60bf379dc0a3be931a419d"
        },
        "date": 1776614630360,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 11.498842255356724,
            "unit": "iter/sec",
            "range": "stddev: 0.014025519775697857",
            "extra": "mean: 86.96527683333954 msec\nrounds: 12"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.981738618495424,
            "unit": "iter/sec",
            "range": "stddev: 0.007008339842740093",
            "extra": "mean: 167.1754758571394 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.9706439967134006,
            "unit": "iter/sec",
            "range": "stddev: 0.1833749933567762",
            "extra": "mean: 1.0302438416000086 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 6.177835364734935,
            "unit": "iter/sec",
            "range": "stddev: 0.01542690532093251",
            "extra": "mean: 161.86899471428464 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 6.293145903837926,
            "unit": "iter/sec",
            "range": "stddev: 0.005142451262900633",
            "extra": "mean: 158.90303757142223 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.584904019328904,
            "unit": "iter/sec",
            "range": "stddev: 0.04584581968499913",
            "extra": "mean: 179.05410666666435 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d109890b59f7c0bc15bd2411b197162980c8fb61",
          "message": "feat(catalog): add -p/--params to `catalog run` (#1840)\n\n## Summary\n- Add repeatable `-p/--params key=value` to `xorq catalog run`,\nmirroring `xorq run`.\n- Values are coerced via each `NamedScalarParameter`'s declared dtype,\nthen bound after `--rename-params` is applied (so values bind to the\nrenamed names).\n- Extract a small `_apply_cli_params(expr, raw_params)` helper in\n`xorq.cli` and reuse it from `run_cached_command` and the new catalog\n`run` path.\n\n## Example\n```\nxorq catalog run src trn \\\n  --rename-params src,threshold,src_threshold \\\n  -p src_threshold=0.5 -p year=2024 \\\n  -o out.csv -f csv\n```\n\n## Test plan\n- [x] `xorq catalog run <entry> -p name=value` binds the value and\nexecutes.\n- [x] `-p unknown=1` reports unknown parameter with available names.\n- [x] `-p badint=nope` on an Int64 param reports a type-coercion error.\n- [x] Composed multi-entry run binds params defined in any sub-entry.\n- [x] `--rename-params` + `-p <new_name>=...` works end-to-end.\n- [x] Existing `xorq run` / `xorq run-cached` still accept `-p`\nunchanged.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-19T19:23:22-04:00",
          "tree_id": "038e65e33c82fd314ea6a6bc6d17885aec1b9007",
          "url": "https://github.com/xorq-labs/xorq/commit/d109890b59f7c0bc15bd2411b197162980c8fb61"
        },
        "date": 1776641215522,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.089615984109983,
            "unit": "iter/sec",
            "range": "stddev: 0.017497578741624928",
            "extra": "mean: 110.01564881818447 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.128558509651305,
            "unit": "iter/sec",
            "range": "stddev: 0.05291717753378005",
            "extra": "mean: 242.21529079999868 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7145902320130343,
            "unit": "iter/sec",
            "range": "stddev: 0.21654157555445022",
            "extra": "mean: 1.39940339960001 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.184765011828595,
            "unit": "iter/sec",
            "range": "stddev: 0.016222208555277406",
            "extra": "mean: 192.8727720000012 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.217474420983708,
            "unit": "iter/sec",
            "range": "stddev: 0.009395305254048579",
            "extra": "mean: 191.66361333333745 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.597783448692024,
            "unit": "iter/sec",
            "range": "stddev: 0.05426051055036784",
            "extra": "mean: 217.4961068000016 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "77161655e221e37a3047f28fa0206718d25fa9ff",
          "message": "perf(catalog): template-clone fixtures to cut annex test setup (#1843)\n\n## Summary\n\n- Make `data_dict` and `catalog_populated` session-scoped **templates**;\neach test gets a per-test `shutil.copy` / `shutil.copytree(...,\nsymlinks=True)` from the template instead of rebuilding from scratch\n- Eliminates the 4× `git annex add` + commit that every `[annex]` test\npaid in setup — the dominant cost in\n`python/xorq/catalog/tests/test_cli.py` under xdist `--dist=loadfile`\n\n## Motivation\n\nAnalysis of [ci-test job\n72022776701](https://github.com/xorq-labs/xorq/actions/runs/24632718743/job/72022776701)\n(10 min wall) showed:\n\n| Worker | Cumulative | Dominant file |\n|---|---|---|\n| gw1 | **557.6 s** | `catalog/tests/test_cli.py` (entire worker) |\n| gw0 | 416.2 s | `catalog/tests/test_catalog.py` (262 s) |\n| gw2 | 404.1 s | `catalog/tests/test_tui.py` (182 s) |\n| gw3 | 407.1 s | mixed |\n\nWithin `test_cli.py`: `[annex]` parametrization = 517 s, `[git]` = 40 s\nfor the same 87 functions — a 13× gap. Slowest-20 durations on that job\nwere 6–8 s of fixture **setup** per annex test. Root cause: the annex\nfixture chain builds 4 zip archives and runs `catalog.add()` (= `git\nannex add` + commit) on each, for every test.\n\n## Local measurements\n\nRunning `pytest python/xorq/catalog/tests/ -k 'not script_execution and\nnot slow and not benchmark' --no-cov -n auto --dist=loadfile`:\n\n| | Baseline | After | Δ |\n|---|---|---|---|\n| Full `catalog/tests/` | 142.8 s | 116.9 s | **−18%** |\n| `test_cli.py` (`-n 4`, mirrors CI worker count) | 139.8 s | 113.1 s |\n**−19%** |\n\nSame pass/fail set both runs: 550 passed / 21 skipped / 8 pre-existing\nfailures. No new regressions.\n\nCI per-test setup (~8 s) is ~5× slower than local (~1.7 s), so CI impact\nshould be meaningfully larger than the local 18%.\n\n## Implementation notes\n\n- `backend_type` is now `scope=\"session\"` so session-scoped fixtures can\ndepend on it while preserving `[\"git\", \"annex\"]` parametrization.\n- `_catalog_populated_template` builds one pristine annex/git repo per\n(session, backend); `catalog_populated` `copytree`s it per test and\nrebuilds the `Catalog` handle on the copy.\n- `symlinks=True` is required — git-annex stores content via symlinks\ninto `.git/annex/objects`, which must be copied verbatim.\n- Templates share UUID/commit-SHA across copies within a session. No\ntests currently assert UUID uniqueness across catalog-populated\ninstances; `repo_cloned_bare` still gets a fresh annex UUID via its own\n`git annex init`.\n\n## Test plan\n\n- [x] Full `catalog/tests/` passes with same pass/fail set as baseline\n- [ ] CI `ci-test` green\n- [ ] CI wall-time delta captured for comparison with the 606 s baseline\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-20T08:25:45+02:00",
          "tree_id": "7014982c14ae29f4acf9a22f51114e0d8f11303c",
          "url": "https://github.com/xorq-labs/xorq/commit/77161655e221e37a3047f28fa0206718d25fa9ff"
        },
        "date": 1776666560511,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.807598650329558,
            "unit": "iter/sec",
            "range": "stddev: 0.010798160907110315",
            "extra": "mean: 128.08035412498953 msec\nrounds: 8"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.776738299442886,
            "unit": "iter/sec",
            "range": "stddev: 0.010735006339465362",
            "extra": "mean: 209.34787240000787 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.703822771038913,
            "unit": "iter/sec",
            "range": "stddev: 0.1730591272212348",
            "extra": "mean: 1.420812228800014 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.169578036888986,
            "unit": "iter/sec",
            "range": "stddev: 0.04461270734373754",
            "extra": "mean: 239.83242216666176 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.719267531861408,
            "unit": "iter/sec",
            "range": "stddev: 0.02331387036026454",
            "extra": "mean: 211.89728983335954 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.000503854935986,
            "unit": "iter/sec",
            "range": "stddev: 0.007553114409072334",
            "extra": "mean: 199.97984783331427 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "d2b1155f5c5575b1ddb24a255570093157c4dd8c",
          "message": "feat: remove xo.read_csv and xo.read_parquet from top-level API (#1834)\n\nRemoves `xo.read_csv` and `xo.read_parquet` to steer users toward the\npreferred `xo.deferred_read_csv` and `xo.deferred_read_parquet`. Updates\nall test callsites to use backend-level `con.read_*` methods instead.\n\nCloses #1830\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-20T09:37:10+02:00",
          "tree_id": "80b2a43fb35278f2cc4900ae0bfb615c17d52848",
          "url": "https://github.com/xorq-labs/xorq/commit/d2b1155f5c5575b1ddb24a255570093157c4dd8c"
        },
        "date": 1776670842638,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.881327118672488,
            "unit": "iter/sec",
            "range": "stddev: 0.012192242039686359",
            "extra": "mean: 126.8821843000012 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 5.017452103301853,
            "unit": "iter/sec",
            "range": "stddev: 0.0071565434275206044",
            "extra": "mean: 199.3043440000008 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7268325200733023,
            "unit": "iter/sec",
            "range": "stddev: 0.1715856441804281",
            "extra": "mean: 1.3758327708000024 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.476519809773792,
            "unit": "iter/sec",
            "range": "stddev: 0.042823191808162735",
            "extra": "mean: 223.38781966666468 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.157268810798332,
            "unit": "iter/sec",
            "range": "stddev: 0.004984411464786902",
            "extra": "mean: 193.901081499997 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.190112782832405,
            "unit": "iter/sec",
            "range": "stddev: 0.008915202193814073",
            "extra": "mean: 192.6740403999986 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "b423c838fc95464e6d18e23cce1c634bf279d03b",
          "message": "chore(deps): bump orjson from 3.11.4 to 3.11.6 (#1845)\n\nBumps [orjson](https://github.com/ijl/orjson) from 3.11.4 to 3.11.6.\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/ijl/orjson/releases\">orjson's\nreleases</a>.</em></p>\n<blockquote>\n<h2>3.11.6</h2>\n<h3>Changed</h3>\n<ul>\n<li>orjson now includes code licensed under the Mozilla Public License\n2.0 (MPL-2.0).</li>\n<li>Drop support for Python 3.9.</li>\n<li>ABI compatibility with CPython 3.15 alpha 5.</li>\n<li>Build now depends on Rust 1.89 or later instead of 1.85.</li>\n</ul>\n<h3>Fixed</h3>\n<ul>\n<li>Fix sporadic crash serializing deeply nested <code>list</code> of\n<code>dict</code>.</li>\n</ul>\n<h2>3.11.5</h2>\n<h3>Changed</h3>\n<ul>\n<li>Show simple error message instead of traceback when attempting to\nbuild on unsupported Python versions.</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/ijl/orjson/blob/master/CHANGELOG.md\">orjson's\nchangelog</a>.</em></p>\n<blockquote>\n<h2>3.11.6 - 2026-01-29</h2>\n<h3>Changed</h3>\n<ul>\n<li>orjson now includes code licensed under the Mozilla Public License\n2.0 (MPL-2.0).</li>\n<li>Drop support for Python 3.9.</li>\n<li>ABI compatibility with CPython 3.15 alpha 5.</li>\n<li>Build now depends on Rust 1.89 or later instead of 1.85.</li>\n</ul>\n<h3>Fixed</h3>\n<ul>\n<li>Fix sporadic crash serializing deeply nested <code>list</code> of\n<code>dict</code>.</li>\n</ul>\n<h2>3.11.5 - 2025-12-06</h2>\n<h3>Changed</h3>\n<ul>\n<li>Show simple error message instead of traceback when attempting to\nbuild on unsupported Python versions.</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/ec02024c3837255064f248c0d2d331319b75e9ad\"><code>ec02024</code></a>\n3.11.6</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/d58168733189f82b3fd0c058dff73e05d09202e6\"><code>d581687</code></a>\nbuild, clippy misc</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/4105b29b2275f200f6fae01349bef02ccf1bc2e2\"><code>4105b29</code></a>\nwriter::num</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/62bb185b70785ded49c79c26f8c9781f1e6fe370\"><code>62bb185</code></a>\nFix sporadic crash on serializing object close</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/d860078a973f44401265c5c4ad12a7dbe4f839ad\"><code>d860078</code></a>\nPyRef idiom refactors</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/343ae2f148197918aba9f8562db42c364620e4b8\"><code>343ae2f</code></a>\nDeserializer, Utf8Buffer</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/7835f58d1c56947d1cf7a18acdfc07a2bca9b0f2\"><code>7835f58</code></a>\nPyBytesRef and other input refactor</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/71e0516424ce1e11613eb1780f18e8cde83989fd\"><code>71e0516</code></a>\nPyStrRef</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/1096df42dc585fde837ed0c930a346f5ef7dbb94\"><code>1096df4</code></a>\nMSRV 1.89</li>\n<li><a\nhref=\"https://github.com/ijl/orjson/commit/b718e75b8ba18a707c2b44b6de14d52547573771\"><code>b718e75</code></a>\nDrop support for python3.9</li>\n<li>Additional commits viewable in <a\nhref=\"https://github.com/ijl/orjson/compare/3.11.4...3.11.6\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\n[![Dependabot compatibility\nscore](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=orjson&package-manager=uv&previous-version=3.11.4&new-version=3.11.6)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore this major version` will close this PR and stop\nDependabot creating any more for this major version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this minor version` will close this PR and stop\nDependabot creating any more for this minor version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this dependency` will close this PR and stop\nDependabot creating any more for this dependency (unless you reopen the\nPR or upgrade to it yourself)\nYou can disable automated security fix PRs for this repo from the\n[Security Alerts\npage](https://github.com/xorq-labs/xorq/network/alerts).\n\n</details>\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-20T12:03:44+02:00",
          "tree_id": "3c29a04ef1d1d644bcf4dae03b3a616b5ba395ec",
          "url": "https://github.com/xorq-labs/xorq/commit/b423c838fc95464e6d18e23cce1c634bf279d03b"
        },
        "date": 1776679644798,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.6133975402953435,
            "unit": "iter/sec",
            "range": "stddev: 0.012514478985561142",
            "extra": "mean: 131.34740366666935 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.706653499909937,
            "unit": "iter/sec",
            "range": "stddev: 0.019237705474999672",
            "extra": "mean: 212.4651836000112 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7013978948982559,
            "unit": "iter/sec",
            "range": "stddev: 0.1644845911781846",
            "extra": "mean: 1.4257242676000033 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.359510085754789,
            "unit": "iter/sec",
            "range": "stddev: 0.03550418364933445",
            "extra": "mean: 229.3835730000069 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.789334485438599,
            "unit": "iter/sec",
            "range": "stddev: 0.027518732640326303",
            "extra": "mean: 208.79727716666707 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.024182180141933,
            "unit": "iter/sec",
            "range": "stddev: 0.009519189662182598",
            "extra": "mean: 199.0373684999914 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "870ce12010a3f44d5ade2719f3019812f8049ddb",
          "message": "fix(ci): reduce false positives in benchmark (#1844)\n\nraise benchmark alert threshold and enable warmup to reduce false\npositives\n\nSubprocess-based CLI benchmarks have high natural variance (~10-20%) on\nshared GitHub Actions runners due to process startup time, OS scheduler\njitter, and filesystem cache state. The previous 130% threshold was too\ntight for this workload.\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-20T08:01:24-04:00",
          "tree_id": "4afd276d58f4188a217a1f6431792f9b28d7eac8",
          "url": "https://github.com/xorq-labs/xorq/commit/870ce12010a3f44d5ade2719f3019812f8049ddb"
        },
        "date": 1776686720160,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.594409108408444,
            "unit": "iter/sec",
            "range": "stddev: 0.004372068169799169",
            "extra": "mean: 94.38940763636663 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.901766900210148,
            "unit": "iter/sec",
            "range": "stddev: 0.00927791819298896",
            "extra": "mean: 204.00806900000248 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.7006035869026874,
            "unit": "iter/sec",
            "range": "stddev: 0.19906131653345163",
            "extra": "mean: 1.4273406797999997 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.107445126352397,
            "unit": "iter/sec",
            "range": "stddev: 0.03670584068529984",
            "extra": "mean: 243.46034316666496 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.000549339514941,
            "unit": "iter/sec",
            "range": "stddev: 0.015051157251708353",
            "extra": "mean: 199.97802883332838 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 3.686249474002165,
            "unit": "iter/sec",
            "range": "stddev: 0.0247052123373977",
            "extra": "mean: 271.2784381666656 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "28957a93f752c3092dc77816518e2c84bbd3036d",
          "message": "ref(packager): build wheel directly instead of sdist (#1790)\n\n## Summary\n\nWheel-based packager pipeline, organized as five reviewable commits:\n\n1. **`ref(packager): replace sdist pipeline with wheel-based build`**\n- Switch from `uv build --sdist` to `uv build --wheel`, eliminating the\nsdist→wheel conversion uv performs internally (~45% cold-build speedup).\n- Replace `SdistPackager`/`SdistArchive` with\n`WheelPackager`/`WheelBundle`; store `requirements.txt` as a sidecar\nnext to the wheel rather than embedding it.\n- Remove dead archive-manipulation helpers\n(`append_toplevel`/`replace_toplevel`/`tgz_to_zip`/`calc_zip_content_hexdigest`)\nand simplify subprocess plumbing.\n- Validate lockfile freshness via `uv export --locked` (not `--frozen`);\nbyte-exact sync-check of in-tree `requirements.txt` against `uv export`\noutput, with an actionable error message.\n   - Add otel tracing spans in the build path.\n- Harden stdout parsing in `PackagedBuilder._build` and surface `uv\nexport` stderr on failure.\n- Rewrite packager tests for the new API and fill coverage gaps for\n`_read_requires_python`, `find_file_upwards`, `_nix_env`,\n`WheelBundle.from_build_path`, `_write_requirements_path`,\n`_ensure_wheel_artifacts`, and `uv_export_requirements`.\n\n2. **`feat(cli): add --project-path, --extra, --all-extras to uv\nbuild`**\n   - Wire the new `WheelPackager` options through `xorq uv build`.\n- Simplify `uv_build_command`/`uv_run_command` onto the new\n`.build()`/`.run()` API.\n- CLI tests for `--project-path`, `--no-all-extras`, and `--extra`;\nremove stale `requirements.txt` from template tmpdirs so uv-version\ndrift doesn't break the sync check.\n\n3. **`feat(catalog): require wheel + requirements in build archives`**\n- Add `_ensure_wheel_artifacts` to inject wheel and `requirements.txt`\ninto build dirs before archiving.\n- Thread `project_path` through `Catalog.add`, `_add_build_dir`,\n`_add_expr`, `build_expr_context`, and `build_expr_context_zip` so\ncallers outside the project cwd (e.g. Jupyter kernels) can opt out of\nthe upward pyproject search.\n   - Validate archives contain exactly one `.whl` file.\n- Add `DumpFiles.requirements` to `REQUIRED_ARCHIVE_NAMES`; move the\ntest-only `_TEST_WHEEL_NAME` constant into the catalog test conftest.\n- Clear error when the upward pyproject search fails, naming the\n`project_path=` escape hatch.\n\n4. **`docs(adr): supersede ADR-0004 with ADR-0008 for wheel-based\npipeline`**\n- ADR-0004 documented the sdist pipeline, which has since been replaced\n— this is a different architectural decision, not a correction. Per the\nproject's ADR template, preserve ADR-0004 as a historical record and\nintroduce a new ADR for the current design of record.\n- Flip ADR-0004 status to \"Superseded by ADR-0008\" and add a pointer\nbanner; leave its original sdist content intact.\n- Add ADR-0008 covering the wheel pipeline end to end: `uv build\n--wheel`, `uv export --locked` with byte-exact sync-check, `uv tool run\n--isolated` with `--with <wheel>` + `--with-requirements`, the sidecar\narchive contract and the `_ensure_wheel_artifacts` ownership split\nbetween `Catalog._add_build_dir` and `build_expr_context_zip`,\n`--python` threading, and Nix LD_LIBRARY_PATH handling. Retains the\nRationale/Consequences sections since those uv-as-sole-runtime choices\ncarry over.\n\n5. **`ref(packager): consolidate _ensure_wheel_artifacts ownership`**\n- Drop the redundant `_ensure_wheel_artifacts` call from\n`build_expr_context`; the catalog-entry path is now owned solely by\n`Catalog._add_build_dir`, and the standalone-zip path (Flight exchange,\n`catalog.push`) by `build_expr_context_zip`. Gives each flow exactly one\ncheckpoint.\n- Default `uv_export_requirements(all_extras=True)` to match every\ncaller in the pipeline.\n- Tighten the stdout-parse comment in `PackagedBuilder._build` so it no\nlonger contradicts its own fallback.\n\n## Test plan\n\n- [ ] `python -m pytest python/xorq/ibis_yaml/tests/test_packager.py -x\n-q -m slow` — wheel build/run end-to-end\n- [ ] `python -m pytest python/xorq/tests/test_cli.py -x -q -k \"uv_build\nor uv_run\"` — CLI integration (includes\n`test_uv_build_with_project_path`, `--no-all-extras`, `--extra`)\n- [ ] `python -m pytest python/xorq/catalog/tests/test_catalog.py -x -q\n-m \"not slow\"` — catalog archive validation (missing wheel, missing\nrequired names, `project_path` threading)\n\n## Reviewer notes\n\n- Commits are path-separated, so file-by-file review maps cleanly onto\neach commit. Caveat: `test_packager.py` in commit 1 imports\n`_ensure_wheel_artifacts` from `catalog.catalog`, which only exists\nafter commit 3, so test collection is imperfect under `git bisect`; the\nfinal merged tree is unaffected.\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-20T18:28:52+02:00",
          "tree_id": "408740ab0c8e76594854c54fe61ad162c00e24b0",
          "url": "https://github.com/xorq-labs/xorq/commit/28957a93f752c3092dc77816518e2c84bbd3036d"
        },
        "date": 1776702781624,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.883485812835238,
            "unit": "iter/sec",
            "range": "stddev: 0.0303838162175024",
            "extra": "mean: 112.56842427272834 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.841135629045137,
            "unit": "iter/sec",
            "range": "stddev: 0.012994541839541112",
            "extra": "mean: 206.5631035000024 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6386740478486974,
            "unit": "iter/sec",
            "range": "stddev: 0.22003845465655106",
            "extra": "mean: 1.5657439085999953 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.0590711353319415,
            "unit": "iter/sec",
            "range": "stddev: 0.011440943946836696",
            "extra": "mean: 197.66474383333352 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.6404917620686676,
            "unit": "iter/sec",
            "range": "stddev: 0.03214664667493982",
            "extra": "mean: 274.6881644999964 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.029017430574621,
            "unit": "iter/sec",
            "range": "stddev: 0.008096731860757296",
            "extra": "mean: 198.84599999999182 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "49699333+dependabot[bot]@users.noreply.github.com",
            "name": "dependabot[bot]",
            "username": "dependabot[bot]"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "48571b9971730ab9e8ce4832eeb815f37d6e674e",
          "message": "chore(deps): bump pyasn1 from 0.6.2 to 0.6.3 (#1846)\n\nBumps [pyasn1](https://github.com/pyasn1/pyasn1) from 0.6.2 to 0.6.3.\n<details>\n<summary>Release notes</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/pyasn1/pyasn1/releases\">pyasn1's\nreleases</a>.</em></p>\n<blockquote>\n<h2>Release 0.6.3</h2>\n<p>It's a minor release.</p>\n<ul>\n<li>Added nesting depth limit to ASN.1 decoder to prevent stack overflow\nfrom deeply nested structures (CVE-2026-30922).</li>\n<li>Fixed OverflowError from oversized BER length field.</li>\n<li>Fixed DeprecationWarning stacklevel for deprecated attributes.</li>\n<li>Fixed asDateTime incorrect fractional seconds parsing.</li>\n</ul>\n<p>All changes are noted in the <a\nhref=\"https://github.com/pyasn1/pyasn1/blob/master/CHANGES.rst\">CHANGELOG</a>.</p>\n</blockquote>\n</details>\n<details>\n<summary>Changelog</summary>\n<p><em>Sourced from <a\nhref=\"https://github.com/pyasn1/pyasn1/blob/main/CHANGES.rst\">pyasn1's\nchangelog</a>.</em></p>\n<blockquote>\n<h2>Revision 0.6.3, released 16-03-2026</h2>\n<ul>\n<li>CVE-2026-30922 (GHSA-jr27-m4p2-rc6r): Added nesting depth\nlimit to ASN.1 decoder to prevent stack overflow from deeply\nnested structures (thanks for reporting, romanticpragmatism)</li>\n<li>Fixed OverflowError from oversized BER length field\n[issue <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/54\">#54</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/54\">pyasn1/pyasn1#54</a>)\n[pr <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/100\">#100</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/pull/100\">pyasn1/pyasn1#100</a>)</li>\n<li>Fixed DeprecationWarning stacklevel for deprecated attributes\n[issue <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/86\">#86</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/86\">pyasn1/pyasn1#86</a>)\n[pr <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/101\">#101</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/pull/101\">pyasn1/pyasn1#101</a>)</li>\n<li>Fixed asDateTime incorrect fractional seconds parsing\n[issue <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/81\">#81</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/81\">pyasn1/pyasn1#81</a>)\n[pr <a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/102\">#102</a>](<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/pull/102\">pyasn1/pyasn1#102</a>)</li>\n</ul>\n</blockquote>\n</details>\n<details>\n<summary>Commits</summary>\n<ul>\n<li><a\nhref=\"https://github.com/pyasn1/pyasn1/commit/af65c3b92e9deeae50db4de390982dd970d87f98\"><code>af65c3b</code></a>\nPrepare release 0.6.3</li>\n<li><a\nhref=\"https://github.com/pyasn1/pyasn1/commit/5a49bd1fe93b5b866a1210f6bf0a3924f21572c8\"><code>5a49bd1</code></a>\nMerge commit from fork</li>\n<li><a\nhref=\"https://github.com/pyasn1/pyasn1/commit/5494ba43f738e700ca9f7c7a69ec5c44908c9a9f\"><code>5494ba4</code></a>\nFix asDateTime incorrect fractional seconds parsing (<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/102\">#102</a>)</li>\n<li><a\nhref=\"https://github.com/pyasn1/pyasn1/commit/71f486e6c32d0f270868aa1b2bb5ceb7d5fd5476\"><code>71f486e</code></a>\nFix DeprecationWarning stacklevel for deprecated attributes (<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/101\">#101</a>)</li>\n<li><a\nhref=\"https://github.com/pyasn1/pyasn1/commit/d7cb42dcaa9a66e18f14c4609c2ed00c5b65f7e8\"><code>d7cb42d</code></a>\nFix OverflowError from oversized BER length field (<a\nhref=\"https://redirect.github.com/pyasn1/pyasn1/issues/100\">#100</a>)</li>\n<li>See full diff in <a\nhref=\"https://github.com/pyasn1/pyasn1/compare/v0.6.2...v0.6.3\">compare\nview</a></li>\n</ul>\n</details>\n<br />\n\n\n[![Dependabot compatibility\nscore](https://dependabot-badges.githubapp.com/badges/compatibility_score?dependency-name=pyasn1&package-manager=uv&previous-version=0.6.2&new-version=0.6.3)](https://docs.github.com/en/github/managing-security-vulnerabilities/about-dependabot-security-updates#about-compatibility-scores)\n\nDependabot will resolve any conflicts with this PR as long as you don't\nalter it yourself. You can also trigger a rebase manually by commenting\n`@dependabot rebase`.\n\n[//]: # (dependabot-automerge-start)\n[//]: # (dependabot-automerge-end)\n\n---\n\n<details>\n<summary>Dependabot commands and options</summary>\n<br />\n\nYou can trigger Dependabot actions by commenting on this PR:\n- `@dependabot rebase` will rebase this PR\n- `@dependabot recreate` will recreate this PR, overwriting any edits\nthat have been made to it\n- `@dependabot show <dependency name> ignore conditions` will show all\nof the ignore conditions of the specified dependency\n- `@dependabot ignore this major version` will close this PR and stop\nDependabot creating any more for this major version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this minor version` will close this PR and stop\nDependabot creating any more for this minor version (unless you reopen\nthe PR or upgrade to it yourself)\n- `@dependabot ignore this dependency` will close this PR and stop\nDependabot creating any more for this dependency (unless you reopen the\nPR or upgrade to it yourself)\nYou can disable automated security fix PRs for this repo from the\n[Security Alerts\npage](https://github.com/xorq-labs/xorq/network/alerts).\n\n</details>\n\nSigned-off-by: dependabot[bot] <support@github.com>\nCo-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
          "timestamp": "2026-04-21T11:28:36+02:00",
          "tree_id": "70c8bed6f04edee25b9eb6455d79820e524a1571",
          "url": "https://github.com/xorq-labs/xorq/commit/48571b9971730ab9e8ce4832eeb815f37d6e674e"
        },
        "date": 1776763962428,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 7.321487313290236,
            "unit": "iter/sec",
            "range": "stddev: 0.017833023990614493",
            "extra": "mean: 136.58426999999887 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 3.669733901209656,
            "unit": "iter/sec",
            "range": "stddev: 0.03437843988769731",
            "extra": "mean: 272.4993219999874 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6370223422282983,
            "unit": "iter/sec",
            "range": "stddev: 0.2529581611682898",
            "extra": "mean: 1.5698036531999946 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 3.806476313169006,
            "unit": "iter/sec",
            "range": "stddev: 0.017603325704130715",
            "extra": "mean: 262.71015966666295 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.05835371219672,
            "unit": "iter/sec",
            "range": "stddev: 0.011068280895574852",
            "extra": "mean: 197.69277850000813 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.061759835447815,
            "unit": "iter/sec",
            "range": "stddev: 0.059064305971377316",
            "extra": "mean: 246.19870216668005 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "10171e00f0038d0741c9471fef66e114f31b6e0d",
          "message": "docs: remove read_csv and read_parquet from API reference (#1848)\n\nFollows d2b1155f which removed xo.read_csv and xo.read_parquet from the\ntop-level API.\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-21T14:31:56+02:00",
          "tree_id": "a66c43f3f47a172af79c669805dc8f81e3bbe31c",
          "url": "https://github.com/xorq-labs/xorq/commit/10171e00f0038d0741c9471fef66e114f31b6e0d"
        },
        "date": 1776774963334,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.990453433512304,
            "unit": "iter/sec",
            "range": "stddev: 0.0033856147432950704",
            "extra": "mean: 100.09555688889628 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.666019466125896,
            "unit": "iter/sec",
            "range": "stddev: 0.008098767672063915",
            "extra": "mean: 214.31543679998413 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6126097138449934,
            "unit": "iter/sec",
            "range": "stddev: 0.18118098445192032",
            "extra": "mean: 1.632360665200008 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.92318618933169,
            "unit": "iter/sec",
            "range": "stddev: 0.00965070515939465",
            "extra": "mean: 203.12049179999576 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.9755949727344175,
            "unit": "iter/sec",
            "range": "stddev: 0.04645953401874853",
            "extra": "mean: 251.53467766667364 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.6139811331841525,
            "unit": "iter/sec",
            "range": "stddev: 0.015522461402541746",
            "extra": "mean: 216.7325723999852 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "9db3be53ec73bd87cbb3802f031862cb40f0a73c",
          "message": "refactor: rename xorq backend to xorq_datafusion (#1851)\n\nRenames the backend from `xorq`/`xorq-datafusion` to `xorq_datafusion`\nfor a consistent, valid Python identifier across directory name, entry\npoint, Backend.name, pytest marker, and all string references.\n\nAlso removes leftover \"let\" references (the backend's previous name):\nstale FIXME comment, error message, docs import, and pytest marker\nentry.\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-21T11:53:06-04:00",
          "tree_id": "8fcb75f89fc59e0ac700e84af9014c02321a8c17",
          "url": "https://github.com/xorq-labs/xorq/commit/9db3be53ec73bd87cbb3802f031862cb40f0a73c"
        },
        "date": 1776787021222,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.402404947812432,
            "unit": "iter/sec",
            "range": "stddev: 0.005430911751763806",
            "extra": "mean: 96.13161619999175 msec\nrounds: 10"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.836329067760649,
            "unit": "iter/sec",
            "range": "stddev: 0.008017438392219415",
            "extra": "mean: 206.76839520000385 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6524504670243647,
            "unit": "iter/sec",
            "range": "stddev: 0.18974099087275728",
            "extra": "mean: 1.5326833997999985 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.08680396981503,
            "unit": "iter/sec",
            "range": "stddev: 0.014507026178058656",
            "extra": "mean: 196.58709200000146 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.9121664258288273,
            "unit": "iter/sec",
            "range": "stddev: 0.03493164462011153",
            "extra": "mean: 255.61284749999896 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.145463667744138,
            "unit": "iter/sec",
            "range": "stddev: 0.006460927396434137",
            "extra": "mean: 194.34594520000132 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "mesejoleon@gmail.com",
            "name": "Daniel Mesejo",
            "username": "mesejo"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "6bdddbbaf36fc21521b6a431740f3c61a932e5da",
          "message": "fix(cache): invalidate on file changes for DuckDB and DataFusion (#1833)\n\n## Summary\n\n- **DataFusion path extraction**: Replace fragile regex with\n`yaml12.parse_yaml` on the `file_groups=` section of execution plans,\ncorrectly handling multi-group, multi-file, URL, and\nwhitespace-containing paths. Apply `_canonicalize_plan_path` to strip\n`xorq-catalog-*` tempdir prefixes for stable catalog-zip tokens.\n- **DuckDB path extraction**: Parse DDL via `sqlglot` AST\n(`ReadParquet`/`ReadCSV` nodes) instead of hashing the raw DDL string.\nFalls back to DDL-string-based token for tables without\n`read_parquet`/`read_csv` (e.g. plain `CREATE TABLE`).\n- **File-metadata-based cache keys**: Both DataFusion and DuckDB\nfile-backed tables now include per-file stat metadata (mtime, size,\ninode) in the cache token via `_normalize_path_stat`, so overwriting a\nbacking file invalidates the cache.\n- **`_normalize_path_stat` consolidation**: Extract shared HTTP HEAD /\nS3 / local stat logic from `normalize_read` into a reusable helper, used\nby all three code paths. Includes `timeout=10` on HTTP requests,\n`FileNotFoundError` for missing local paths, and a descriptive\n`User-Agent` header.\n- **Scheme prefix bugfix**: Fix overly broad `startswith(\"http\")` /\n`startswith(\"s3\")` checks in `normalize_read` to use full scheme\nprefixes (`\"http://\"`, `\"s3://\"`, etc.).\n- **Known limitation**: DataFusion strips scheme+host from HTTP URLs in\nexecution plans, so HTTP-backed CSV tables on the xorq/DataFusion\nbackend cannot be tokenized (documented as `xfail(strict=True)`).\n\n## Test plan\n\n- [x] Parametrized `test_extract_plan_file_paths` — unit tests for\nDataFusion plan parsing (single/multi group, absolute paths, URLs, no\nfile_groups)\n- [x] Parametrized `test_extract_duckdb_file_paths` — unit tests for\nDuckDB DDL parsing (parquet, csv, multi-path, plain CREATE TABLE, URLs)\n- [x] `test_parquet_invalidates_on_file_change` — token changes when\nbacking parquet is overwritten (datafusion, duckdb, xorq)\n- [x] `test_token_stable_for_same_file` — same file produces same token\nacross connections (parquet+csv × all backends)\n- [x] `test_parquet_different_files_produce_different_tokens` /\n`test_parquet_same_content_different_path_produces_different_token`\n- [x] `test_duckdb_multi_path_cache_key_invalidates_on_file_change` /\n`test_xorq_multi_csv_path_cache_key_invalidates_on_file_change`\n- [x] `test_parquet_cache_storage` — asserts cache populated after first\nexecute, invalidated after file change, then re-populated with correct\ndata\n- [x] `test_http_csv_token_is_stable` — stable for DuckDB, xfail for\nxorq/DataFusion\n- [x] Snapshot test updated (`datafusion_key.txt`) with mocked\n`_normalize_path_stat` for machine-stable hashes\n\nCloses #1821\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Sonnet 4.6 <noreply@anthropic.com>\nCo-authored-by: dlovell <dlovell@gmail.com>",
          "timestamp": "2026-04-21T13:33:09-04:00",
          "tree_id": "dae275882d82e5215117a954ebaefed8e0e01947",
          "url": "https://github.com/xorq-labs/xorq/commit/6bdddbbaf36fc21521b6a431740f3c61a932e5da"
        },
        "date": 1776793023424,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 10.036275325413301,
            "unit": "iter/sec",
            "range": "stddev: 0.005939750810575565",
            "extra": "mean: 99.63855788888685 msec\nrounds: 9"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.726156130437488,
            "unit": "iter/sec",
            "range": "stddev: 0.010842889275201488",
            "extra": "mean: 211.5884393999977 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.591412109279543,
            "unit": "iter/sec",
            "range": "stddev: 0.21128331669056521",
            "extra": "mean: 1.6908683205999921 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.08298213332397,
            "unit": "iter/sec",
            "range": "stddev: 0.007948594606746323",
            "extra": "mean: 196.73490359999732 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 4.144476889880848,
            "unit": "iter/sec",
            "range": "stddev: 0.0476473341588515",
            "extra": "mean: 241.28497433333487 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.993079132958348,
            "unit": "iter/sec",
            "range": "stddev: 0.007066570462230169",
            "extra": "mean: 200.27721840000368 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "3d7e9aeb147bd98888f8d702be3634697ec66b9d",
          "message": "fix(catalog): deterministic catalog entry zip bytes (#1852)\n\n## Summary\n\nTwo sources of non-determinism made `catalog.add(expr)` produce a\ndifferent zip on every run for the same expression, plus a downstream\ncrash when UDF closures contained Read nodes with the newly-relative\npaths. After this change, two runs of a catalog addition produce\nbyte-identical zips and normalize correctly.\n\n### 1. `expr.yaml` leaked the build tempdir prefix\n\n`read_kwargs.hash_path` was serialized as the absolute build-time path,\ne.g. `/tmp/tmpXXXX/<hash>/database_tables/...`. The existing rewrite in\n`Registry.register_node` only fired for `InMemoryTable` memtables (gated\non `\"memtables\" in read_kwargs`), so `database_table` Reads leaked the\ntempdir prefix.\n\nFix: gate on `\"read_path\" in read_kwargs` (the builder-materialized\nmarker introduced by ADR-0006) and set `hash_path = Path(read_path)`.\nLossless because `ExprLoader.deferred_reads_to_memtables` overwrites\n`hash_path` with `expr_path.joinpath(read_path)` at load time whenever\n`read_path` is present.\n\n### 2. Zip member mtimes came from the filesystem\n\n`make_zip_context` used `zf.write(p, arcname=...)`, which captures each\nmember's mtime and permissions. Switched to `zf.writestr(ZipInfo(...,\ndate_time=(1980,1,1,0,0,0)), bytes)` with `external_attr=0o644<<16` and\nsorted `rglob` output for stable ordering.\n\n### 3. `normalize_read` crashed on relative build-relative paths\n\nThe deterministic YAML serialization rewrites `hash_path` to a relative\nbuild-relative path for all Read nodes. When a UDF's pickled closure\ncontains a Read, `deferred_reads_to_memtables` can't resolve it, so\n`normalize_read` encounters the relative path during tokenization and\nfails. Fixed by using the path string directly as the normalization\ntoken — the filename already embeds a content hash, so this is safe and\ndeterministic.\n\n## Test plan\n\n- [x] `python/xorq/ibis_yaml/tests/` — 435 passed (5 pre-existing\npostgres-env failures unrelated)\n- [x] `python/xorq/catalog/tests/test_catalog.py` +\n`test_git_backend.py` — 117 passed\n- [x] Run `scripts/2026-04-17-paddy-issue.py` twice; assert resulting\nentry zip md5s match\n- [x] Verify `expr.yaml` `hash_path` is now\n`database_tables/<hash>.parquet` (relative)\n- [x] Verify zip member mtimes are `(1980, 1, 1, 0, 0, 0)`\n- [x] UDF expressions with Read nodes in their closures normalize\nwithout crashing\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\n---------\n\nCo-authored-by: Claude Opus 4.7 (1M context) <noreply@anthropic.com>",
          "timestamp": "2026-04-21T21:30:07+02:00",
          "tree_id": "1e57f934602cc7da35d7eaa40f83eea182e627de",
          "url": "https://github.com/xorq-labs/xorq/commit/3d7e9aeb147bd98888f8d702be3634697ec66b9d"
        },
        "date": 1776800054059,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 8.037092123167467,
            "unit": "iter/sec",
            "range": "stddev: 0.022813328030573655",
            "extra": "mean: 124.4231103333296 msec\nrounds: 12"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.242345913720465,
            "unit": "iter/sec",
            "range": "stddev: 0.059602180170504775",
            "extra": "mean: 235.71863783333433 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6625471104835132,
            "unit": "iter/sec",
            "range": "stddev: 0.2007170881891149",
            "extra": "mean: 1.5093266338000035 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.208389847683788,
            "unit": "iter/sec",
            "range": "stddev: 0.05491362605182098",
            "extra": "mean: 237.62057133332823 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 5.147328679047843,
            "unit": "iter/sec",
            "range": "stddev: 0.004674979256353038",
            "extra": "mean: 194.27552860000787 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.1292577848044685,
            "unit": "iter/sec",
            "range": "stddev: 0.05005053712915977",
            "extra": "mean: 242.17427250000392 msec\nrounds: 6"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "dlovell@gmail.com",
            "name": "Dan Lovell",
            "username": "dlovell"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "833c3a56674f0078c094f77f7b6557beb938f9e7",
          "message": "fix(flight): reduce test flakiness from CI resource contention (#1855)\n\n## Summary\n\n- Serialize all Flight server tests under `xdist_group(\"serve\")` so they\ndon't compete for CPU when pytest-xdist fans out across workers — the\ndirect cause of `test_serve_unbound_tag_get_exchange_udf` timing out in\nCI\n- Move port-announcement log lines in `unbind_and_serve_command` and\n`serve_command` to after the gRPC server is actually listening, so\nhealthcheck retries aren't burned during server startup\n- Fix `_wait_on_healthcheck` sleeping 1s unconditionally in a `finally`\nblock (including after successful healthcheck)\n- Explicitly close the reserved socket in `FlightUrl.unbind_socket`\ninstead of relying on refcount/GC\n- Double the default healthcheck budget from 20 to 40 attempts\n\n## Test plan\n\n- [ ] CI `ci-test` passes (specifically\n`test_serve_unbound_tag_get_exchange_udf` on split 1)\n- [ ] `test_serve_command` still passes (affected by `serve_command`\nblock→wait change)\n- [ ] No regressions in other serve tests (`test_serve_unbound_hash`,\n`test_serve_unbound_tag`, `test_serve_penguins_template`)\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)\n\nCo-authored-by: Claude Opus 4.6 <noreply@anthropic.com>",
          "timestamp": "2026-04-21T21:53:48+02:00",
          "tree_id": "5492c206cfff39bfe038ca7c8a0349c0fdc416e1",
          "url": "https://github.com/xorq-labs/xorq/commit/833c3a56674f0078c094f77f7b6557beb938f9e7"
        },
        "date": 1776801538867,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 12.49558476922511,
            "unit": "iter/sec",
            "range": "stddev: 0.0043372850187969126",
            "extra": "mean: 80.0282674615486 msec\nrounds: 13"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 6.139992152640216,
            "unit": "iter/sec",
            "range": "stddev: 0.002314145204462755",
            "extra": "mean: 162.86665766665465 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.8277343833162019,
            "unit": "iter/sec",
            "range": "stddev: 0.12521948530100602",
            "extra": "mean: 1.2081170242000099 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 5.3496186935093,
            "unit": "iter/sec",
            "range": "stddev: 0.0213857329897478",
            "extra": "mean: 186.9292107142704 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 6.36527424128428,
            "unit": "iter/sec",
            "range": "stddev: 0.007116510539947546",
            "extra": "mean: 157.10242200000428 msec\nrounds: 7"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 5.019859460429033,
            "unit": "iter/sec",
            "range": "stddev: 0.033485593150147056",
            "extra": "mean: 199.20876428571026 msec\nrounds: 7"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hussainz@gmail.com",
            "name": "Hussain Sultan",
            "username": "hussainsultan"
          },
          "committer": {
            "email": "noreply@github.com",
            "name": "GitHub",
            "username": "web-flow"
          },
          "distinct": true,
          "id": "646f882b5e27eae9bf744b639b5fe9fd687dfc5d",
          "message": "release: 0.3.20 (#1856)\n\n## Summary\nRelease v0.3.20.\n\nSee `CHANGELOG.md` for the full list of changes since v0.3.19.\n\n## Test plan\n- [x] Trigger the [ci-pre-release\nworkflow](https://github.com/xorq-labs/xorq/actions/workflows/ci-pre-release.yml)\nagainst branch `release-0.3.20`\n- [x] All ci-pre-release checks pass\n- [ ] Squash-merge this PR\n- [ ] Tag `v0.3.20` on `origin/main` and push\n- [ ] Create GitHub release to trigger publishing\n\n🤖 Generated with [Claude Code](https://claude.com/claude-code)",
          "timestamp": "2026-04-21T16:44:11-04:00",
          "tree_id": "551acfe283003d19c93f89f01378ba0006ce79d5",
          "url": "https://github.com/xorq-labs/xorq/commit/646f882b5e27eae9bf744b639b5fe9fd687dfc5d"
        },
        "date": 1776804497132,
        "tool": "pytest",
        "benches": [
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_help",
            "value": 9.838633644833303,
            "unit": "iter/sec",
            "range": "stddev: 0.007825179114903517",
            "extra": "mean: 101.64012972727608 msec\nrounds: 11"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_init",
            "value": 4.7294098203063095,
            "unit": "iter/sec",
            "range": "stddev: 0.01455654215181987",
            "extra": "mean: 211.44287300000428 msec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_add",
            "value": 0.6067366229600133,
            "unit": "iter/sec",
            "range": "stddev: 0.10996194453476416",
            "extra": "mean: 1.648161594600009 sec\nrounds: 5"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_list",
            "value": 4.839458353334928,
            "unit": "iter/sec",
            "range": "stddev: 0.009105143632617201",
            "extra": "mean: 206.63469483333566 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_info",
            "value": 3.957305031743087,
            "unit": "iter/sec",
            "range": "stddev: 0.049956722237160085",
            "extra": "mean: 252.6972249999962 msec\nrounds: 6"
          },
          {
            "name": "python/xorq/catalog/tests/test_benchmark_cli.py::test_benchmark_catalog_check",
            "value": 4.626043084126652,
            "unit": "iter/sec",
            "range": "stddev: 0.026620136869060926",
            "extra": "mean: 216.16746360000434 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}