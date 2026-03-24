window.BENCHMARK_DATA = {
  "lastUpdate": 1774367289404,
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
      }
    ]
  }
}