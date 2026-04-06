window.BENCHMARK_DATA = {
  "lastUpdate": 1775477330964,
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
      }
    ]
  }
}