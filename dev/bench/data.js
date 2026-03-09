window.BENCHMARK_DATA = {
  "lastUpdate": 1773064552731,
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
      }
    ]
  }
}