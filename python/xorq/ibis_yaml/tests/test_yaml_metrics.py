"""Characterization + size/perf metrics harness for ibis_yaml serialization.

Run on demand:

    pytest python/xorq/ibis_yaml/tests/test_yaml_metrics.py -m metrics -s

It serializes a fixed set of exprs (the TPC-H suite + a few simple shapes),
records emitted-YAML byte/line counts and ``definitions`` entry counts, asserts
round-trip equality, and writes a JSON report next to this file
(``yaml_metrics_report.json``). If a baseline report exists
(``yaml_metrics_baseline.json``) it prints a per-expr delta table so size wins
are quantified phase-over-phase.

This module is the gate referenced by the ibis-yaml node.map refactor plan:
round-trip is the real correctness contract; bytes/lines/defs are the win
metric.
"""

import json
import time
from pathlib import Path

import pytest
import yaml12

import xorq.vendor.ibis as ibis
from xorq.ibis_yaml.common import translate_from_yaml, translate_to_yaml
from xorq.ibis_yaml.compiler import YamlExpressionTranslator, _to_yaml_safe
from xorq.ibis_yaml.tests.test_tpch import TPC_H


HERE = Path(__file__).parent
REPORT_PATH = HERE / "yaml_metrics_report.json"
BASELINE_PATH = HERE / "yaml_metrics_baseline.json"

pytestmark = pytest.mark.metrics


def _simple_exprs():
    t = ibis.table(
        {
            "a": "int64",
            "b": "string",
            "c": "float64",
            "d": "timestamp",
            "e": "date",
        },
        name="test_table",
    )
    yield "literal", ibis.literal(1)
    yield "filter_project", t.filter(t.a > 1).select(t.a, t.b, doubled=t.c * 2.0)
    yield "aggregate", t.group_by("b").aggregate(n=t.a.count(), mean_c=t.c.mean())
    wide = t.select(*[t.a.name(f"col_{i}") for i in range(12)])
    yield "wide_project", wide


def _yaml_bytes(yaml_dict, tmp_path):
    # write through the real serializer (incl. document markers) so byte counts
    # match the on-disk expr.yaml artifact exactly
    out = tmp_path / "expr.yaml"
    yaml12.write_yaml(_to_yaml_safe(yaml_dict), out)
    return out.read_bytes()


def _measure(name, expr, con, tmp_path):
    compiler = YamlExpressionTranslator
    profiles = {con._profile.hash_name: con} if con is not None else {}

    # perf: median of a few to_yaml calls (cold lru_cache cleared each round)
    timings = []
    yaml_dict = None
    for _ in range(5):
        translate_to_yaml.cache_clear()
        translate_from_yaml.cache_clear()
        t0 = time.perf_counter()
        yaml_dict = compiler.to_yaml(expr, profiles)
        timings.append(time.perf_counter() - t0)
    timings.sort()
    to_yaml_median_ms = timings[len(timings) // 2] * 1e3

    raw = _yaml_bytes(yaml_dict, tmp_path)
    defs = yaml_dict.get("definitions", {})

    roundtrip_ok = None
    try:
        rt = compiler.from_yaml(yaml_dict, profiles)
        roundtrip_ok = bool(rt.equals(expr))
    except Exception as e:  # record, don't crash the harness
        roundtrip_ok = f"ERROR: {type(e).__name__}: {e}"

    return {
        "bytes": len(raw),
        "lines": raw.count(b"\n") + 1,
        "n_dtypes": len(defs.get("dtypes", {})),
        "n_nodes": len(defs.get("nodes", {})),
        "n_schemas": len(defs.get("schemas", {})),
        "to_yaml_median_ms": round(to_yaml_median_ms, 3),
        "roundtrip_ok": roundtrip_ok,
    }


def test_yaml_metrics(request, con, tmp_path):
    report = {}

    for name, expr in _simple_exprs():
        report[name] = _measure(name, expr, None, tmp_path)

    for fixture_name in TPC_H:
        expr = request.getfixturevalue(fixture_name)
        report[fixture_name] = _measure(fixture_name, expr, con, tmp_path)

    REPORT_PATH.write_text(json.dumps(report, indent=2, sort_keys=True))

    # ---- print summary + delta vs baseline -------------------------------
    totals = {k: sum(r[k] for r in report.values()) for k in ("bytes", "lines")}
    baseline = json.loads(BASELINE_PATH.read_text()) if BASELINE_PATH.exists() else {}

    print("\n=== ibis_yaml metrics ===")
    header = f"{'expr':<20}{'bytes':>9}{'Δbytes':>9}{'lines':>7}{'dtypes':>7}{'nodes':>6}{'ms':>8}{'rt':>5}"
    print(header)
    print("-" * len(header))
    for name in sorted(report):
        r = report[name]
        b0 = baseline.get(name, {}).get("bytes")
        delta = f"{r['bytes'] - b0:+d}" if b0 is not None else "-"
        rt = "ok" if r["roundtrip_ok"] is True else "FAIL"
        print(
            f"{name:<20}{r['bytes']:>9}{delta:>9}{r['lines']:>7}"
            f"{r['n_dtypes']:>7}{r['n_nodes']:>6}{r['to_yaml_median_ms']:>8}{rt:>5}"
        )
    print("-" * len(header))
    base_total = sum(v.get("bytes", 0) for v in baseline.values()) if baseline else None
    dtot = f"{totals['bytes'] - base_total:+d}" if base_total else "-"
    print(f"{'TOTAL':<20}{totals['bytes']:>9}{dtot:>9}{totals['lines']:>7}")
    print(f"report written: {REPORT_PATH}")

    # correctness gate: every expr must round-trip
    failures = {
        k: v["roundtrip_ok"] for k, v in report.items() if v["roundtrip_ok"] is not True
    }
    assert not failures, f"round-trip failures: {failures}"
