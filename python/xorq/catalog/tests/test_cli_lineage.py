"""`xorq catalog lineage` -- the three detail levels over a stored sidecar.

The command never re-walks the expression: every level is read or derived from
the `LineageDAG` in `expr_metadata.json`, so these tests build entries once and
assert on what the sidecar carries.
"""

from __future__ import annotations

import json

import pytest
import yaml12
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli


@pytest.fixture
def catalog_with_udxf(catalog_path: str) -> tuple[str, str]:
    """A catalog holding one duckdb -> xorq join over a FlightUDXF.

    Construction and build only -- no Flight server is started.
    """
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    duck = xo.duckdb.connect()
    con = xo.connect()

    customers = duck.create_table(
        "raw_customers",
        xo.memtable({"id": [1, 2], "name": ["a", "b"]}, name="c_src").to_pyarrow(),
    )
    orders = con.create_table(
        "raw_orders",
        xo.memtable({"id": [1, 2], "amount": [1.0, 2.0]}, name="o_src").to_pyarrow(),
    )
    enriched = xo.expr.relations.flight_udxf(
        orders,
        process_df=lambda df: df.assign(score=df.amount * 2),
        maybe_schema_in=orders.schema(),
        maybe_schema_out=xo.schema(orders.schema() | {"score": "float64"}),
        con=con,
        make_udxf_kwargs={"name": "OrderEnricher"},
    )
    expr = customers.into_backend(con, "customers_in").join(enriched, "id")

    entry = catalog.add(expr, aliases=("udxf-demo",))
    return catalog_path, entry.name


def test_lineage_compact_is_the_default_level(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(cli, ["--path", catalog_path, "lineage", name])

    assert result.exit_code == 0, result.output
    assert "UDXF[OrderEnricher] : 2→3 cols" in result.output
    assert "(+score)" in result.output
    # a tree, not a flat chain, with the nested input under the UDXF
    assert "└── " in result.output
    assert "↳ " in result.output
    # and the same output as asking for it explicitly
    explicit = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--level", "compact"]
    )
    assert explicit.output == result.output


def test_lineage_accepts_an_alias(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    by_alias = runner.invoke(cli, ["--path", catalog_path, "lineage", "udxf-demo"])
    by_name = runner.invoke(cli, ["--path", catalog_path, "lineage", name])

    assert by_alias.exit_code == 0, by_alias.output
    assert by_alias.output == by_name.output


def test_lineage_boundaries_is_one_tab_separated_line_per_boundary(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-l", "boundaries"]
    )

    assert result.exit_code == 0, result.output
    rows = [
        line.split("\t")
        for line in result.output.splitlines()
        if line and not line.startswith("# nested")
    ]
    assert rows
    assert all(len(row) == 3 for row in rows), rows

    kinds = {kind for _, kind, _ in rows}
    assert {"join", "engine_crossing", "table", "flight_udxf"} <= kinds
    assert all(node_id.startswith("@") for node_id, _, _ in rows)
    # no tree glyphs: this level is for grep and cut
    assert "└──" not in result.output

    # the Flight boundary reports its nested scope on a comment line
    [nested] = [
        line for line in result.output.splitlines() if line.startswith("# nested")
    ]
    assert "root=@" in nested
    assert "node" in nested


def test_lineage_raw_is_the_stored_dag_as_json(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(cli, ["--path", catalog_path, "lineage", name, "-l", "raw"])

    assert result.exit_code == 0, result.output
    raw = json.loads(result.output)

    assert raw["version"] == 2
    assert raw["root"].startswith("@")
    # every node, not just the boundaries the compact view keeps
    assert len(raw["nodes"]) > len([n for n in raw["nodes"] if n.get("is_boundary")])
    assert set(raw["edges"][0]) == {"from", "to", "scope"}

    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    stored = catalog.get_catalog_entry(name).metadata.lineage
    assert raw == json.loads(json.dumps(stored.to_dict(), default=str))


def test_lineage_rejects_an_unknown_level(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-l", "verbose"]
    )

    assert result.exit_code != 0
    assert "'compact', 'boundaries', 'raw'" in result.output


def test_lineage_of_an_unknown_entry_points_at_list(
    runner: CliRunner, catalog_path: str
) -> None:
    result = runner.invoke(cli, ["--path", catalog_path, "lineage", "no-such-entry"])

    assert result.exit_code != 0
    assert "not found" in result.output
    assert "xorq catalog list" in result.output


def test_lineage_of_a_sidecar_without_lineage_explains_itself(
    runner: CliRunner, catalog_path: str
) -> None:
    """A sidecar written before lineage existed has `lineage: None`; the command
    says how to get one rather than printing an empty tree."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    entry = catalog.add(xo.memtable({"a": [1]}, name="no_lineage"))
    sidecar = yaml12.parse_yaml(entry.metadata_path.read_text())
    sidecar["expr_metadata"]["lineage"] = None
    entry.metadata_path.write_text(yaml12.format_yaml(sidecar))

    result = runner.invoke(cli, ["--path", catalog_path, "lineage", entry.name])

    assert result.exit_code != 0
    assert "no lineage in its metadata sidecar" in result.output
