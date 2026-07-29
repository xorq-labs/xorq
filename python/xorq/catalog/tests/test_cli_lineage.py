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
from xorq.common.utils.lineage_utils import LineageDAG


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
        if line and not line.startswith("#")
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


def _lineage_dag(catalog_path: str, name: str) -> LineageDAG:
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    return catalog.get_catalog_entry(name).metadata.lineage


def test_lineage_node_by_kind_prints_the_subtree_feeding_it(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """`--node` scopes the compact tree to what feeds that node -- including the
    UDXF's nested input lineage, which lives in its own scope."""
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--node", "flight_udxf"]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].startswith("UDXF[OrderEnricher] : 2→3 cols")
    assert "↳ " in lines[1]
    # scoped: the join and the duckdb side of the graph are not in this subtree
    assert "Join(" not in result.output
    assert "raw_customers" not in result.output


def test_lineage_node_by_hash_and_by_label_agree(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """The content hash is the durable handle; the @label is the doc-local one."""
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)
    [udxf] = dag.boundaries(kind="flight_udxf")

    by_hash = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--node", udxf["snapshot_hash"]]
    )
    by_label = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--node", udxf["id"]]
    )

    assert by_hash.exit_code == 0, by_hash.output
    assert by_hash.output == by_label.output
    assert "UDXF[OrderEnricher]" in by_hash.output


def test_lineage_node_by_tag_value(runner: CliRunner, catalog_path: str) -> None:
    """A tag resolves by its *value*, not just by the `tag:<value>` kind."""
    catalog = Catalog.from_kwargs(path=catalog_path, init=False)
    tagged = xo.memtable({"x": [1, 2]}, name="sales").tag(tag="bsl", name="flights")
    entry = catalog.add(tagged)

    by_value = runner.invoke(
        cli, ["--path", catalog_path, "lineage", entry.name, "--node", "bsl"]
    )
    by_kind = runner.invoke(
        cli, ["--path", catalog_path, "lineage", entry.name, "--node", "tag"]
    )

    assert by_value.exit_code == 0, by_value.output
    assert "Tag[bsl]" in by_value.output
    assert by_value.output == by_kind.output


def test_lineage_node_prints_every_match(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """A kind handle resolves to several nodes; each gets its own subtree."""
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)
    tables = dag.boundaries(kind="table")
    assert len(tables) > 1

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--node", "table"]
    )

    assert result.exit_code == 0, result.output
    blocks = [b for b in result.output.split("\n\n") if b.strip()]
    assert len(blocks) == len(tables)
    assert "raw_customers" in result.output
    assert "raw_orders" in result.output


def test_lineage_node_with_raw_prints_only_the_matched_nodes(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli,
        ["--path", catalog_path, "lineage", name, "--node", "flight_udxf", "-l", "raw"],
    )

    assert result.exit_code == 0, result.output
    [node] = json.loads(result.output)
    assert node["boundary_kind"] == "flight_udxf"
    assert node["udxf_class"] == "OrderEnricher"
    # the whole DAG's keys are absent: this is a node list, not the document
    assert "nodes" not in node and "edges" not in node


def test_lineage_node_with_boundaries_reports_capabilities_and_scope(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "lineage",
            name,
            "--node",
            "flight_udxf",
            "-l",
            "boundaries",
        ],
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0].split("\t")[1] == "flight_udxf"
    assert any(line.startswith("# capabilities\t") for line in lines)
    assert any(line.startswith("# nested\t") for line in lines)
    # capability lines are for the expansion case only
    listing = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-l", "boundaries"]
    )
    assert "# capabilities" not in listing.output


def test_lineage_node_of_an_unknown_handle_points_at_boundaries(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--node", "no-such-node"]
    )

    assert result.exit_code != 0
    assert "No lineage node matches 'no-such-node'" in result.output
    assert "--level boundaries" in result.output


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
