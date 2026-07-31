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
from xorq.common.utils.lineage_utils import LineageDAG, _mermaid_id


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


def test_lineage_format_defaults_to_text(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    default = runner.invoke(cli, ["--path", catalog_path, "lineage", name])
    explicit = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--format", "text"]
    )

    assert default.exit_code == 0, default.output
    assert default.output == explicit.output
    assert "flowchart" not in default.output


def test_lineage_format_mermaid_emits_a_flowchart(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-f", "mermaid"]
    )

    assert result.exit_code == 0, result.output
    lines = result.output.splitlines()
    assert lines[0] == "flowchart TD"
    assert any("UDXF[OrderEnricher]" in line for line in lines)
    assert any(line.strip().startswith("subgraph ") for line in lines)
    assert any("-->" in line for line in lines)
    assert any(line.strip().startswith("classDef ") for line in lines)
    # tree glyphs belong to the text renderer only
    assert "└──" not in result.output


def test_lineage_format_mermaid_honours_node(
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
            "-f",
            "mermaid",
        ],
    )

    assert result.exit_code == 0, result.output
    assert result.output.splitlines()[0] == "flowchart TD"
    assert "UDXF[OrderEnricher]" in result.output
    assert "Join(" not in result.output


def test_lineage_format_mermaid_rejects_the_boundaries_level(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """`boundaries` is a flat listing: keeping the edges would just re-spell the
    compact graph, dropping them would draw node soup."""
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli,
        ["--path", catalog_path, "lineage", name, "-f", "mermaid", "-l", "boundaries"],
    )

    assert result.exit_code != 0, result.output
    assert "'boundaries' is a flat listing" in result.output


def test_lineage_raw_mermaid_expands_the_collapsed_runs(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """`raw` is the stored graph, so mermaid draws it un-collapsed: the ops that
    the compact view folds into `via` become real nodes."""
    catalog_path, name = catalog_with_udxf

    compact = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-f", "mermaid"]
    )
    expanded = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-f", "mermaid", "-l", "raw"]
    )

    assert expanded.exit_code == 0, expanded.output
    assert expanded.output.splitlines()[0] == "flowchart TD"
    # not JSON: --format wins over the level's own listing format
    assert not expanded.output.lstrip().startswith("{")

    def declared(output: str) -> set[str]:
        return {
            line.strip().split("[", 1)[0]
            for line in output.splitlines()
            if "[" in line and not line.strip().startswith(("subgraph", "class"))
        }

    assert declared(compact.output) < declared(expanded.output)
    # the plumbing the compact view hides behind `via`
    assert "Field:" in expanded.output
    assert "|via " in compact.output
    assert "|via " not in expanded.output
    # and it carries a receding class of its own
    assert "classDef unknown " in expanded.output


def test_lineage_raw_mermaid_honours_node(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """Expanding one node: the ops under it, nothing above it."""
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)
    crossing, *_ = dag.boundaries(kind="engine_crossing")

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "lineage",
            name,
            "--node",
            crossing["id"],
            "-l",
            "raw",
            "-f",
            "mermaid",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _mermaid_id(crossing["id"]) in result.output
    assert _mermaid_id(dag.root) not in result.output


def test_lineage_expand_keeps_the_whole_graph(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """`--expand` is the mixed-detail dial: everything stays compact except from
    the matched node downward, which is what `--node` cannot do (it narrows)."""
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)

    compact = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-f", "mermaid"]
    )
    expanded = runner.invoke(
        cli,
        ["--path", catalog_path, "lineage", name, "-f", "mermaid", "--expand", "join"],
    )

    assert expanded.exit_code == 0, expanded.output
    # the root and the far side of the graph survive, unlike with --node
    assert _mermaid_id(dag.root) in expanded.output
    assert "raw_customers" in expanded.output
    # and the expanded region gained the ops the compact view folded into `via`
    assert "Field:" not in compact.output
    assert "Field:" in expanded.output
    assert "|via " in compact.output
    assert "|via " not in expanded.output


def test_lineage_expand_reaches_the_run_above_the_node(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """Expanding a node whose own subtree has no intermediates still shows
    something: the run collapsed onto its consumer's edge."""
    catalog_path, name = catalog_with_udxf

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "lineage",
            name,
            "-f",
            "mermaid",
            "--expand",
            "flight_udxf",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "Field:" in result.output or "JoinReference" in result.output


def test_lineage_node_bounds_expand(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    """`--node` narrows; an `--expand` outside that subtree must not drag the rest
    of the graph back in."""
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)
    [duck_table] = [
        n for n in dag.boundaries(kind="table") if n.get("backend") == "duckdb"
    ]

    result = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "lineage",
            name,
            "-f",
            "mermaid",
            "--node",
            duck_table["id"],
            "--expand",
            "flight_udxf",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _mermaid_id(duck_table["id"]) in result.output
    # nothing above or beside the selected leaf
    assert _mermaid_id(dag.root) not in result.output
    assert "UDXF[" not in result.output
    assert "Cache[" not in result.output


def test_lineage_expand_accepts_the_same_handles_as_node(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf
    dag = _lineage_dag(catalog_path, name)
    [join] = dag.boundaries(kind="join")

    by_kind, by_label, by_hash = (
        runner.invoke(
            cli,
            ["--path", catalog_path, "lineage", name, "-f", "mermaid", "--expand", h],
        )
        for h in ("join", join["id"], join["snapshot_hash"])
    )

    assert by_kind.exit_code == 0, by_kind.output
    assert by_kind.output == by_label.output == by_hash.output


def test_lineage_expand_needs_mermaid_and_conflicts_with_raw(
    runner: CliRunner, catalog_with_udxf: tuple[str, str]
) -> None:
    catalog_path, name = catalog_with_udxf

    text = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "--expand", "join"]
    )
    with_raw = runner.invoke(
        cli,
        [
            "--path",
            catalog_path,
            "lineage",
            name,
            "-f",
            "mermaid",
            "-l",
            "raw",
            "--expand",
            "join",
        ],
    )
    unknown = runner.invoke(
        cli, ["--path", catalog_path, "lineage", name, "-f", "mermaid", "--expand", "x"]
    )

    assert text.exit_code != 0
    assert "--expand needs --format mermaid" in text.output
    assert with_raw.exit_code != 0
    assert "redundant with --level raw" in with_raw.output
    assert unknown.exit_code != 0
    assert "No lineage node matches 'x'" in unknown.output


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
