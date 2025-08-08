import pandas as pd
import pytest

import xorq as xo
from xorq.cli import lineage_command
from xorq.ibis_yaml.compiler import BuildManager


def test_lineage_no_target(capsys):
    # When build target does not exist, exit with code 2 and print error
    with pytest.raises(SystemExit) as exc:
        lineage_command("nonexistent")
    assert exc.value.code == 2
    out = capsys.readouterr().out
    assert "Build target not found: nonexistent" in out


def test_lineage_simple(tmp_path, capsys):
    # Create a simple build of a mutate expression
    builds_dir = tmp_path / "builds"
    cache_dir = tmp_path / "cache"
    builds_dir.mkdir()
    cache_dir.mkdir()
    # BuildManager writes expr.yaml, metadata.json, etc.
    bm = BuildManager(builds_dir, cache_dir)
    # Create an in-memory table and expression
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    table = xo.memtable(df, name="tbl")
    expr = table.mutate(c=table["a"] + table["b"])
    # Compile expression to a build directory
    expr_hash = bm.compile_expr(expr)
    build_path = builds_dir / expr_hash
    # Invoke lineage command
    lineage_command(str(build_path))
    out = capsys.readouterr().out
    # Expect lineage for each column
    # Check that lineage sections appear for each column
    assert "Lineage for column 'a':" in out
    assert "Field:a" in out
    assert "Lineage for column 'b':" in out
    assert "Field:b" in out
    assert "Lineage for column 'c':" in out
    # The root of 'c' lineage is an Add operation
    assert "Add #1" in out
