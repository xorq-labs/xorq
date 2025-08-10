import os
import sys
import yaml
import pytest
from pathlib import Path

import xorq
from xorq.cli import parse_args, catalog_command
from xorq.catalog import load_catalog, DEFAULT_CATALOG_PATH


@pytest.fixture(autouse=True)
def isolate_catalog(tmp_path, monkeypatch):
    """Redirect catalog file to a temporary location for each test."""
    # Point XDG_CONFIG_HOME to tmp_path so DEFAULT_CATALOG_PATH is under tmp_path
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Ensure parent exists
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    yield


def test_ls_empty_catalog(capsys):
    args = parse_args(["ls"])
    catalog_command(args)
    out = capsys.readouterr().out
    # No aliases, just Entries header
    assert "Entries:" in out


def test_add_and_ls_and_inspect(tmp_path, capsys):
    # Create minimal build directory
    build_dir = tmp_path / "build1"
    build_dir.mkdir()
    # metadata.json is required
    (build_dir / "metadata.json").write_text("{}")
    # Add build to catalog
    args = parse_args(["catalog", "add", str(build_dir)])
    catalog_command(args)
    out_add = capsys.readouterr().out
    assert "Added build build1 as entry" in out_add
    # ls shows the entry
    args = parse_args(["ls"])
    catalog_command(args)
    out_ls = capsys.readouterr().out
    assert "build1" in out_ls
    # inspect shows summary and metadata
    # Extract entry_id from add output (fifth token)
    entry_id = out_add.split()[5]
    args = parse_args(["catalog", "inspect", entry_id])
    catalog_command(args)
    out_inspect = capsys.readouterr().out
    assert "Summary:" in out_inspect
    assert f"Expr Hash    : build1" in out_inspect


def test_diff_builds_identical(tmp_path, capsys):
    # Create two identical build dirs
    for name in ("b1", "b2"):
        d = tmp_path / name
        d.mkdir()
        (d / "expr.yaml").write_text("foo: bar")
    # Run diff-builds
    args = parse_args(["catalog", "diff-builds", str(tmp_path / "b1"), str(tmp_path / "b2")])
    with pytest.raises(SystemExit) as e:
        catalog_command(args)
    # No differences -> exit code 0
    assert e.value.code == 0


def test_diff_builds_no_files(tmp_path):
    # Empty builds: no expr.yaml
    for name in ("b1", "b2"):
        (tmp_path / name).mkdir()
    args = parse_args(["catalog", "diff-builds", str(tmp_path / "b1"), str(tmp_path / "b2")])
    with pytest.raises(SystemExit) as e:
        catalog_command(args)
    # No files to diff -> exit code 2
    assert e.value.code == 2
    
def test_inspect_full_profiles(tmp_path, capsys):
    # Prepare a fake build directory with metadata.json and profiles.yaml
    build_dir = tmp_path / "b1"
    build_dir.mkdir()
    # metadata.json is required by validate_build
    (build_dir / "metadata.json").write_text("{}")
    # Create a profiles.yaml file
    profiles = {
        "pg": {"host": "localhost", "port": 5432},
        "duckdb": {"path": "db.duckdb"},
    }
    (build_dir / "profiles.yaml").write_text(yaml.safe_dump(profiles))
    # Add the build to the catalog
    args = parse_args(["catalog", "add", str(build_dir)])
    catalog_command(args)
    out_add = capsys.readouterr().out
    # Extract entry_id (6th token)
    entry_id = out_add.split()[5]
    # Inspect with --full to show profiles and expression DAG
    args = parse_args(["catalog", "inspect", entry_id, "--full"])
    catalog_command(args)
    out = capsys.readouterr().out
    # Since expr.yaml is missing, expect an error loading the DAG
    assert "Error loading expression for DAG" in out
    # Should print Profiles section and our two profiles
    assert "Profiles:" in out
    assert "pg: {'host': 'localhost', 'port': 5432}" in out
    assert "duckdb: {'path': 'db.duckdb'}" in out
    # Should include Node hashes section even if empty
    assert "Node hashes:" in out
    assert "No node hashes recorded." in out

def test_inspect_print_nodes_only(tmp_path, capsys):
    # Prepare a fake build directory with metadata.json but no profiles
    build_dir = tmp_path / "b2"
    build_dir.mkdir()
    (build_dir / "metadata.json").write_text("{}")
    # Add the build to the catalog
    args = parse_args(["catalog", "add", str(build_dir)])
    catalog_command(args)
    out_add = capsys.readouterr().out
    entry_id = out_add.split()[5]
    # Inspect with --print-nodes only
    args = parse_args(["catalog", "inspect", entry_id, "--print-nodes"])
    catalog_command(args)
    out = capsys.readouterr().out
    # Should print Node hashes section with no recorded hashes
    assert "Node hashes:" in out
    assert "No node hashes recorded." in out
    # Since expr.yaml is missing, expect an error loading the DAG
    assert "Error loading expression for DAG" in out
    
def test_rm_entry(tmp_path, capsys):
    # Add and then remove a build entry
    build_dir = tmp_path / "b1"
    build_dir.mkdir()
    (build_dir / "metadata.json").write_text("{}")
    # Add build
    args = parse_args(["catalog", "add", str(build_dir)])
    catalog_command(args)
    out_add = capsys.readouterr().out
    entry_id = out_add.split()[5]
    # Remove entry
    args = parse_args(["catalog", "rm", entry_id])
    catalog_command(args)
    out_rm = capsys.readouterr().out
    assert f"Removed entry {entry_id}" in out_rm
    # ls should no longer list the entry
    args = parse_args(["ls"])
    catalog_command(args)
    out_ls = capsys.readouterr().out
    assert entry_id not in out_ls

def test_rm_alias(tmp_path, capsys):
    # Add build with alias, then remove alias only
    build_dir = tmp_path / "b2"
    build_dir.mkdir()
    (build_dir / "metadata.json").write_text("{}")
    alias = "myalias"
    args = parse_args(["catalog", "add", "-a", alias, str(build_dir)])
    catalog_command(args)
    _ = capsys.readouterr()
    # Remove alias
    args = parse_args(["catalog", "rm", alias])
    catalog_command(args)
    out_rm = capsys.readouterr().out
    assert f"Removed alias {alias}" in out_rm
    # ls should not show any aliases
    args = parse_args(["ls"])
    catalog_command(args)
    out_ls = capsys.readouterr().out
    assert "Aliases:" not in out_ls

def test_rm_not_found(tmp_path, capsys):
    # Attempt to remove non-existent entry
    args = parse_args(["catalog", "rm", "noexist"] )
    catalog_command(args)
    out = capsys.readouterr().out
    assert "Entry noexist not found in catalog" in out