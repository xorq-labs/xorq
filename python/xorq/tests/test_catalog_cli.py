import pytest

from xorq.cli import catalog_command, parse_args
from xorq.ibis_yaml.compiler import DumpFiles


LS_BASELINE = "Entries:\n"


@pytest.fixture(autouse=True)
def isolate_catalog(tmp_path, monkeypatch):
    """Redirect catalog file to a temporary location for each test."""
    # Point XDG_CONFIG_HOME to tmp_path so DEFAULT_CATALOG_PATH is under tmp_path
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # Ensure parent exists
    (tmp_path / "config").mkdir(parents=True, exist_ok=True)
    yield


def test_ls_empty_catalog(capsys):
    args = parse_args(["catalog", "ls"])
    catalog_command(args)
    out = capsys.readouterr().out
    # No aliases, just Entries header
    assert out == LS_BASELINE


def test_diff_builds_identical(tmp_path, capsys):
    # Create two identical build dirs
    for name in ("b1", "b2"):
        d = tmp_path / name
        d.mkdir()
        (d / DumpFiles.expr).write_text("foo: bar")
    # Run diff-builds
    args = parse_args(
        ["catalog", "diff-builds", str(tmp_path / "b1"), str(tmp_path / "b2")]
    )
    with pytest.raises(SystemExit) as e:
        catalog_command(args)
    # No differences -> exit code 0
    assert e.value.code == 0


def test_diff_builds_no_files(tmp_path):
    # Empty builds: no expr.yaml
    for name in ("b1", "b2"):
        (tmp_path / name).mkdir()
    args = parse_args(
        ["catalog", "diff-builds", str(tmp_path / "b1"), str(tmp_path / "b2")]
    )
    with pytest.raises(SystemExit) as e:
        catalog_command(args)
    # No files to diff -> exit code 2
    assert e.value.code == 2


def test_rm_entry(tmp_path, capsys):
    # Add and then remove a build entry
    build_dir = tmp_path / "b1"
    build_dir.mkdir()
    (build_dir / DumpFiles.metadata).write_text("{}")
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
    args = parse_args(["catalog", "ls"])
    catalog_command(args)
    out_ls = capsys.readouterr().out
    assert entry_id not in out_ls


def test_rm_alias(tmp_path, capsys):
    # Add build with alias, then remove alias only
    build_dir = tmp_path / "b2"
    build_dir.mkdir()
    (build_dir / DumpFiles.metadata).write_text("{}")
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
    args = parse_args(["catalog", "ls"])
    catalog_command(args)
    out_ls = capsys.readouterr().out
    assert "Aliases:" not in out_ls


def test_rm_not_found(tmp_path, capsys):
    # Attempt to remove non-existent entry
    args = parse_args(["catalog", "rm", "noexist"])
    catalog_command(args)
    out = capsys.readouterr().out
    assert "Entry noexist not found in catalog" in out


def test_info_empty_catalog(tmp_path, capsys):
    # On an empty catalog, info should show zero entries and aliases
    args = parse_args(["catalog", "info"])
    catalog_command(args)
    out = capsys.readouterr().out
    assert "Catalog path:" in out
    assert "Entries: 0" in out
    assert "Aliases: 0" in out


def test_info_after_add(tmp_path, capsys):
    # After adding one entry and one alias, info should update counts
    build_dir = tmp_path / "b1"
    build_dir.mkdir()
    (build_dir / DumpFiles.metadata).write_text("{}")
    # Add with alias
    alias = "foo"
    args = parse_args(["catalog", "add", "-a", alias, str(build_dir)])
    catalog_command(args)
    _ = capsys.readouterr()
    # Info now should report 1 entry and 1 alias
    args = parse_args(["catalog", "info"])
    catalog_command(args)
    out = capsys.readouterr().out
    assert "Entries: 1" in out
    assert "Aliases: 1" in out


def test_catalog_init_default(tmp_path, capsys, monkeypatch):
    # Test initializing a catalog in .xorq directory
    monkeypatch.chdir(tmp_path)
    args = parse_args(["catalog", "init"])
    catalog_command(args)
    out = capsys.readouterr().out

    # Check output
    assert "Initialized project catalog" in out
    assert ".xorq/catalog.yaml" in out

    # Check files were created
    catalog_path = tmp_path / ".xorq" / "catalog.yaml"
    builds_dir = tmp_path / ".xorq" / "builds"
    assert catalog_path.exists()
    assert builds_dir.exists()
    assert builds_dir.is_dir()


def test_catalog_init_custom_namespace(tmp_path, capsys, monkeypatch):
    # Test initializing a catalog with custom namespace
    monkeypatch.chdir(tmp_path)
    custom_dir = tmp_path / "custom_namespace"
    args = parse_args(["catalog", "--namespace", str(custom_dir), "init"])
    catalog_command(args)
    out = capsys.readouterr().out

    # Check output
    assert "Initialized custom catalog" in out

    # Check files were created
    catalog_path = custom_dir / "catalog.yaml"
    builds_dir = custom_dir / "catalog-builds"  # Custom paths use catalog-builds
    assert catalog_path.exists()
    assert builds_dir.exists()


def test_catalog_init_already_exists(tmp_path, capsys, monkeypatch):
    # Test that init reports when catalog already exists
    monkeypatch.chdir(tmp_path)
    # Initialize once
    args = parse_args(["catalog", "init"])
    catalog_command(args)
    _ = capsys.readouterr()

    # Try to initialize again
    args = parse_args(["catalog", "init"])
    catalog_command(args)
    out = capsys.readouterr().out
    assert "Catalog already exists" in out
