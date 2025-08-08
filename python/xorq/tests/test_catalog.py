import json
import yaml
import pytest
from pathlib import Path

from xorq.catalog import (
    load_catalog,
    save_catalog,
    resolve_target,
    resolve_build_dir,
    build_virtual_export_tree,
    write_tree,
    unified_dir_diff,
)


def test_load_and_save_catalog(tmp_path):
    # Initial load on missing path returns minimal structure
    cat = load_catalog(path=str(tmp_path / 'catalog.yaml'))
    assert 'entries' in cat and isinstance(cat['entries'], list)
    # Save and reload
    save_catalog(cat, path=str(tmp_path / 'catalog.yaml'))
    cat2 = load_catalog(path=str(tmp_path / 'catalog.yaml'))
    assert cat2['entries'] == cat['entries']


def test_resolve_target_alias_and_entry():
    catalog = {
        'aliases': {'foo': {'entry_id': 'e1', 'revision_id': 'r1'}},
        'entries': [
            {'entry_id': 'e1', 'current_revision': 'r1', 'history': [{'revision_id': 'r1'}]}
        ],
    }
    t = resolve_target('foo', catalog)
    assert t.entry_id == 'e1'
    assert t.rev == 'r1'
    assert t.alias is True
    t2 = resolve_target('e1@r1', catalog)
    assert t2.entry_id == 'e1'
    assert t2.rev == 'r1'
    assert t2.alias is False


def test_resolve_build_dir_by_path(tmp_path):
    # If path exists, return it
    d = tmp_path / 'x'
    d.mkdir()
    cat = {'aliases': {}, 'entries': []}
    p = resolve_build_dir(str(d), cat)
    assert p == d

def test_resolve_build_dir_by_build_id(tmp_path):
    # Build id case
    # Create dummy build dir
    d = tmp_path / 'build1'
    d.mkdir()
    entry = {
        'entry_id': 'e1',
        'current_revision': 'r1',
        'history': [
            {'revision_id': 'r1', 'build': {'build_id': 'b1', 'path': str(d)}}
        ],
    }
    cat = {'aliases': {}, 'entries': [entry]}
    p = resolve_build_dir('b1', cat)
    assert p == d

def test_unified_dir_diff(tmp_path):
    # Create two directories with same and different files
    dir1 = tmp_path / 'd1'
    dir2 = tmp_path / 'd2'
    dir1.mkdir()
    dir2.mkdir()
    # identical file
    f1 = 'a.txt'
    (dir1 / f1).write_text('hello')
    (dir2 / f1).write_text('hello')
    # differing file
    f2 = 'b.txt'
    (dir1 / f2).write_text('x')
    (dir2 / f2).write_text('y')
    diff, text = unified_dir_diff(dir1, dir2)
    assert diff is True
    assert 'hello' not in text or 'x' in text
    # identical dirs
    diff2, text2 = unified_dir_diff(dir1, dir1)
    assert diff2 is False
    assert text2 == ''