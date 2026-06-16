"""Unit coverage for pyiceberg Backend helpers and error paths."""

from __future__ import annotations

import pandas as pd
import pytest

from xorq.backends.pyiceberg import Backend, parse_url


def test_parse_url_path_only():
    cfg = parse_url("file:///tmp/wh")
    assert cfg["warehouse_path"] == "tmp/wh"
    assert cfg["namespace"] == "default"
    assert cfg["catalog_type"] == "sql"


def test_parse_url_netloc_and_query():
    cfg = parse_url("s3://bucket/path?namespace=ns&catalog_name=cat")
    assert cfg["warehouse_path"] == "bucket/path"
    assert cfg["namespace"] == "ns"
    assert cfg["catalog_name"] == "cat"


def test_from_url_delegates_to_parse_url():
    assert Backend._from_url("file:///tmp/wh") == parse_url("file:///tmp/wh")


def test_version_is_str(fresh_con):
    assert isinstance(fresh_con.version, str)


def test_create_table_exists_raises(fresh_con):
    fresh_con.create_table("t", pd.DataFrame({"a": [1]}))
    with pytest.raises(ValueError, match="already exists"):
        fresh_con.create_table("t", pd.DataFrame({"a": [2]}))


def test_create_table_overwrite(fresh_con):
    fresh_con.create_table("t", pd.DataFrame({"a": [1]}))
    fresh_con.create_table("t", pd.DataFrame({"a": [9, 9]}), overwrite=True)
    assert fresh_con.table("t").execute()["a"].tolist() == [9, 9]


def test_insert_nonexistent_raises(fresh_con):
    with pytest.raises(ValueError, match="does not exist"):
        fresh_con.insert("nope", pd.DataFrame({"a": [1]}))


def test_insert_append_accumulates(fresh_con):
    fresh_con.create_table("t", pd.DataFrame({"a": [1, 2, 3]}))
    fresh_con.insert("t", pd.DataFrame({"a": [4]}))
    assert sorted(fresh_con.table("t").execute()["a"].tolist()) == [1, 2, 3, 4]


def test_list_tables_like_filters(fresh_con):
    fresh_con.create_table("cats", pd.DataFrame({"a": [1]}))
    fresh_con.create_table("dogs", pd.DataFrame({"a": [1]}))
    assert fresh_con.list_tables(like="cat*") == ["cats"]


def test_get_schema_using_query_not_implemented(fresh_con):
    with pytest.raises(NotImplementedError):
        fresh_con._get_schema_using_query("select 1")
