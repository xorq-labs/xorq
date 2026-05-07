"""Unit tests for ``_three_way_list_merge`` — the set-with-removal
semantics used by ``Catalog.pull()``'s ``catalog.yaml`` resolver.

The function treats lists as ordered sets and drops any item that was
in the merge base but missing from at least one side ("removed by one
side").  Items added by either side survive; duplicates collapse.

The full pull stack exercises these branches via the conflict matrix
in ``test_catalog_pull_conflicts.py``; pinning them at the function
level here means a regression in ``_three_way_list_merge`` surfaces as
a focused unit failure rather than a confusing pull-integration one.
"""

from __future__ import annotations

import io

import pytest

from xorq.catalog.catalog import _parse_catalog_yaml_blob, _three_way_list_merge
from xorq.catalog.constants import CatalogInfix


def test_empty_inputs():
    assert _three_way_list_merge([], [], []) == []


def test_no_changes_passes_through():
    assert _three_way_list_merge(["a", "b"], ["a", "b"], ["a", "b"]) == ["a", "b"]


def test_both_sides_add_same_new_item_collapses():
    # mirrors matrix case 02 / 08
    assert _three_way_list_merge([], ["x"], ["x"]) == ["x"]


def test_both_sides_add_different_items_unions():
    # mirrors matrix case 01
    result = _three_way_list_merge([], ["x"], ["y"])
    assert sorted(result) == ["x", "y"]


def test_ours_removes_theirs_keeps_drops_item():
    # base had x, ours dropped it, theirs still has it → removed
    assert _three_way_list_merge(["x"], [], ["x"]) == []


def test_theirs_removes_ours_keeps_drops_item():
    assert _three_way_list_merge(["x"], ["x"], []) == []


def test_both_remove_same_item():
    # mirrors matrix case 04
    assert _three_way_list_merge(["x"], [], []) == []


def test_each_side_removes_the_other_sides_item():
    # mirrors matrix case 03: base = [x, y]; ours kept y, theirs kept x
    assert _three_way_list_merge(["x", "y"], ["y"], ["x"]) == []


def test_add_and_remove_independent():
    # mirrors matrix case 05: base = [x]; ours kept x and added z;
    # theirs removed x and added w
    result = _three_way_list_merge(["x"], ["x", "z"], ["w"])
    assert sorted(result) == ["w", "z"]


def test_ours_first_ordering_then_theirs_only():
    # ordering contract: ours items appear in their order, then
    # theirs-only additions in their order
    result = _three_way_list_merge([], ["a", "b"], ["c", "a"])
    assert result == ["a", "b", "c"]


def test_duplicates_within_one_side_collapse():
    assert _three_way_list_merge([], ["x", "x"], []) == ["x"]


def test_duplicates_across_sides_collapse():
    assert _three_way_list_merge([], ["x"], ["x", "x"]) == ["x"]


def test_input_lists_not_mutated():
    base = ["a"]
    ours = ["a", "b"]
    theirs = ["a", "c"]
    _three_way_list_merge(base, ours, theirs)
    assert base == ["a"]
    assert ours == ["a", "b"]
    assert theirs == ["a", "c"]


# ---------------------------------------------------------------------------
# Unit tests for _parse_catalog_yaml_blob
# ---------------------------------------------------------------------------


class _FakeBlob:
    """Minimal stand-in for a gitpython Blob with a data_stream."""

    def __init__(self, text: str):
        self._stream = io.BytesIO(text.encode())

    def data_stream_read(self):
        return self._stream.read()

    @property
    def data_stream(self):
        class _DS:
            def __init__(self, stream):
                self._s = stream

            def read(self):
                return self._s.read()

        return _DS(self._stream)


def test_parse_blob_none_returns_empty_defaults():
    result = _parse_catalog_yaml_blob(None)
    assert result == {CatalogInfix.ENTRY: [], CatalogInfix.ALIAS: []}


def test_parse_blob_legacy_list_format():
    blob = _FakeBlob("[a, b, c]\n")
    result = _parse_catalog_yaml_blob(blob)
    assert result[CatalogInfix.ENTRY] == ["a", "b", "c"]
    assert result[CatalogInfix.ALIAS] == []


def test_parse_blob_dict_format():
    blob = _FakeBlob("entries:\n  - x\n  - y\naliases:\n  - alpha\n")
    result = _parse_catalog_yaml_blob(blob)
    assert result[CatalogInfix.ENTRY] == ["x", "y"]
    assert result[CatalogInfix.ALIAS] == ["alpha"]


def test_parse_blob_dict_missing_keys_default_to_empty():
    blob = _FakeBlob("{}\n")
    result = _parse_catalog_yaml_blob(blob)
    assert result[CatalogInfix.ENTRY] == []
    assert result[CatalogInfix.ALIAS] == []


# ---------------------------------------------------------------------------
# _three_way_list_merge edge-case
# ---------------------------------------------------------------------------


def test_unhashable_items_raise_type_error():
    # documents the schema constraint: list items must be hashable.
    # If a future schema makes entries dicts instead of strings, this
    # helper has to change shape — see RESOLVE_LIST_ITEM_NOT_HASHABLE
    # in the failure-mode catalog (PR-1902 open-items).
    with pytest.raises(TypeError):
        _three_way_list_merge([], [{"a": 1}], [])
