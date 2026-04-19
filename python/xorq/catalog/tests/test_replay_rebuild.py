from pathlib import Path

import pytest
from click.testing import CliRunner

import xorq.api as xo
from xorq.catalog.backend import GitBackend
from xorq.catalog.bind import CatalogTag, bind
from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.composer import ExprComposer
from xorq.catalog.replay import Replayer
from xorq.common.utils.graph_utils import walk_nodes
from xorq.expr.relations import HashingTag
from xorq.vendor.ibis.expr import operations as ops


@pytest.fixture
def source_catalog(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("source-repo"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    source_expr = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    source_entry = catalog.add(source_expr, aliases=("my-source",))
    unbound = ops.UnboundTable(
        name="placeholder", schema=source_expr.schema()
    ).to_expr()
    transform_expr = unbound.filter(unbound.amount > 0).select("user_id", "amount")
    transform_entry = catalog.add(transform_expr, aliases=("my-transform",))
    bound = bind(source_entry, transform_entry)
    bound_entry = catalog.add(bound, aliases=("bound1",))
    return catalog, source_entry, transform_entry, bound_entry


@pytest.fixture
def target_path(tmpdir):
    return Path(tmpdir).joinpath("target-repo")


@pytest.fixture
def saved_registry():
    """Save and restore the builder handler registry around a test."""
    import xorq.expr.builders as _builders_mod  # noqa: PLC0415
    from xorq.expr.builders import _FROM_TAG_NODE_REGISTRY  # noqa: PLC0415

    saved = dict(_FROM_TAG_NODE_REGISTRY)
    saved_keys = _builders_mod._BUILTIN_KEYS
    saved_init = _builders_mod._initialized
    yield
    _FROM_TAG_NODE_REGISTRY.clear()
    _FROM_TAG_NODE_REGISTRY.update(saved)
    _builders_mod._BUILTIN_KEYS = saved_keys
    _builders_mod._initialized = saved_init


def _replay_rebuild(source_catalog_obj, target_path):
    target = Catalog.from_repo_path(target_path, init=True)
    Replayer(from_catalog=source_catalog_obj, rebuild=True).replay(target)
    return target


def test_rebuild_produces_consistent_target(source_catalog, target_path):
    catalog, *_ = source_catalog
    target = _replay_rebuild(catalog, target_path)
    assert len(target.list()) == len(catalog.list())
    assert set(target.list_aliases()) == set(catalog.list_aliases())
    target.assert_consistency()


def test_rebuild_preserves_composed_from_linkage(source_catalog, target_path):
    catalog, source_entry, transform_entry, bound_entry = source_catalog
    target = _replay_rebuild(catalog, target_path)

    new_bound = target.get_catalog_entry("bound1", maybe_alias=True)
    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_transform = target.get_catalog_entry("my-transform", maybe_alias=True)

    composed_entry_names = {c["entry_name"] for c in new_bound.composed_from}
    assert new_source.name in composed_entry_names
    assert new_transform.name in composed_entry_names
    for name in composed_entry_names:
        assert name in target.list(), (
            f"bound composed_from refers to {name} which is not in target"
        )


def test_rebuild_rewrites_remote_expr_not_just_metadata(source_catalog, target_path):
    catalog, _source, _transform, _bound = source_catalog
    target = _replay_rebuild(catalog, target_path)

    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_transform = target.get_catalog_entry("my-transform", maybe_alias=True)
    new_bound = target.get_catalog_entry("bound1", maybe_alias=True)

    # Strong invariant: the rebuilt expression's recovered recipe references
    # target entries by name. ExprComposer.from_expr walks HashingTag.metadata
    # only, so this is independent of RemoteTable.name (gen_name-tainted) and
    # of remote_expr op identity (which can shift across code versions).
    recovered = ExprComposer.from_expr(new_bound.lazy_expr, target)
    assert recovered.source.name == new_source.name
    assert tuple(t.name for t in recovered.transforms) == (new_transform.name,)
    assert recovered.code is None


def test_rebuild_expr_for_target_returns_input_when_no_catalog_tags(source_catalog):
    from types import SimpleNamespace  # noqa: PLC0415

    from xorq.catalog.replay import _rebuild_expr_for_target  # noqa: PLC0415

    catalog, _source, _transform, _bound = source_catalog
    # Stub a minimal source_entry so lazy_expr is a stable object — the
    # real CatalogEntry.lazy_expr reloads from zip on each access, which
    # defeats identity comparison.
    expr = xo.memtable({"x": [1, 2, 3]})
    stub = SimpleNamespace(lazy_expr=expr, catalog=catalog, name="stub")
    result = _rebuild_expr_for_target(stub, catalog, remap={})
    assert result is expr


def test_bind_is_hash_deterministic(source_catalog, tmpdir):
    # Use two separate catalogs: Catalog.add(exist_ok=True) returns None when
    # an entry already exists, so a second add into the same catalog can't
    # reveal the hash. Scratch catalogs populated with the same source and
    # transform will produce matching hashes iff bind() is deterministic.
    _, source_entry, transform_entry, _ = source_catalog
    a_cat = Catalog.from_repo_path(Path(tmpdir).joinpath("det-a"), init=True)
    a_src = a_cat.add(source_entry.lazy_expr)
    a_tr = a_cat.add(transform_entry.lazy_expr)
    a = a_cat.add(bind(a_src, a_tr))
    b_cat = Catalog.from_repo_path(Path(tmpdir).joinpath("det-b"), init=True)
    b_src = b_cat.add(source_entry.lazy_expr)
    b_tr = b_cat.add(transform_entry.lazy_expr)
    b = b_cat.add(bind(b_src, b_tr))
    assert a.name == b.name, (
        "bind() is non-deterministic — gen_name is leaking into entry hash"
    )


def test_rebuild_matches_fresh_bind(source_catalog, target_path, tmpdir):
    catalog, *_ = source_catalog
    target = _replay_rebuild(catalog, target_path)

    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_transform = target.get_catalog_entry("my-transform", maybe_alias=True)
    new_bound = target.get_catalog_entry("bound1", maybe_alias=True)

    # Re-add into a SCRATCH catalog so add()'s exist_ok-collision behavior
    # cannot mask a name mismatch. Sanity-check that source/transform hash
    # is content-pure first — otherwise the bind comparison below is moot.
    # Aliases must match too: _make_source_tag falls back to the first
    # alias of the source entry, and rebuild pins that alias explicitly.
    scratch = Catalog.from_repo_path(Path(tmpdir).joinpath("scratch"), init=True)
    scratch_source = scratch.add(new_source.lazy_expr, aliases=("my-source",))
    scratch_transform = scratch.add(new_transform.lazy_expr, aliases=("my-transform",))
    assert scratch_source.name == new_source.name
    assert scratch_transform.name == new_transform.name

    fresh = bind(scratch_source, scratch_transform)
    fresh_entry = scratch.add(fresh)
    assert fresh_entry.name == new_bound.name


def test_rebuild_chained_bind_of_bind(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    source_expr = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    source_entry = catalog.add(source_expr, aliases=("my-source",))

    # First transform keeps schema (filter + select).
    u1 = ops.UnboundTable(name="p1", schema=source_expr.schema()).to_expr()
    t1 = u1.filter(u1.amount > 0).select("user_id", "amount")
    t1_entry = catalog.add(t1, aliases=("t1",))

    # Second transform keeps schema too.
    u2 = ops.UnboundTable(name="p2", schema=source_expr.schema()).to_expr()
    t2 = u2.select("user_id", "amount")
    t2_entry = catalog.add(t2, aliases=("t2",))

    # bound1 = bind(source, t1); bound2 = bind(bound1_as_source, t2) via a
    # fresh source entry — the interesting case is chained transforms in
    # one bind() call, which ExprComposer handles as transforms=(t1, t2).
    bound = bind(source_entry, t1_entry, t2_entry)
    catalog.add(bound, aliases=("chained",))

    target = _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    new_chained = target.get_catalog_entry("chained", maybe_alias=True)
    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_t1 = target.get_catalog_entry("t1", maybe_alias=True)
    new_t2 = target.get_catalog_entry("t2", maybe_alias=True)

    recovered = ExprComposer.from_expr(new_chained.lazy_expr, target)
    assert recovered.source.name == new_source.name
    assert tuple(t.name for t in recovered.transforms) == (new_t1.name, new_t2.name)
    assert recovered.code is None


def test_rebuild_code_only_composition(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    source_expr = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    source_entry = catalog.add(source_expr, aliases=("my-source",))

    composed = ExprComposer(
        source=source_entry,
        code="source.filter(source.amount > 0)",
    )
    catalog.add(composed.expr, aliases=("code-only",))

    target = _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    new_code_only = target.get_catalog_entry("code-only", maybe_alias=True)
    new_source = target.get_catalog_entry("my-source", maybe_alias=True)

    # (a) rebuilt entry has a CODE tag with the expected code metadata
    code_tags = tuple(
        ht
        for ht in walk_nodes(HashingTag, new_code_only.lazy_expr)
        if ht.metadata.get("tag") == CatalogTag.CODE
    )
    assert len(code_tags) == 1
    assert code_tags[0].metadata["code"] == "source.filter(source.amount > 0)"

    # (b) every catalog-tag entry_name is in target.list()
    tag_names = {
        ht.metadata["entry_name"]
        for ht in walk_nodes(HashingTag, new_code_only.lazy_expr)
        if ht.metadata.get("tag") in frozenset(CatalogTag)
        and ht.metadata.get("entry_name")
    }
    assert tag_names <= set(target.list())

    # (c) recovered recipe matches
    recovered = ExprComposer.from_expr(new_code_only.lazy_expr, target)
    assert recovered.source.name == new_source.name
    assert recovered.transforms == ()
    assert recovered.code == "source.filter(source.amount > 0)"


def test_rebuild_source_transform_code_composition(source_catalog, target_path):
    catalog, source_entry, transform_entry, _bound = source_catalog

    composed = ExprComposer(
        source=source_entry,
        transforms=(transform_entry,),
        code="source.select('user_id')",
    )
    catalog.add(composed.expr, aliases=("full-recipe",))

    target = _replay_rebuild(catalog, target_path)
    new_full = target.get_catalog_entry("full-recipe", maybe_alias=True)
    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_transform = target.get_catalog_entry("my-transform", maybe_alias=True)

    code_tags = tuple(
        ht
        for ht in walk_nodes(HashingTag, new_full.lazy_expr)
        if ht.metadata.get("tag") == CatalogTag.CODE
    )
    assert len(code_tags) == 1
    assert code_tags[0].metadata["code"] == "source.select('user_id')"

    tag_names = {
        ht.metadata["entry_name"]
        for ht in walk_nodes(HashingTag, new_full.lazy_expr)
        if ht.metadata.get("tag") in frozenset(CatalogTag)
        and ht.metadata.get("entry_name")
    }
    assert tag_names <= set(target.list())

    recovered = ExprComposer.from_expr(new_full.lazy_expr, target)
    assert recovered.source.name == new_source.name
    assert tuple(t.name for t in recovered.transforms) == (new_transform.name,)
    assert recovered.code == "source.select('user_id')"


def test_rebuild_preserves_outer_builder_wrapping(tmpdir, saved_registry):
    from xorq.expr.builders import TagHandler, register_tag_handler  # noqa: PLC0415
    from xorq.expr.relations import Tag  # noqa: PLC0415
    from xorq.ibis_yaml.enums import ExprKind  # noqa: PLC0415

    register_tag_handler(
        TagHandler(
            tag_names=("test_rebuild_builder",),
            extract_metadata=lambda tag_node: {"type": "test_rebuild_builder"},
        )
    )

    # Build a fresh catalog that has no pre-existing bound entry — Tag is
    # transparent to dask.tokenize, so composed.tag(...) and composed hash
    # to the same entry name; if bound were pre-added, the Tag-wrapped
    # expression would collide.
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    source_expr = xo.memtable({"user_id": [1, 2, 3], "amount": [10.0, 20.0, 30.0]})
    source_entry = catalog.add(source_expr, aliases=("my-source",))
    unbound = ops.UnboundTable(name="p", schema=source_expr.schema()).to_expr()
    transform_entry = catalog.add(
        unbound.filter(unbound.amount > 0).select("user_id", "amount"),
        aliases=("my-transform",),
    )
    composed = bind(source_entry, transform_entry)
    builder_expr = composed.tag("test_rebuild_builder")
    catalog.add(builder_expr, aliases=("builder1",))

    target = _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    new_builder = target.get_catalog_entry("builder1", maybe_alias=True)
    new_source = target.get_catalog_entry("my-source", maybe_alias=True)
    new_transform = target.get_catalog_entry("my-transform", maybe_alias=True)

    # Outer wrapping preserved.
    assert new_builder.kind == ExprKind.ExprBuilder
    outer = new_builder.lazy_expr.op()
    assert isinstance(outer, Tag)
    assert outer.metadata["tag"] == "test_rebuild_builder"

    # Inner catalog refs translated. Recover recipe from the inner
    # composition subtree.
    inner_root = next(
        ht
        for ht in walk_nodes(HashingTag, new_builder.lazy_expr)
        if ht.metadata.get("tag") in frozenset(CatalogTag)
    )
    recovered = ExprComposer.from_expr(inner_root.to_expr(), target)
    assert recovered.source.name == new_source.name
    assert tuple(t.name for t in recovered.transforms) == (new_transform.name,)


def test_rebuild_pure_builder_without_catalog_refs(tmpdir, saved_registry):
    import dask.base  # noqa: PLC0415

    from xorq.expr.builders import TagHandler, register_tag_handler  # noqa: PLC0415

    register_tag_handler(
        TagHandler(
            tag_names=("test_pure_builder",),
            extract_metadata=lambda tag_node: {"type": "test_pure_builder"},
        )
    )

    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    # Pure builder: no catalog refs anywhere.
    raw = xo.memtable({"x": [1, 2, 3]}).tag("test_pure_builder")
    catalog.add(raw, aliases=("pure",))

    target = _replay_rebuild(catalog, Path(tmpdir).joinpath("tgt"))
    new_pure = target.get_catalog_entry("pure", maybe_alias=True)
    src_entry = catalog.get_catalog_entry("pure", maybe_alias=True)
    # Same content hash — no catalog-tag rewrite should have run.
    assert dask.base.tokenize(new_pure.lazy_expr) == dask.base.tokenize(
        src_entry.lazy_expr
    )
    assert new_pure.name == src_entry.name


def test_rebuild_raises_when_recipe_references_unknown_target_entry(
    source_catalog, target_path
):
    from xorq.catalog.replay import AddEntry  # noqa: PLC0415

    catalog, _source, _transform, bound_entry = source_catalog
    target = Catalog.from_repo_path(target_path, init=True)

    op = AddEntry(entry_hash=bound_entry.name, aliases=("bound1",))
    with pytest.raises(RuntimeError, match="not been rebuilt in target"):
        op.do(catalog, target, rebuild=True, remap={})


def test_rebuild_preserves_aliases(source_catalog, target_path):
    catalog, *_ = source_catalog
    target = _replay_rebuild(catalog, target_path)

    for alias in ("my-source", "my-transform", "bound1"):
        target_entry = target.get_catalog_entry(alias, maybe_alias=True)
        assert target_entry.exists()
        assert target_entry.name in target.list()


def test_remove_entry_translates_via_remap(tmpdir):
    from xorq.catalog.replay import RemoveEntry  # noqa: PLC0415

    src_repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    src = Catalog(backend=GitBackend(repo=src_repo))
    tgt_repo = Catalog.init_repo_path(Path(tmpdir).joinpath("tgt"))
    tgt = Catalog(backend=GitBackend(repo=tgt_repo))
    entry = tgt.add(xo.memtable({"a": [1]}))

    op = RemoveEntry(entry_name="OLD_HASH", aliases=())
    op.do(src, tgt, rebuild=True, remap={"OLD_HASH": entry.name})
    assert entry.name not in tgt.list()


def test_add_alias_translates_via_remap(tmpdir):
    from xorq.catalog.replay import AddAlias  # noqa: PLC0415

    src_repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    src = Catalog(backend=GitBackend(repo=src_repo))
    tgt_repo = Catalog.init_repo_path(Path(tmpdir).joinpath("tgt"))
    tgt = Catalog(backend=GitBackend(repo=tgt_repo))
    entry = tgt.add(xo.memtable({"a": [1]}))

    op = AddAlias(alias="my-alias", entry_name="OLD_HASH")
    op.do(src, tgt, rebuild=True, remap={"OLD_HASH": entry.name})
    resolved = tgt.get_catalog_entry("my-alias", maybe_alias=True)
    assert resolved.name == entry.name


def test_rebuild_preserves_commit_metadata(source_catalog, target_path):
    catalog, *_ = source_catalog
    target = _replay_rebuild(catalog, target_path)

    src_commits = list(catalog.repo.iter_commits(reverse=True))
    tgt_commits = list(target.repo.iter_commits(reverse=True))
    assert len(src_commits) == len(tgt_commits)
    for s, t in zip(src_commits, tgt_commits):
        assert s.author.name == t.author.name
        assert s.author.email == t.author.email
        assert s.authored_date == t.authored_date


def test_rebuild_raises_on_unknown_op(tmpdir):
    repo = Catalog.init_repo_path(Path(tmpdir).joinpath("src"))
    catalog = Catalog(backend=GitBackend(repo=repo))
    (Path(repo.working_dir) / "stray.txt").write_text("hello")
    repo.index.add(["stray.txt"])
    repo.index.commit("not a catalog op")

    target_path = Path(tmpdir).joinpath("target")
    target = Catalog.from_repo_path(target_path, init=True)
    replayer = Replayer(from_catalog=catalog, rebuild=True, verify=False)
    with pytest.raises(RuntimeError, match="Cannot rebuild unknown op"):
        replayer.replay(target)


def test_cli_replay_rebuild_smoke(source_catalog, tmpdir):
    catalog, *_ = source_catalog
    target = str(Path(tmpdir).joinpath("cli-target"))
    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["--path", str(catalog.repo_path), "replay", target, "--rebuild"],
    )
    assert result.exit_code == 0, result.output
    target_catalog = Catalog.from_repo_path(target, init=False)
    target_catalog.assert_consistency()
    assert set(target_catalog.list_aliases()) == set(catalog.list_aliases())
