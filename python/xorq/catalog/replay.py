"""Parse a catalog's git log into structured operations and optionally replay them.

Parses the four commit-message formats produced by xorq's Catalog API:
  - "add: {hash} (aliases {a1}, {a2}, ...)"
  - "add alias: {alias} -> {entry}"
  - "rm: {hash} (aliases {a1}, {a2}, ...)"
  - "rm alias: {alias}"

The :class:`Replayer` replays these operations in order against a target
catalog, reproducing the source catalog's state.

Rebuild mode (``rebuild=True``) re-executes each ``AddEntry`` against the
target catalog under the current code. Entries whose expression contains no
rebuildable tags are re-added from their stored ``lazy_expr``. Entries
containing rebuildable tags walk outermost-first: the first ``CatalogTag``
(SOURCE / TRANSFORM / CODE) goes through :class:`ExprComposer` as a
single-output builder; any other tag registered via a ``TagHandler``
goes through the handler's rebuild protocol. The fresh subtree is spliced
back under any outer wrapping.

Rebuild protocols (see ``xorq.expr.builders.get_rebuild_dispatch``):
  1. Handler-level ``reemit`` callable on the ``TagHandler``.
  2. Domain-object ``reemit(tag_node, rebuild_subexpr)`` method
     (multi-output builders like ``FittedPipeline``).
  3. Domain-object ``with_inputs_translated(remap, to_catalog)`` +
     ``expr`` (single-output builders like ``ExprComposer``).
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager, nullcontext
from functools import cached_property, partial

from attr import field, frozen, validators

from xorq.catalog.constants import CATALOG_YAML_NAME, CatalogInfix


# Only exercised on rebuild=True — translates old catalog names to rebuilt ones.
def _translate(name, remap):
    return remap.get(name, name) if remap else name


def _rebuild_expr_for_target(source_entry, to_catalog, remap):
    """Return an expression for re-adding *source_entry* into *to_catalog*.

    Thin wrapper around :func:`_rebuild_subexpr` that reads ``lazy_expr``
    and ``catalog`` from *source_entry*.
    """
    return _rebuild_subexpr(
        source_entry.lazy_expr,
        from_catalog=source_entry.catalog,
        to_catalog=to_catalog,
        remap=remap,
    )


def _rebuild_subexpr(expr, *, from_catalog, to_catalog, remap):
    """Rebuild every outermost rebuildable tag in *expr* under current code.

    Walks tags outermost-first and collects all outermost rebuildable
    tags — those with no rebuildable tag among their ancestors. Each is
    rebuilt atomically and spliced back, so parallel rebuildable
    subtrees (e.g., a ``FittedPipeline``'s training subtree reached via
    fit UDF, and its predict-input subtree reached via the direct tree
    path) all get rebuilt in one pass.

    - Catalog tag (``ExprKind.Composed``): recovers the full
      ``SOURCE [TRANSFORM*] [CODE]`` chain via ``ExprComposer.from_expr``
      and rebuilds it atomically via ``with_inputs_translated``. Inner
      catalog tags inside the chain are rebuilt as part of the same
      composition.
    - Registered builder tag (``ExprKind.ExprBuilder``): dispatches via
      the handler protocol. The handler's ``reemit`` recursively
      rebuilds its own inputs via the ``rebuild_subexpr`` closure
      passed in; outer processing does not re-enter its subtree.

    Returns *expr* unchanged when no rebuildable tag is found.
    """
    from xorq.catalog.bind import CatalogTag  # noqa: PLC0415
    from xorq.catalog.composer import ExprComposer  # noqa: PLC0415
    from xorq.common.utils.graph_utils import walk_nodes  # noqa: PLC0415
    from xorq.expr.builders import (  # noqa: PLC0415
        get_rebuild_dispatch,
    )
    from xorq.expr.relations import HashingTag, Tag  # noqa: PLC0415

    catalog_tags = frozenset(CatalogTag)
    rebuild_subexpr = partial(
        _rebuild_subexpr,
        from_catalog=from_catalog,
        to_catalog=to_catalog,
        remap=remap,
    )

    def _is_catalog_tag(node):
        return isinstance(node, HashingTag) and node.metadata.get("tag") in catalog_tags

    def _rebuild_tag(tag):
        if _is_catalog_tag(tag):
            try:
                composer = ExprComposer.from_expr(tag.to_expr(), from_catalog)
            except ValueError as e:
                raise RuntimeError(
                    f"rebuild: cannot recover composition recipe: {e}. "
                    "Only bind()/ExprComposer-produced composition shapes "
                    "can be rebuilt."
                ) from e
            return composer.with_inputs_translated(remap, to_catalog).expr
        dispatch = get_rebuild_dispatch(tag)
        if dispatch is None:
            return None
        if callable(dispatch):
            return dispatch(rebuild_subexpr)
        _, builder = dispatch
        return builder.with_inputs_translated(remap, to_catalog).expr

    # Collect outermost rebuildable tags. walk_nodes yields outermost-first,
    # so a tag is "outermost rebuildable" iff no already-collected
    # rebuildable tag contains it in its subtree.
    outermost = []
    claimed = set()
    for tag in walk_nodes((Tag, HashingTag), expr):
        if id(tag) in claimed:
            continue
        if not (_is_catalog_tag(tag) or get_rebuild_dispatch(tag) is not None):
            continue
        outermost.append(tag)
        for descendant in walk_nodes((Tag, HashingTag), tag.to_expr()):
            claimed.add(id(descendant))

    for tag in outermost:
        fresh = _rebuild_tag(tag)
        if fresh is None:
            continue
        _assert_schema_preserved(tag, fresh)
        expr = _splice_or_return(expr, old=tag, new=fresh)

    return expr


def _assert_schema_preserved(tag_node, fresh):
    old_schema = tag_node.to_expr().schema()
    new_schema = fresh.schema()
    if old_schema != new_schema:
        raise RuntimeError(
            f"rebuild: schema changed for tag {tag_node.metadata.get('tag')!r} — "
            f"old: {dict(old_schema.items())}, new: {dict(new_schema.items())}. "
            "Remove and re-add the entry manually under current code."
        )


def _splice_or_return(expr, *, old, new):
    if expr.op() is old:
        return new
    return _splice(expr, old=old, new=new.op())


def _splice(expr, *, old, new):
    """Return expr with node `old` replaced by `new`, rebuilding parents.

    PRECONDITION: `new` must have the same schema and node type as `old`.
    `replace_nodes` rebuilds every parent of `old` with `new` as its
    child, so any parent validator that checks schema/type compatibility
    (e.g. builder tags that pre-computed schema-dependent metadata) will
    reject a mismatched substitution. Schema preservation is enforced by
    :func:`_assert_schema_preserved` before calling this function.
    """
    from xorq.common.utils.graph_utils import replace_nodes  # noqa: PLC0415

    def replacer(node, kwargs):
        if node is old:
            return new
        if kwargs:
            return node.__recreate__(kwargs)
        return node

    return replace_nodes(replacer, expr).to_expr()


# -- Commit metadata ----------------------------------------------------------


@frozen
class CommitMetadata:
    sha: str = field(validator=validators.instance_of(str))
    author_name: str = field(validator=validators.instance_of(str))
    author_email: str = field(validator=validators.instance_of(str))
    authored_date: str = field(validator=validators.instance_of(str))
    committed_date: str = field(validator=validators.instance_of(str))

    @contextmanager
    def git_env(self):
        """Temporarily override git author/committer env vars."""
        env_vars = {
            "GIT_AUTHOR_NAME": self.author_name,
            "GIT_AUTHOR_EMAIL": self.author_email,
            "GIT_COMMITTER_NAME": self.author_name,
            "GIT_COMMITTER_EMAIL": self.author_email,
            "GIT_AUTHOR_DATE": self.authored_date,
            "GIT_COMMITTER_DATE": self.committed_date,
        }
        old = {k: os.environ.get(k) for k in env_vars}
        os.environ.update(env_vars)
        try:
            yield
        finally:
            for k, v in old.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    @staticmethod
    def _git_date(unix_ts, tz_offset_seconds):
        """Format as git-internal date: '<unix_ts> <+/-HHMM>'.

        GitPython's tz_offset is seconds west of UTC (positive = behind UTC),
        but git's date format uses the opposite convention (negative = behind).
        """
        sign = "+" if tz_offset_seconds < 0 else "-"
        total_minutes = abs(tz_offset_seconds) // 60
        hours, minutes = divmod(total_minutes, 60)
        return f"{unix_ts} {sign}{hours:02d}{minutes:02d}"

    @classmethod
    def from_commit(cls, commit):
        return cls(
            sha=str(commit.hexsha),
            author_name=str(commit.author),
            author_email=commit.author.email,
            authored_date=cls._git_date(commit.authored_date, commit.author_tz_offset),
            committed_date=cls._git_date(
                commit.committed_date, commit.committer_tz_offset
            ),
        )


# -- Operation types ----------------------------------------------------------

_ADD_RE = re.compile(r"^add: (?P<hash>[a-f0-9]+)(?:\s+\(aliases\s+(?P<aliases>.+)\))?$")
_ADD_ALIAS_RE = re.compile(r"^add alias: (?P<alias>.+?) -> (?P<entry>.+)$")
_RM_RE = re.compile(r"^rm: (?P<hash>[a-f0-9]+)(?:\s+\(aliases\s+(?P<aliases>.+)\))?$")
_RM_ALIAS_RE = re.compile(r"^rm alias: (?P<alias>.+)$")


def _make_commit_metadata_field():
    return field(
        default=None,
        validator=validators.optional(validators.instance_of(CommitMetadata)),
    )


def _parse_aliases(raw):
    if raw:
        return tuple(a.strip() for a in raw.split(","))
    return ()


def _changed_paths(commit):
    """Return the set of file paths changed by a commit."""
    if not commit.parents:
        # initial commit — all files are new
        return {item.path for item in commit.tree.traverse()}
    parent = commit.parents[0]
    return {diff.b_path or diff.a_path for diff in parent.diff(commit)}


@frozen
class InitCatalog:
    """First commit: bare repo initialization."""

    message: str = field(validator=validators.instance_of(str))
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        return f"[init]  {sha}  {self.message}"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        pass

    @staticmethod
    def verify_commit(commit):
        assert not commit.parents, f"InitCatalog commit {commit.hexsha[:8]} has parents"

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if msg == "initial commit":
            return cls(message=msg, commit_metadata=CommitMetadata.from_commit(commit))
        return None


@frozen
class AddCatalogYAML:
    """Second commit: catalog.yaml creation."""

    message: str = field(validator=validators.instance_of(str))
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        return f"[init]  {sha}  {self.message}"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        pass

    @staticmethod
    def verify_commit(commit):
        paths = _changed_paths(commit)
        assert any(
            p == CATALOG_YAML_NAME or p.endswith(CATALOG_YAML_NAME) for p in paths
        ), (
            f"AddCatalogYAML commit {commit.hexsha[:8]} did not touch {CATALOG_YAML_NAME}: {paths}"
        )

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if msg.startswith("add catalog"):
            return cls(message=msg, commit_metadata=CommitMetadata.from_commit(commit))
        return None


@frozen
class AddEntry:
    """catalog.add(build_dir, aliases=(...))"""

    entry_hash: str = field(validator=validators.instance_of(str))
    aliases: tuple[str, ...] = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(tuple),
        )
    )
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        alias_str = ", ".join(self.aliases) if self.aliases else "(none)"
        return f"[add]   {sha}  entry={self.entry_hash}  aliases=[{alias_str}]"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        if not rebuild:
            catalog_entry = from_catalog.get_catalog_entry(self.entry_hash)
            to_catalog.add(
                catalog_entry.catalog_path,
                sync=False,
                aliases=self.aliases,
                exist_ok=True,
            )
            return

        source_entry = from_catalog.get_catalog_entry(self.entry_hash)
        try:
            expr = _rebuild_subexpr(
                source_entry.lazy_expr,
                from_catalog=from_catalog,
                to_catalog=to_catalog,
                remap=remap,
            )
        except RuntimeError as e:
            raise RuntimeError(
                f"rebuild: failed to rebuild entry {source_entry.name!r}: {e}"
            ) from e
        new_entry = to_catalog.add(
            expr,
            sync=False,
            aliases=self.aliases,
            exist_ok=True,
        )
        if remap is not None:
            remap[self.entry_hash] = new_entry.name

    def verify_commit(self, commit):
        paths = _changed_paths(commit)
        entry_prefix = f"{CatalogInfix.ENTRY}/{self.entry_hash}"
        assert any(p.startswith(entry_prefix) for p in paths), (
            f"AddEntry commit {commit.hexsha[:8]} has no changes under {entry_prefix}: {paths}"
        )
        for alias in self.aliases:
            alias_prefix = f"{CatalogInfix.ALIAS}/{alias}"
            assert any(p.startswith(alias_prefix) for p in paths), (
                f"AddEntry commit {commit.hexsha[:8]} missing alias {alias!r} under {alias_prefix}: {paths}"
            )

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if m := _ADD_RE.match(msg):
            return cls(
                entry_hash=m["hash"],
                aliases=_parse_aliases(m["aliases"]),
                commit_metadata=CommitMetadata.from_commit(commit),
            )
        return None


@frozen
class AddAlias:
    """catalog.add_alias(entry, alias)"""

    alias: str = field(validator=validators.instance_of(str))
    entry_name: str = field(validator=validators.instance_of(str))
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        return f"[alias] {sha}  {self.alias} -> {self.entry_name}"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        entry_name = _translate(self.entry_name, remap) if rebuild else self.entry_name
        to_catalog.add_alias(entry_name, self.alias, sync=False)

    def verify_commit(self, commit):
        paths = _changed_paths(commit)
        alias_prefix = f"{CatalogInfix.ALIAS}/{self.alias}"
        assert any(p.startswith(alias_prefix) for p in paths), (
            f"AddAlias commit {commit.hexsha[:8]} missing {alias_prefix}: {paths}"
        )

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if m := _ADD_ALIAS_RE.match(msg):
            return cls(
                alias=m["alias"],
                entry_name=m["entry"],
                commit_metadata=CommitMetadata.from_commit(commit),
            )
        return None


@frozen
class RemoveEntry:
    """catalog.remove(entry) -- removes entry and its aliases."""

    entry_name: str = field(validator=validators.instance_of(str))
    aliases: tuple[str, ...] = field(
        validator=validators.deep_iterable(
            member_validator=validators.instance_of(str),
            iterable_validator=validators.instance_of(tuple),
        )
    )
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        alias_str = ", ".join(self.aliases) if self.aliases else "(none)"
        return f"[rm]    {sha}  entry={self.entry_name}  aliases=[{alias_str}]"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        entry_name = _translate(self.entry_name, remap) if rebuild else self.entry_name
        to_catalog.remove(entry_name, sync=False)

    def verify_commit(self, commit):
        paths = _changed_paths(commit)
        entry_prefix = f"{CatalogInfix.ENTRY}/{self.entry_name}"
        assert any(p.startswith(entry_prefix) for p in paths), (
            f"RemoveEntry commit {commit.hexsha[:8]} has no changes under {entry_prefix}: {paths}"
        )

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if m := _RM_RE.match(msg):
            return cls(
                entry_name=m["hash"],
                aliases=_parse_aliases(m["aliases"]),
                commit_metadata=CommitMetadata.from_commit(commit),
            )
        return None


@frozen
class RemoveAlias:
    """CatalogAlias.remove()"""

    alias: str = field(validator=validators.instance_of(str))
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        return f"[rm-a]  {sha}  alias={self.alias}"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        from xorq.catalog.catalog import CatalogAlias  # noqa: PLC0415

        CatalogAlias.from_name(self.alias, to_catalog).remove()

    def verify_commit(self, commit):
        paths = _changed_paths(commit)
        alias_prefix = f"{CatalogInfix.ALIAS}/{self.alias}"
        assert any(p.startswith(alias_prefix) for p in paths), (
            f"RemoveAlias commit {commit.hexsha[:8]} missing {alias_prefix}: {paths}"
        )

    @classmethod
    def from_commit(cls, commit):
        msg = commit.message.strip()
        if m := _RM_ALIAS_RE.match(msg):
            return cls(
                alias=m["alias"],
                commit_metadata=CommitMetadata.from_commit(commit),
            )
        return None


@frozen
class UnknownOp:
    """Commit that doesn't match any known catalog operation."""

    message: str = field(validator=validators.instance_of(str))
    hexsha: str = field(validator=validators.instance_of(str))
    commit_metadata: CommitMetadata | None = _make_commit_metadata_field()

    def __str__(self):
        sha = self.commit_metadata.sha[:8] if self.commit_metadata else "--------"
        return f"[???]   {sha}  {self.message}"

    def do(self, from_catalog, to_catalog, *, rebuild=False, remap=None):
        if rebuild:
            raise RuntimeError(
                f"Cannot rebuild unknown op at commit {self.hexsha[:8]}: {self.message!r}. "
                "Rebuild requires all ops be recognized catalog operations."
            )
        patch = from_catalog.repo.git.format_patch("-1", self.hexsha, stdout=True)
        to_catalog.repo.git.am(input=patch)

    def verify_commit(self, commit):
        pass


CatalogOp = (
    InitCatalog
    | AddCatalogYAML
    | AddEntry
    | AddAlias
    | RemoveEntry
    | RemoveAlias
    | UnknownOp
)

_OP_TYPES = (InitCatalog, AddCatalogYAML, AddEntry, AddAlias, RemoveEntry, RemoveAlias)


def parse_commit(commit, *, verify=True) -> CatalogOp:
    msg = commit.message.strip()
    for op_type in _OP_TYPES:
        if (op := op_type.from_commit(commit)) is not None:
            if verify:
                op.verify_commit(commit)
            return op
    return UnknownOp(
        message=msg,
        hexsha=commit.hexsha,
        commit_metadata=CommitMetadata.from_commit(commit),
    )


# -- Replayer -----------------------------------------------------------------


def _parse_catalog_ops(catalog, *, verify=True) -> tuple[CatalogOp, ...]:
    commits = tuple(catalog.repo.iter_commits(reverse=True))
    return tuple(parse_commit(c, verify=verify) for c in commits)


@frozen(hash=False)
class Replayer:
    from_catalog: object = field()  # Catalog — avoid import-time dep
    verify: bool = field(default=True)
    rebuild: bool = field(default=False)

    @cached_property
    def ops(self) -> tuple[CatalogOp, ...]:
        return _parse_catalog_ops(self.from_catalog, verify=self.verify)

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for op in self.ops:
            key = type(op).__name__
            counts[key] = counts.get(key, 0) + 1
        return counts

    def print_plan(self) -> None:
        header = "--- rebuild plan ---" if self.rebuild else "--- replay plan ---"
        print(header)
        for op in self.ops:
            print(op)
        print("\n--- summary ---")
        for key, count in self.summary.items():
            print(f"  {key}: {count}")
        print(f"  total commits: {len(self.ops)}")

    def replay(self, to_catalog, *, preserve_commits=True) -> None:
        remap: dict[str, str] = {}
        for op in self.ops:
            ctx = (
                op.commit_metadata.git_env()
                if preserve_commits and op.commit_metadata is not None
                else nullcontext()
            )
            with ctx:
                op.do(self.from_catalog, to_catalog, rebuild=self.rebuild, remap=remap)
        to_catalog.assert_consistency()
