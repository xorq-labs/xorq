"""Parse a catalog's git log into structured operations and optionally replay them.

Parses the four commit-message formats produced by xorq's Catalog API:
  - "add: {hash} (aliases {a1}, {a2}, ...)"
  - "add alias: {alias} -> {entry}"
  - "rm: {hash} (aliases {a1}, {a2}, ...)"
  - "rm alias: {alias}"

The :class:`Replayer` replays these operations in order against a target
catalog, reproducing the source catalog's state.
"""

from __future__ import annotations

import os
import re
from contextlib import contextmanager
from functools import cached_property

from attr import field, frozen, validators

from xorq.catalog.constants import CATALOG_YAML_NAME, CatalogInfix


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

    def do(self, from_catalog, to_catalog):
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

    def do(self, from_catalog, to_catalog):
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

    def do(self, from_catalog, to_catalog):
        catalog_entry = from_catalog.get_catalog_entry(self.entry_hash)
        to_catalog.add(
            catalog_entry.catalog_path,
            sync=False,
            aliases=self.aliases,
            exist_ok=True,
        )

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

    def do(self, from_catalog, to_catalog):
        to_catalog.add_alias(self.entry_name, self.alias, sync=False)

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

    def do(self, from_catalog, to_catalog):
        to_catalog.remove(self.entry_name, sync=False)

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

    def do(self, from_catalog, to_catalog):
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

    def do(self, from_catalog, to_catalog):
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
        for op in self.ops:
            print(op)
        print("\n--- summary ---")
        for key, count in self.summary.items():
            print(f"  {key}: {count}")
        print(f"  total commits: {len(self.ops)}")

    def replay(self, to_catalog, *, preserve_commits=True) -> None:
        for op in self.ops:
            if preserve_commits and op.commit_metadata is not None:
                with op.commit_metadata.git_env():
                    op.do(self.from_catalog, to_catalog)
            else:
                op.do(self.from_catalog, to_catalog)
        to_catalog.assert_consistency()
