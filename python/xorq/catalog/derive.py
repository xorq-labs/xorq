"""Catalog a build derived from an existing entry, and move aliases onto it.

The add half of a command that turns one entry into another: the build is
cataloged, the aliases the caller asked for are moved, and a failed move rolls
everything back. The sync is split into its pull and its push, so a caller can
tell a pull that wrote nothing from a push that failed after a local commit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from attr import field, frozen
from attr.validators import deep_iterable, instance_of

from xorq.catalog.catalog import Catalog, CatalogAlias, CatalogEntry
from xorq.catalog.exceptions import (
    AliasRefusedError,
    PullError,
    PushError,
    RollbackError,
)
from xorq.common.utils.logging_utils import get_logger


logger = get_logger(__name__)


@frozen
class Derived:
    """``created`` is false when ``new_entry`` was already cataloged."""

    new_entry = field(validator=instance_of(CatalogEntry))
    moved_aliases = field(
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )
    created = field(validator=instance_of(bool))
    # False without a sync or a remote; a failed push raises `PushError`.
    pushed = field(validator=instance_of(bool))
    # Registered by the derivation: an `alias` that was on no entry.
    added_aliases = field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )


def alias_targets(catalog: Catalog, aliases: Iterable[str]) -> dict[str, str | None]:
    """The entry each of ``aliases`` points at, ``None`` for an unregistered one."""
    registered = set(catalog.list_aliases())
    return {
        alias: CatalogAlias.from_name(alias, catalog).catalog_entry.name
        if alias in registered
        else None
        for alias in aliases
    }


def plan_aliases(
    old_entry: CatalogEntry,
    alias: str | None,
    move_aliases: bool,
    only_aliases: tuple[str, ...],
    new_name: str | None = None,
) -> tuple[tuple[str, ...], dict[str, str | None]]:
    """The aliases to move onto the new entry; refuses any the request can't have.

    ``move_aliases`` moves every alias ``old_entry`` has now, ``only_aliases``
    just these, each of which must be on it. ``alias`` must be unregistered or
    on ``old_entry``: one on another entry is refused, not taken. An alias
    already on ``new_name`` is where it was asked to be. Run after the sync's
    pull, which can move an alias, or with no ``new_name`` when nothing is
    derived.

    Returned with them: the entry each of them and ``alias`` points at now
    (``None`` for an unregistered one), for a rollback to restore.
    """
    here = {old_entry.name, new_name} - {None}
    added = (alias,) if alias else ()
    targets = alias_targets(old_entry.catalog, (*only_aliases, *added))
    if lacking := [name for name in only_aliases if targets[name] not in here]:
        raise AliasRefusedError(
            f"{old_entry.name} has no alias {', '.join(sorted(lacking))}"
        )
    if alias and targets[alias] not in {None, *here}:
        # `add-alias` moves it in one commit; removing it first would leave
        # it resolving to nothing until a later derivation lands.
        raise AliasRefusedError(
            f"alias {alias!r} points at {targets[alias]}; pick another name, "
            "or leave it out and then run "
            f"`xorq catalog add-alias {new_name} {alias}`"
        )
    if move_aliases:
        moving = tuple(catalog_alias.alias for catalog_alias in old_entry.aliases)
        # Each was found on `old_entry`; no need to read it again.
        return moving, {**targets, **dict.fromkeys(moving, old_entry.name)}
    return only_aliases, targets


def check_entry_alias(old_entry: CatalogEntry, entry_alias: str | None) -> None:
    """Refuse if the pull took ``entry_alias``, which named ``old_entry``, off it.

    The derivation would otherwise start from an entry the name no longer means.
    """
    if not entry_alias:
        return
    old = old_entry.name
    now = alias_targets(old_entry.catalog, (entry_alias,))[entry_alias]
    if now != old:
        raise AliasRefusedError(
            f"alias {entry_alias!r} points at {now or 'nothing'} after the pull, "
            f"not {old}; rerun with {old} to use {old}"
        )


def left_state(catalog: Catalog, name: str) -> str:
    """What a failed rollback left of ``name``: whether it is cataloged, and
    which aliases point at it. Best effort: it reads a catalog that just failed.
    """
    try:
        if not catalog.contains(name):
            return f"{name} not cataloged"
        on_it = sorted(
            alias
            for alias, target in alias_targets(catalog, catalog.list_aliases()).items()
            if target == name
        )
    except Exception as e:
        return f"state of {name} unknown ({type(e).__name__}: {e})"
    return f"{name} cataloged, aliases [{', '.join(on_it)}] -> {name}"


def roll_back(
    new_entry: CatalogEntry, prior: Mapping[str, str | None], added: bool
) -> None:
    """Remove ``new_entry`` if ``added``, and point each alias back at ``prior``.

    An alias ``prior`` maps to ``None`` did not exist, and is removed.
    """
    catalog = new_entry.catalog
    # First: removing the entry takes the aliases on it along.
    if added:
        catalog.remove(new_entry.name, sync=False)
    for alias, target in prior.items():
        if target is not None:
            if target != new_entry.name:
                catalog.add_alias(target, alias, sync=False)
        elif alias in catalog.list_aliases():
            catalog.remove_alias(alias, sync=False)


def add_and_move(
    old_entry: CatalogEntry,
    build_path: Path,
    alias: str | None,
    moving: tuple[str, ...],
    prior: Mapping[str, str | None],
) -> Derived:
    """Catalog ``build_path`` and move ``moving`` onto it, all or nothing.

    ``prior`` is where each of ``moving`` and ``alias`` points now, from
    ``plan_aliases``: ``catalog.add`` and ``add_alias`` overwrite an alias, so
    a rollback restores it.
    """
    catalog = old_entry.catalog
    added_aliases = (alias,) if alias else ()
    # A derivation can land on an entry that already exists (an earlier one
    # of the same entry, or one the pull brought in); a rollback must not
    # remove that one.
    created = not catalog.contains(build_path.name)
    # `catalog.add` moves `alias` itself; one already on the new entry stays.
    moving = tuple(
        name for name in moving if name != alias and prior[name] != build_path.name
    )
    new_entry = catalog.add(
        build_path, sync=False, aliases=added_aliases, exist_ok=True
    )
    moved = [alias] if alias and prior[alias] == old_entry.name else []
    added = (alias,) if alias and prior[alias] is None else ()
    attempted = []
    try:
        for name in moving:
            # Recorded before the call: a move that fails after writing the
            # symlink must be restored too.
            attempted.append(name)
            catalog.add_alias(new_entry.name, name, sync=False)
            moved.append(name)
    except Exception as e:
        touched = (*added_aliases, *attempted)
        try:
            roll_back(new_entry, {name: prior[name] for name in touched}, created)
        except Exception as rollback_error:
            logger.exception("rollback failed for %s", new_entry.name)
            state = left_state(catalog, new_entry.name)
            raise RollbackError(
                f"{type(e).__name__}: {e}; rollback failed: "
                f"{type(rollback_error).__name__}: {rollback_error}; left {state}",
                state,
            ) from e
        raise
    return Derived(new_entry, moved, created, pushed=False, added_aliases=added)


def add_derived(
    old_entry: CatalogEntry,
    build_path: Path,
    *,
    alias: str | None = None,
    move_aliases: bool = False,
    only_aliases: tuple[str, ...] = (),
    sync: bool = True,
    entry_alias: str | None = None,
) -> Derived:
    """Catalog ``build_path`` as derived from ``old_entry``, all or nothing.

    ``entry_alias`` is the alias ``old_entry`` was named by, if any; a pull
    that moves or removes it is refused.

    With ``sync``, pulls first and pushes after, as ``maybe_synchronizing``
    does, but raises ``PullError`` for a pull (nothing written) and
    ``PushError`` for a push (committed locally). Raises ``AliasRefusedError``
    after the pull and before any write, ``RollbackError`` if a failed alias
    move could not be undone.
    """
    catalog = old_entry.catalog
    if sync:
        try:
            catalog.pull()
        except Exception as e:
            raise PullError(f"pull failed: {type(e).__name__}: {e}") from e
    # Checked after the pull, which can move an alias: a refusal leaves what
    # the pull merged, and nothing else.
    check_entry_alias(old_entry, entry_alias)
    moving, prior = plan_aliases(
        old_entry, alias, move_aliases, only_aliases, build_path.name
    )
    derived = add_and_move(old_entry, build_path, alias, moving, prior)
    if not sync:
        return derived
    try:
        # `()` when the catalog has no remote: nothing was pushed.
        pushed = bool(catalog.push())
    except Exception as e:
        raise PushError(f"push failed: {type(e).__name__}: {e}", derived) from e
    return Derived(
        derived.new_entry,
        derived.moved_aliases,
        derived.created,
        pushed,
        added_aliases=derived.added_aliases,
    )
