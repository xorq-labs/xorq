"""The aliases a new catalog entry derived from an old one takes.

Shared by ``xorq catalog rebase``, ``pin`` and ``unpin``: one ``AliasPlan``
over one read of the catalog's aliases, holding every alias refusal, and one
all-or-nothing ``add_with_aliases`` that catalogs the build and points the
planned aliases at it, planning again after the sync's pull. ``AliasOutcome``
is what it reports back.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from attr import evolve, field, frozen
from attr.validators import deep_iterable, instance_of, optional

from xorq.catalog.catalog import Catalog, CatalogEntry
from xorq.catalog.enums import RebaseFailure
from xorq.catalog.exceptions import RebaseError
from xorq.common.utils.logging_utils import get_logger


logger = get_logger(__name__)


def str_tuple() -> Any:
    """An attrs field holding a tuple of ``str``, empty by default."""
    return field(
        default=(),
        converter=tuple,
        validator=deep_iterable(instance_of(str), instance_of(tuple)),
    )


@frozen
class AliasOutcome:
    """What became of the aliases once the new entry was in.

    Built by ``of`` from two ``AliasPlan``: ``before``, over the aliases the
    user saw, and ``landed``, over them after the sync's pull, which can move
    or remove one in between.
    """

    # Pointed at the new entry by this run: were on the old entry.
    moved = str_tuple()
    # Registered by this run: `alias`, which was on no entry.
    added = str_tuple()
    # Asked for, or seen on the old entry, and on the new entry already (a
    # pull or an earlier run put them there).
    already = str_tuple()
    # On the old entry before the sync's pull, on neither entry after it; only
    # with `move_aliases`, which takes what the old entry has once the pull is in.
    unmoved = str_tuple()
    # Still on the old entry, not asked for.
    left = str_tuple()

    @classmethod
    def of(cls, before: AliasPlan, landed: AliasPlan) -> AliasOutcome:
        # `move_aliases` takes what the old entry has after the pull; what the
        # user saw on it before is accounted for too.
        pulled = landed.pulled_away(before) if landed.move_aliases else ()
        return cls(
            moved=landed.moved,
            added=landed.added,
            already=(
                *landed.already,
                *(n for n in pulled if landed.target(n) == landed.new),
            ),
            unmoved=tuple(n for n in pulled if landed.target(n) != landed.new),
            left=landed.left,
        )


def alias_targets(catalog: Catalog) -> dict[str, str]:
    """Every registered alias and the entry it points at, in one read."""
    return {
        catalog_alias.alias: catalog_alias.catalog_entry.name
        for catalog_alias in catalog.catalog_aliases
    }


@frozen
class AliasPlan:
    """The aliases a new entry derived from ``old_entry`` takes, over one read of them.

    ``move_aliases`` takes every alias ``old_entry`` has, ``only_aliases`` just
    these, each of which must be on it; ``alias`` is registered besides, and
    must be unregistered or on ``old_entry``: one on another entry is refused,
    not taken. One already on ``new`` is where it was asked to be. Until the
    derivation knows its name, ``new`` is ``None`` and only an unregistered
    name is refused: an alias elsewhere may be on the new entry (a rerun
    after a failed push). ``entry_alias`` is the alias ``old_entry`` was named
    by, if any; the sync's pull moving it off ``old_entry`` is refused, since
    the name no longer means this entry.

    ``targets`` is the read: each registered alias and the entry it points at.
    ``pulled`` says the read came after the sync's pull: a name the read lacks
    is then one the pull removed, not one that was never registered, and is
    refused as such.
    """

    old_entry = field(validator=instance_of(CatalogEntry))
    targets = field(converter=dict, validator=instance_of(dict))
    new = field(default=None, validator=optional(instance_of(str)))
    alias = field(default=None, validator=optional(instance_of(str)))
    move_aliases = field(default=False, validator=instance_of(bool))
    only_aliases = str_tuple()
    entry_alias = field(default=None, validator=optional(instance_of(str)))
    pulled = field(default=False, validator=instance_of(bool))

    @classmethod
    def read(cls, old_entry: CatalogEntry, **kwargs: Any) -> AliasPlan:
        """The plan over the aliases as they are now, checked."""
        return cls(old_entry, alias_targets(old_entry.catalog), **kwargs).checked()

    def evolved(self, **changes: Any) -> AliasPlan:
        """``self`` with ``changes``, checked; ``targets`` to re-read after a pull."""
        return evolve(self, **changes).checked()

    @property
    def old(self) -> str:
        return self.old_entry.name

    @property
    def here(self) -> frozenset[str]:
        return frozenset({self.old, self.new} - {None})

    def target(self, name: str) -> str | None:
        """The entry ``name`` points at, ``None`` for an unregistered one."""
        return self.targets.get(name)

    @property
    def on_old(self) -> tuple[str, ...]:
        return tuple(
            sorted(name for name, target in self.targets.items() if target == self.old)
        )

    @property
    def asked(self) -> tuple[str, ...]:
        """The aliases to end up on ``new``, ``alias`` first."""
        named = self.on_old if self.move_aliases else self.only_aliases
        return tuple(dict.fromkeys((*((self.alias,) if self.alias else ()), *named)))

    def checked(self) -> AliasPlan:
        """``self``; raises ``RebaseError`` for an alias the request can't have."""
        if unknown := tuple(n for n in self.only_aliases if self.target(n) is None):
            raise RebaseError(
                "; ".join(
                    f"alias {n!r} points at nothing after the pull, not {self.old}"
                    for n in unknown
                )
                if self.pulled
                else f"{self.old} has no alias {', '.join(sorted(unknown))}",
                RebaseFailure.REFUSED,
            )
        if self.new is None:
            return self
        here = self.here
        if elsewhere := tuple(
            n for n in self.only_aliases if self.target(n) not in here
        ):
            # Says where it went.
            raise RebaseError(
                "; ".join(
                    f"alias {n!r} points at {self.target(n)}, not {self.old}"
                    for n in elsewhere
                ),
                RebaseFailure.REFUSED,
            )
        if self.alias and self.target(self.alias) not in {None, *here}:
            # `add-alias` moves it in one commit; removing it first would leave
            # it resolving to nothing until the new entry lands.
            raise RebaseError(
                f"alias {self.alias!r} points at {self.target(self.alias)}; pick "
                "another name, or leave it off and then run "
                f"`xorq catalog add-alias {self.new} {self.alias}`",
                RebaseFailure.REFUSED,
            )
        if self.entry_alias and self.target(self.entry_alias) not in here:
            now = self.target(self.entry_alias)
            raise RebaseError(
                f"alias {self.entry_alias!r} points at {now or 'nothing'} after "
                f"the pull, not {self.old}; rerun with {self.old}",
                RebaseFailure.REFUSED,
            )
        return self

    @property
    def moving(self) -> tuple[str, ...]:
        """To point at ``new`` after the add, which registers ``alias`` itself."""
        return tuple(
            n for n in self.asked if n != self.alias and self.target(n) != self.new
        )

    @property
    def derives(self) -> bool:
        """Whether ``new`` is another entry than ``old``.

        A pin of an entry with no caches hashes to the entry itself: nothing
        moves onto it and nothing is left behind on it.
        """
        return self.new != self.old

    @property
    def moved(self) -> tuple[str, ...]:
        if not self.derives:
            return ()
        return tuple(n for n in self.asked if self.target(n) == self.old)

    @property
    def added(self) -> tuple[str, ...]:
        return (self.alias,) if self.alias and self.target(self.alias) is None else ()

    @property
    def already(self) -> tuple[str, ...]:
        return tuple(n for n in self.asked if self.target(n) == self.new)

    @property
    def left(self) -> tuple[str, ...]:
        if not self.derives:
            return ()
        return tuple(n for n in self.on_old if n not in self.asked)

    def pulled_away(self, before: AliasPlan) -> tuple[str, ...]:
        """Of the aliases ``before`` saw on the old entry, the ones not asked now.

        With ``move_aliases``, the pull between the two plans took them off it.
        """
        return tuple(n for n in before.on_old if n not in self.asked)


def roll_back(
    catalog: Catalog, name: str, prior: Mapping[str, str | None], added: bool
) -> None:
    """Remove entry ``name`` if ``added``, and point each alias back at ``prior``.

    An alias ``prior`` maps to ``None`` did not exist, and is removed. Logged,
    not raised, so the error that caused it is the one that surfaces.
    """
    try:
        # First: removing the entry takes the aliases on it along. An add that
        # failed part-way left some of the entry's pieces; `remove` takes
        # whichever are there.
        pieces = CatalogEntry(name, catalog, require_exists=False)._exists_components  # noqa: SLF001  # xorq-style: disable=protected-access
        if added and any(pieces.values()):
            catalog.remove(name, sync=False)
        for alias, target in prior.items():
            if target is not None:
                if target != name:
                    catalog.add_alias(target, alias, sync=False)
            elif alias in catalog.list_aliases():
                catalog.remove_alias(alias, sync=False)
    except Exception:
        logger.exception("rollback failed for %s", name)


def add_with_aliases(
    plan: AliasPlan, build_path: Path, pulled: bool
) -> tuple[CatalogEntry, AliasOutcome]:
    """Catalog ``build_path`` and point ``plan``'s aliases at it, all or nothing.

    Returns the new entry and what became of the aliases, planned again here,
    after the sync's pull (``pulled``), which can move one. The caller owns the
    sync: it pulls before and pushes after, and decides what a failed push is.
    """
    catalog = plan.old_entry.catalog
    name = build_path.name
    # Planned again after the pull: a refusal here still comes before the
    # first write of our own, though after whatever the pull merged.
    landed = plan.evolved(new=name, targets=alias_targets(catalog), pulled=pulled)
    # Read after the pull, which can bring in the entry (an earlier
    # derivation of the same one); a rollback must not remove that one.
    added = not catalog.contains(name)
    # `catalog.add` and `add_alias` overwrite an alias, so each is recorded
    # before the call that writes it, for its prior target to be restored.
    registered = (landed.alias,) if landed.alias else ()
    attempted = [*registered]
    try:
        new_entry = catalog.add(
            build_path, sync=False, aliases=registered, exist_ok=True
        )
        for alias in landed.moving:
            attempted.append(alias)
            catalog.add_alias(name, alias, sync=False)
    except Exception:
        roll_back(catalog, name, {n: landed.target(n) for n in attempted}, added)
        raise
    return new_entry, AliasOutcome.of(plan, landed)
