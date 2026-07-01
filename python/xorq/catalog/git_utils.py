from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from git import IndexFile, Repo
from toolz import curry

from xorq.catalog.constants import DEFAULT_REMOTE


@contextmanager
@curry
def commit_context(repo: Repo, message: str) -> Iterator[IndexFile]:
    """Commit whatever the wrapped body stages against *repo*'s index.

    The single commit primitive for the catalog (``CatalogBackend.commit_context``
    delegates here binding ``self.repo``, and the root/submodule wiring calls it
    directly). An empty commit is skipped: a no-op mutation -- e.g. re-adding an
    alias that already resolves to its target -- stages nothing against HEAD, and
    committing anyway would leave an empty commit in the catalog history. Every
    git-tracked catalog mutation funnels through ``repo.index.add``/``remove``
    (see the backend ``stage``/``stage_content``/``stage_unlink`` methods), so
    ``index.diff(HEAD)`` faithfully reflects whether anything changed.
    """
    yield repo.index
    if not repo.head.is_valid() or repo.index.diff(repo.head.commit):
        repo.index.commit(message)


def add_as_submodule(repo, subrepo, remote=DEFAULT_REMOTE):
    strpath = "./" + str(Path(subrepo.working_dir).relative_to(repo.working_dir))
    match subrepo.remotes:
        case ():
            # local
            url = strpath
        case remotes if remote in remotes:
            (url, *rest) = subrepo.remote(remote).urls
            if rest:
                raise ValueError(
                    f"remote {remote!r} has multiple URLs: {list(subrepo.remote(remote).urls)}"
                )
        case _:
            raise ValueError(f"no remote named {remote!r} found in subrepo")
    repo.git.submodule("add", url, strpath)
