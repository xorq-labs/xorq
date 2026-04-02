from contextlib import contextmanager
from pathlib import Path

from toolz import curry

from xorq.catalog.constants import DEFAULT_REMOTE


@contextmanager
@curry
def commit_context(repo, message):
    yield repo.index
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
