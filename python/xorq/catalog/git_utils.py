from contextlib import contextmanager
from pathlib import Path

from toolz import curry


@contextmanager
@curry
def commit_context(repo, message):
    yield repo.index
    repo.index.commit(message)


def add_as_submodule(repo, subrepo, remote="origin"):
    strpath = "./" + str(Path(subrepo.working_dir).relative_to(repo.working_dir))
    match subrepo.remotes:
        case ():
            # local
            url = strpath
        case remotes if remote in remotes:
            (url, *rest) = subrepo.remote(remote).urls
            if rest:
                raise ValueError
        case _:
            raise ValueError
    repo.git.submodule("add", url, strpath)
