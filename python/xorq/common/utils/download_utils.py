from pathlib import Path
from shutil import move
from tarfile import TarFile
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve


def download_github_archive(org, repo, branch, suffix=".tar.gz", target=None):
    archive_name = f"{branch}{suffix}"
    target = Path(target or archive_name)
    assert not target.exists()
    archive_url = f"https://github.com/{org}/{repo}/archive/refs/heads/{archive_name}"
    _, _ = urlretrieve(archive_url, target)
    return target


def extract_tar(source, target):
    (source, target) = map(Path, (source, target))
    assert not target.exists()
    with TarFile.open(source, mode="r:*") as tf:
        (first, *rest) = (member.name for member in tf.members)
        assert all(member.startswith(first) for member in rest)
        with TemporaryDirectory() as td:
            tf.extractall(td)
            move(Path(td).joinpath(first), target)
    assert target.exists()
    return target


def download_xorq_template(template, branch="main", target=None):
    return download_github_archive(
        org="xorq-labs",
        repo=f"xorq-template-{template}",
        branch=branch,
        target=target,
    )


def download_unpacked_xorq_template(target, template, branch="main"):
    target = Path(target)
    if target.exists():
        raise ValueError(
            f"download_unpacked_xorq_template: target `{target}` already exists"
        )
    with TemporaryDirectory() as td:
        archive_target = Path(td).joinpath("repo.tar.gz")
        download_xorq_template(
            template=template,
            branch=branch,
            target=archive_target,
        )
        extract_tar(archive_target, target)
        return target
