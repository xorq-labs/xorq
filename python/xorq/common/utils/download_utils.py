import zipfile
from pathlib import Path
from shutil import move
from tempfile import TemporaryDirectory
from urllib.request import urlretrieve

from xorq.init_templates import InitTemplates


def download_github_archive(org, repo, branch, suffix=".zip", target=None):
    target = Path(target or f"{branch}{suffix}")
    assert not target.exists()
    archive_url = f"https://github.com/{org}/{repo}/archive/{branch}.zip"
    _, _ = urlretrieve(archive_url, target)
    return target


def extract_zip(source, target):
    (source, target) = map(Path, (source, target))
    assert not target.exists()
    with zipfile.ZipFile(source, "r") as zf:
        names = zf.namelist()
        (first, *rest) = names
        # strip trailing slash from directory entry
        first = first.rstrip("/")
        assert all(member.startswith(first) for member in rest)
        with TemporaryDirectory() as td:
            zf.extractall(td)
            move(Path(td).joinpath(first), target)
    assert target.exists()
    return target


def download_xorq_template(template, branch=None, target=None):
    branch = branch or InitTemplates.get_default_branch(template)
    return download_github_archive(
        org="xorq-labs",
        repo=f"xorq-template-{template}",
        branch=branch,
        target=target,
    )


def download_unpacked_xorq_template(target, template, branch=None):
    target = Path(target)
    if target.exists():
        raise ValueError(
            f"download_unpacked_xorq_template: target `{target}` already exists"
        )
    with TemporaryDirectory() as td:
        archive_target = Path(td).joinpath("repo.zip")
        download_xorq_template(
            template=template,
            branch=branch,
            target=archive_target,
        )
        extract_zip(archive_target, target)
        return target
