from pathlib import Path
from types import SimpleNamespace

import yaml12
from git import PushInfo, Remote, Repo

from xorq.catalog.catalog import Catalog
from xorq.catalog.cli import cli
from xorq.catalog.constants import CONTENT_STORE_YAML
from xorq.catalog.replay import Replayer


CATALOG_ID = "11111111-1111-4111-8111-111111111111"
SERVICE_URL = "https://catalog.example/"
REMOTE_URL = f"{SERVICE_URL}alice/demo.git"


def _set_hosted_env(monkeypatch) -> None:
    monkeypatch.setenv("XORQ_CONTENT_STORE_PRESIGNED_CATALOG_ID", CATALOG_ID)
    monkeypatch.setenv("XORQ_CONTENT_STORE_PRESIGNED_SERVICE_URL", SERVICE_URL)


def _invoke_init(runner, target: Path, *options: str):
    return runner.invoke(
        cli,
        ["--path", str(target), "init", "--content-store", "presigned", *options],
    )


def _invoke_replay(runner, source: Catalog, target: Path, *options: str):
    return runner.invoke(
        cli,
        [
            "--path",
            str(source.repo_path),
            "replay",
            str(target),
            "--content-store",
            "presigned",
            *options,
        ],
    )


def test_init_presigned_commits_only_public_config_and_sets_remote(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    target = tmp_path / "hosted"

    result = _invoke_init(runner, target, "--remote-url", REMOTE_URL)

    assert result.exit_code == 0, result.output
    assert yaml12.read_yaml(target / CONTENT_STORE_YAML) == {
        "type": "presigned",
        "catalog_id": CATALOG_ID,
        "service_url": SERVICE_URL,
    }
    repo = Repo(target)
    assert len(repo.remotes) == 1
    assert tuple(repo.remotes.origin.urls) == (REMOTE_URL,)
    catalog = Catalog.from_repo_path(target, init=False)
    assert catalog.backend.content_store.remote_url == REMOTE_URL


def test_init_presigned_requires_remote_before_creating_repo(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    target = tmp_path / "hosted"

    result = _invoke_init(runner, target)

    assert result.exit_code != 0
    assert "--remote-url is required" in result.output
    assert not target.exists()


def test_init_presigned_rejects_gcs(runner, tmp_path: Path, monkeypatch) -> None:
    _set_hosted_env(monkeypatch)
    target = tmp_path / "hosted"

    result = _invoke_init(runner, target, "--gcs", "--remote-url", REMOTE_URL)

    assert result.exit_code != 0
    assert "--gcs cannot be combined" in result.output
    assert not target.exists()


def test_init_presigned_rejects_remote_outside_service(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    target = tmp_path / "hosted"

    result = _invoke_init(
        runner,
        target,
        "--remote-url",
        "https://other.example/alice/demo.git",
    )

    assert result.exit_code != 0
    assert "does not match the catalog Git remote" in result.output
    assert not target.exists()


def test_replay_presigned_dry_run_does_not_require_remote(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    source = Catalog.from_repo_path(tmp_path / "source")
    target = tmp_path / "hosted"

    result = _invoke_replay(runner, source, target, "--dry-run")

    assert result.exit_code == 0, result.output
    assert not target.exists()


def test_replay_presigned_requires_remote_before_creating_target(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    source = Catalog.from_repo_path(tmp_path / "source")
    target = tmp_path / "hosted"

    result = _invoke_replay(runner, source, target)

    assert result.exit_code != 0
    assert "--remote-url is required" in result.output
    assert not target.exists()


def test_replay_sets_hosted_remote_before_materializing_store(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    source = Catalog.from_repo_path(tmp_path / "source")
    target = tmp_path / "hosted"
    observed = {}

    def inspect_target_then_stop(self, target_catalog, preserve_commits=True):
        observed["remote_urls"] = tuple(target_catalog.repo.remotes.origin.urls)
        observed["store_remote_url"] = target_catalog.backend.content_store.remote_url
        raise RuntimeError("stop before remote push")

    monkeypatch.setattr(Replayer, "replay", inspect_target_then_stop)

    result = _invoke_replay(
        runner,
        source,
        target,
        "--remote-url",
        REMOTE_URL,
    )

    assert result.exit_code != 0
    assert "stop before remote push" in result.output
    assert observed == {
        "remote_urls": (REMOTE_URL,),
        "store_remote_url": REMOTE_URL,
    }


def test_replay_reports_a_remote_hook_rejection(
    runner, tmp_path: Path, monkeypatch
) -> None:
    _set_hosted_env(monkeypatch)
    source = Catalog.from_repo_path(tmp_path / "source")
    target = tmp_path / "hosted"
    monkeypatch.setattr(Replayer, "replay", lambda *args, **kwargs: None)

    def reject_push(self, *args, **kwargs):
        return [
            SimpleNamespace(
                flags=PushInfo.REMOTE_REJECTED,
                local_ref="main",
                summary="remote rejected by pre-receive hook",
            )
        ]

    monkeypatch.setattr(Remote, "push", reject_push)

    result = _invoke_replay(
        runner,
        source,
        target,
        "--remote-url",
        REMOTE_URL,
    )

    assert result.exit_code != 0
    assert "remote rejected by pre-receive hook" in result.output
    assert "Pushed to" not in result.output
