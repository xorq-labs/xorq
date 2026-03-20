#!/usr/bin/env python
"""Set up a local xorq Catalog backed by minio S3.

Discovers the minio container IP from Docker, creates a catalog repo with
git-annex initialized and an S3 special remote pointing at minio, then
prints the catalog path so other tools can use it.

Usage:
    python -m xorq.catalog.scripts.setup_minio_catalog [--name NAME] [--bucket BUCKET]

Requires:
    - docker compose minio service running  (docker compose up -d minio)
    - git-annex on PATH                     (uv pip install git-annex)
"""

import argparse
import json
import subprocess
import sys

from git import Repo

from xorq.catalog.annex import Annex, GitAnnex, S3RemoteConfig
from xorq.catalog.catalog import Catalog


MINIO_CONTAINER = "xorq-minio-1"
MINIO_PORT = "9000"
MINIO_USER = "accesskey"
MINIO_PASSWORD = "secretkey"


def get_minio_ip():
    result = subprocess.run(
        [
            "docker",
            "inspect",
            MINIO_CONTAINER,
            "--format",
            "{{range $k,$v := .NetworkSettings.Networks}}{{$v.IPAddress}}{{end}}",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"error: cannot inspect {MINIO_CONTAINER} — is it running?\n"
            f"  docker compose up -d minio",
            file=sys.stderr,
        )
        raise SystemExit(1)
    ip = result.stdout.strip()
    if not ip:
        print(f"error: {MINIO_CONTAINER} has no IP address", file=sys.stderr)
        raise SystemExit(1)
    return ip


def check_minio_health(host, port):
    result = subprocess.run(
        ["curl", "-sf", f"http://{host}:{port}/minio/health/live"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(
            f"error: minio at {host}:{port} not healthy\n  docker compose up -d minio",
            file=sys.stderr,
        )
        raise SystemExit(1)


def setup_catalog(name, bucket):
    host = get_minio_ip()
    check_minio_health(host, MINIO_PORT)

    remote_config = S3RemoteConfig.make_minio_remote(
        name="minio",
        bucket=bucket,
        host=host,
        aws_access_key_id=MINIO_USER,
        aws_secret_access_key=MINIO_PASSWORD,
        port=MINIO_PORT,
    )

    catalog_path = Catalog.by_name_base_path / name
    if catalog_path.exists():
        print(f"error: catalog already exists at {catalog_path}", file=sys.stderr)
        print("  run teardown first, or pick a different --name", file=sys.stderr)
        raise SystemExit(1)

    # create repo + initial commit
    catalog_path.mkdir(parents=True)
    repo = Repo.init(catalog_path)
    repo.index.commit("initial commit")

    # allow private IPs (minio is on a docker bridge network)
    subprocess.run(
        ["git", "config", "annex.security.allowed-ip-addresses", "all"],
        cwd=catalog_path,
        check=True,
    )

    # init git-annex + S3 remote
    Annex.init_repo_path(catalog_path, external_remote_config=remote_config)

    annex = Annex(repo_path=catalog_path, env=remote_config.env)
    git_annex = GitAnnex(repo=repo, annex=annex)
    catalog = Catalog(git_annex=git_annex)

    # persist remote config (without secrets) to catalog.yaml
    catalog.set_remote_config(remote_config)

    info = {
        "name": name,
        "path": str(catalog_path),
        "bucket": bucket,
        "minio_host": host,
        "minio_port": MINIO_PORT,
        "remote_name": "minio",
    }
    # persist config for teardown
    catalog_path.joinpath(".minio-catalog.json").write_text(
        json.dumps(info, indent=2) + "\n"
    )

    print(json.dumps(info, indent=2))
    return catalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="minio-catalog",
        help="Catalog name (default: minio-catalog)",
    )
    parser.add_argument(
        "--bucket",
        default="xorq-catalog",
        help="S3 bucket name (default: xorq-catalog)",
    )
    args = parser.parse_args()
    setup_catalog(args.name, args.bucket)


if __name__ == "__main__":
    main()
