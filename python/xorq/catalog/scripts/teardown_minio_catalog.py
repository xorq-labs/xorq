#!/usr/bin/env python
"""Tear down a local minio-backed xorq Catalog.

Drops all git-annex content, uninits annex, removes the catalog directory,
and deletes the S3 bucket from minio.

Usage:
    python -m xorq.catalog.scripts.teardown_minio_catalog [--name NAME]

Requires:
    - docker compose minio service running  (for bucket cleanup)
"""

import argparse
import json
import shutil
import subprocess
import sys

from xorq.catalog.annex import Annex, S3RemoteConfig
from xorq.catalog.catalog import Catalog


def delete_bucket(host, port, bucket, access_key, secret_key):
    """Delete all objects in the bucket, then remove it via the minio mc CLI inside the container."""
    # use mc inside the minio container — it's already configured as "data"
    cmds = [
        f"mc rb --force data/{bucket}",
    ]
    for cmd in cmds:
        result = subprocess.run(
            ["docker", "exec", "xorq-minio-1", "bash", "-c", cmd],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            # bucket may already be gone
            print(f"  warning: {cmd!r} → {result.stderr.strip()}", file=sys.stderr)


def teardown_catalog(name):
    catalog_path = Catalog.by_name_base_path / name

    if not catalog_path.exists():
        print(f"error: no catalog at {catalog_path}", file=sys.stderr)
        raise SystemExit(1)

    config_path = catalog_path / ".minio-catalog.json"
    if config_path.exists():
        info = json.loads(config_path.read_text())
    else:
        info = {}
        print(
            "warning: no .minio-catalog.json found, skipping bucket cleanup",
            file=sys.stderr,
        )

    # drop annex content + uninit
    annex_path = catalog_path / ".git" / "annex"
    if annex_path.exists():
        env = None
        if info:
            env = S3RemoteConfig.make_minio_remote(
                name=info.get("remote_name", "minio"),
                bucket=info["bucket"],
                host=info["minio_host"],
                aws_access_key_id="accesskey",
                aws_secret_access_key="secretkey",
                port=info.get("minio_port", "9000"),
            ).env
        annex = Annex(repo_path=catalog_path, env=env)
        try:
            annex.teardown()
            print("  annex content dropped and uninited")
        except (AssertionError, OSError) as e:
            print(f"  warning: annex teardown: {e}", file=sys.stderr)

    # delete S3 bucket
    if info:
        delete_bucket(
            host=info["minio_host"],
            port=info.get("minio_port", "9000"),
            bucket=info["bucket"],
            access_key="accesskey",
            secret_key="secretkey",
        )
        print(f"  bucket {info['bucket']} deleted")

    # remove catalog directory
    shutil.rmtree(catalog_path)
    print(f"  {catalog_path} removed")
    print(f"teardown complete: {name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--name",
        default="minio-catalog",
        help="Catalog name (default: minio-catalog)",
    )
    args = parser.parse_args()
    teardown_catalog(args.name)


if __name__ == "__main__":
    main()
