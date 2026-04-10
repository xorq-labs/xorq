#!/usr/bin/env python
"""Initialize a new xorq Catalog with a git-annex remote.

Supports directory, S3, and GCS remote types.  For S3/GCS, credentials
can be passed via CLI flags or XORQ_CATALOG_S3_* environment variables.

Usage:
    # directory remote
    python -m xorq.catalog.scripts.init_catalog_with_remote \
        --name my-catalog --remote-type directory --directory /tmp/store

    # S3 remote
    python -m xorq.catalog.scripts.init_catalog_with_remote \
        --name my-catalog --remote-type s3 --bucket my-bucket \
        --aws-access-key-id AKID --aws-secret-access-key SECRET

    # GCS remote (uses S3-compatible interop API)
    python -m xorq.catalog.scripts.init_catalog_with_remote \
        --name my-catalog --remote-type gcs --bucket my-bucket \
        --aws-access-key-id HMAC_KEY --aws-secret-access-key HMAC_SECRET

    # S3 remote (credentials from env)
    export XORQ_CATALOG_S3_AWS_ACCESS_KEY_ID=AKID
    export XORQ_CATALOG_S3_AWS_SECRET_ACCESS_KEY=SECRET
    python -m xorq.catalog.scripts.init_catalog_with_remote \
        --name my-catalog --remote-type s3 --bucket my-bucket

Requires:
    - git-annex on PATH  (uv pip install git-annex)
"""

import argparse
import json
import sys

from xorq.catalog.annex import DirectoryRemoteConfig, S3RemoteConfig
from xorq.catalog.catalog import Catalog


def build_directory_remote(args):
    if not args.directory:
        print("error: --directory is required for directory remotes", file=sys.stderr)
        raise SystemExit(1)
    return DirectoryRemoteConfig(
        name=args.remote_name,
        directory=args.directory,
    )


def _require_bucket(args):
    if not args.bucket:
        print("error: --bucket is required for S3/GCS remotes", file=sys.stderr)
        raise SystemExit(1)


def _get_s3_credentials(args):
    key_id = args.aws_access_key_id
    secret = args.aws_secret_access_key
    if key_id and secret:
        return key_id, secret
    return None, None


def build_s3_remote(args):
    _require_bucket(args)
    key_id, secret = _get_s3_credentials(args)
    if key_id and secret:
        kwargs = {}
        for attr in ("host", "port", "protocol", "region"):
            value = getattr(args, attr, None)
            if value is not None:
                kwargs[attr] = value
        return S3RemoteConfig(
            name=args.remote_name,
            bucket=args.bucket,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
            **kwargs,
        )
    # fall back to environment variables
    try:
        return S3RemoteConfig.from_env(
            name=args.remote_name,
            bucket=args.bucket,
        )
    except Exception as exc:
        print(
            f"error: S3 credentials not provided and env lookup failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


def build_gcs_remote(args):
    _require_bucket(args)
    key_id, secret = _get_s3_credentials(args)
    if key_id and secret:
        return S3RemoteConfig.make_gcs_remote(
            name=args.remote_name,
            bucket=args.bucket,
            aws_access_key_id=key_id,
            aws_secret_access_key=secret,
        )
    # fall back to environment variables
    try:
        return S3RemoteConfig.from_env(
            name=args.remote_name,
            bucket=args.bucket,
            host="storage.googleapis.com",
            protocol="https",
            requeststyle="path",
        )
    except Exception as exc:
        print(
            f"error: GCS credentials not provided and env lookup failed: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc


REMOTE_BUILDERS = {
    "directory": build_directory_remote,
    "s3": build_s3_remote,
    "gcs": build_gcs_remote,
}


def init_catalog(args):
    remote_config = REMOTE_BUILDERS[args.remote_type](args)

    catalog = Catalog.from_name(args.name, init=True, annex=remote_config)

    info = {
        "name": args.name,
        "path": str(catalog.repo_path),
        "remote_type": args.remote_type,
        "remote_name": args.remote_name,
    }
    print(json.dumps(info, indent=2))
    return catalog


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--name",
        default="default",
        help="Catalog name (default: default)",
    )
    parser.add_argument(
        "--remote-type",
        choices=("directory", "s3", "gcs"),
        required=True,
        help="Type of git-annex remote",
    )
    parser.add_argument(
        "--remote-name",
        default="origin-annex",
        help="Name for the git-annex special remote (default: origin-annex)",
    )

    # directory remote options
    parser.add_argument(
        "--directory",
        help="Local directory path (for directory remotes)",
    )

    # S3/GCS remote options
    parser.add_argument("--bucket", help="S3/GCS bucket name")
    parser.add_argument(
        "--aws-access-key-id",
        help="AWS access key ID (or GCS HMAC key)",
    )
    parser.add_argument(
        "--aws-secret-access-key",
        help="AWS secret access key (or GCS HMAC secret)",
    )
    parser.add_argument("--host", help="S3 host (for S3-compatible stores)")
    parser.add_argument("--port", help="S3 port")
    parser.add_argument("--protocol", help="S3 protocol (http/https)")
    parser.add_argument("--region", help="AWS region")

    args = parser.parse_args()
    init_catalog(args)


if __name__ == "__main__":
    main()
